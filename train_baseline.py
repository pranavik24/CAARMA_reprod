from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping, Optional
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from feature.build_feature import build_feature
from functions.dataset import load_audio
from model.model_build import build_model


SPLIT_COLUMNS = ("VoxCeleb1_ID", "VGGFace1_ID", "Gender", "Nationality", "Set")


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _none_like(value: object) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _resolve_path(value: object, base_dir: Optional[Path] = None) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def _ensure_root_suffix(root: object) -> str:
    root_text = str(root)
    return root_text if root_text.endswith(os.sep) else root_text + os.sep


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in (
        "dataset",
        "trial_path",
        "checkpoint_path",
        "save_dir",
        "vox1_wav_root",
        "vox1_split_dir",
        "train_split_csv",
        "val_split_csv",
        "test_split_csv",
        "root",
    ):
        if key in config and not _none_like(config[key]):
            config[key] = str(_resolve_path(config[key], config_path.parent))
    if "root" in config:
        config["root"] = _ensure_root_suffix(config["root"])
    return config


def is_voxceleb_split_csv(path: str | Path) -> bool:
    columns = pd.read_csv(path, nrows=0).columns
    return set(SPLIT_COLUMNS).issubset(set(columns))


def split_csv_path(config: Mapping[str, Any], split_name: str) -> Path:
    configured = config.get(f"{split_name}_split_csv")
    if configured:
        return Path(str(configured))
    split_dir = Path(str(config["vox1_split_dir"]))
    return split_dir / f"vox1_{split_name}.csv"


def speaker_wavs(wav_root: Path, speaker_id: str) -> list[Path]:
    return sorted((wav_root / speaker_id).rglob("*.wav"))


def examples_from_voxceleb_split(split_path: str | Path, wav_root: str | Path) -> pd.DataFrame:
    split = pd.read_csv(split_path)
    missing_columns = sorted(set(SPLIT_COLUMNS) - set(split.columns))
    if missing_columns:
        raise ValueError(f"{split_path} is missing columns: {', '.join(missing_columns)}")

    speakers = sorted(str(value).strip() for value in split["VoxCeleb1_ID"])
    label_by_speaker = {speaker_id: index for index, speaker_id in enumerate(speakers)}
    row_by_speaker = {
        str(row["VoxCeleb1_ID"]).strip(): row for _, row in split.iterrows()
    }
    wav_root_path = Path(wav_root)
    records = []
    missing = []
    for speaker_id in speakers:
        wav_paths = speaker_wavs(wav_root_path, speaker_id)
        if not wav_paths:
            missing.append(speaker_id)
            continue
        for wav_path in wav_paths:
            row = row_by_speaker[speaker_id]
            records.append(
                {
                    "utt_spk_int_labels": label_by_speaker[speaker_id],
                    "utt_spk_id": speaker_id,
                    "utt_paths": str(wav_path),
                    "gender": row["Gender"],
                    "nationality": row["Nationality"],
                    "split": row["Set"],
                }
            )
    if missing:
        preview = ", ".join(missing[:20])
        raise FileNotFoundError(f"No WAV files found for split speakers: {preview}")
    return pd.DataFrame.from_records(records)


def training_dataframe(config: Mapping[str, Any]) -> pd.DataFrame:
    dataset = config.get("dataset")
    if not _none_like(dataset):
        dataset_path = Path(str(dataset))
        if is_voxceleb_split_csv(dataset_path):
            return examples_from_voxceleb_split(dataset_path, config["vox1_wav_root"])
        return pd.read_csv(dataset_path)

    active_split = str(config.get("active_split", "train")).strip().lower()
    return examples_from_voxceleb_split(split_csv_path(config, active_split), config["vox1_wav_root"])


def relative_or_absolute(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def session_key(path: str | Path, wav_root: Path, speaker_id: str) -> str:
    path = Path(path)
    try:
        relative = path.relative_to(wav_root / speaker_id)
    except ValueError:
        relative = path
    return relative.parts[0] if relative.parts else str(path.parent)


def cross_session_pairs(paths: list[str], wav_root: Path, speaker_id: str, max_pairs: int):
    by_session: dict[str, str] = {}
    for path in sorted(paths):
        by_session.setdefault(session_key(path, wav_root, speaker_id), path)
    sessions = sorted(by_session)
    pairs = []
    for left_index, left_session in enumerate(sessions):
        for right_session in sessions[left_index + 1 :]:
            pairs.append((by_session[left_session], by_session[right_session]))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_trials_from_split(config: Mapping[str, Any], split_name: str) -> np.ndarray:
    wav_root = Path(str(config["vox1_wav_root"]))
    examples = examples_from_voxceleb_split(split_csv_path(config, split_name), wav_root)
    grouped = {
        speaker_id: sorted(group["utt_paths"].tolist())
        for speaker_id, group in examples.groupby("utt_spk_id")
    }
    pos_per_speaker = int(config.get("validation_pos_pairs_per_speaker", 1))
    neg_per_speaker = int(config.get("validation_neg_pairs_per_speaker", 1))
    max_speakers_value = config.get("validation_max_speakers", 300)
    max_speakers = int(max_speakers_value) if max_speakers_value else None

    positive_pairs = {
        speaker_id: pairs
        for speaker_id in sorted(grouped)
        if (
            pairs := cross_session_pairs(
                grouped[speaker_id],
                wav_root,
                speaker_id,
                max(pos_per_speaker, 1),
            )
        )
    }
    speakers = sorted(positive_pairs)
    if max_speakers and max_speakers > 0:
        speakers = speakers[:max_speakers]
    if len(speakers) < 2:
        raise ValueError("Generated validation trials require at least two eligible speakers.")

    records = []
    for speaker_id in speakers:
        for enroll_wav, test_wav in positive_pairs[speaker_id][:pos_per_speaker]:
            records.append(
                [
                    "1",
                    relative_or_absolute(enroll_wav, wav_root),
                    relative_or_absolute(test_wav, wav_root),
                ]
            )
    for left_index, left in enumerate(speakers):
        left_wav, _ = positive_pairs[left][0]
        for offset in range(1, min(neg_per_speaker, len(speakers) - 1) + 1):
            right = speakers[(left_index + offset) % len(speakers)]
            right_wav, _ = positive_pairs[right][0]
            records.append(
                [
                    "0",
                    relative_or_absolute(left_wav, wav_root),
                    relative_or_absolute(right_wav, wav_root),
                ]
            )
    return np.array(records, dtype=str)


def load_trials(config: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    trial_path = config.get("trial_path")
    if not _none_like(trial_path) and Path(str(trial_path)).exists():
        return np.loadtxt(str(trial_path), str), str(config["root"])
    if not _as_bool(config.get("generate_validation_trials", False)):
        raise FileNotFoundError(f"{trial_path} not found.")
    split_name = str(config.get("validation_split", "val"))
    return build_trials_from_split(config, split_name), str(config["vox1_wav_root"])


class BaselineTrainDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, second: int):
        self.paths = dataframe["utt_paths"].astype(str).tolist()
        self.labels = dataframe["utt_spk_int_labels"].astype(int).tolist()
        self.second = second
        print(f"Train Dataset load {len(set(self.labels))} speakers")
        print(f"Train Dataset load {len(self.paths)} utterance")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return {
            "waveform": load_audio(self.paths[index], self.second),
            "mapped_id": int(self.labels[index]),
            "path": self.paths[index],
        }


class BaselineEvalDataset(Dataset):
    def __init__(self, paths: np.ndarray, root: str):
        self.paths = [str(path) for path in paths]
        self.root = Path(root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = Path(self.paths[index])
        wav_path = path if path.is_absolute() else self.root / path
        return {
            "waveform": load_audio(str(wav_path), -1),
            "path": str(wav_path),
        }


class BaselineDataModule(LightningDataModule):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.train_df: Optional[pd.DataFrame] = None
        self.trials: Optional[np.ndarray] = None
        self.eval_root = str(self.config["root"])

    def setup(self, stage: Optional[str] = None):
        if stage in {None, "fit"}:
            self.train_df = training_dataframe(self.config)
            if _as_bool(self.config.get("derive_num_spk", True)):
                labels = self.train_df["utt_spk_int_labels"].astype(int)
                self.config["num_spk"] = int(labels.max() + 1)
        if stage in {None, "fit", "validate", "test"}:
            self.trials, self.eval_root = load_trials(self.config)

    def train_dataloader(self):
        if self.train_df is None:
            self.setup("fit")
        dataset = BaselineTrainDataset(self.train_df, int(self.config.get("second", 3)))
        return DataLoader(
            dataset,
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 4)),
            batch_size=int(self.config.get("batch_size", 200)),
            pin_memory=True,
            drop_last=_as_bool(self.config.get("train_drop_last", True)),
        )

    def val_dataloader(self):
        if self.trials is None:
            self.setup("validate")
        eval_path = np.unique(np.concatenate((self.trials.T[1], self.trials.T[2])))
        print(f"number of enroll: {len(set(self.trials.T[1]))}")
        print(f"number of test: {len(set(self.trials.T[2]))}")
        print(f"number of evaluation: {len(eval_path)}")
        return DataLoader(
            BaselineEvalDataset(eval_path, self.eval_root),
            num_workers=int(self.config.get("val_num_workers", self.config.get("num_workers", 4))),
            shuffle=False,
            batch_size=1,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class AMSoftmax(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        print(f"Initialised AM-Softmax m={margin:.3f} s={scale:.3f}")
        print(f"Embedding dim is {embedding_dim}, number of speakers is {num_classes}")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings_norm = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        weights_norm = self.weight / torch.norm(self.weight, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        cosine = torch.mm(embeddings_norm, weights_norm)
        label_view = labels.view(-1, 1).to(device=embeddings.device)
        margin = torch.zeros_like(cosine).scatter_(1, label_view, self.margin)
        logits = self.scale * (cosine - margin)
        loss = self.ce(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean() * 100.0
        return loss, accuracy


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    if np.any((fpr == 0.0) & (1.0 - tpr == 0.0)):
        return 0.0, thresholds[int(np.argmax(tpr))]
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return float(eer), float(threshold)


def compute_min_dcf(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for index in range(0, len(fnr)):
        c_det = c_miss * fnr[index] * p_target + c_fa * fpr[index] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[index]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(min_c_det / c_def), float(min_c_det_threshold)


class BaselineTask(LightningModule):
    def __init__(self, features, model, loss, config: Mapping[str, Any]):
        super().__init__()
        self.features = features
        self.model = model
        self.loss = loss
        self.config = dict(config)
        self.learning_rate = float(self.config["init_lr"])
        self.weight_decay = float(self.config["weight_decay"])
        self.eval_vectors: list[np.ndarray] = []
        self.index_mapping: dict[str, int] = {}
        self.trials: Optional[np.ndarray] = None
        self.eval_root = str(self.config["root"])

    def forward(self, waveform):
        return self.model(self.features(waveform))

    def configure_optimizers(self):
        optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": StepLR(optimizer, step_size=4, gamma=0.5),
        }

    def training_step(self, batch, batch_idx):
        embeddings = self(batch["waveform"])
        labels = batch["mapped_id"].long().to(embeddings.device)
        loss, accuracy = self.loss(embeddings, labels)
        self.log("am_loss", loss, prog_bar=True)
        self.log("acc", accuracy, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.eval_vectors = []
        self.index_mapping = {}
        self.trials, self.eval_root = load_trials(self.config)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            embedding = self(batch["waveform"])
        self.eval_vectors.append(embedding.detach().cpu().numpy()[0])
        self.index_mapping[batch["path"][0]] = batch_idx

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def trial_key(self, item: str) -> str:
        path = Path(item)
        return str(path if path.is_absolute() else Path(self.eval_root) / path)

    def similarity_scores(self):
        eval_vectors = np.vstack(self.eval_vectors)
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
        labels = []
        scores = []
        for item in self.trials:
            enroll_vector = eval_vectors[self.index_mapping[self.trial_key(item[1])]]
            test_vector = eval_vectors[self.index_mapping[self.trial_key(item[2])]]
            denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
            score = enroll_vector.dot(test_vector.T) / (denom + 1e-8)
            labels.append(int(item[0]))
            scores.append(float(score))
        return labels, scores

    def finish_eval_epoch(self, stage: str):
        labels, scores = self.similarity_scores()
        eer, threshold = compute_eer(labels, scores)
        print(f"\ncosine EER: {eer * 100:.2f}% with threshold {threshold:.2f}")
        self.log("cosine_eer", eer * 100, prog_bar=True)

        min_dcf, threshold = compute_min_dcf(labels, scores, p_target=0.01)
        print(f"cosine minDCF(10-2): {min_dcf:.2f} with threshold {threshold:.2f}")
        self.log("cosine_minDCF(10-2)", min_dcf, prog_bar=True)

        min_dcf, threshold = compute_min_dcf(labels, scores, p_target=0.001)
        print(f"cosine minDCF(10-3): {min_dcf:.2f} with threshold {threshold:.2f}")
        self.log("cosine_minDCF(10-3)", min_dcf, prog_bar=True)

        prefix = self.config.get("score_output_prefix")
        if prefix:
            np.savetxt(f"{prefix}_{stage}_labels.txt", np.array(labels, dtype=int), fmt="%d")
            np.savetxt(f"{prefix}_{stage}_scores.txt", np.array(scores, dtype=float), fmt="%.8f")

    def on_validation_epoch_end(self):
        self.finish_eval_epoch("val")

    def on_test_epoch_end(self):
        self.finish_eval_epoch("test")


def build_task(config: Mapping[str, Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = build_feature(config)
    model = build_model(config, device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"MFA-Conformer parameters: {total_params:,}")
    loss = AMSoftmax(
        embedding_dim=int(config.get("embedding_dim", 192)),
        num_classes=int(config["num_spk"]),
        margin=float(config.get("margin", 0.2)),
        scale=float(config.get("scale", 30)),
    )
    return BaselineTask(features, model, loss, config)


def build_trainer(config: Mapping[str, Any], mode: str):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if accelerator == "gpu" else 32
    logger = False
    if mode in {"train", "test"} and _as_bool(config.get("USE_WANDB", False)):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=config.get("wandb_project", "caarma-baseline"),
            name=config.get("title", "caarma_baseline_mfa_conformer"),
            save_dir=config["save_dir"],
        )
        logger.experiment.config.update(config)

    callbacks = []
    if mode == "train":
        callbacks.append(
            ModelCheckpoint(
                dirpath=config["save_dir"],
                monitor="cosine_eer",
                mode="min",
                save_top_k=int(config.get("save_top_k", 3)),
                save_last=True,
                filename="{epoch}_{cosine_eer:.2f}",
            )
        )
        if logger:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

    return Trainer(
        strategy="auto",
        accelerator=accelerator,
        devices=int(config.get("devices", 1)),
        max_epochs=int(config["epochs"]),
        logger=logger,
        num_sanity_val_steps=0,
        precision=precision,
        callbacks=callbacks,
        default_root_dir=config["save_dir"],
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=1.0 if _as_bool(config.get("validate_during_train", True)) else 0,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
        profiler="simple" if mode == "train" else None,
    )


def parse_args():
    parser = ArgumentParser(description="Run clean MFA-Conformer AMSoftmax baseline")
    parser.add_argument("--config", default="configs/baseline_mfa_conformer_bridges2.yaml")
    parser.add_argument("--mode", choices=("train", "validate", "test"), default="train")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--trial-path", default=None)
    parser.add_argument("--score-output-prefix", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["_mode"] = args.mode
    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path
    if args.trial_path:
        config["trial_path"] = args.trial_path
    if args.score_output_prefix:
        config["score_output_prefix"] = args.score_output_prefix
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

    datamodule = BaselineDataModule(config)
    if args.mode == "train":
        datamodule.setup("fit")
        if _as_bool(config.get("derive_num_spk", True)):
            config["num_spk"] = datamodule.config["num_spk"]
    task = build_task(config)
    trainer = build_trainer(config, args.mode)

    if args.mode == "train":
        trainer.fit(task, datamodule=datamodule)
    elif args.mode == "validate":
        trainer.validate(task, datamodule=datamodule, ckpt_path=config.get("checkpoint_path"))
    else:
        trainer.test(task, datamodule=datamodule, ckpt_path=config.get("checkpoint_path"))


if __name__ == "__main__":
    main()
