from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.nn.utils import spectral_norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from feature.build_feature import build_feature
from functions.dataset import Evaluation_Dataset
from functions.loader import super_dataset
from functions.nationality import (
    build_label_nationality_map,
    canonical_speaker_pair,
    eligible_peer_count,
    same_nationality_candidates,
)
from functions.voxceleb_split import (
    build_label_metadata_map,
    is_voxceleb_split_csv,
    load_validation_trials,
    resolve_split_csv_path,
)
from helper.diffusion_mixup import diffusion_mixup
from model.discriminator_mix import InterDiscrim, SimpleDiscriminator
from model.model_build import build_model


SPEAKER_ID_RE = re.compile(r"id\d{5}")
EXPERIMENT_TYPES = {"base", "gender", "nationality"}
SYNTHETIC_STRATEGIES = {"none", "avg", "diffusion"}
CONDITION_ATTRIBUTES = {"none", "gender", "nationality"}
PATH_KEYS = (
    "dataset",
    "trial_path",
    "checkpoint_path",
    "vox1_meta_path",
    "save_dir",
    "vox1_wav_root",
    "vox1_split_dir",
    "train_split_csv",
    "val_split_csv",
    "test_split_csv",
)


def load_config(config_file_path: str) -> Dict[str, Any]:
    with open(config_file_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _none_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _path_base_dir(config_dir: Path) -> Path:
    if config_dir.name == "configs":
        return config_dir.parent
    return config_dir


def _resolve_path(value: Any, base_dir: Path) -> Any:
    if _none_like(value):
        return value
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _ensure_root_suffix(root: str) -> str:
    if _none_like(root):
        return root
    return root if str(root).endswith(os.sep) else f"{root}{os.sep}"


def extract_speaker_id(value: Any) -> Optional[str]:
    match = SPEAKER_ID_RE.search(str(value))
    return match.group(0) if match else None


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def _preferred_columns(columns: Iterable[str]) -> Iterable[str]:
    preferred = (
        "utt_spk_id",
        "speaker_id",
        "spk_id",
        "VoxCeleb1 ID",
        "vox1_id",
        "utt_paths",
        "path",
        "wav",
        "file",
    )
    existing = list(columns)
    lower_to_original = {column.lower(): column for column in existing}
    for column in preferred:
        if column in existing:
            yield column
        elif column.lower() in lower_to_original:
            yield lower_to_original[column.lower()]


def load_vox1_gender_by_speaker(meta_path: str) -> Dict[str, str]:
    meta = pd.read_csv(meta_path, sep=None, engine="python")
    meta.columns = [str(column).strip() for column in meta.columns]

    id_column = next(
        (
            column
            for column in meta.columns
            if column.lower().replace(" ", "") in {"voxceleb1id", "vox1id", "speakerid"}
        ),
        meta.columns[0],
    )
    gender_column = next(
        (column for column in meta.columns if column.lower().strip() == "gender"),
        None,
    )
    if gender_column is None:
        raise ValueError(f"No Gender column found in {meta_path}")

    speaker_to_gender = {}
    for _, row in meta.iterrows():
        speaker_id = extract_speaker_id(row[id_column])
        gender = str(row[gender_column]).strip().lower()
        if speaker_id and gender:
            speaker_to_gender[speaker_id] = gender
    return speaker_to_gender


def build_label_gender_map(train_csv_path: str, meta_path: str) -> Dict[int, str]:
    speaker_to_gender = load_vox1_gender_by_speaker(meta_path)
    train_df = pd.read_csv(train_csv_path)

    label_column = next(
        (
            column
            for column in train_df.columns
            if column in {"utt_spk_int_labels", "mapped_id", "label", "speaker_label"}
        ),
        None,
    )
    if label_column is None:
        raise ValueError(
            f"No integer speaker label column found in {train_csv_path}; "
            "expected utt_spk_int_labels, mapped_id, label, or speaker_label."
        )

    candidate_columns = list(dict.fromkeys(_preferred_columns(train_df.columns)))
    if not candidate_columns:
        candidate_columns = list(train_df.columns)

    label_to_gender: Dict[int, str] = {}
    for _, row in train_df.iterrows():
        try:
            label = int(row[label_column])
        except (TypeError, ValueError):
            continue

        speaker_id = None
        for column in candidate_columns:
            speaker_id = extract_speaker_id(row[column])
            if speaker_id is not None:
                break
        if speaker_id is None:
            for value in row.values:
                speaker_id = extract_speaker_id(value)
                if speaker_id is not None:
                    break

        gender = speaker_to_gender.get(speaker_id)
        if gender is not None:
            label_to_gender[label] = gender

    return label_to_gender


def infer_num_speakers(config: Mapping[str, Any]) -> Optional[int]:
    dataset = config.get("dataset")
    split_path = Path(str(dataset)) if not _none_like(dataset) else resolve_split_csv_path(config)
    if not split_path.exists():
        return None

    if is_voxceleb_split_csv(split_path):
        split_df = pd.read_csv(split_path, usecols=["VoxCeleb1_ID"])
        return int(split_df["VoxCeleb1_ID"].astype(str).str.strip().nunique())

    df = pd.read_csv(split_path)
    if "utt_spk_int_labels" not in df.columns:
        return None
    labels = pd.to_numeric(df["utt_spk_int_labels"], errors="coerce").dropna().astype(int)
    if labels.empty:
        return None
    return int(labels.max() + 1)


def _apply_experiment_defaults(config: Dict[str, Any], default_experiment_type: str) -> None:
    experiment_type = str(config.get("experiment_type", default_experiment_type)).strip().lower()
    config["experiment_type"] = experiment_type

    if "condition_attribute" not in config:
        config["condition_attribute"] = (
            experiment_type if experiment_type in {"gender", "nationality"} else "none"
        )
    config["condition_attribute"] = str(config["condition_attribute"]).strip().lower()

    if "synthetic_strategy" not in config:
        config["synthetic_strategy"] = "avg" if experiment_type in {"gender", "nationality"} else "none"
    config["synthetic_strategy"] = str(config["synthetic_strategy"]).strip().lower()

    if "adversarial_enabled" not in config:
        config["adversarial_enabled"] = experiment_type in {"gender", "nationality"}
    config["adversarial_enabled"] = _as_bool(config["adversarial_enabled"])

    if experiment_type == "base" and config["synthetic_strategy"] == "diffusion":
        config.setdefault("lambda_syn", 0.01)
        config.setdefault("diffusion_fake_fraction", 0.25)
        config.setdefault("diffusion_t_min", 0)
        config.setdefault("diffusion_t_max", 3)
        config.setdefault("diffusion_embedding_noise", 0.0)
        config["adversarial_enabled"] = _as_bool(config.get("adversarial_enabled", False))


def _validate_experiment_config(config: Mapping[str, Any]) -> None:
    experiment_type = str(config["experiment_type"]).strip().lower()
    condition_attribute = str(config["condition_attribute"]).strip().lower()
    synthetic_strategy = str(config["synthetic_strategy"]).strip().lower()

    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(f"experiment_type must be one of {sorted(EXPERIMENT_TYPES)}")
    if condition_attribute not in CONDITION_ATTRIBUTES:
        raise ValueError(f"condition_attribute must be one of {sorted(CONDITION_ATTRIBUTES)}")
    if synthetic_strategy not in SYNTHETIC_STRATEGIES:
        raise ValueError(f"synthetic_strategy must be one of {sorted(SYNTHETIC_STRATEGIES)}")
    if experiment_type == "base" and condition_attribute != "none":
        raise ValueError("base experiment_type must use condition_attribute: none")
    if experiment_type in {"gender", "nationality"} and condition_attribute != experiment_type:
        raise ValueError(
            f"{experiment_type} experiment_type must use condition_attribute: {experiment_type}"
        )

    other_tokens = sorted(EXPERIMENT_TYPES - {experiment_type})
    named_fields = (
        "wandb_project",
        "title",
        "save_dir",
        "score_output_prefix",
    )
    for field in named_fields:
        value = str(config.get(field, "")).lower()
        for token in other_tokens:
            if token in value:
                raise ValueError(
                    f"{experiment_type} experiment has mismatched {field}: contains {token}"
                )

    fake_fraction = float(config.get("diffusion_fake_fraction", 1.0))
    if fake_fraction <= 0.0 or fake_fraction > 1.0:
        raise ValueError("diffusion_fake_fraction must be in the range (0, 1]")


def prepare_experiment_config(
    config: Dict[str, Any],
    config_path: Path,
    args: Any,
    default_experiment_type: str = "base",
) -> Dict[str, Any]:
    config_dir = config_path.resolve().parent
    path_base_dir = _path_base_dir(config_dir)
    prepared = dict(config)
    prepared["_mode"] = getattr(args, "mode", "train")

    if getattr(args, "sl_mixup", None) is not None:
        prepared["sl_mixup"] = args.sl_mixup
    if getattr(args, "nationality_mixup", None) is not None:
        prepared["nationality_mixup"] = args.nationality_mixup
    if getattr(args, "vox1_meta_path", None) is not None:
        prepared["vox1_meta_path"] = args.vox1_meta_path
    prepared.setdefault("vox1_meta_path", "vox1_meta.csv")

    if getattr(args, "checkpoint_path", None) is not None:
        prepared["checkpoint_path"] = args.checkpoint_path
    if getattr(args, "trial_path", None) is not None:
        prepared["trial_path"] = args.trial_path
    if getattr(args, "root", None) is not None:
        prepared["root"] = args.root
    if getattr(args, "validation_split", None) is not None:
        prepared["validation_split"] = args.validation_split
        prepared["generate_validation_trials"] = True
    if getattr(args, "score_output_prefix", None) is not None:
        prepared["score_output_prefix"] = args.score_output_prefix

    _apply_experiment_defaults(prepared, default_experiment_type)

    for key in PATH_KEYS:
        if key in prepared:
            prepared[key] = _resolve_path(prepared[key], path_base_dir)

    if "root" in prepared:
        prepared["root"] = _ensure_root_suffix(prepared["root"])

    inferred_num_spk = infer_num_speakers(prepared)
    if inferred_num_spk is not None and _as_bool(prepared.get("derive_num_spk", True)):
        prepared["num_spk"] = inferred_num_spk

    _validate_experiment_config(prepared)
    return prepared


def _nearest_candidate(
    speaker: int,
    candidate_labels: Iterable[int],
    weights: torch.Tensor,
) -> int:
    candidates = list(candidate_labels)
    if not candidates:
        return speaker

    with torch.no_grad():
        source = weights[:, speaker].detach()
        distances = torch.stack(
            [torch.norm(source - weights[:, candidate].detach(), p=2) for candidate in candidates]
        )
        return candidates[int(torch.argmin(distances).item())]


def mixup_data_euc_avg_conditioned(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    label_to_condition: Optional[Dict[int, str]] = None,
    condition_mixup: bool = False,
):
    batch_size = x.size(0)
    device = x.device
    label_values = [int(label) for label in labels.detach().cpu().tolist()]
    invalid_labels = [label for label in label_values if label < 0 or label >= weights.size(1)]
    if invalid_labels:
        raise ValueError(
            f"Speaker labels must be in [0, {weights.size(1)}); "
            f"found {sorted(set(invalid_labels))[:10]}"
        )

    unique_labels = sorted(set(label_values))
    label_positions = defaultdict(list)
    for position, label in enumerate(label_values):
        label_positions[label].append(position)

    pair_by_label: Dict[int, int] = {}
    for speaker in unique_labels:
        if condition_mixup:
            condition = (label_to_condition or {}).get(speaker)
            if condition is None:
                candidates = []
            else:
                candidates = [
                    candidate
                    for candidate in unique_labels
                    if candidate != speaker and (label_to_condition or {}).get(candidate) == condition
                ]
        else:
            candidates = [candidate for candidate in unique_labels if candidate != speaker]
        pair_by_label[speaker] = _nearest_candidate(speaker, candidates, weights)

    mixed_weights = weights.new_zeros((weights.size(0), batch_size))
    mixed_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    pair_indices = []
    synthetic_label_by_pair: Dict[tuple[int, int], int] = {}

    for row_index, speaker in enumerate(label_values):
        paired_speaker = pair_by_label[speaker]
        paired_positions = label_positions.get(paired_speaker, [row_index])
        pair_indices.append(paired_positions[0] if paired_speaker != speaker else row_index)

        pair_key = canonical_speaker_pair(speaker, paired_speaker)
        if pair_key not in synthetic_label_by_pair:
            synthetic_label = len(synthetic_label_by_pair)
            synthetic_label_by_pair[pair_key] = synthetic_label
            mixed_weights[:, synthetic_label] = (
                weights[:, speaker] + weights[:, paired_speaker]
            ) / 2
        mixed_labels[row_index] = synthetic_label_by_pair[pair_key]

    pair_index_tensor = torch.tensor(pair_indices, dtype=torch.long, device=device)
    mixed_embeddings = 0.5 * (x + x.index_select(0, pair_index_tensor))
    used_classes = len(synthetic_label_by_pair)
    return mixed_embeddings, mixed_labels, mixed_weights[:, :used_classes].to(device)


def mixup_data_euc_avg_gender(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    label_to_gender: Optional[Dict[int, str]] = None,
    sl_mixup: bool = False,
):
    return mixup_data_euc_avg_conditioned(
        x,
        weights,
        labels,
        label_to_condition=label_to_gender,
        condition_mixup=sl_mixup,
    )


def mixup_data_euc_avg_nationality(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    label_to_nationality: Optional[Dict[int, str]] = None,
    nationality_mixup: bool = True,
):
    return mixup_data_euc_avg_conditioned(
        x,
        weights,
        labels,
        label_to_condition=label_to_nationality,
        condition_mixup=nationality_mixup,
    )


class AMSoftmaxGANExperiment(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30,
        synthetic_strategy: str = "avg",
        condition_attribute: str = "none",
        label_to_condition: Optional[Dict[int, str]] = None,
        diffusion_timesteps: int = 100,
        diffusion_t_min: int = 1,
        diffusion_t_max: int = 20,
        diffusion_beta_start: float = 0.0001,
        diffusion_beta_end: float = 0.02,
        diffusion_embedding_noise: float = 0.0,
        diffusion_fake_fraction: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.synthetic_strategy = str(synthetic_strategy).strip().lower()
        self.condition_attribute = str(condition_attribute).strip().lower()
        self.label_to_condition = dict(label_to_condition or {})
        self.diffusion_timesteps = diffusion_timesteps
        self.diffusion_t_min = diffusion_t_min
        self.diffusion_t_max = diffusion_t_max
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        self.diffusion_embedding_noise = diffusion_embedding_noise
        self.diffusion_fake_fraction = diffusion_fake_fraction
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print(f"Initialised AM-Softmax m={self.m:.3f} s={self.s:.3f}")
        print(f"Embedding dim is {embedding_dim}, number of speakers is {num_classes}")
        print(
            "Synthetic strategy: "
            f"{self.synthetic_strategy}, condition_attribute={self.condition_attribute}"
        )

    def _zero_synthetic_loss(self, x: torch.Tensor):
        return x.sum() * 0.0, torch.zeros((), device=x.device), x

    def _synthetic_batch(self, x: torch.Tensor, label: torch.Tensor):
        if self.synthetic_strategy == "none":
            return x, label, self.W
        if self.synthetic_strategy == "diffusion":
            return diffusion_mixup(
                x,
                self.W,
                label,
                diffusion_timesteps=self.diffusion_timesteps,
                diffusion_t_min=self.diffusion_t_min,
                diffusion_t_max=self.diffusion_t_max,
                diffusion_beta_start=self.diffusion_beta_start,
                diffusion_beta_end=self.diffusion_beta_end,
                diffusion_embedding_noise=self.diffusion_embedding_noise,
                diffusion_fake_fraction=self.diffusion_fake_fraction,
            )
        if self.synthetic_strategy == "avg":
            return mixup_data_euc_avg_conditioned(
                x,
                self.W,
                label,
                label_to_condition=self.label_to_condition,
                condition_mixup=self.condition_attribute != "none",
            )
        raise ValueError(f"Unsupported synthetic strategy: {self.synthetic_strategy}")

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        flagSyn: bool = False,
    ):
        if label is None:
            raise ValueError("AMSoftmaxGANExperiment requires speaker labels")
        if x.size(0) != label.size(0) or x.size(1) != self.in_feats:
            raise ValueError("Embedding and label shapes do not match the configured criterion")

        if flagSyn and self.synthetic_strategy == "none":
            return self._zero_synthetic_loss(x)

        synthetic_embeddings, synthetic_labels, synthetic_weights = self._synthetic_batch(x, label)
        if flagSyn:
            embeddings_for_loss = synthetic_embeddings
            weights_for_loss = synthetic_weights
            labels_for_loss = synthetic_labels
        else:
            embeddings_for_loss = x
            weights_for_loss = self.W
            labels_for_loss = label

        embeddings_norm = embeddings_for_loss / torch.norm(
            embeddings_for_loss,
            p=2,
            dim=1,
            keepdim=True,
        ).clamp(min=1e-12)
        weights_norm = weights_for_loss / torch.norm(
            weights_for_loss,
            p=2,
            dim=0,
            keepdim=True,
        ).clamp(min=1e-12)
        cosine = torch.mm(embeddings_norm, weights_norm)
        label_view = labels_for_loss.view(-1, 1).to(device=x.device)
        margin = torch.zeros_like(cosine).scatter_(1, label_view, self.m)
        logits = self.s * (cosine - margin)

        loss = self.ce(logits, labels_for_loss)
        acc = accuracy(logits.detach(), labels_for_loss.detach(), topk=(1,))[0]
        return loss, acc, synthetic_embeddings


class AMSoftmaxGANGender(AMSoftmaxGANExperiment):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30,
        label_to_gender: Optional[Dict[int, str]] = None,
        sl_mixup: bool = False,
        synthetic_strategy: str = "avg",
        **kwargs: Any,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
            synthetic_strategy=synthetic_strategy,
            condition_attribute="gender" if sl_mixup and synthetic_strategy != "diffusion" else "none",
            label_to_condition=label_to_gender,
            **kwargs,
        )
        self.sl_mixup = sl_mixup
        self.label_to_gender = dict(label_to_gender or {})


class AMSoftmaxGANNationality(AMSoftmaxGANExperiment):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30,
        label_to_nationality: Optional[Dict[int, str]] = None,
        nationality_mixup: bool = True,
        synthetic_strategy: str = "avg",
        **kwargs: Any,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
            synthetic_strategy=synthetic_strategy,
            condition_attribute=(
                "nationality" if nationality_mixup and synthetic_strategy != "diffusion" else "none"
            ),
            label_to_condition=label_to_nationality,
            **kwargs,
        )
        self.nationality_mixup = nationality_mixup
        self.label_to_nationality = dict(label_to_nationality or {})


class LegacyMixupDiscriminator(nn.Module):
    def __init__(
        self,
        emb_dim: int = 192,
        hidden_dim: int = 256,
        mid_dim: int = 128,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.discriminator = nn.Sequential(
            ResidualBlock(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(mid_dim, 1)),
        )

    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        if input_embedding.dim() == 3 and input_embedding.size(1) == 1:
            input_embedding = input_embedding.squeeze(1)
        if input_embedding.dim() != 2:
            raise ValueError(
                "LegacyMixupDiscriminator expects pooled embeddings of shape (B, E), "
                f"but got {tuple(input_embedding.shape)}"
            )
        return self.discriminator(input_embedding)


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear1 = spectral_norm(nn.Linear(in_features, out_features))
        self.linear2 = spectral_norm(nn.Linear(out_features, out_features))
        self.shortcut = (
            spectral_norm(nn.Linear(in_features, out_features))
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        out = self.linear2(out)
        return F.leaky_relu(out + identity, negative_slope=0.2)


def build_embedding_discriminator(config: Dict[str, Any]) -> nn.Module:
    discriminator_name = str(config.get("discriminator", "simple")).strip().lower()
    embedding_dim = int(config.get("embedding_dim", 192))
    hidden_dim = int(config.get("discriminator_hidden_dim", 256))
    mid_dim = int(config.get("discriminator_mid_dim", max(hidden_dim // 2, 1)))
    head_dim = int(config.get("discriminator_head_dim", 128))
    dropout_rate = float(config.get("discriminator_dropout", 0.1))

    if discriminator_name == "simple":
        return SimpleDiscriminator(
            emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
    if discriminator_name in {"inter", "intermediate", "interdiscrim"}:
        return InterDiscrim(
            emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            mid_dim=mid_dim,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
        )
    if discriminator_name in {"mixup", "legacy"}:
        return LegacyMixupDiscriminator(
            emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            mid_dim=mid_dim,
            dropout_rate=dropout_rate,
        )
    raise ValueError(f"Unsupported discriminator: {config.get('discriminator')}")


def _build_label_condition_map(config: Dict[str, Any]) -> Dict[int, str]:
    condition_attribute = str(config.get("condition_attribute", "none")).strip().lower()
    if condition_attribute == "none":
        return {}

    try:
        label_to_condition = build_label_metadata_map(config, condition_attribute)
        if label_to_condition:
            return label_to_condition
        if condition_attribute == "gender":
            return build_label_gender_map(config["dataset"], config.get("vox1_meta_path", "vox1_meta.csv"))
        if condition_attribute == "nationality":
            return build_label_nationality_map(config["dataset"], config["vox1_meta_path"])
    except FileNotFoundError:
        if config.get("_mode") == "train":
            raise
        print(f"Warning: metadata unavailable; proceeding without {condition_attribute} map.")

    return {}


def build_experiment_criterion(config: Dict[str, Any]) -> nn.Module:
    if config["criterion"] != "AMSoftmaxGAN":
        raise NotImplementedError("The shared CAARMA experiment runner supports AMSoftmaxGAN only.")

    synthetic_strategy = str(config.get("synthetic_strategy", "avg")).strip().lower()
    condition_attribute = str(config.get("condition_attribute", "none")).strip().lower()
    active_condition_attribute = condition_attribute if synthetic_strategy == "avg" else "none"
    condition_config = dict(config)
    condition_config["condition_attribute"] = active_condition_attribute
    label_to_condition = _build_label_condition_map(condition_config)

    if config.get("_mode") == "train" and active_condition_attribute != "none":
        if not label_to_condition:
            raise ValueError(
                f"{active_condition_attribute} mixup is enabled, but no training labels could be mapped."
            )
        if active_condition_attribute == "nationality" and eligible_peer_count(label_to_condition) == 0:
            raise ValueError("No mapped speaker has a distinct same-nationality peer.")

    return AMSoftmaxGANExperiment(
        embedding_dim=int(config.get("embedding_dim", 192)),
        num_classes=int(config.get("num_spk", 1211)),
        margin=float(config.get("margin", 0.2)),
        scale=float(config.get("scale", 30)),
        synthetic_strategy=synthetic_strategy,
        condition_attribute=active_condition_attribute,
        label_to_condition=label_to_condition,
        diffusion_timesteps=int(config.get("diffusion_timesteps", 100)),
        diffusion_t_min=int(config.get("diffusion_t_min", 1)),
        diffusion_t_max=int(config.get("diffusion_t_max", 20)),
        diffusion_beta_start=float(config.get("diffusion_beta_start", 0.0001)),
        diffusion_beta_end=float(config.get("diffusion_beta_end", 0.02)),
        diffusion_embedding_noise=float(config.get("diffusion_embedding_noise", 0.0)),
        diffusion_fake_fraction=float(config.get("diffusion_fake_fraction", 1.0)),
    )


def build_gender_criterion(config: Dict[str, Any]) -> nn.Module:
    gender_config = dict(config)
    gender_config.setdefault("experiment_type", "gender")
    gender_config.setdefault("condition_attribute", "gender")
    return build_experiment_criterion(gender_config)


def build_nationality_criterion(config: Dict[str, Any]) -> nn.Module:
    nationality_config = dict(config)
    nationality_config.setdefault("experiment_type", "nationality")
    nationality_config.setdefault("condition_attribute", "nationality")
    return build_experiment_criterion(nationality_config)


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
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(min_c_det / c_def), float(min_c_det_threshold)


class Task(LightningModule):
    def __init__(
        self,
        features,
        model,
        loss,
        config,
        learning_rate=0.2,
        weight_decay=1.5e-6,
        batch_size=32,
        num_workers=10,
        max_epochs=1000,
        trial_path="data/vox1_test.txt",
        warmup_step=2000,
        **kwargs,
    ):
        super().__init__()
        self.features = features
        self.model = model
        self.loss = loss
        self.loss_syn = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.config = config
        self.automatic_optimization = False
        self.adversarial_enabled = _as_bool(self.config.get("adversarial_enabled", True))
        self.synthetic_strategy = str(self.config.get("synthetic_strategy", "avg")).strip().lower()
        self.discriminator_learning_rate = float(
            self.config.get("discriminator_lr", self.learning_rate * 0.01)
        )
        self.lambda_adv = float(self.config.get("lambda_adv", 0.01))
        self.lambda_adv_min = float(self.config.get("lambda_adv_min", 0.0001))
        self.lambda_adv_max = float(self.config.get("lambda_adv_max", 0.01))
        self.lambda_adv_pretrain = float(self.config.get("lambda_adv_pretrain", 0.0005))
        default_lambda_syn = 1.0 / max(int(self.config.get("num_spk", 1)), 1)
        self.lambda_syn = float(self.config.get("lambda_syn", default_lambda_syn))
        self.log_training_steps = _as_bool(self.config.get("log_training_steps", False))
        self.trials = None

        should_load_trials = (
            config.get("_mode") != "train"
            or _as_bool(config.get("validate_during_train", True))
            or Path(trial_path).exists()
        )
        if should_load_trials:
            self.trials, validation_root = load_validation_trials(config)
            self.config["root"] = validation_root
        else:
            print(f"Skipping training validation because trial_path was not found: {trial_path}")

        self.discriminator = None
        if self.adversarial_enabled:
            self.discriminator = build_embedding_discriminator(self.config).train()
            self.BCE_loss = nn.BCEWithLogitsLoss()
        self.pretrain_eps = int(self.config.get("pretrain_eps", 15))

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return torch.div(x, x_norm)

    def set_discriminator_requires_grad(self, requires_grad: bool):
        if self.discriminator is None:
            return
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(requires_grad)

    def forward(self, x):
        feature = self.features(x)
        return self.model(feature)

    def adjust_weight(self, amsoftmax_loss, g_loss):
        loss_ratio = amsoftmax_loss / (g_loss + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, self.lambda_adv_max)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, self.lambda_adv_min)
        return self.lambda_adv

    def _main_optimizer_and_scheduler(self):
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        optimizer_main = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        scheduler_main = schedulers[0] if isinstance(schedulers, (list, tuple)) else schedulers
        return optimizer_main, scheduler_main

    def _main_loss(self, embedding, label, include_synthetic=True):
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
        amsoftmax_syn_loss = embedding.sum() * 0.0
        acc_syn = torch.zeros((), device=embedding.device)
        if include_synthetic and self.synthetic_strategy != "none":
            amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(
                embedding,
                label,
                flagSyn=True,
            )
        total_loss = amsoftmax_loss + self.lambda_syn * amsoftmax_syn_loss
        return total_loss, amsoftmax_loss, amsoftmax_syn_loss, acc, acc_syn, synthetic_embeddings

    def _step_main_only(self, embedding, label):
        optimizer_main, _ = self._main_optimizer_and_scheduler()
        optimizer_main.zero_grad()
        total_loss, am_loss, syn_loss, acc, _, _ = self._main_loss(embedding, label)
        self.manual_backward(total_loss)
        self.log("am_loss", am_loss, prog_bar=True)
        self.log("am_loss_syn", syn_loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        optimizer_main.step()
        if self.trainer.global_step < self.config["warmup_step"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / float(self.config["warmup_step"]),
            )
            for pg in optimizer_main.param_groups:
                pg["lr"] = lr_scale * self.learning_rate
        return total_loss

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        label = batch["mapped_id"]

        feature = self.features(waveform)
        embedding = self.model(feature)

        if not self.adversarial_enabled:
            return self._step_main_only(embedding, label)

        opt = self.optimizers()
        optimizer_main, d_optimizer = opt

        _, _, _, _, _, synthetic_embeddings = self._main_loss(
            embedding,
            label,
            include_synthetic=False,
        )

        if not hasattr(self, "d_step_counter"):
            self.d_step_counter = 0
        if not hasattr(self, "g_step_counter"):
            self.g_step_counter = 0

        if self.d_step_counter >= 1 and self.g_step_counter >= 5:
            self.d_step_counter = 0
            self.g_step_counter = 0
        elif (
            self.d_step_counter >= 1
            and self.g_step_counter >= 1
            and self.current_epoch > self.pretrain_eps
        ):
            self.d_step_counter = 0
            self.g_step_counter = 0

        if self.current_epoch <= self.pretrain_eps and self.g_step_counter < 5:
            optimizer_main.zero_grad()
            self.set_discriminator_requires_grad(False)
            real_preds = self.discriminator(self.normalize(embedding.detach()))
            fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
            fake_labels = torch.zeros(real_preds.size(), device=self.device)
            real_labels = torch.ones(fake_preds.size(), device=self.device)
            g_loss = (
                self.BCE_loss(fake_preds, real_labels)
                + self.BCE_loss(real_preds, fake_labels)
            ) / 2
            total_loss, am_loss, syn_loss, acc, _, _ = self._main_loss(embedding, label)
            total_loss = total_loss + self.lambda_adv_pretrain * g_loss
            self.manual_backward(total_loss)
            self.log("am_loss", am_loss, prog_bar=True)
            self.log("am_loss_syn", syn_loss, prog_bar=True)
            self.log("acc", acc, prog_bar=True)
            self.log("g_loss", g_loss, prog_bar=True)
            self.log("total_loss", total_loss, prog_bar=True)
            optimizer_main.step()
            if self.trainer.global_step < self.config["warmup_step"]:
                lr_scale = min(
                    1.0,
                    float(self.trainer.global_step + 1) / float(self.config["warmup_step"]),
                )
                for pg in optimizer_main.param_groups:
                    pg["lr"] = lr_scale * self.learning_rate
            self.g_step_counter += 1
            self.set_discriminator_requires_grad(True)
            return total_loss

        if self.d_step_counter < 1:
            self.set_discriminator_requires_grad(True)
            d_optimizer.zero_grad()
            real_preds = self.discriminator(self.normalize(embedding.detach()))
            fake_preds = self.discriminator(self.normalize(synthetic_embeddings.detach()))
            real_labels = torch.ones(real_preds.size(), device=self.device)
            fake_labels = torch.zeros(fake_preds.size(), device=self.device)
            d_loss = (
                self.BCE_loss(real_preds, real_labels)
                + self.BCE_loss(fake_preds, fake_labels)
            ) / 2
            self.manual_backward(d_loss)
            self.log("d_loss", d_loss, prog_bar=True)
            d_optimizer.step()
            self.d_step_counter += 1
            return d_loss

        optimizer_main.zero_grad()
        self.set_discriminator_requires_grad(False)
        real_preds = self.discriminator(self.normalize(embedding.detach()))
        fake_preds = self.discriminator(self.normalize(synthetic_embeddings))
        fake_labels = torch.zeros(real_preds.size(), device=self.device)
        real_labels = torch.ones(fake_preds.size(), device=self.device)
        g_loss = (
            self.BCE_loss(fake_preds, real_labels)
            + self.BCE_loss(real_preds, fake_labels)
        ) / 2
        total_loss, am_loss, syn_loss, acc, _, _ = self._main_loss(embedding, label)
        self.lambda_adv = self.adjust_weight(am_loss, g_loss)
        total_loss = total_loss + self.lambda_adv * g_loss
        self.manual_backward(total_loss)
        self.log("am_loss", am_loss, prog_bar=True)
        self.log("am_loss_syn", syn_loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        optimizer_main.step()
        if self.trainer.global_step < self.config["warmup_step"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / float(self.config["warmup_step"]),
            )
            for pg in optimizer_main.param_groups:
                pg["lr"] = lr_scale * self.learning_rate
        self.g_step_counter += 1
        self.set_discriminator_requires_grad(True)
        return total_loss

    def configure_optimizers(self):
        embedding_optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        embedding_scheduler = StepLR(embedding_optimizer, step_size=4, gamma=0.5)

        if not self.adversarial_enabled:
            return [embedding_optimizer], [embedding_scheduler]

        discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999),
        )
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size=4, gamma=0.5)
        return [embedding_optimizer, discriminator_optimizer], [
            embedding_scheduler,
            discriminator_scheduler,
        ]

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, (list, tuple)):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def validation_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        path = batch["path"]
        with torch.no_grad():
            x = self.features(waveform)
            self.model.eval()
            x = self.model(x)

        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[os.path.normpath(path[0])] = batch_idx

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def _trial_key(self, relative_or_abs_path: str) -> str:
        if os.path.isabs(relative_or_abs_path):
            return os.path.normpath(relative_or_abs_path)
        return os.path.normpath(os.path.join(self.config["root"], relative_or_abs_path))

    def similarity_score(self, trials, index_mapping, eval_vectors):
        labels = []
        scores = []
        epsilon = 1e-8
        for item in trials:
            enroll_vector = eval_vectors[index_mapping[self._trial_key(item[1])]]
            test_vector = eval_vectors[index_mapping[self._trial_key(item[2])]]
            score = enroll_vector.dot(test_vector.T)
            denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
            score = score / (denom + epsilon)
            if np.isnan(score):
                print("Warning: NaN detected in score calculation. Setting score to 0.")
                score = 0.0
            labels.append(int(item[0]))
            scores.append(score)

        print(torch.mean(torch.tensor(scores)))
        return labels, scores

    def compute_eer(self, labels, scores):
        return compute_eer(labels, scores)

    def compute_minDCF(self, labels, scores, p_target=0.01, c_miss=1, c_fa=1):
        return compute_min_dcf(labels, scores, p_target=p_target, c_miss=c_miss, c_fa=c_fa)

    def _gather_eval_outputs(self):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_vectors = [None for _ in range(world_size)]
            gathered_maps = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_vectors, self.eval_vectors)
            dist.all_gather_object(gathered_maps, self.index_mapping)

            eval_vectors = []
            index_mapping = {}
            offset = 0
            for vectors, mapping in zip(gathered_vectors, gathered_maps):
                vectors = vectors or []
                mapping = mapping or {}
                eval_vectors.extend(vectors)
                for path, local_index in mapping.items():
                    index_mapping[path] = offset + local_index
                offset += len(vectors)
            return eval_vectors, index_mapping

        return self.eval_vectors, self.index_mapping

    def on_validation_epoch_end(self):
        eval_vectors, index_mapping = self._gather_eval_outputs()
        eval_vectors = np.vstack(eval_vectors)
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
        labels, scores = self.similarity_score(self.trials, index_mapping, eval_vectors)
        eer, threshold = compute_eer(labels, scores)

        output_prefix = self.config.get("score_output_prefix", "org_inf")
        with open(f"{output_prefix}_labels.txt", "w", encoding="utf-8") as handle:
            for line in labels:
                handle.write(f"{line}\n")
        with open(f"{output_prefix}_scores.txt", "w", encoding="utf-8") as handle:
            for line in scores:
                handle.write(f"{line}\n")

        print(f"\ncosine EER: {eer * 100:.2f}% with threshold {threshold:.2f}")
        self.log("cosine_eer", eer * 100, on_epoch=True, sync_dist=True)

        min_dcf, threshold = compute_min_dcf(labels, scores, p_target=0.01)
        print(f"cosine minDCF(10-2): {min_dcf:.2f} with threshold {threshold:.2f}")
        self.log("cosine_minDCF(10-2)", min_dcf, on_epoch=True, sync_dist=True)

        min_dcf, threshold = compute_min_dcf(labels, scores, p_target=0.001)
        print(f"cosine minDCF(10-3): {min_dcf:.2f} with threshold {threshold:.2f}")
        self.log("cosine_minDCF(10-3)", min_dcf, on_epoch=True, sync_dist=True)


class TestOnlyDataModule(LightningDataModule):
    def __init__(
        self,
        trial_path: str,
        root: str,
        num_workers: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.config = dict(config or {})
        self.config.setdefault("trial_path", trial_path)
        self.config.setdefault("root", root)
        self.trial_path = trial_path
        self.root = _ensure_root_suffix(root)
        self.num_workers = num_workers

    def test_dataloader(self):
        trials, root = load_validation_trials(self.config)
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print(f"number of enroll: {len(set(trials.T[1]))}")
        print(f"number of test: {len(set(trials.T[2]))}")
        print(f"number of evaluation: {len(eval_path)}")

        eval_dataset = Evaluation_Dataset(eval_path, root=root)
        return torch.utils.data.DataLoader(
            eval_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=1,
        )


def build_trainer(config: Dict[str, Any], mode: str) -> Trainer:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = int(config.get("devices", 1)) if accelerator == "gpu" else 1
    precision = "16-mixed" if accelerator == "gpu" else 32

    logger = False
    if mode in {"train", "test"} and _as_bool(config.get("USE_WANDB", False)):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=config.get("wandb_project", "mixup"),
            name=config.get("title", "caarma_experiment"),
            save_dir=config["save_dir"],
        )
        logger.experiment.config.update(config)

    callbacks = []
    if mode == "train":
        validate_during_train = _as_bool(config.get("validate_during_train", True))
        checkpoint_kwargs = (
            {
                "monitor": "cosine_eer",
                "mode": "min",
                "save_top_k": int(config.get("save_top_k", 3)),
                "save_last": True,
                "filename": "{epoch}_{cosine_eer:.2f}",
            }
            if validate_during_train
            else {
                "save_last": True,
                "save_top_k": 1,
                "filename": "{epoch}",
            }
        )
        callbacks.append(ModelCheckpoint(dirpath=config["save_dir"], **checkpoint_kwargs))
        if logger:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

    strategy = "auto"
    if accelerator == "gpu" and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    return Trainer(
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(config["epochs"]),
        logger=logger,
        num_sanity_val_steps=0,
        sync_batchnorm=accelerator == "gpu" and torch.cuda.device_count() > 1,
        precision=precision,
        callbacks=callbacks,
        default_root_dir=config["save_dir"],
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=(
            1.0
            if mode != "train" or _as_bool(config.get("validate_during_train", True))
            else 0
        ),
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
        profiler="simple" if mode == "train" else None,
    )


def _load_checkpoint_state_dict(checkpoint_path: str):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("Checkpoint must contain a state_dict mapping")
    state_dict = checkpoint["state_dict"]
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()):
        raise ValueError("Checkpoint state_dict must contain only string keys and tensors")
    return state_dict


def build_task(config: Dict[str, Any], device: str) -> Task:
    features = build_feature(config)
    model = build_model(config, device)
    criterion = build_experiment_criterion(config)
    task = Task(
        features,
        model,
        criterion,
        config,
        learning_rate=config["init_lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_epochs=config["epochs"],
        trial_path=config["trial_path"],
        warmup_step=config["warmup_step"],
    )

    checkpoint_path = config.get("checkpoint_path")
    if not _none_like(checkpoint_path):
        task.load_state_dict(_load_checkpoint_state_dict(checkpoint_path), strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    return task


def parse_experiment_args(default_config: str, description: str):
    parser = ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="Path to config YAML")
    parser.add_argument("--mode", choices=("train", "validate", "test"), default="train")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--trial-path", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--vox1-meta-path", default=None)
    parser.add_argument("--validation-split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--score-output-prefix", default=None)
    parser.add_argument("--sl-mixup", dest="sl_mixup", action="store_true", default=None)
    parser.add_argument("--no-sl-mixup", dest="sl_mixup", action="store_false")
    parser.add_argument(
        "--nationality-mixup",
        dest="nationality_mixup",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-nationality-mixup",
        dest="nationality_mixup",
        action="store_false",
    )
    return parser.parse_args()


def cli_main(
    default_config: str = "configs/base_clean_bridges2.yaml",
    default_experiment_type: str = "base",
    description: str = "Run a modular CAARMA experiment",
) -> None:
    args = parse_experiment_args(default_config, description)
    config_path = Path(args.config)
    config = prepare_experiment_config(
        load_config(str(config_path)),
        config_path,
        args,
        default_experiment_type=default_experiment_type,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Experiment:", config.get("experiment_type"))
    print("Synthetic strategy:", config.get("synthetic_strategy"))
    print("Condition attribute:", config.get("condition_attribute"))
    print("Adversarial enabled:", config.get("adversarial_enabled"))

    datamodule = super_dataset(config)
    task = build_task(config, device)
    trainer = build_trainer(config, args.mode)

    if args.mode == "train":
        trainer.fit(task, datamodule=datamodule)
    elif args.mode == "validate":
        trainer.validate(task, datamodule=datamodule)
    else:
        test_datamodule = TestOnlyDataModule(
            trial_path=config["trial_path"],
            root=config["root"],
            num_workers=min(int(config.get("num_workers", 10)), 10),
            config=config,
        )
        trainer.test(task, datamodule=test_datamodule)
