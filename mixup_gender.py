from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import spectral_norm

from feature.build_feature import build_feature
from functions.dataset import Evaluation_Dataset
from functions.loader import super_dataset
from functions.voxceleb_split import build_label_metadata_map, load_validation_trials
from model.discriminator_mix import InterDiscrim, SimpleDiscriminator
from model.model_build import build_model


SPEAKER_ID_RE = re.compile(r"id\d{5}")


def load_config(config_file_path: str) -> Dict[str, Any]:
    with open(config_file_path) as file:
        return yaml.safe_load(file)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _none_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


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


def prepare_config(config: Dict[str, Any], config_path: Path, args: Any) -> Dict[str, Any]:
    config_dir = config_path.resolve().parent
    config = dict(config)
    config["_mode"] = args.mode

    if args.sl_mixup is not None:
        config["sl_mixup"] = args.sl_mixup
    else:
        config.setdefault("sl_mixup", True)

    if args.vox1_meta_path is not None:
        config["vox1_meta_path"] = args.vox1_meta_path
    config.setdefault("vox1_meta_path", "vox1_meta.csv")

    if args.checkpoint_path is not None:
        config["checkpoint_path"] = args.checkpoint_path
    if args.trial_path is not None:
        config["trial_path"] = args.trial_path
    if args.root is not None:
        config["root"] = args.root

    for key in ("dataset", "trial_path", "checkpoint_path", "vox1_meta_path", "save_dir"):
        if key in config:
            config[key] = _resolve_path(config[key], config_dir)

    if "root" in config:
        resolved_root = _resolve_path(config["root"], config_dir)
        config["root"] = _ensure_root_suffix(resolved_root)

    return config


def extract_speaker_id(value: Any) -> Optional[str]:
    match = SPEAKER_ID_RE.search(str(value))
    return match.group(0) if match else None


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


def mixup_data_euc_avg_gender(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    label_to_gender: Optional[Dict[int, str]] = None,
    sl_mixup: bool = False,
):
    batch_size = x.size(0)
    device = x.device
    label_values = [int(label) for label in labels.detach().cpu().tolist()]
    unique_labels = sorted(set(label_values))
    label_positions = defaultdict(list)
    for position, label in enumerate(label_values):
        label_positions[label].append(position)

    pair_by_label: Dict[int, int] = {}
    for speaker in unique_labels:
        candidates = [candidate for candidate in unique_labels if candidate != speaker]

        if sl_mixup:
            gender = (label_to_gender or {}).get(speaker)
            if gender is None:
                candidates = []
            else:
                candidates = [
                    candidate
                    for candidate in candidates
                    if (label_to_gender or {}).get(candidate) == gender
                ]

        pair_by_label[speaker] = _nearest_candidate(speaker, candidates, weights)

    w_mix = weights.new_zeros((weights.size(0), batch_size))
    y_mix = torch.zeros(batch_size, dtype=torch.long, device=device)
    pair_indices = []
    synthetic_label_by_pair = {}
    next_synthetic_label = 0

    for row_idx, speaker in enumerate(label_values):
        paired_speaker = pair_by_label[speaker]
        paired_positions = label_positions.get(paired_speaker, [row_idx])
        pair_indices.append(paired_positions[0] if paired_speaker != speaker else row_idx)

        pair_key = (min(speaker, paired_speaker), max(speaker, paired_speaker))
        if pair_key not in synthetic_label_by_pair:
            synthetic_label_by_pair[pair_key] = next_synthetic_label
            w_mix[:, next_synthetic_label] = (
                weights[:, speaker] + weights[:, paired_speaker]
            ) / 2
            next_synthetic_label += 1
        y_mix[row_idx] = synthetic_label_by_pair[pair_key]

    pair_index_tensor = torch.tensor(pair_indices, dtype=torch.long, device=device)
    x_mix = 0.5 * (x + x.index_select(0, pair_index_tensor))
    return x_mix, y_mix, w_mix[:, :next_synthetic_label].to(device)


def diffusion_mixup(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    diffusion_timesteps: int = 100,
    diffusion_t_min: int = 1,
    diffusion_t_max: int = 20,
    diffusion_beta_start: float = 0.0001,
    diffusion_beta_end: float = 0.02,
    diffusion_embedding_noise: float = 0.0,
):
    """Create non-conditional diffusion-style extensions of speaker weights."""
    if diffusion_timesteps <= 0:
        raise ValueError("diffusion_timesteps must be positive")
    if diffusion_t_min < 0 or diffusion_t_max < diffusion_t_min:
        raise ValueError("diffusion timestep range is invalid")
    if diffusion_t_max >= diffusion_timesteps:
        raise ValueError("diffusion_t_max must be smaller than diffusion_timesteps")

    batch_size = x.size(0)
    device = x.device
    dtype = x.dtype
    labels_for_weights = labels.to(device=weights.device, dtype=torch.long)
    selected_weights = weights.index_select(1, labels_for_weights).to(device=device, dtype=dtype)

    betas = torch.linspace(
        diffusion_beta_start,
        diffusion_beta_end,
        diffusion_timesteps,
        device=device,
        dtype=dtype,
    )
    alphas = 1.0 - betas
    alpha_bar_tail = torch.cumprod(alphas, dim=0)
    alpha_bars = torch.cat((torch.ones(1, device=device, dtype=dtype), alpha_bar_tail[:-1]))

    timesteps = torch.randint(
        diffusion_t_min,
        diffusion_t_max + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    alpha_bar_t = alpha_bars.index_select(0, timesteps).view(1, batch_size)
    epsilon = torch.randn_like(selected_weights)
    synthetic_weights = (
        torch.sqrt(alpha_bar_t) * selected_weights
        + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    )
    synthetic_weights = synthetic_weights / torch.norm(
        synthetic_weights,
        p=2,
        dim=0,
        keepdim=True,
    ).clamp(min=1e-12)

    if diffusion_embedding_noise > 0:
        synthetic_embeddings = x + diffusion_embedding_noise * torch.randn_like(x)
    else:
        synthetic_embeddings = x

    synthetic_labels = torch.arange(batch_size, dtype=torch.long, device=device)
    return synthetic_embeddings, synthetic_labels, synthetic_weights


class AMSoftmaxGANGender(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30,
        label_to_gender: Optional[Dict[int, str]] = None,
        sl_mixup: bool = False,
        synthetic_strategy: str = "avg",
        diffusion_timesteps: int = 100,
        diffusion_t_min: int = 1,
        diffusion_t_max: int = 20,
        diffusion_beta_start: float = 0.0001,
        diffusion_beta_end: float = 0.02,
        diffusion_embedding_noise: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.sl_mixup = sl_mixup
        self.label_to_gender = label_to_gender or {}
        self.synthetic_strategy = str(synthetic_strategy).strip().lower()
        self.diffusion_timesteps = diffusion_timesteps
        self.diffusion_t_min = diffusion_t_min
        self.diffusion_t_max = diffusion_t_max
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        self.diffusion_embedding_noise = diffusion_embedding_noise
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print("Initialised AM-Softmax m=%.3f s=%.3f" % (self.m, self.s))
        print(f"Embedding dim is {embedding_dim}, number of speakers is {num_classes}")
        if self.sl_mixup:
            print(f"Gender-aware sl_mixup enabled for {len(self.label_to_gender)} labels")
        if self.synthetic_strategy == "diffusion":
            print(
                "Diffusion mixup enabled: "
                f"timesteps={self.diffusion_timesteps}, "
                f"t=[{self.diffusion_t_min}, {self.diffusion_t_max}]"
            )

    def forward(self, x: torch.Tensor, label: torch.Tensor = None, flagSyn: bool = False):
        assert label is not None
        assert x.size(0) == label.size(0)
        assert x.size(1) == self.in_feats

        if self.synthetic_strategy == "diffusion":
            synthetic_embeddings, y_combined, w_combined = diffusion_mixup(
                x,
                self.W,
                label,
                diffusion_timesteps=self.diffusion_timesteps,
                diffusion_t_min=self.diffusion_t_min,
                diffusion_t_max=self.diffusion_t_max,
                diffusion_beta_start=self.diffusion_beta_start,
                diffusion_beta_end=self.diffusion_beta_end,
                diffusion_embedding_noise=self.diffusion_embedding_noise,
            )
        else:
            synthetic_embeddings, y_combined, w_combined = mixup_data_euc_avg_gender(
                x,
                self.W,
                label,
                label_to_gender=self.label_to_gender,
                sl_mixup=self.sl_mixup,
            )

        if flagSyn:
            x_for_loss = synthetic_embeddings.to(x.device)
            w_for_loss = w_combined.to(x.device)
            y_for_loss = y_combined.to(x.device)
        else:
            x_for_loss = x
            w_for_loss = self.W
            y_for_loss = label

        x_norm = x_for_loss / torch.norm(x_for_loss, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        w_norm = w_for_loss / torch.norm(w_for_loss, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        costh = torch.mm(x_norm, w_norm)

        label_view = y_for_loss.view(-1, 1).to(device=x.device)
        delt_costh = torch.zeros(costh.size(), device=x.device).scatter_(1, label_view, self.m)
        logits = self.s * (costh - delt_costh)

        loss = self.ce(logits, y_for_loss)
        acc = accuracy(logits.detach(), y_for_loss.detach(), topk=(1,))[0]
        return loss, acc, synthetic_embeddings


def build_gender_criterion(config: Dict[str, Any]) -> nn.Module:
    if config["criterion"] != "AMSoftmaxGAN":
        raise NotImplementedError(
            "mixup_gender.py currently implements gender-aware mixup for AMSoftmaxGAN only."
        )

    synthetic_strategy = str(config.get("synthetic_strategy", "avg")).strip().lower()
    sl_mixup = _as_bool(config.get("sl_mixup", True))
    uses_gender_mixup = sl_mixup and synthetic_strategy != "diffusion"
    label_to_gender = {}
    if uses_gender_mixup:
        try:
            label_to_gender = build_label_metadata_map(config, "gender")
            if not label_to_gender:
                label_to_gender = build_label_gender_map(
                    config["dataset"],
                    config.get("vox1_meta_path", "vox1_meta.csv"),
                )
        except FileNotFoundError:
            if config.get("_mode") == "train":
                raise
            print(
                "Warning: train CSV or vox1_meta.csv was not found; "
                "test/validate will proceed without a gender map."
            )
        if config.get("_mode") == "train" and not label_to_gender:
            raise ValueError(
                "sl_mixup is enabled, but no training labels could be matched to vox1_meta.csv. "
                "Make sure the train CSV contains VoxCeleb speaker ids in a speaker/path column."
            )

    return AMSoftmaxGANGender(
        embedding_dim=int(config.get("embedding_dim", 192)),
        num_classes=int(config.get("num_spk", 1211)),
        margin=float(config.get("margin", 0.2)),
        scale=float(config.get("scale", 30)),
        label_to_gender=label_to_gender,
        sl_mixup=uses_gender_mixup,
        synthetic_strategy=synthetic_strategy,
        diffusion_timesteps=int(config.get("diffusion_timesteps", 100)),
        diffusion_t_min=int(config.get("diffusion_t_min", 1)),
        diffusion_t_max=int(config.get("diffusion_t_max", 20)),
        diffusion_beta_start=float(config.get("diffusion_beta_start", 0.0001)),
        diffusion_beta_end=float(config.get("diffusion_beta_end", 0.02)),
        diffusion_embedding_noise=float(config.get("diffusion_embedding_noise", 0.0)),
    )


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


class MixupDiscriminator(nn.Module):
    def __init__(
        self,
        hubert_model_name: str = "facebook/hubert-large-ls960-ft",
        cache_dir: str = "",
        proj_dim: int = 256,
        emb_dim: int = 192,
        hidden_dim: int = 256,
        mid_dim: int = 128,
        dropout_rate: float = 0.1,
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
                "MixupDiscriminator expects pooled embeddings of shape (B, E), "
                f"but got {tuple(input_embedding.shape)}"
            )
        return self.discriminator(input_embedding)


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
        return MixupDiscriminator(
            emb_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
    raise ValueError(f"Unsupported discriminator: {config.get('discriminator')}")


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
        self.discriminator_learning_rate = float(
            self.config.get("discriminator_lr", self.learning_rate * 0.01)
        )
        self.lambda_adv = float(self.config.get("lambda_adv", 0.01))
        self.lambda_adv_min = float(self.config.get("lambda_adv_min", 0.0001))
        self.lambda_adv_max = float(self.config.get("lambda_adv_max", 0.01))
        self.lambda_adv_pretrain = float(self.config.get("lambda_adv_pretrain", 0.0005))
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
        self.automatic_optimization = False

        self.discriminator = build_embedding_discriminator(self.config).train()
        self.BCE_loss = nn.BCEWithLogitsLoss()

        self.pretrain_eps = int(self.config.get("pretrain_eps", 15))
        self.pretrain_discriminator = True
        self.discriminator_steps = 0

    def normalize(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return torch.div(x, x_norm)

    def set_discriminator_requires_grad(self, requires_grad: bool):
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(requires_grad)

    def forward(self, x):
        feature = self.features(x)
        embedding = self.model(feature)
        return embedding

    def adjust_weight(self, amsoftmax_loss, g_loss):
        loss_ratio = amsoftmax_loss / (g_loss + 1e-8)
        if loss_ratio > 1.5:
            self.lambda_adv = min(self.lambda_adv * 1.1, self.lambda_adv_max)
        elif loss_ratio < 0.5:
            self.lambda_adv = max(self.lambda_adv * 0.9, self.lambda_adv_min)
        return self.lambda_adv

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        d_sch = self.lr_schedulers()
        optimizer_main, d_optimizer = opt
        main_scheduler, d_scheduler = d_sch

        waveform = batch["waveform"]
        label = batch["mapped_id"]

        feature = self.features(waveform)
        embedding = self.model(feature)
        amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)

        if not hasattr(self, "d_step_counter"):
            self.d_step_counter = 0
        if not hasattr(self, "g_step_counter"):
            self.g_step_counter = 0

        if self.d_step_counter >= 1 and self.g_step_counter >= 5:
            self.d_step_counter = 0
            self.g_step_counter = 0
            if self.log_training_steps:
                print("set 0 pre-training")
        elif (
            self.d_step_counter >= 1
            and self.g_step_counter >= 1
            and self.current_epoch > self.pretrain_eps
        ):
            self.d_step_counter = 0
            self.g_step_counter = 0
            if self.log_training_steps:
                print("set 0 discriminator")

        if self.current_epoch <= self.pretrain_eps:
            if self.g_step_counter < 5:
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
                amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
                amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(
                    embedding,
                    label,
                    flagSyn=True,
                )
                total_loss = (
                    amsoftmax_loss
                    + (1 / self.config["num_spk"]) * amsoftmax_syn_loss
                    + self.lambda_adv_pretrain * g_loss
                )
                self.manual_backward(total_loss)
                self.log("am_loss", amsoftmax_loss, prog_bar=True)
                self.log("am_loss_syn", amsoftmax_syn_loss, prog_bar=True)
                self.log("acc", acc, prog_bar=True)
                self.log("g_loss", g_loss, prog_bar=True)
                self.log("total_loss", total_loss, prog_bar=True)
                optimizer_main.step()
                if self.trainer.global_step < self.config["warmup_step"]:
                    lr_scale = min(
                        1.0,
                        float(self.trainer.global_step + 1)
                        / float(self.config["warmup_step"]),
                    )
                    for pg in optimizer_main.param_groups:
                        pg["lr"] = lr_scale * self.learning_rate
                if self.log_training_steps:
                    print("gloss")
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
                if self.log_training_steps:
                    print("d_loss")
                return d_loss

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
            if self.log_training_steps:
                print("d_loss")
            return d_loss

        if self.d_step_counter >= 1 and self.g_step_counter < 1:
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

            amsoftmax_loss, acc, synthetic_embeddings = self.loss(embedding, label)
            amsoftmax_syn_loss, acc_syn, synthetic_embeddings = self.loss_syn(
                embedding,
                label,
                flagSyn=True,
            )
            self.lambda_adv = self.adjust_weight(amsoftmax_loss, g_loss)
            total_loss = (
                amsoftmax_loss
                + (1 / self.config["num_spk"]) * amsoftmax_syn_loss
                + self.lambda_adv * g_loss
            )
            self.manual_backward(total_loss)
            self.log("am_loss", amsoftmax_loss, prog_bar=True)
            self.log("am_loss_syn", amsoftmax_syn_loss, prog_bar=True)
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
            if self.log_training_steps:
                print("gloss")
            self.g_step_counter += 1
            self.set_discriminator_requires_grad(True)
            return total_loss

        return amsoftmax_loss

    def configure_optimizers(self):
        embedding_optimizer = AdamW(
            list(self.model.parameters()) + list(self.loss.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        discriminator_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.5, 0.999),
        )

        embedding_scheduler = StepLR(embedding_optimizer, step_size=4, gamma=0.5)
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size=4, gamma=0.5)

        return [embedding_optimizer, discriminator_optimizer], [
            embedding_scheduler,
            discriminator_scheduler,
        ]

    def on_train_epoch_end(self):
        main_scheduler, d_scheduler = self.lr_schedulers()
        main_scheduler.step()
        d_scheduler.step()

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
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                score = enroll_vector.dot(test_vector.T)
                denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
                score = score / (denom + epsilon)
            if np.isnan(score):
                print("Warning: NaN detected in score calculation. Setting score to 0.")
                score = 0.0
            labels.append(int(item[0]))
            scores.append(score)

        scoress = torch.tensor(scores)
        meanscores = torch.mean(scoress)
        print(meanscores)
        return labels, scores

    def compute_eer(self, labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        threshold = interp1d(fpr, thresholds)(eer)
        return eer, threshold

    def compute_minDCF(self, labels, scores, p_target=0.01, c_miss=1, c_fa=1):
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
        min_dcf = min_c_det / c_def
        return min_dcf, min_c_det_threshold

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
        eer, threshold = self.compute_eer(labels, scores)

        output_prefix = self.config.get("score_output_prefix", "org_inf")
        with open(f"{output_prefix}_labels.txt", "w") as f:
            for line in labels:
                f.write(f"{line}\n")
        with open(f"{output_prefix}_scores.txt", "w") as f:
            for line in scores:
                f.write(f"{line}\n")

        print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(eer * 100, threshold))
        self.log("cosine_eer", eer * 100, on_epoch=True, sync_dist=True)

        minDCF, threshold = self.compute_minDCF(labels, scores, p_target=0.01)
        print("cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-2)", minDCF, on_epoch=True, sync_dist=True)

        minDCF, threshold = self.compute_minDCF(labels, scores, p_target=0.001)
        print("cosine minDCF(10-3): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-3)", minDCF, on_epoch=True, sync_dist=True)


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
        print("number of enroll:", len(set(trials.T[1])))
        print("number of test:", len(set(trials.T[2])))
        print("number of evaluation:", len(eval_path))

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
    precision = 16 if accelerator == "gpu" else 32

    logger = False
    if mode in {"train", "test"} and _as_bool(config.get("USE_WANDB", False)):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=config.get("wandb_project", "mixup"),
            name=config.get("title", "mixup_gender"),
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
        callbacks.append(
            ModelCheckpoint(
                dirpath=config["save_dir"],
                **checkpoint_kwargs,
            )
        )
        if logger:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

    strategy = "auto"
    if accelerator == "gpu" and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    return Trainer(
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        max_epochs=config["epochs"],
        logger=logger,
        num_sanity_val_steps=0,
        sync_batchnorm=accelerator == "gpu" and torch.cuda.device_count() > 1,
        precision=precision,
        callbacks=callbacks,
        default_root_dir=config["save_dir"],
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=1.0 if mode != "train" or _as_bool(config.get("validate_during_train", True)) else 0,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        benchmark=True,
        deterministic=False,
        profiler="simple" if mode == "train" else None,
    )


def build_task(config: Dict[str, Any], device: str) -> Task:
    features = build_feature(config)
    model = build_model(config, device)
    criterion = build_gender_criterion(config)
    final_project = Task(
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
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        final_project.load_state_dict(state_dict, strict=False)
        print(f"load weight from {checkpoint_path}")

    return final_project


def parse_args():
    parser = ArgumentParser(description="CAARMA training/testing with gender-aware sl_mixup")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--mode",
        choices=("train", "validate", "test"),
        default="train",
        help="Run trainer.fit, trainer.validate, or trainer.test",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--trial-path", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--vox1-meta-path", default=None)
    parser.add_argument("--sl-mixup", dest="sl_mixup", action="store_true", default=None)
    parser.add_argument("--no-sl-mixup", dest="sl_mixup", action="store_false")
    return parser.parse_args()


def cli_main():
    args = parse_args()
    config_path = Path(args.config)
    config = prepare_config(load_config(str(config_path)), config_path, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("sl_mixup:", config.get("sl_mixup"))

    dataloader = super_dataset(config)
    final_project = build_task(config, device)
    trainer = build_trainer(config, args.mode)

    if args.mode == "train":
        trainer.fit(final_project, datamodule=dataloader)
    elif args.mode == "validate":
        trainer.validate(final_project, datamodule=dataloader)
    else:
        test_dm = TestOnlyDataModule(
            trial_path=config["trial_path"],
            root=config["root"],
            num_workers=min(int(config.get("num_workers", 10)), 10),
        )
        trainer.test(final_project, datamodule=test_dm)


if __name__ == "__main__":
    cli_main()
