"""Train CAARMA with synthetic pairs constrained by speaker nationality."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import os

import torch
import torch.nn as nn

from feature.build_feature import build_feature
from functions.loader import super_dataset
from functions.nationality import (
    build_label_nationality_map,
    canonical_speaker_pair,
    eligible_peer_count,
    same_nationality_candidates,
)
from mixup_gender import (
    Task,
    TestOnlyDataModule,
    _as_bool,
    _ensure_root_suffix,
    _nearest_candidate,
    _none_like,
    _resolve_path,
    accuracy,
    build_trainer,
    load_config,
)
from model.model_build import build_model


def prepare_config(config: Dict[str, Any], config_path: Path, args: Any) -> Dict[str, Any]:
    """Apply CLI overrides and make paths independent of the launch directory."""

    config_dir = config_path.resolve().parent
    prepared = dict(config)
    prepared["_mode"] = args.mode

    if args.nationality_mixup is not None:
        prepared["nationality_mixup"] = args.nationality_mixup
    else:
        prepared.setdefault("nationality_mixup", True)

    if args.vox1_meta_path is not None:
        prepared["vox1_meta_path"] = args.vox1_meta_path
    prepared.setdefault("vox1_meta_path", "vox1_meta.csv")

    for argument, key in (
        (args.checkpoint_path, "checkpoint_path"),
        (args.trial_path, "trial_path"),
        (args.root, "root"),
    ):
        if argument is not None:
            prepared[key] = argument

    for key in ("dataset", "trial_path", "checkpoint_path", "vox1_meta_path", "save_dir"):
        if key in prepared:
            prepared[key] = _resolve_path(prepared[key], config_dir)

    if "root" in prepared:
        prepared["root"] = _ensure_root_suffix(_resolve_path(prepared["root"], config_dir))
    return prepared


def mixup_data_euc_avg_nationality(
    x: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    label_to_nationality: Optional[Dict[int, str]] = None,
    nationality_mixup: bool = True,
):
    """Average each embedding with its nearest eligible in-batch speaker.

    When nationality conditioning is enabled, a speaker with no distinct
    same-nationality peer in the local batch is paired with itself. This safe
    fallback never violates the requested conditioning.
    """

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
        if nationality_mixup:
            candidates = same_nationality_candidates(
                speaker,
                unique_labels,
                label_to_nationality or {},
            )
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


class AMSoftmaxGANNationality(nn.Module):
    """AM-Softmax GAN criterion using nationality-conditioned synthetic classes."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30,
        label_to_nationality: Optional[Dict[int, str]] = None,
        nationality_mixup: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.nationality_mixup = nationality_mixup
        self.label_to_nationality = dict(label_to_nationality or {})
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print(f"Initialised AM-Softmax m={self.m:.3f} s={self.s:.3f}")
        print(f"Embedding dim is {embedding_dim}, number of speakers is {num_classes}")
        if self.nationality_mixup:
            nationality_count = len(set(self.label_to_nationality.values()))
            peer_count = eligible_peer_count(self.label_to_nationality)
            print(
                "Nationality-aware mixup enabled: "
                f"{len(self.label_to_nationality)} labels, {nationality_count} nationalities, "
                f"{peer_count} labels with an eligible global peer"
            )

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        flagSyn: bool = False,
    ):
        if label is None:
            raise ValueError("AMSoftmaxGANNationality requires speaker labels")
        if x.size(0) != label.size(0) or x.size(1) != self.in_feats:
            raise ValueError("Embedding and label shapes do not match the configured criterion")

        synthetic_embeddings, synthetic_labels, synthetic_weights = (
            mixup_data_euc_avg_nationality(
                x,
                self.W,
                label,
                label_to_nationality=self.label_to_nationality,
                nationality_mixup=self.nationality_mixup,
            )
        )

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


def build_nationality_criterion(config: Dict[str, Any]) -> nn.Module:
    if config["criterion"] != "AMSoftmaxGAN":
        raise NotImplementedError(
            "mixup_nationality.py implements nationality-conditioned mixup for AMSoftmaxGAN only."
        )

    enabled = _as_bool(config.get("nationality_mixup", True))
    label_to_nationality: Dict[int, str] = {}
    if enabled:
        try:
            label_to_nationality = build_label_nationality_map(
                config["dataset"],
                config["vox1_meta_path"],
            )
        except FileNotFoundError:
            if config.get("_mode") == "train":
                raise
            print("Warning: metadata unavailable; evaluation will proceed without a nationality map.")

        if config.get("_mode") == "train" and not label_to_nationality:
            raise ValueError(
                "Nationality mixup is enabled, but no training labels matched vox1_meta.csv. "
                "Include VoxCeleb speaker ids in utt_spk_id or utt_paths."
            )
        if config.get("_mode") == "train" and eligible_peer_count(label_to_nationality) == 0:
            raise ValueError("No mapped speaker has a distinct same-nationality peer.")

    return AMSoftmaxGANNationality(
        embedding_dim=int(config.get("embedding_dim", 192)),
        num_classes=int(config.get("num_spk", 1211)),
        margin=float(config.get("margin", 0.2)),
        scale=float(config.get("scale", 30)),
        label_to_nationality=label_to_nationality,
        nationality_mixup=enabled,
    )


def build_task(config: Dict[str, Any], device: str) -> Task:
    features = build_feature(config)
    model = build_model(config, device)
    criterion = build_nationality_criterion(config)
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
            raise ValueError("Checkpoint must contain a state_dict mapping")
        state_dict = checkpoint["state_dict"]
        if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()):
            raise ValueError("Checkpoint state_dict must contain only string keys and tensors")
        task.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    return task


def parse_args():
    parser = ArgumentParser(description="CAARMA training/testing with nationality-conditioned mixup")
    parser.add_argument("--config", default="config_nationality.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=("train", "validate", "test"), default="train")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--trial-path", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--vox1-meta-path", default=None)
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


def cli_main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = prepare_config(load_config(str(config_path)), config_path, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train" and device != "cuda":
        raise RuntimeError(
            "Training requires a CUDA allocation. On Bridges-2, submit through the GPU or "
            "GPU-shared partition instead of running on a login node."
        )

    print("Device:", device)
    print("Nationality mixup:", config.get("nationality_mixup"))
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
        )
        trainer.test(task, datamodule=test_datamodule)


if __name__ == "__main__":
    cli_main()
