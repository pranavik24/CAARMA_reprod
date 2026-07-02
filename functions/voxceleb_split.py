"""Utilities for using server-staged VoxCeleb1 speaker split CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import os

import numpy as np
import pandas as pd


SPLIT_COLUMNS = ("VoxCeleb1_ID", "VGGFace1_ID", "Gender", "Nationality", "Set")
LEGACY_COLUMNS = ("utt_spk_int_labels", "utt_paths")
SERVER_WAV_ROOT = "/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1/wav"
SERVER_SPLIT_DIR = "/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1/data"


def _resolve_path(value: object) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def is_voxceleb_split_csv(path: str | Path) -> bool:
    """Return whether ``path`` has the server VoxCeleb1 split CSV shape."""

    columns = pd.read_csv(_resolve_path(path), nrows=0).columns
    return set(SPLIT_COLUMNS).issubset(set(columns))


def resolve_split_csv_path(config: Mapping[str, object]) -> Path:
    """Resolve the configured train/val/test split CSV path."""

    active_split = str(config.get("active_split", "train")).strip().lower()
    split_key = f"{active_split}_split_csv"
    configured_split = config.get(split_key)
    if configured_split:
        return _resolve_path(configured_split)

    split_dir = _resolve_path(config.get("vox1_split_dir", SERVER_SPLIT_DIR))
    return split_dir / f"vox1_{active_split}.csv"


def resolve_named_split_csv_path(config: Mapping[str, object], split_name: str) -> Path:
    split = str(split_name).strip().lower()
    split_key = f"{split}_split_csv"
    configured_split = config.get(split_key)
    if configured_split:
        return _resolve_path(configured_split)

    split_dir = _resolve_path(config.get("vox1_split_dir", SERVER_SPLIT_DIR))
    return split_dir / f"vox1_{split}.csv"


def _normalize_speaker_id(value: object) -> str:
    speaker_id = str(value).strip()
    if not speaker_id:
        raise ValueError("VoxCeleb1_ID values must be non-empty.")
    return speaker_id


def _speaker_wavs(wav_root: Path, speaker_id: str) -> list[Path]:
    return sorted((wav_root / speaker_id).rglob("*.wav"))


def build_examples_from_split_csv(
    split_csv_path: str | Path,
    wav_root: str | Path,
) -> pd.DataFrame:
    """Expand a speaker-level split CSV into one dataloader row per WAV file."""

    split_csv = _resolve_path(split_csv_path)
    wav_root_path = _resolve_path(wav_root)
    df = pd.read_csv(split_csv)
    missing_columns = sorted(set(SPLIT_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{split_csv} is missing required columns: {', '.join(missing_columns)}")

    speaker_rows = {}
    for _, row in df.iterrows():
        speaker_id = _normalize_speaker_id(row["VoxCeleb1_ID"])
        speaker_rows.setdefault(speaker_id, row)

    label_by_speaker = {
        speaker_id: index for index, speaker_id in enumerate(sorted(speaker_rows))
    }
    missing_speakers = []
    records = []
    for speaker_id in sorted(speaker_rows):
        row = speaker_rows[speaker_id]
        wav_paths = _speaker_wavs(wav_root_path, speaker_id)
        if not wav_paths:
            missing_speakers.append(speaker_id)
            continue

        for wav_path in wav_paths:
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

    if missing_speakers:
        preview = ", ".join(missing_speakers[:20])
        suffix = "" if len(missing_speakers) <= 20 else f", ... ({len(missing_speakers)} total)"
        raise FileNotFoundError(
            f"No WAV files found under {wav_root_path} for split speakers: {preview}{suffix}"
        )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "utt_spk_int_labels",
            "utt_spk_id",
            "utt_paths",
            "gender",
            "nationality",
            "split",
        ],
    )


def load_training_dataframe(config_or_path: Mapping[str, object] | str | Path) -> pd.DataFrame:
    """Load either the legacy dataloader CSV or server split CSV into dataloader rows."""

    if isinstance(config_or_path, Mapping):
        config = config_or_path
        dataset = config.get("dataset")
        candidate_path = _resolve_path(dataset) if dataset else resolve_split_csv_path(config)
        if is_voxceleb_split_csv(candidate_path):
            wav_root = config.get("vox1_wav_root", SERVER_WAV_ROOT)
            return build_examples_from_split_csv(candidate_path, str(wav_root))
        return pd.read_csv(candidate_path)

    return pd.read_csv(_resolve_path(config_or_path))


def build_label_metadata_map(
    config_or_path: Mapping[str, object] | str | Path,
    metadata_column: str,
) -> dict[int, str]:
    """Build ``speaker label -> metadata`` from an expanded training dataframe."""

    df = load_training_dataframe(config_or_path)
    if "utt_spk_int_labels" not in df.columns or metadata_column not in df.columns:
        return {}

    result = {}
    for _, row in df.iterrows():
        try:
            label = int(row["utt_spk_int_labels"])
        except (TypeError, ValueError):
            continue
        value = str(row[metadata_column]).strip().lower()
        if value and value not in {"nan", "none", "null", "n/a", "na"}:
            result[label] = value
    return result


def _relative_or_absolute(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_trials_from_split_csv(
    split_csv_path: str | Path,
    wav_root: str | Path,
    max_speakers: int | None = None,
) -> np.ndarray:
    """Build deterministic positive/negative verification trials from a split CSV."""

    wav_root_path = _resolve_path(wav_root)
    examples = build_examples_from_split_csv(split_csv_path, wav_root_path)
    grouped = {
        speaker_id: sorted(group["utt_paths"].tolist())
        for speaker_id, group in examples.groupby("utt_spk_id")
    }
    speakers = [speaker for speaker in sorted(grouped) if len(grouped[speaker]) >= 2]
    if max_speakers is not None and max_speakers > 0:
        speakers = speakers[:max_speakers]
    if len(speakers) < 2:
        raise ValueError("At least two speakers with two WAV files each are required for validation trials.")

    records = []
    for speaker in speakers:
        wavs = grouped[speaker]
        records.append(
            [
                "1",
                _relative_or_absolute(wavs[0], wav_root_path),
                _relative_or_absolute(wavs[1], wav_root_path),
            ]
        )

    for left, right in zip(speakers, speakers[1:] + speakers[:1]):
        records.append(
            [
                "0",
                _relative_or_absolute(grouped[left][0], wav_root_path),
                _relative_or_absolute(grouped[right][0], wav_root_path),
            ]
        )

    return np.array(records, dtype=str)


def load_validation_trials(config: Mapping[str, object]) -> tuple[np.ndarray, str]:
    """Load official trials or generate split-based validation trials."""

    trial_path = config.get("trial_path")
    if trial_path and _resolve_path(trial_path).exists():
        return np.loadtxt(_resolve_path(trial_path), str), str(config["root"])

    if str(config.get("generate_validation_trials", False)).strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
    }:
        raise FileNotFoundError(f"{trial_path} not found.")

    split_name = str(config.get("validation_split", "val"))
    split_path = resolve_named_split_csv_path(config, split_name)
    wav_root = str(config.get("vox1_wav_root", SERVER_WAV_ROOT))
    max_speakers_value = config.get("validation_max_speakers", 300)
    max_speakers = int(max_speakers_value) if max_speakers_value is not None else None
    return build_trials_from_split_csv(split_path, wav_root, max_speakers=max_speakers), wav_root
