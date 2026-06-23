"""Nationality metadata helpers for conditioned CAARMA mixup."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional


SPEAKER_ID_RE = re.compile(r"id\d{5}", re.IGNORECASE)
MISSING_VALUES = {"", "nan", "none", "null", "n/a", "na"}
LABEL_COLUMNS = ("utt_spk_int_labels", "mapped_id", "label", "speaker_label")
SPEAKER_COLUMNS = (
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


def extract_speaker_id(value: object) -> Optional[str]:
    """Return a normalized VoxCeleb1 speaker id embedded in ``value``."""

    match = SPEAKER_ID_RE.search(str(value))
    return match.group(0).lower() if match else None


def normalize_nationality(value: object) -> Optional[str]:
    """Normalize a nationality, treating common CSV missing values as absent."""

    normalized = str(value).strip().casefold()
    return None if normalized in MISSING_VALUES else normalized


def _read_dict_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {source}")
        fieldnames = [field.strip() for field in reader.fieldnames]
        rows = [
            {str(key).strip(): value for key, value in row.items() if key is not None}
            for row in reader
        ]
    return fieldnames, rows


def _find_column(columns: Iterable[str], accepted: Iterable[str]) -> Optional[str]:
    by_normalized = {column.strip().casefold().replace(" ", ""): column for column in columns}
    for candidate in accepted:
        match = by_normalized.get(candidate.strip().casefold().replace(" ", ""))
        if match is not None:
            return match
    return None


def load_vox1_nationality_by_speaker(meta_path: str | Path) -> dict[str, str]:
    """Load ``speaker id -> nationality`` from the official VoxCeleb1 metadata."""

    columns, rows = _read_dict_rows(meta_path)
    id_column = _find_column(columns, ("VoxCeleb1 ID", "vox1_id", "speaker_id"))
    if id_column is None:
        id_column = columns[0]
    nationality_column = _find_column(columns, ("Nationality",))
    if nationality_column is None:
        raise ValueError(f"No Nationality column found in {meta_path}")

    result: dict[str, str] = {}
    for row in rows:
        speaker_id = extract_speaker_id(row.get(id_column, ""))
        nationality = normalize_nationality(row.get(nationality_column, ""))
        if speaker_id is not None and nationality is not None:
            result[speaker_id] = nationality
    return result


def build_label_nationality_map(
    train_csv_path: str | Path,
    meta_path: str | Path,
) -> dict[int, str]:
    """Join integer training labels to nationalities through VoxCeleb speaker ids."""

    speaker_to_nationality = load_vox1_nationality_by_speaker(meta_path)
    columns, rows = _read_dict_rows(train_csv_path)
    label_column = _find_column(columns, LABEL_COLUMNS)
    if label_column is None:
        expected = ", ".join(LABEL_COLUMNS)
        raise ValueError(f"No integer speaker label column found in {train_csv_path}; expected {expected}.")

    preferred_columns = [
        column
        for candidate in SPEAKER_COLUMNS
        if (column := _find_column(columns, (candidate,))) is not None
    ]
    search_columns = list(dict.fromkeys(preferred_columns)) or columns

    result: dict[int, str] = {}
    for row in rows:
        try:
            label = int(row.get(label_column, ""))
        except (TypeError, ValueError):
            continue

        speaker_id = next(
            (
                candidate
                for column in search_columns
                if (candidate := extract_speaker_id(row.get(column, ""))) is not None
            ),
            None,
        )
        if speaker_id is None:
            speaker_id = next(
                (
                    candidate
                    for value in row.values()
                    if (candidate := extract_speaker_id(value)) is not None
                ),
                None,
            )

        nationality = speaker_to_nationality.get(speaker_id or "")
        if nationality is not None:
            result[label] = nationality
    return result


def same_nationality_candidates(
    speaker: int,
    candidate_labels: Iterable[int],
    label_to_nationality: Mapping[int, str],
) -> list[int]:
    """Return distinct candidate labels with the speaker's nationality."""

    nationality = label_to_nationality.get(speaker)
    if nationality is None:
        return []
    return [
        candidate
        for candidate in candidate_labels
        if candidate != speaker and label_to_nationality.get(candidate) == nationality
    ]


def canonical_speaker_pair(speaker: int, paired_speaker: int) -> tuple[int, int]:
    """Return one stable key for a symmetric speaker average."""

    return (min(speaker, paired_speaker), max(speaker, paired_speaker))


def eligible_peer_count(label_to_nationality: Mapping[int, str]) -> int:
    """Count mapped labels that have at least one distinct same-nationality peer."""

    counts = Counter(label_to_nationality.values())
    return sum(counts[nationality] > 1 for nationality in label_to_nationality.values())
