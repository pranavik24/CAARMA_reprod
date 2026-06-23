import csv
import tempfile
import unittest
from pathlib import Path

from functions.nationality import (
    build_label_nationality_map,
    canonical_speaker_pair,
    eligible_peer_count,
    load_vox1_nationality_by_speaker,
    same_nationality_candidates,
)


class NationalityMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_rows(self, path: Path, fieldnames, rows, delimiter=","):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_and_normalizes_nationalities(self):
        meta_path = self.root / "vox1_meta.csv"
        self.write_rows(
            meta_path,
            ["VoxCeleb1 ID", "Nationality"],
            [
                {"VoxCeleb1 ID": "id10001", "Nationality": " USA "},
                {"VoxCeleb1 ID": "id10002", "Nationality": "India"},
                {"VoxCeleb1 ID": "id10003", "Nationality": ""},
                {"VoxCeleb1 ID": "id10004", "Nationality": "NaN"},
            ],
            delimiter="\t",
        )

        self.assertEqual(
            load_vox1_nationality_by_speaker(meta_path),
            {"id10001": "usa", "id10002": "india"},
        )

    def test_missing_nationality_column_is_rejected(self):
        meta_path = self.root / "vox1_meta.csv"
        self.write_rows(
            meta_path,
            ["VoxCeleb1 ID", "Gender"],
            [{"VoxCeleb1 ID": "id10001", "Gender": "m"}],
            delimiter="\t",
        )

        with self.assertRaisesRegex(ValueError, "Nationality"):
            load_vox1_nationality_by_speaker(meta_path)

    def test_builds_label_map_from_speaker_columns_and_paths(self):
        meta_path = self.root / "vox1_meta.csv"
        train_path = self.root / "train.csv"
        self.write_rows(
            meta_path,
            ["VoxCeleb1 ID", "Nationality"],
            [
                {"VoxCeleb1 ID": "id10001", "Nationality": "USA"},
                {"VoxCeleb1 ID": "id10002", "Nationality": "India"},
            ],
            delimiter="\t",
        )
        self.write_rows(
            train_path,
            ["utt_spk_int_labels", "utt_spk_id", "utt_paths"],
            [
                {"utt_spk_int_labels": "0", "utt_spk_id": "id10001", "utt_paths": ""},
                {
                    "utt_spk_int_labels": "1",
                    "utt_spk_id": "",
                    "utt_paths": "/data/id10002/video/audio.wav",
                },
                {"utt_spk_int_labels": "2", "utt_spk_id": "unknown", "utt_paths": "x.wav"},
            ],
        )

        self.assertEqual(
            build_label_nationality_map(train_path, meta_path),
            {0: "usa", 1: "india"},
        )


class NationalityCandidateTests(unittest.TestCase):
    def test_symmetric_pairs_share_one_synthetic_class_key(self):
        self.assertEqual(canonical_speaker_pair(4, 2), (2, 4))
        self.assertEqual(canonical_speaker_pair(2, 4), (2, 4))
        self.assertEqual(canonical_speaker_pair(3, 3), (3, 3))

    def test_candidates_are_restricted_to_same_nationality(self):
        mapping = {0: "usa", 1: "india", 2: "usa", 3: "usa"}
        self.assertEqual(same_nationality_candidates(0, [0, 1, 2, 3], mapping), [2, 3])
        self.assertEqual(same_nationality_candidates(1, [0, 1, 2, 3], mapping), [])
        self.assertEqual(same_nationality_candidates(4, [0, 1, 2, 3, 4], mapping), [])

    def test_reports_labels_that_can_actually_mix(self):
        mapping = {0: "usa", 1: "india", 2: "usa", 3: "canada"}
        self.assertEqual(eligible_peer_count(mapping), 2)


if __name__ == "__main__":
    unittest.main()
