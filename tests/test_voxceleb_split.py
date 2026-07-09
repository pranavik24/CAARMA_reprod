import csv
import tempfile
import unittest
from pathlib import Path

from functions.voxceleb_split import (
    build_examples_from_split_csv,
    build_label_metadata_map,
    build_trials_from_split_csv,
    is_voxceleb_split_csv,
    load_training_dataframe,
    load_validation_trials,
    resolve_split_csv_path,
)


class VoxCelebSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.wav_root = self.root / "wav"
        self.split_dir = self.root / "data"
        self.split_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_split(self, name, rows):
        path = self.split_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["VoxCeleb1_ID", "VGGFace1_ID", "Gender", "Nationality", "Set"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def touch_wav(self, speaker_id, relative_path):
        wav_path = self.wav_root / speaker_id / relative_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"fake wav")
        return wav_path

    def test_detects_server_split_csv_shape(self):
        split_path = self.write_split(
            "vox1_train.csv",
            [
                {
                    "VoxCeleb1_ID": "id10002",
                    "VGGFace1_ID": "person",
                    "Gender": "m",
                    "Nationality": "USA",
                    "Set": "train",
                }
            ],
        )

        self.assertTrue(is_voxceleb_split_csv(split_path))

    def test_builds_deterministic_examples_from_speaker_split(self):
        split_path = self.write_split(
            "vox1_train.csv",
            [
                {
                    "VoxCeleb1_ID": "id10002",
                    "VGGFace1_ID": "person_b",
                    "Gender": "m",
                    "Nationality": "USA",
                    "Set": "train",
                },
                {
                    "VoxCeleb1_ID": "id10001",
                    "VGGFace1_ID": "person_a",
                    "Gender": "f",
                    "Nationality": "India",
                    "Set": "train",
                },
            ],
        )
        id10002_wav = self.touch_wav("id10002", "video_b/00001.wav")
        id10001_wav_a = self.touch_wav("id10001", "video_a/00002.wav")
        id10001_wav_b = self.touch_wav("id10001", "video_a/00001.wav")

        df = build_examples_from_split_csv(split_path, self.wav_root)

        self.assertEqual(
            df[
                [
                    "utt_spk_int_labels",
                    "utt_spk_id",
                    "utt_paths",
                    "gender",
                    "nationality",
                    "split",
                ]
            ].to_dict("records"),
            [
                {
                    "utt_spk_int_labels": 0,
                    "utt_spk_id": "id10001",
                    "utt_paths": str(id10001_wav_b),
                    "gender": "f",
                    "nationality": "India",
                    "split": "train",
                },
                {
                    "utt_spk_int_labels": 0,
                    "utt_spk_id": "id10001",
                    "utt_paths": str(id10001_wav_a),
                    "gender": "f",
                    "nationality": "India",
                    "split": "train",
                },
                {
                    "utt_spk_int_labels": 1,
                    "utt_spk_id": "id10002",
                    "utt_paths": str(id10002_wav),
                    "gender": "m",
                    "nationality": "USA",
                    "split": "train",
                },
            ],
        )

    def test_rejects_split_speakers_without_wav_files(self):
        split_path = self.write_split(
            "vox1_train.csv",
            [
                {
                    "VoxCeleb1_ID": "id10001",
                    "VGGFace1_ID": "person_a",
                    "Gender": "f",
                    "Nationality": "India",
                    "Set": "train",
                }
            ],
        )

        with self.assertRaisesRegex(FileNotFoundError, "id10001"):
            build_examples_from_split_csv(split_path, self.wav_root)

    def test_resolves_active_split_from_config_defaults(self):
        config = {
            "vox1_split_dir": str(self.split_dir),
            "active_split": "val",
        }

        self.assertEqual(
            resolve_split_csv_path(config),
            self.split_dir / "vox1_val.csv",
        )

    def test_load_training_dataframe_preserves_existing_format(self):
        legacy_path = self.root / "voxceleb_full.csv"
        with legacy_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["utt_spk_int_labels", "utt_paths"])
            writer.writeheader()
            writer.writerow({"utt_spk_int_labels": 7, "utt_paths": "x.wav"})

        df = load_training_dataframe({"dataset": str(legacy_path)})

        self.assertEqual(df.to_dict("records"), [{"utt_spk_int_labels": 7, "utt_paths": "x.wav"}])

    def test_builds_label_metadata_map_from_expanded_split(self):
        split_path = self.write_split(
            "vox1_train.csv",
            [
                {
                    "VoxCeleb1_ID": "id10002",
                    "VGGFace1_ID": "person_b",
                    "Gender": "M",
                    "Nationality": "USA",
                    "Set": "train",
                },
                {
                    "VoxCeleb1_ID": "id10001",
                    "VGGFace1_ID": "person_a",
                    "Gender": "F",
                    "Nationality": "India",
                    "Set": "train",
                },
            ],
        )
        self.touch_wav("id10002", "video_b/00001.wav")
        self.touch_wav("id10001", "video_a/00001.wav")

        config = {"dataset": str(split_path), "vox1_wav_root": str(self.wav_root)}

        self.assertEqual(build_label_metadata_map(config, "gender"), {0: "f", 1: "m"})
        self.assertEqual(build_label_metadata_map(config, "nationality"), {0: "india", 1: "usa"})

    def test_builds_balanced_validation_trials_from_split(self):
        split_path = self.write_split(
            "vox1_val.csv",
            [
                {
                    "VoxCeleb1_ID": "id10002",
                    "VGGFace1_ID": "person_b",
                    "Gender": "m",
                    "Nationality": "USA",
                    "Set": "val",
                },
                {
                    "VoxCeleb1_ID": "id10001",
                    "VGGFace1_ID": "person_a",
                    "Gender": "f",
                    "Nationality": "India",
                    "Set": "val",
                },
            ],
        )
        self.touch_wav("id10002", "video_b/00001.wav")
        self.touch_wav("id10002", "video_c/00001.wav")
        self.touch_wav("id10001", "video_a/00001.wav")
        self.touch_wav("id10001", "video_b/00001.wav")

        trials = build_trials_from_split_csv(split_path, self.wav_root)

        self.assertEqual(
            trials.tolist(),
            [
                ["1", "id10001/video_a/00001.wav", "id10001/video_b/00001.wav"],
                ["1", "id10002/video_b/00001.wav", "id10002/video_c/00001.wav"],
                ["0", "id10001/video_a/00001.wav", "id10002/video_b/00001.wav"],
                ["0", "id10002/video_b/00001.wav", "id10001/video_a/00001.wav"],
            ],
        )

    def test_load_validation_trials_generates_when_trial_file_is_missing(self):
        split_path = self.write_split(
            "vox1_val.csv",
            [
                {
                    "VoxCeleb1_ID": "id10001",
                    "VGGFace1_ID": "person_a",
                    "Gender": "f",
                    "Nationality": "India",
                    "Set": "val",
                },
                {
                    "VoxCeleb1_ID": "id10002",
                    "VGGFace1_ID": "person_b",
                    "Gender": "m",
                    "Nationality": "USA",
                    "Set": "val",
                },
            ],
        )
        self.touch_wav("id10001", "video_a/00001.wav")
        self.touch_wav("id10001", "video_b/00001.wav")
        self.touch_wav("id10002", "video_b/00001.wav")
        self.touch_wav("id10002", "video_c/00001.wav")

        trials, root = load_validation_trials(
            {
                "trial_path": str(self.root / "missing_trials.txt"),
                "generate_validation_trials": True,
                "val_split_csv": str(split_path),
                "vox1_wav_root": str(self.wav_root),
                "validation_split": "val",
            }
        )

        self.assertEqual(root, str(self.wav_root))
        self.assertEqual(trials.shape, (4, 3))

    def test_generated_validation_trials_require_cross_session_positive_pairs(self):
        split_path = self.write_split(
            "vox1_val.csv",
            [
                {
                    "VoxCeleb1_ID": "id10001",
                    "VGGFace1_ID": "person_a",
                    "Gender": "f",
                    "Nationality": "India",
                    "Set": "val",
                },
                {
                    "VoxCeleb1_ID": "id10002",
                    "VGGFace1_ID": "person_b",
                    "Gender": "m",
                    "Nationality": "USA",
                    "Set": "val",
                },
            ],
        )
        self.touch_wav("id10001", "video_a/00001.wav")
        self.touch_wav("id10001", "video_a/00002.wav")
        self.touch_wav("id10002", "video_b/00001.wav")
        self.touch_wav("id10002", "video_b/00002.wav")

        with self.assertRaisesRegex(ValueError, "different sessions"):
            build_trials_from_split_csv(split_path, self.wav_root)


if __name__ == "__main__":
    unittest.main()
