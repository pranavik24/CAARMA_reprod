import tempfile
import unittest
from pathlib import Path

import pandas as pd

from functions.loader import super_dataset


class TrainLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_legacy_train_csv(self, utterance_count):
        path = self.root / "train.csv"
        rows = [
            {
                "utt_spk_int_labels": index % 2,
                "utt_spk_id": f"id1000{index % 2}",
                "utt_paths": str(self.root / f"{index}.wav"),
            }
            for index in range(utterance_count)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def base_config(self, dataset_path, batch_size=2):
        return {
            "dataset": str(dataset_path),
            "second": 3,
            "do_augmentation": False,
            "num_workers": 0,
            "batch_size": batch_size,
            "train_drop_last": False,
        }

    def test_train_loader_drops_unsafe_singleton_remainder(self):
        dataset_path = self.write_legacy_train_csv(5)

        loader = super_dataset(self.base_config(dataset_path)).train_dataloader()

        self.assertTrue(loader.drop_last)

    def test_train_loader_keeps_non_singleton_partial_batch_when_configured(self):
        dataset_path = self.write_legacy_train_csv(6)

        loader = super_dataset(self.base_config(dataset_path, batch_size=4)).train_dataloader()

        self.assertFalse(loader.drop_last)


if __name__ == "__main__":
    unittest.main()
