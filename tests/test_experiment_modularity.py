import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiments.caarma import (
    AMSoftmaxGANExperiment,
    compute_eer,
    compute_min_dcf,
    prepare_experiment_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.split_dir = self.root / "data"
        self.split_dir.mkdir()
        self.train_split = self.split_dir / "vox1_train.csv"
        with self.train_split.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["VoxCeleb1_ID", "VGGFace1_ID", "Gender", "Nationality", "Set"],
            )
            writer.writeheader()
            writer.writerows(
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
                    {
                        "VoxCeleb1_ID": "id10002",
                        "VGGFace1_ID": "person_b_dup",
                        "Gender": "m",
                        "Nationality": "USA",
                        "Set": "train",
                    },
                ]
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def args(self, mode="train"):
        return Namespace(
            mode=mode,
            checkpoint_path=None,
            trial_path=None,
            root=None,
            vox1_meta_path=None,
            validation_split=None,
            score_output_prefix=None,
            sl_mixup=None,
            nationality_mixup=None,
        )

    def test_derives_num_speakers_from_active_server_split(self):
        config = prepare_experiment_config(
            {
                "experiment_type": "base",
                "condition_attribute": "none",
                "synthetic_strategy": "diffusion",
                "train_split_csv": str(self.train_split),
                "save_dir": str(self.root / "caarma-output" / "base-diffusion"),
                "wandb_project": "caarma-diffusion",
                "title": "caarma_base_diffusion",
                "score_output_prefix": "caarma_base_diffusion",
            },
            self.root / "configs" / "base_diffusion_bridges2.yaml",
            self.args(),
            default_experiment_type="base",
        )

        self.assertEqual(config["num_spk"], 2)
        self.assertEqual(config["condition_attribute"], "none")
        self.assertFalse(config["adversarial_enabled"])

    def test_config_relative_paths_preserve_legacy_repo_root_resolution(self):
        repo_data = self.split_dir
        trial_path = repo_data / "vox1_test.txt"
        trial_path.write_text("1 id10001/a.wav id10001/b.wav\n", encoding="utf-8")
        meta_path = self.root / "vox1_meta.csv"
        meta_path.write_text("VoxCeleb1_ID,Gender\nid10001,m\n", encoding="utf-8")

        config = prepare_experiment_config(
            {
                "experiment_type": "base",
                "condition_attribute": "none",
                "synthetic_strategy": "none",
                "train_split_csv": str(self.train_split),
                "trial_path": "data/vox1_test.txt",
                "vox1_meta_path": "vox1_meta.csv",
                "save_dir": "caarma-output/base-baseline",
                "wandb_project": "caarma-base",
                "title": "caarma_base_baseline",
                "score_output_prefix": "caarma_base_baseline",
            },
            self.root / "configs" / "base_baseline_no_false_bridges2.yaml",
            self.args(),
            default_experiment_type="base",
        )

        self.assertEqual(config["trial_path"], str(trial_path.resolve()))
        self.assertEqual(config["vox1_meta_path"], str(meta_path.resolve()))
        self.assertEqual(
            config["save_dir"],
            str((self.root / "caarma-output" / "base-baseline").resolve()),
        )

    def test_rejects_base_config_that_logs_to_gender_project(self):
        with self.assertRaisesRegex(ValueError, "base.*gender"):
            prepare_experiment_config(
                {
                    "experiment_type": "base",
                    "condition_attribute": "none",
                    "synthetic_strategy": "diffusion",
                    "train_split_csv": str(self.train_split),
                    "save_dir": str(self.root / "caarma-output" / "gender"),
                    "wandb_project": "caarma-gender",
                    "title": "caarma_gender",
                },
                self.root / "configs" / "base_diffusion_bridges2.yaml",
                self.args(),
                default_experiment_type="base",
            )


class ExperimentCriterionTests(unittest.TestCase):
    def test_none_strategy_has_zero_synthetic_loss(self):
        criterion = AMSoftmaxGANExperiment(
            embedding_dim=4,
            num_classes=3,
            synthetic_strategy="none",
        )
        embeddings = torch.randn(2, 4)
        labels = torch.tensor([0, 1])

        loss, acc, synthetic_embeddings = criterion(embeddings, labels, flagSyn=True)

        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(tuple(acc.shape), ())
        self.assertEqual(tuple(synthetic_embeddings.shape), (2, 4))

    def test_diffusion_fake_fraction_can_reduce_synthetic_batch(self):
        torch.manual_seed(7)
        criterion = AMSoftmaxGANExperiment(
            embedding_dim=4,
            num_classes=6,
            synthetic_strategy="diffusion",
            diffusion_t_min=0,
            diffusion_t_max=0,
            diffusion_fake_fraction=0.5,
        )
        embeddings = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 3])

        _, _, synthetic_embeddings = criterion(embeddings, labels, flagSyn=True)

        self.assertEqual(tuple(synthetic_embeddings.shape), (2, 4))

    def test_avg_strategy_can_condition_on_metadata(self):
        criterion = AMSoftmaxGANExperiment(
            embedding_dim=2,
            num_classes=3,
            synthetic_strategy="avg",
            condition_attribute="gender",
            label_to_condition={0: "f", 1: "m", 2: "f"},
        )
        with torch.no_grad():
            criterion.W.copy_(torch.tensor([[1.0, 3.0, 5.0], [0.0, 0.0, 0.0]]))
        embeddings = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        labels = torch.tensor([0, 1, 2])

        _, _, synthetic_embeddings = criterion(embeddings, labels, flagSyn=False)

        self.assertTrue(torch.equal(synthetic_embeddings[0], torch.tensor([3.0, 0.0])))
        self.assertTrue(torch.equal(synthetic_embeddings[1], torch.tensor([3.0, 0.0])))
        self.assertTrue(torch.equal(synthetic_embeddings[2], torch.tensor([3.0, 0.0])))


class SharedMetricTests(unittest.TestCase):
    def test_shared_metrics_compute_expected_range(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]

        eer, _ = compute_eer(labels, scores)
        mindcf, _ = compute_min_dcf(labels, scores)

        self.assertEqual(eer, 0.0)
        self.assertGreaterEqual(mindcf, 0.0)
        self.assertLessEqual(mindcf, 1.0)


if __name__ == "__main__":
    unittest.main()
