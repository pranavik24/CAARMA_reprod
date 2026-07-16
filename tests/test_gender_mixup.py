import unittest
from tempfile import TemporaryDirectory

import torch

from mixup_gender import Task, build_trainer, mixup_data_euc_avg_gender


class GenderMixupTests(unittest.TestCase):
    def test_symmetric_gender_pair_uses_one_synthetic_class(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [3.0, 0.0],
            ]
        )
        weights = torch.tensor(
            [
                [1.0, 3.0],
                [0.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1])

        mixed_embeddings, mixed_labels, mixed_weights = mixup_data_euc_avg_gender(
            embeddings,
            weights,
            labels,
            label_to_gender={0: "m", 1: "m"},
            sl_mixup=True,
        )

        self.assertEqual(mixed_labels.tolist(), [0, 0])
        self.assertEqual(tuple(mixed_weights.shape), (2, 1))
        self.assertTrue(torch.equal(mixed_embeddings, torch.tensor([[2.0, 0.0], [2.0, 0.0]])))

    def test_training_checkpoint_monitors_lowest_cosine_eer(self):
        with TemporaryDirectory() as save_dir:
            trainer = build_trainer(
                {
                    "USE_WANDB": False,
                    "epochs": 1,
                    "save_dir": save_dir,
                    "validate_during_train": True,
                    "save_top_k": 2,
                },
                "train",
            )

        checkpoints = [
            callback
            for callback in trainer.callbacks
            if callback.__class__.__name__ == "ModelCheckpoint"
        ]

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].monitor, "cosine_eer")
        self.assertEqual(checkpoints[0].mode, "min")
        self.assertEqual(checkpoints[0].save_top_k, 2)
        self.assertTrue(checkpoints[0].save_last)

    def test_task_uses_configured_discriminator_lr_and_adv_bounds(self):
        task = Task(
            features=torch.nn.Identity(),
            model=torch.nn.Linear(4, 4),
            loss=torch.nn.Linear(4, 2),
            config={
                "discriminator": "simple",
                "embedding_dim": 4,
                "discriminator_hidden_dim": 8,
                "discriminator_dropout": 0.0,
                "_mode": "train",
                "validate_during_train": False,
                "discriminator_lr": 0.0001,
                "lambda_adv": 0.02,
                "lambda_adv_min": 0.001,
                "lambda_adv_max": 0.05,
                "lambda_adv_pretrain": 0.0005,
                "log_training_steps": False,
            },
            learning_rate=0.0003,
            trial_path="missing-trials.txt",
        )

        optimizers, _ = task.configure_optimizers()

        self.assertEqual(optimizers[0].param_groups[0]["lr"], 0.0003)
        self.assertEqual(optimizers[1].param_groups[0]["lr"], 0.0001)
        self.assertEqual(task.lambda_adv, 0.02)
        self.assertEqual(task.lambda_adv_min, 0.001)
        self.assertEqual(task.lambda_adv_max, 0.05)
        self.assertEqual(task.lambda_adv_pretrain, 0.0005)


if __name__ == "__main__":
    unittest.main()
