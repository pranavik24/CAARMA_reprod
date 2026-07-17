import unittest
from tempfile import TemporaryDirectory

import torch

from mixup_gender import (
    AMSoftmaxGANGender,
    Task,
    build_gender_criterion,
    build_trainer,
    mixup_data_euc_avg_gender,
)


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

    def test_diffusion_strategy_flag_syn_produces_scalar_loss(self):
        criterion = AMSoftmaxGANGender(
            embedding_dim=4,
            num_classes=5,
            synthetic_strategy="diffusion",
            diffusion_t_min=0,
            diffusion_t_max=0,
        )
        embeddings = torch.randn(3, 4)
        labels = torch.tensor([0, 2, 4])

        loss, acc, synthetic_embeddings = criterion(embeddings, labels, flagSyn=True)

        self.assertEqual(tuple(loss.shape), ())
        self.assertEqual(tuple(acc.shape), ())
        self.assertEqual(tuple(synthetic_embeddings.shape), (3, 4))

    def test_default_strategy_still_uses_average_mixup(self):
        criterion = AMSoftmaxGANGender(
            embedding_dim=2,
            num_classes=2,
            sl_mixup=False,
        )
        with torch.no_grad():
            criterion.W.copy_(torch.tensor([[1.0, 3.0], [0.0, 0.0]]))
        embeddings = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        labels = torch.tensor([0, 1])

        _, _, synthetic_embeddings = criterion(embeddings, labels, flagSyn=False)

        self.assertTrue(torch.equal(synthetic_embeddings, torch.tensor([[2.0, 0.0], [2.0, 0.0]])))

    def test_diffusion_strategy_does_not_require_gender_metadata(self):
        criterion = build_gender_criterion(
            {
                "criterion": "AMSoftmaxGAN",
                "embedding_dim": 4,
                "num_spk": 5,
                "_mode": "train",
                "sl_mixup": True,
                "synthetic_strategy": "diffusion",
                "dataset": "missing-train.csv",
            }
        )

        self.assertEqual(criterion.synthetic_strategy, "diffusion")

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
