import unittest

import torch

from criterion.build_criterion import build_criterion
from criterion.amsoftmax_mix_gan import amsoftmax_gan
from helper.diffusion_mixup import diffusion_mixup


class DiffusionMixupTests(unittest.TestCase):
    def test_diffusion_mixup_shapes_labels_and_finite_weights(self):
        embeddings = torch.randn(3, 4)
        weights = torch.randn(4, 5)
        labels = torch.tensor([0, 2, 4])

        synthetic_embeddings, synthetic_labels, synthetic_weights = diffusion_mixup(
            embeddings,
            weights,
            labels,
            diffusion_timesteps=100,
            diffusion_t_min=1,
            diffusion_t_max=20,
            diffusion_beta_start=0.0001,
            diffusion_beta_end=0.02,
            diffusion_embedding_noise=0.0,
        )

        self.assertEqual(tuple(synthetic_embeddings.shape), (3, 4))
        self.assertEqual(synthetic_labels.tolist(), [0, 1, 2])
        self.assertEqual(tuple(synthetic_weights.shape), (4, 3))
        self.assertTrue(torch.isfinite(synthetic_weights).all())
        self.assertTrue(torch.all(torch.norm(synthetic_weights, p=2, dim=0) > 0))

    def test_zero_timestep_diffusion_reproduces_normalized_speaker_weights(self):
        embeddings = torch.randn(2, 3)
        weights = torch.tensor(
            [
                [3.0, 0.0, 4.0],
                [4.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        labels = torch.tensor([0, 2])

        _, synthetic_labels, synthetic_weights = diffusion_mixup(
            embeddings,
            weights,
            labels,
            diffusion_timesteps=100,
            diffusion_t_min=0,
            diffusion_t_max=0,
        )

        expected = weights.index_select(1, labels)
        expected = expected / torch.norm(expected, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        self.assertEqual(synthetic_labels.tolist(), [0, 1])
        self.assertTrue(torch.allclose(synthetic_weights, expected, atol=1e-6))

    def test_base_amsoftmax_gan_supports_diffusion_strategy(self):
        criterion = amsoftmax_gan(
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
        self.assertEqual(acc.numel(), 1)
        self.assertEqual(tuple(synthetic_embeddings.shape), (3, 4))

    def test_build_criterion_passes_base_diffusion_config(self):
        criterion = build_criterion(
            {
                "criterion": "AMSoftmaxGAN",
                "embedding_dim": 4,
                "num_spk": 5,
                "synthetic_strategy": "diffusion",
                "diffusion_timesteps": 100,
                "diffusion_t_min": 1,
                "diffusion_t_max": 20,
                "diffusion_beta_start": 0.0001,
                "diffusion_beta_end": 0.02,
                "diffusion_embedding_noise": 0.0,
            }
        )

        self.assertEqual(criterion.synthetic_strategy, "diffusion")
        self.assertEqual(criterion.diffusion_t_min, 1)
        self.assertEqual(criterion.diffusion_t_max, 20)


if __name__ == "__main__":
    unittest.main()
