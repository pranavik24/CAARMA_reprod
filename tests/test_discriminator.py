import unittest

import torch

from mixup_gender import build_embedding_discriminator
from model.discriminator_mix import InterDiscrim, SimpleDiscriminator


class SimpleDiscriminatorTests(unittest.TestCase):
    def test_scores_pooled_embeddings(self):
        discriminator = SimpleDiscriminator(emb_dim=4, hidden_dim=8, dropout_rate=0.0)
        embeddings = torch.randn(3, 4)

        scores = discriminator(embeddings)

        self.assertEqual(tuple(scores.shape), (3, 1))

    def test_accepts_singleton_sequence_dimension(self):
        discriminator = SimpleDiscriminator(emb_dim=4, hidden_dim=8, dropout_rate=0.0)
        embeddings = torch.randn(3, 1, 4)

        scores = discriminator(embeddings)

        self.assertEqual(tuple(scores.shape), (3, 1))

    def test_rejects_unpooled_sequences(self):
        discriminator = SimpleDiscriminator(emb_dim=4, hidden_dim=8, dropout_rate=0.0)

        with self.assertRaisesRegex(ValueError, "pooled embeddings"):
            discriminator(torch.randn(3, 2, 4))


class InterDiscrimTests(unittest.TestCase):
    def test_scores_pooled_embeddings(self):
        discriminator = InterDiscrim(
            emb_dim=4,
            hidden_dim=16,
            mid_dim=8,
            head_dim=4,
            dropout_rate=0.0,
        )
        embeddings = torch.randn(3, 4)

        scores = discriminator(embeddings)

        self.assertEqual(tuple(scores.shape), (3, 1))

    def test_accepts_singleton_sequence_dimension(self):
        discriminator = InterDiscrim(
            emb_dim=4,
            hidden_dim=16,
            mid_dim=8,
            head_dim=4,
            dropout_rate=0.0,
        )
        embeddings = torch.randn(3, 1, 4)

        scores = discriminator(embeddings)

        self.assertEqual(tuple(scores.shape), (3, 1))

    def test_rejects_unpooled_sequences(self):
        discriminator = InterDiscrim(
            emb_dim=4,
            hidden_dim=16,
            mid_dim=8,
            head_dim=4,
            dropout_rate=0.0,
        )

        with self.assertRaisesRegex(ValueError, "pooled embeddings"):
            discriminator(torch.randn(3, 2, 4))

    def test_factory_builds_intermediate_discriminator(self):
        discriminator = build_embedding_discriminator(
            {
                "discriminator": "intermediate",
                "embedding_dim": 4,
                "discriminator_hidden_dim": 16,
                "discriminator_mid_dim": 8,
                "discriminator_head_dim": 4,
                "discriminator_dropout": 0.0,
            }
        )

        self.assertIsInstance(discriminator, InterDiscrim)


if __name__ == "__main__":
    unittest.main()
