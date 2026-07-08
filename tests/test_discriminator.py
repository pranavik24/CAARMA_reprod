import unittest

import torch

from model.discriminator_mix import SimpleDiscriminator


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


if __name__ == "__main__":
    unittest.main()
