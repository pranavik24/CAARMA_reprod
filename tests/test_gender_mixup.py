import unittest

import torch

from mixup_gender import mixup_data_euc_avg_gender


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


if __name__ == "__main__":
    unittest.main()
