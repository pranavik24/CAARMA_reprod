import unittest

import torch

from feature.fbanks import PreEmphasis


class PreEmphasisTests(unittest.TestCase):
    def test_matches_reflect_padded_filter_math_without_convolution(self):
        pre_emphasis = PreEmphasis(coef=0.5)
        inputs = torch.tensor([[1.0, 3.0, 7.0]])

        output = pre_emphasis(inputs)

        expected = torch.tensor([[-0.5, 2.5, 5.5]])
        self.assertTrue(torch.equal(output, expected))


if __name__ == "__main__":
    unittest.main()
