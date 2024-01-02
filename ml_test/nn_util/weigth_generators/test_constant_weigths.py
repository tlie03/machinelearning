import unittest
import numpy as np

from ml import ConstantWeights


class ConstantWeightsTest(unittest.TestCase):

    def test_generate(self):
        generator = ConstantWeights(weight_value=2)
        A = generator.generate((2, 3))
        B = np.array([[2, 2, 2], [2, 2, 2]])

        self.assertEqual(A.shape, (2, 3))
        self.assertTrue(np.allclose(A, B, atol=1e-8))