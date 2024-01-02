import unittest
import numpy as np

from ml import UniformWeights


class UniformWeightsTest(unittest.TestCase):

    def test_generate(self):
        generator = UniformWeights(min_value=0, max_value=1, seed=1000000)
        A = generator.generate((2, 3))
        B = np.array([[0.41067769, 0.20853126, 0.54301682], [0.68764139, 0.71686117, 0.05250152]])

        self.assertEqual(A.shape, (2, 3))
        self.assertTrue(np.allclose(A, B, atol=1e-8))
