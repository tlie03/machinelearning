import unittest
import numpy as np

from ml import shape_fill_none



class ShapeTest(unittest.TestCase):

    def test_shape_fill_none(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(shape_fill_none(m1, 1), (2, 3))
        self.assertEqual(shape_fill_none(m1, 2), (2, 3))
        self.assertEqual(shape_fill_none(m1, 3), (2, 3, None))
        self.assertEqual(shape_fill_none(m1, 4), (2, 3, None, None))