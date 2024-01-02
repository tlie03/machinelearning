import unittest
import numpy as np

from ml import ReLu


class ReLuTest(unittest.TestCase):

    def test_eval(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]])
        m2 = np.array([[-1, -2, -3], [-4, -5, -6]])

        relu = ReLu()

        self.assertTrue(np.array_equal(np.maximum(0, m1), relu.eval(m1)))
        self.assertTrue(np.array_equal(np.maximum(0, m2), relu.eval(m2)))

    def test_eval_deriv(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]])
        m2 = np.array([[-1, -2, -3], [-4, -5, -6]])

        relu = ReLu()

        self.assertTrue(np.array_equal(np.where(m1 > 0, 1, 0), relu.eval_deriv(m1)))
        self.assertTrue(np.array_equal(np.where(m2 > 0, 1, 0), relu.eval_deriv(m2)))