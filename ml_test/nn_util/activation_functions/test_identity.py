import unittest
import numpy as np

from ml import Identity


class IdentityTest(unittest.TestCase):

    def test_eval(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]])
        m2 = np.array([[-1, -2, -3], [-4, -5, -6]])

        identity = Identity()

        self.assertTrue(np.array_equal(m1, identity.eval(m1)))
        self.assertTrue(np.array_equal(m2, identity.eval(m2)))

    def test_eval_deriv(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]])
        m2 = np.array([[-1, -2, -3], [-4, -5, -6]])

        identity = Identity()

        self.assertTrue(np.array_equal(np.ones(m1.shape), identity.eval_deriv(m1)))
        self.assertTrue(np.array_equal(np.ones(m2.shape), identity.eval_deriv(m2)))