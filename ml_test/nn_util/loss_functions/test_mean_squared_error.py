import unittest
import numpy as np

from ml import MeanSquaredError


class MeanSquaredErrorTest(unittest.TestCase):

    def test_loss(self):
        loss = MeanSquaredError()
        y_pred = np.array([1, 2, 3])
        y_real = np.array([4, 5, 6])
        self.assertEqual(loss.loss(y_pred, y_real), 9)

        y_pred = np.array([[1, 2, 3], [4, 5, 6]])
        y_real = np.array([[7, 8, 9], [10, 11, 12]])
        self.assertEqual(loss.loss(y_pred, y_real), 36)

    def test_loss_deriv(self):
        loss = MeanSquaredError()
        y_pred = np.array([1, 2, 3])
        y_real = np.array([4, 5, 6])
        deriv = [-6, -6, -6]
        self.assertTrue((loss.loss_deriv(y_pred, y_real) == deriv).all())

        y_pred = np.array([[1, 2, 3], [4, 5, 6]])
        y_real = np.array([[7, 8, 9], [10, 11, 12]])
        deriv = [[-12, -12, -12], [-12, -12, -12]]
        self.assertTrue((loss.loss_deriv(y_pred, y_real) == deriv).all())
