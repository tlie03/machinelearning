import numpy as np

from .i_label_loss import ILabelLoss


class MeanSquaredError(ILabelLoss):

    def loss(self, y_pred: np.ndarray, y_real: np.ndarray) -> float:
        return np.mean(np.square(y_pred - y_real))

    def loss_deriv(self, y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_real)