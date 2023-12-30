import numpy as np

from nn_util.loss_functions.i_label_loss import ILabelLoss


class MeanSquaredError(ILabelLoss):

    def loss(self, y_pred: np.ndarray, y_real: np.ndarray) -> float:
        pass

    def loss_deriv(self, y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
        pass