from abc import ABC, abstractmethod
import numpy as np


class ILabelLoss(ABC):


    @abstractmethod
    def loss(self, y_pred: np.ndarray, y_real: np.ndarray) -> float:
        """
        Evaluates the loss of the prediction.
        :param y_pred: predicted label
        :param y_real: true label
        :return: loss
        """
        pass

    @abstractmethod
    def loss_deriv(self, y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
        """
        Evaluates the gradient of the loss function.
        :param y_pred: predicted label
        :param y_real: true label
        :return: the derivative of the loss function w.r.t the predicted label
        """
        pass

