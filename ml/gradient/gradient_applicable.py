from abc import ABC, abstractmethod
import numpy as np


class GradientApplicable(ABC):

    @abstractmethod
    def apply_gradient(self, grad: np.ndarray):
        pass