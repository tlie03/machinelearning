from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ILayer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        The activation function is applied on the weighted sum of the input and the bias.
        :param x: can be a vector or a matrix
        :return: resulting vector or matrix depending on the input
        """
        pass

    @abstractmethod
    def __str__(self):
        pass