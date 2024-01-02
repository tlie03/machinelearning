from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class IWeightGenerator(ABC):

    @abstractmethod
    def generate(self, shape: Tuple) -> np.ndarray:
        """
        Generates a weight.
        :param shape: shape of the weight
        :return: generated weight
        """
        pass