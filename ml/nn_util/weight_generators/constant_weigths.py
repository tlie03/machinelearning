from typing import Tuple
import numpy as np

from .i_weight_generator import IWeightGenerator


class ConstantWeights(IWeightGenerator):

    def __init__(self, weight_value=0):
        self.value = weight_value

    def generate(self, shape: Tuple) -> np.array:
        return np.full(shape, self.value)