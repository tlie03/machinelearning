from typing import Tuple
import random
import numpy as np

from .i_weight_generator import IWeightGenerator


MAX_SEED = 1000000


class UniformWeights(IWeightGenerator):
    def __init__(self, min_value=-1, max_value=1, seed=None):
        self.min_value = min_value
        self.max_value = max_value
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        self.generator = np.random.default_rng(seed=seed)

    def generate(self, shape: Tuple) -> np.array:
        return self.generator.uniform(self.min_value, self.max_value, size=shape)
