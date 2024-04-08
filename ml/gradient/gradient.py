from typing import Dict
import numpy as np

from .gradient_applicable import GradientApplicable


class Gradient:

    def __init__(self):
        self.parameter_gradients: Dict[GradientApplicable, np.ndarray] = {

        }

    def apply_gradient(self):
        pass