import numpy as np

from .i_activation_function import IActivationFunction


class Identity(IActivationFunction):

    def __init__(self):
        super().__init__()
        self.name = 'identity'

    def eval(self, x):
        return x

    def eval_deriv(self, x):
        return np.ones(x.shape)

    def __str__(self):
        return self.name