import numpy as np
from nn_util.activation_functions.i_activation_function import IActivationFunction


class ReLu(IActivationFunction):

    def __init__(self):
        super().__init__()
        self.name = 'relu'

    def eval(self, x):
        return np.maximum(0, x)

    def eval_deriv(self, x):
        return np.where(x > 0, 1, 0)
