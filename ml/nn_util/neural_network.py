from typing import List, Tuple
import numpy as np

from ml.nn_util import ILayer
from ml.np_util import shape_fill_none


class NeuralNetwork:

    def __init__(self, layers: List[ILayer], seed: int = None):
        self.layers = layers
        self.seed = seed



    def forward(self, x: np.array) -> np.array:
        # reshape input data so that each datapoint is a column vector in the matrix
        m, n = shape_fill_none(x, 2)
        if n is None:
            x = x.reshape(m, 1)
        else:
            x = x.T

        # the actual forward pass
        for layer in self.layers:
            x = layer.forward(x)

        # reshape output data so that an vector becomes an array and a matrix has the datapoints as rows
        m, n = shape_fill_none(x, 2)
        if n == 1:
            x = x.reshape(m)
        else:
            x = x.T
        return x


    def backprop(self, loss_deriv: np.array) -> List[Tuple[np.array, np.array]]:
        pass

    def __str__(self):
        return '\n'.join([str(layer) for layer in self.layers])


if __name__ == '__main__':
    vec = np.array([1, 2, 3])
    print(vec.shape)
    print(vec.reshape(3, 1))