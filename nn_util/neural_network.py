from typing import List, Tuple
import numpy as np

from nn_util import ILayer


class NeuralNetwork:

    def __init__(self, layers: List[ILayer]):
        self.layers = layers

    def forward(self, x: float) -> float:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backprop(self, loss_deriv: np.array) -> List[Tuple[np.array, np.array]]:
        pass

    def __str__(self):
        return '\n'.join([str(layer) for layer in self.layers])