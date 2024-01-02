import numpy as np

from .i_layer import ILayer
from ml.nn_util.activation_functions.i_activation_function import IActivationFunction
from ml.nn_util.activation_functions.identity import Identity


class Layer(ILayer):

    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: IActivationFunction = Identity(),
                 bias: bool = False,
                 name: str = 'layer'
                 ):

        self.name = name

        # layer after will be set during the model construction
        self.next_layer = None

        # hyperparameters of the layer
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.bias = bias


        # todo change so that values will be set by some kind of weight generator
        # values that will be adjusted during training
        self.weights = np.random.uniform(-1, 1, (output_shape, input_shape))
        if bias:
            self.biases = np.random.uniform(-1, 1, (output_shape, 1))
        else:
            self.biases = np.zeros((output_shape, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Bias is always added but the bias vector is zero if bias is set to False.
        """
        z = self.weights @ x + self.biases
        return self.activation.eval(z)

    def __str__(self):
        return f'{self.name}: {self.input_shape} -> {self.output_shape} | {self.activation} | bias: {self.bias}'
