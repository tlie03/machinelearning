import numpy as np

from nn_util.layers.i_layer import ILayer
from nn_util.activation_functions.i_activation_function import IActivationFunction
from nn_util.activation_functions.identity import Identity


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
        z = np.dot(self.weights, x) + self.biases
        result = self.activation.eval(z)

        if self.next_layer is not None:
            return self.next_layer.forward(result)
        return result

    def append_layer(self, layer: ILayer):
        self.next_layer = layer


if __name__ == '__main__':
    pass