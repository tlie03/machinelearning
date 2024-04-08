import numpy as np

from ..node_alt import Node


class Input(Node):
    """
    An input node in the computational graph.
    Thus, the input node does not have any input ports.
    """

    def __init__(self, output_shape: tuple, output_grad_shape: tuple, name: str = "Input"):
        """
        Initialize the input node with the dimensions of the data points and the batch size.
        :param name: name of the node
        """
        super().__init__(output_shape, output_grad_shape, name)
        self.value: np.ndarray = None
        self.grad: np.ndarray = None

    def calc_result(self) -> np.ndarray:
        pass

    def backward(self):
        pass