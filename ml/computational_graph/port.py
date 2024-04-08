import numpy as np

from ml.computational_graph.node import Node


WRONG_SHAPE_ERROR = "Value shape {0} does not match the expected shape {1} in {2}"


class Port:
    """
    A port is an edge between two nodes in a computational graph.
    It is used as a buffer to store the value that is calculated by the source
    and the gradient that is calculated by the target.
    """
    def __init__(self, source: Node, target: Node, value_shape: tuple, grad_shape: tuple):
        self.source: Node = source
        self.target: Node = target
        self.value_shape: tuple = value_shape
        self.grad_shape: tuple = grad_shape
        self.value: np.ndarray = None
        self.grad: np.ndarray = None

    def set_value(self, value: np.ndarray):
        """
        Set the value that is calculated by the source node.
        :param value: ndarray of shape value_shape
        """
        if value.shape != self.value_shape:
            raise ValueError(WRONG_SHAPE_ERROR.format(value.shape, self.value_shape, str(self)))
        self.value = value

    def set_grad(self, grad: np.ndarray):
        """
        Set the gradient that is calculated by the target node.
        :param grad: ndarray of shape grad_shape
        """
        if grad.shape != self.grad_shape:
            raise ValueError(WRONG_SHAPE_ERROR.format(grad.shape, self.grad_shape, str(self)))
        self.grad = grad

    def get_value(self) -> np.ndarray:
        """
        Get the value that is buffered in the port.
        :return: ndarray of shape value_shape
        """
        return self.value

    def get_grad(self) -> np.ndarray:
        """
        Get the gradient that is buffered in the port.
        :return: ndarray of shape grad_shape
        """
        return self.grad

    def __str__(self):
        return f"Port from {self.source.get_name()} -> {self.target.get_name()}"