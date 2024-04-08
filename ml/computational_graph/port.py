import numpy as np

from .node import Node


class Port:
    """
    A port is an edge in the computational graph that is used to connect nodes and distribute values and gradients.
    The ports can be seen as buffers that are used to stream values and gradients between nodes.
    By using ports, the gradient for the inputs can be pushed back to the input ports and the input
    nodes can access the gradient always via the same port interface.
    This would not be possible if the gradient would be stored in the node itself.
    """

    def __init__(self, target: Node, value_shape: tuple = None, grad_shape: tuple = None):
        """
        Values are streamed from the source to the target node.
        The gradient is streamed from the target to the source node due to backpropagation.
        :param target: The initial target node that is provided during the creation of the port.
        :param value_shape: The shape of the value that is streamed from the source to the target node.
        :param grad_shape: The shape of the gradient that is streamed from the target to the source node.
        The source node will be provided later on during the assembly of the computational graph.
        """
        self.source: Node = None
        self.target: Node = target
        self.value: np.ndarray = None
        self.grad: np.ndarray = None
        self.value_shape: tuple = value_shape
        self.grad_shape: tuple = grad_shape

    def set_source(self, source: Node):
        """
        Set the source node of the port.
        :param source: the source node of the port
        """
        self.source = source

    def set_value(self, value: np.ndarray):
        """
        Set the value that is buffered in the port.
        :param value: ndarray of shape value_shape
        """
        if value.shape != self.value_shape:
            raise ValueError(f"Value shape {value.shape} does not match the expected shape "
                             f"{self.value_shape} in {str(self)}")
        self.value = value

    def set_grad(self, grad: np.ndarray):
        """
        Set the gradient that is buffered in the port.
        :param grad: ndarray of shape grad_shape
        """
        if grad.shape != self.grad_shape:
            raise ValueError(f"Gradient shape {grad.shape} does not match the expected shape "
                             f"{self.grad_shape} in {str(self)}")
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
        """
        String representation of the port.
        """
        return f"Port: {self.source.name} -> {self.target.name}"