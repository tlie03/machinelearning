from typing import List
import numpy as np


class Port:
    """
    Each Node has exactly one port. The port is used to cache the output that was calculated by the Node,
    as well as the gradients of the Node output w.r.t to the Session output. The port was introduced
    to outsource the storage of outputs and gradients from the Node to a separate class.
    """

    def __init__(self, output_shape: tuple):
        self.output_shape: tuple = output_shape
        self.output: np.ndarray = np.ndarray(output_shape)

        # the gradient shape is (dimension of session Output)x(output_shape)
        self.grad_shape: tuple = (..., output_shape)
        # list of gradients of shape output_shape,
        # each gradient was calculated by one of the following connected nodes
        # each gradient represents a gradient of the output w.r.t to the Session output
        self.grads: List[np.ndarray] = []
        
    def get_output_shape(self) -> tuple:
        """
        Returns the shape of the output of the port node.
        This method can be used by following nodes that connect to the port
        to determine its input shape and thus calculate their own output shape
        or detect shape mismatches.
        """
        return self.output_shape

    def set_output(self, output: np.ndarray):
        """
        Sets the output of the node to the given value.
        The output can then be queried by the following connected nodes.
        :param output: output of the node
        """
        self.output: np.ndarray = output

    def get_output(self) -> np.ndarray:
        """
        Returns the output that was calculated by the node and stored in the port.
        :return: output of the node
        """
        return self.output

    def add_grad(self, grad: np.ndarray):
        """
        Adds a gradient that was calculated by a connected following node.
        Each gradient must be a gradient of the output of the ports node w.r.t to the Session output
        to ensure that each gradient has the same shape.
        :param grad: gradient of the output of the ports node w.r.t to the Session output
        """
        self.grads.append(grad)

    def get_grad(self) -> np.ndarray:
        """
        Returns the sum of all gradients of the output of the ports node w.r.t to the Session output.
        :return: sum of all gradients (influence of the ports node output on the Session output)
        """
        stack = np.stack(self.grads)
        return stack.sum(axis=0)

