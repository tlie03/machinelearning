from abc import ABC, abstractmethod
from typing import List

from ml.computational_graph.port import Port



class Node(ABC):
    """
    A node in a computational graph.
    A node can have any number of input ports and output ports.
    In the concrete subclasses, the nodes that provide the inputs for the computation must be passed to the
    constructor. In the constructor the ports are created and connected to the node. Thus, establishing the
    connections between the nodes.
    Each Node has exactly one output value that is streamed to the output ports.
    Furthermore, the gradients that are queried from the output ports have all the same shape.
    Thus, the shape of the output value and the output gradient is defined in the abstract class.
    The shapes of the value and gradient of the input ports must be defined in each concrete subclass, since
    the number of input ports can vary.
    """

    def __init__(self, name: str = "Node", output_value_shape: tuple = None, output_grad_shape: tuple = None):
        """
        Initialize the node with the name.
        :param name: name of the node
        :param output_value_shape: shape of the value that is streamed to the output ports
        :param output_grad_shape: shape of the gradient that is queried from the output ports
        """
        self.name: str = name
        self.output_ports: List[Port] = []
        self.output_value_shape: tuple = output_value_shape
        self.output_grad_shape: tuple = output_grad_shape

    def get_name(self) -> str:
        """
        Get the name of the node.
        :return: name of the node
        """
        return self.name

    @abstractmethod
    def forward(self):
        """
        Calculate the result of the node based on the values of the input ports.
        The result is written to the output ports via the set_value method.
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Calculate the gradient of the node based on the gradients of the output ports.
        The gradient is written to the input ports via the set_grad method.
        Usually a different gradient is calculated for each input port.
        """
        pass

    def _add_output_port(self, port: Port):
        """
        Add an output port to the node.
        The when a new result is calculated it will be streamed to the output port.
        This method should only be called in the constructor of a different node, where the connection via the port
        is established.
        :param port: new output port that will receive the results
        """
        self.output_ports.append(port)