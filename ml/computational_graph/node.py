from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
import numpy as np

from .port import Port


class Node(ABC):
    """
    A node in a computational graph.
    The node has input ports and output ports.
    The input ports are defined in the concrete nodes that are defined and the input ports are created
    and connected to the node if the node is created.
    The mathematical operation that is performed by the node defines the number of input ports.
    The output ports can be dynamically connected to the nodes.
    The result that is calculated by the node can be streamed to any number of output ports.
    """

    def __init__(self, name: str = "Node"):
        self.name: str = name
        self.output_ports: List[Port] = []

    def forward(self):
        """
        Since the result is same for all output ports it can be calculated once
        and is then streamed to all output ports.
        :return the value for all output ports should be set
        """
        result = self.calc_result()
        for ports in self.output_ports:
            ports.set_value(result)

    @abstractmethod
    def calc_result(self) -> np.ndarray:
        """
        Calculate the result of the node.
        Gets all the necessary values from the input ports and returns the result.
        The implementation is highly dependent on the concrete node.
        :return: result for the node that should be streamed to all output ports
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Takes the gradients from all output ports and calculates the gradients for the input ports.
        Thus, the gradients are streamed back to the input ports.
        Since the number of input ports is defined by the concrete node, the implementation is highly dependent on the
        concrete node. Furthermore, the gradients that are calculated usually differ from input port to input port.
        Thus, an individual gradient calculation is necessary for each input port.
        :return: gradients for all input ports should be set
        """
        pass

    def connect_to_output(self, port: Port):
        """
        Connect a given port to the output of the node.
        Thus, the output of that node is streamed to the given port.
        This Node will therefore be the source of the given port.
        :param port: the port that should be connected to the output of the node:
        :return: The node should be connected to the source of the given port
        """
        port.set_source(self)
        self.output_ports.append(port)