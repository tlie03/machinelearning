from abc import ABC, abstractmethod
import numpy as np

from ml.computational_graph.nodes.computational_graph_node import ComputationalGraphNode


class ComputationalGraphEdge(ABC):
    """
    The base class that defines the properties that must be shared by all edges in a computational graph.
    The edges are the connections between the nodes in the computational graph, and hold the input and output values
    that are passed between the nodes.
    """

    def __init__(self, source: ComputationalGraphNode, target: ComputationalGraphNode):
        """
        Initializes the edge with the source and target nodes.
        Forwarding: Values are passed from the source node to the target node.
        Backprop: Gradients are passed from the target node to the source node.
        """
        self.source = source
        self.target = target

    @abstractmethod
    def push_value(self, new_value: np.ndarray):
        """
        Stores the new value in the edge.
        :param new_value: The new value to store in the edge. The dimension of the ndarray
        depends on the concrete edge type.
        """
        pass

    @abstractmethod
    def pull_value(self) -> np.ndarray:
        """
        Returns the value that is currently stored in the edge.
        """
        pass
