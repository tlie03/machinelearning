from abc import ABC, abstractmethod


class ComputationalGraphNode(ABC):
    """
    The base class that defines the properties that must be shared by all nodes in a computational graph.
    A node in a computational graph defines an operation that is applied to the values on the input edges and the
    result is passed to the output edges. Except for hyperparameters the nodes are stateless.
    All input and output values are stored in the edges.
    """

    @abstractmethod
    def forward(self):
        """
        Pulls all the values from the input edges, calculates the result and pushes the result onto the output edges.
        """
        pass