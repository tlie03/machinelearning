from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod

from ml.util.math.computational_graph.port import Port


class Node(ABC):
    # todo add comments

    def __init__(self, output_shape: tuple, topo_number: int = 0):
        self.port: Port = Port(output_shape=output_shape)
        self.topo_number: int = topo_number

    def get_topo_number(self) -> int:
        return self.topo_number

    def get_port(self) -> Port:
        return self.port

    @abstractmethod
    def get_predecessors(self) -> List[Node]:
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
