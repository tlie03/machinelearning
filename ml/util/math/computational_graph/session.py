from typing import List

from ml.util.math.computational_graph.node import Node


class Session:

    def __init__(self, output_node: Node):
        self.output_node: Node = output_node
        self.nodes_sorted: List[Node] = []

        unseen_nodes = [output_node]
        while len(unseen_nodes) > 0:
            node = unseen_nodes.pop()
            self.nodes_sorted.append(node)
            unseen_nodes.extend(node.get_predecessors())

        self.nodes_sorted = sorted(self.nodes_sorted, key=lambda x: x.get_topo_number())

    def forward(self):
        for node in self.nodes_sorted:
            node.forward()

    def backward(self):
        for node in reversed(self.nodes_sorted):
            node.backward()
