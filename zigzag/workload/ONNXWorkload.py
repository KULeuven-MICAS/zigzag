from typing import Any

from zigzag.workload.LayerNodeABC import LayerNodeABC
from zigzag.workload.Workload import Workload


class ONNXWorkload(Workload):

    def __init__(self, **attr: Any):
        """! Collect all the algorithmic workload information here."""
        super().__init__(**attr)

        self.node_id_to_obj: dict[int, LayerNodeABC] = {}
        self.node_list: list[LayerNodeABC] = []

    def add(self, node_id: int, node_obj: LayerNodeABC):
        """! Add a node object to the ONNX workload graph.
        This can be a different object based on if it's an "accelerateable" node or not.
        """
        self.node_list.append(node_obj)
        self.node_id_to_obj[node_id] = node_obj

        self.add_workload_node(node_obj)
        edges: list[tuple[LayerNodeABC, LayerNodeABC]] = []
        for _, parent_id in node_obj.input_operand_source.items():
            # for parent_id in parents:
            parent_node_obj = self.node_id_to_obj[parent_id]
            edges.append((parent_node_obj, node_obj))
            self.add_workload_edges_from(edges)
