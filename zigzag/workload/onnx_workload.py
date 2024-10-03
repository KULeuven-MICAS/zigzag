from copy import deepcopy
from typing import Any

from typeguard import typeguard_ignore

from zigzag.workload.dummy_node import DummyNode
from zigzag.workload.layer_node import LayerNode
from zigzag.workload.layer_node_abc import LayerNodeABC
from zigzag.workload.workload_abc import WorkloadABC, WorkloadNoDummyABC


class ONNXWorkload(WorkloadABC):
    """Represents a workload graph parsed from ONNX"""

    def __init__(self, **attr: Any):
        """! Collect all the algorithmic workload information here."""
        super().__init__(**attr)  # type: ignore

        self.node_id_to_obj: dict[int, LayerNodeABC] = {}

    def add(self, node_id: int, node_obj: LayerNodeABC):
        """! Add a node object to the ONNX workload graph.
        This can be a different object based on if it's an "accelerateable" node or not.
        """
        self.node_id_to_obj[node_id] = node_obj

        self.add_node(node_obj)
        edges: list[tuple[LayerNodeABC, LayerNodeABC]] = []
        for parent_id in node_obj.input_operand_source.values():
            parent_node_obj = self.node_id_to_obj[parent_id]
            edges.append((parent_node_obj, node_obj))
            self.add_edges_from(edges)

    @typeguard_ignore
    def get_copy_no_dummy(self) -> WorkloadNoDummyABC:
        """! Remove dummy nodes (layers) in the graph
        Redirect the outgoing edges of dummy nodes to non-dummy nodes Method: for each dummy node, add edges between its
        predecessor nodes and successor nodes; then remove the dummy node.
        """
        workload_copy = deepcopy(self)

        dummy_nodes = [node for node in workload_copy.node_list if isinstance(node, DummyNode)]
        for dummy_node in dummy_nodes:
            for successor_node in workload_copy.successors(dummy_node):
                for predecessor_node in workload_copy.predecessors(dummy_node):
                    workload_copy.add_edge(predecessor_node, successor_node)

        workload_copy.remove_nodes_from(iter(dummy_nodes))

        assert all([isinstance(x, LayerNode) for x in workload_copy.node_list])
        return workload_copy  # type: ignore
