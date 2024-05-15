from zigzag.workload.Workload import Workload
from zigzag.workload.layer_node import LayerNode
from typing import Any


class DNNWorkload(Workload):

    def __init__(self, nodes: list[LayerNode], **attr: Any):
        """!
        @return (self): Directed Graph with nodes the layers and edges the connections between layers.
        """
        super().__init__(**attr)

        layer_id_to_obj: dict[int, LayerNode] = {}
        self.layer_node_list = nodes

        for layer_node in nodes:
            layer_id_to_obj[layer_node.id] = layer_node

            self.add_workload_node(layer_node)
            # Find all of its operand sources and add edges accordingly
            edges: list[tuple[LayerNode, LayerNode]] = []
            for _, parent_id in layer_node.input_operand_source.items():
                # for parent_id in parent_list:
                assert parent_id in layer_id_to_obj, f"Illegal reference to non-existent layer with id {parent_id}"
                parent_layer = layer_id_to_obj[parent_id]
                edges.append((parent_layer, layer_node))

            self.add_workload_edges_from(edges)
