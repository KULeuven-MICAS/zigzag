from typing import Any

from onnx import NodeProto
from zigzag.parser.onnx.Parser import Parser
from zigzag.workload.DummyNode import DummyNode


class DefaultNodeParser(Parser):
    """! This class parses an ONNX node into a DummyNode."""

    def __init__(self, node_id: int, node: NodeProto, nodes_outputs: dict[int, Any]) -> None:

        super().__init__(node_id, node, nodes_outputs, mapping=None, onnx_model=None)

    def run(self) -> DummyNode:
        """! Run the parser"""
        dummy_node = self.generate_dummy_node()
        return dummy_node

    def generate_dummy_node(self) -> DummyNode:
        preds: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    preds.append(n)

        node_obj = DummyNode(
            self.node_id,
            preds,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        return node_obj
