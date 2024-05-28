from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.workload.DummyNode import DummyNode


class DefaultNodeParser(ONNXOperatorParser):
    """! This class parses an ONNX node into a DummyNode."""

    def run(self) -> DummyNode:
        """! Run the parser"""
        return self.generate_dummy_node()

    def generate_dummy_node(self) -> DummyNode:
        prev_node_id = self.get_predecessor_id()

        node_obj = DummyNode(
            self.node_id,
            prev_node_id,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        return node_obj
