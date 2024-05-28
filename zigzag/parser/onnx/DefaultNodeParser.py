from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.workload.DummyNode import DummyNode


class DefaultNodeParser(ONNXOperatorParser):
    """! This class parses an ONNX node into a DummyNode."""

    def run(self) -> DummyNode:
        """! Run the parser"""
        return self.generate_dummy_node()

    def generate_dummy_node(self) -> DummyNode:
        predecessors = self.get_node_predecessors()

        node_obj = DummyNode(
            node_id=self.node_id,
            predecessors=predecessors,
            node_name=self.node.name,
            node_type=self.node.op_type.lower(),
        )

        return node_obj
