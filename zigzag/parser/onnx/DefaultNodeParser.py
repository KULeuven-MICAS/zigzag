from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.workload.DummyNode import DummyNode


class DefaultNodeParser(ONNXOperatorParser):
    """! This class parses an ONNX node into a DummyNode."""

    def run(self) -> DummyNode:
        """! Run the parser"""
        return self.generate_dummy_node()

    def generate_dummy_node(self) -> DummyNode:
        predecessors: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)

        # TODO DummyLayer cannot deal with two operand sources
        # assert len(predecessors) <= 1, "Only a single layer operand source expected"
        prev_node_id = None if len(predecessors) == 0 else predecessors.pop()

        node_obj = DummyNode(
            self.node_id,
            prev_node_id,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        return node_obj
