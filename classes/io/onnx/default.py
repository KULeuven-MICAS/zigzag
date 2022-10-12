from classes.workload.dummy_node import DummyNode


class DefaultNodeParser:
    """Parse an ONNX node into a DummyNode.
    """
    def __init__(self, node_id, node, nodes_outputs) -> None:
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
    def run(self):
        """Run the parser
        """
        dummy_node = self.generate_dummy_node()
        return dummy_node

    def generate_dummy_node(self):
        preds = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    preds.append(n)
        
        node_obj = DummyNode(self.node_id, preds, node_name=self.node.name)

        return node_obj
