class DummyNode:
    """
    A class to represent an ONNX node that is not "accelerateable".
    This node is created to preserve the original ONNX model graph structure,
    but will be skipped by the underlying engines, treating it as a 0 HW cost node.
    """
    def __init__(self, id, preds) -> None:
        """
        id = node_id
        preds = predecessors of this node.
        """
        self.id = id
        self.input_operand_source = {'I': preds}

    def __str__(self):
        return f"DummyLayerNode_{self.id}"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"id": self.id}