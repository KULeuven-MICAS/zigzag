from zigzag.datatypes import Constants
from zigzag.workload.layer_attributes import InputOperandSource
from zigzag.workload.layer_node_abc import LayerNodeABC


class DummyNode(LayerNodeABC):
    """! A class to represent an ONNX node that is not "accelerateable".
    This node is created to preserve the original ONNX model graph structure,
    but will be skipped by the underlying engines, treating it as a 0 HW cost node.
    """

    def __init__(self, node_id: int, predecessors: list[int], node_type: str, node_name: str = "") -> None:
        """
        Initialize the DummyNode by setting its id, the node's predecessors and optionally giving it a name.
        @param id (int): id for this node
        @param predecessor (list): list of ids of this node's predecessor nodes
        @param node_name (str, optional): a name for this node, e.g. the node's name within the onnx model
        """
        LayerNodeABC.__init__(self, node_id, node_name)

        match len(predecessors):
            case 0:
                self.input_operand_source: InputOperandSource = {}
            case 1:
                self.input_operand_source = {Constants.LAYER_OP_I: predecessors[0]}
            case _:
                # We currently don't support more than 2 sources so we can also use `I` and `W` for the layer operands
                self.input_operand_source = {
                    Constants.LAYER_OP_I: predecessors[0],
                    Constants.LAYER_OP_W: predecessors[1],
                }

        self.type = node_type
        self.runtime = 0
        self.start = None
        self.end = None

    def __str__(self):
        return f"DummyNode({self.id})"

    def set_start(self, start: int):
        """! Set the start time in cyles of this node
        @param start : start time in cycles
        """
        self.start = start

    def set_end(self, end: int):
        """! Set the end time in cycles of this node
        @param end: end time in cycles
        """
        self.end = end

    def get_start(self):
        """! Get the start time in cycles of this node."""
        return self.start

    def get_end(self):
        """! Get the end time in cycles of this node."""
        return self.end

    def get_runtime(self):
        """! Return the runtime of running this node."""
        return self.runtime

    def has_end(self) -> bool:
        """! Check if this node has already been assigned an end time.
        @return (bool) True if this node has been assigned an end time
        """
        return self.end is not None
