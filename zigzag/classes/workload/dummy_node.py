
## A class to represent an ONNX node that is not "accelerateable".
# This node is created to preserve the original ONNX model graph structure,
# but will be skipped by the underlying engines, treating it as a 0 HW cost node.
class DummyNode:

    ## The class constructor
    # Initialize the DummyNode by setting its id, the node's predecessors and optionally giving it a name.
    # @param id (int): id for this node
    # @param preds (list): list of ids of this node's predecessor nodes
    # @param node_name (str, optional): a name for this node, e.g. the node's name within the onnx model
    def __init__(self, id, preds, node_name="", type=None) -> None:
        self.id = id
        self.input_operand_source = {"I": preds}
        self.name = node_name
        self.type = type
        self.core_allocation = (
            -1
        )  # We assume these nodes are mapped on a core with id -1
        self.runtime = 0
        self.start = None
        self.end = None

    def __str__(self):
        return f"DummyNode({self.id})"

    def __repr__(self) -> str:
        return str(self)

    ## JSON representation used for saving this object to a json file.
    def __jsonrepr__(self):
        return {"id": self.id}

    ## Set the start time in ccyles of this node
    # @param start (int): start time in cycles
    def set_start(self, start):
        self.start = start

    ## Set the end time in cycles of this node
    # @param end (int): end time in cycles
    def set_end(self, end):
        self.end = end

    ## Get the start time in cycles of this node.
    def get_start(self):
        return self.start

    ## Get the end time in cycles of this node.
    def get_end(self):
        return self.end

    ## Return the runtime of running this node.
    def get_runtime(self):
        return self.runtime

    ## Check if this node has already been assigned an end time.
    # @return (bool) True if this node has been assigned an end time
    def has_end(self) -> bool:
        return self.end is not None
