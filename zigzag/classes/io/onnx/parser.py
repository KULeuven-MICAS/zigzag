from abc import ABCMeta, abstractmethod


class Parser(metaclass=ABCMeta):
    """Abstract base class that represents a parser of an onnx operator.
    Example: Conv, MatMul, etc.
    """
    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model, accelerator=None) -> None:
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.mapping = mapping
        self.onnx_model = onnx_model
        self.accelerator = accelerator
    
    @abstractmethod
    def run(self):
        ...