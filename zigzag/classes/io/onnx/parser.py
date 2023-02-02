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

    @staticmethod
    def get_spatial_mappings(accelerator, core_allocation):
        # If there is only one possible core allocation, set the spatial mapping as the one(s) of that core
        if isinstance(core_allocation, int):
            core = accelerator.get_core(core_allocation)
            spatial_mappings = core.dataflows
        elif (isinstance(core_allocation, list) and len(core_allocation) == 1):
            core = accelerator.get_core(core_allocation[0])
            spatial_mappings = core.dataflows
        else:
            spatial_mappings = None
        return spatial_mappings
