from abc import ABCMeta

from zigzag.workload.layer_attributes import InputOperandSource


class LayerNodeABC(metaclass=ABCMeta):
    """Represents a single layer of a workload in any form."""

    id: int
    name: str

    def __init__(self, node_id: int, node_name: str):
        self.id = node_id
        self.name = node_name
        self.input_operand_source: InputOperandSource

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.name

    def __jsonrepr__(self):
        """! JSON representation used for saving this object to a json file."""
        return str(self)
