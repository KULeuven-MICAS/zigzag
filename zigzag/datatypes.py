from abc import ABCMeta
import re
from typing import Any, TypeAlias
from typeguard import typechecked


class DimensionABC(metaclass=ABCMeta):
    """! Abstract Base Class for all dimension- and operand-like classes"""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any):
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __lt__(self, other: "DimensionABC"):
        return self.name < other.name

    def __ge__(self, other: "DimensionABC"):
        return self.name >= other.name

    def __jsonrepr__(self):
        return self.name


@typechecked
class LayerOperand(DimensionABC):
    """! Operand from the layer definition, e.g. `I`, `W`, `O`."""

    def is_output(self):
        return self == Constants.OUTPUT_LAYER_OP

    def is_final_output(self):
        return self == Constants.FINAL_OUTPUT_LAYER_OP


@typechecked
class MemoryOperand(DimensionABC):
    """! Operand from the memory definition, e.g. `I1`, `I2`, `O`."""

    def is_output(self):
        return self == Constants.OUTPUT_MEM_OP

    def is_final_output(self):
        return self == Constants.FINAL_OUTPUT_MEM_OP


@typechecked
class LayerDim(DimensionABC):
    """! (for-loop) dimension of a workload layer (e.g. `K`, `C`)
    # TODO make layer_dim_size a property of LayerDim
    """

    def __init__(self, name: str):
        assert name.isalpha(), "LayerDim name contains special characters or numbers"
        super().__init__(name.upper())

    def create_r_version(self) -> "LayerDim":
        """! Create a new LayerDim instance with is tagged `relevant` and can be distinguished from non-tagged
        LayerDims"""
        new_instance = LayerDim(self.name)
        new_instance.name = self.name + "_r"
        return new_instance

    def create_ir_version(self) -> "LayerDim":
        """! Create a new LayerDim instance with is tagged `irrelevant` and can be distinguished from non-tagged LayerDims"""
        new_instance = LayerDim(self.name)
        new_instance.name = self.name + "_ir"
        return new_instance


@typechecked
class Dimension(DimensionABC):
    """! Operational Array Dimension
    # TODO rename to `OADimension
    """

    def __init__(self, index: int = 0, name: str = "", size: int = 1):
        """!  The class constructor
        @param index: The integer index of this Dimension.
        @param name: The user-provided name of this Dimension.
        @param size: The user-provided size of this Dimension.
        """
        super().__init__(name)
        self.id = index
        self.size = size

    def __eq__(self, other: Any):
        if not isinstance(other, Dimension):
            return False
        assert not (
            self.id == other.id and self.name != other.name
        ), "OADims found with same id but different name, perhaps this was a mistake?"
        assert not (
            self.name == other.name and self.id != other.id
        ), "OADims found with same name but different id, perhaps this was a mistake?"
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @staticmethod
    def parse_user_input(x: str):
        assert bool(re.match(r"D\d", x)), f"User specified dimension {x} not recognized"
        # Dimension id starts at 0, user format starts at D1
        idx: int = int(re.findall(r"\d", x).pop()) - 1
        return Dimension(idx, x)


class Constants:
    """! Store constant objects used throughout ZigZag (instead of hardcoding them)"""

    # Intermediate output operand. Hard coded, and must be specified by the user as such
    OUTPUT_OPERAND_STR = "O"
    # Final output operand after scaling. Hard coded, and must be specified by the user as such
    FINAL_OUTPUT_OPERAND_STR = "O_final"

    OUTPUT_LAYER_OP = LayerOperand(OUTPUT_OPERAND_STR)
    FINAL_OUTPUT_LAYER_OP = LayerOperand(FINAL_OUTPUT_OPERAND_STR)
    OUTPUT_MEM_OP = MemoryOperand(OUTPUT_OPERAND_STR)
    FINAL_OUTPUT_MEM_OP = MemoryOperand(FINAL_OUTPUT_OPERAND_STR)

    MEM_OP_1_STR = "I1"
    MEM_OP_2_STR = "I2"
    MEM_OP_1 = MemoryOperand(MEM_OP_1_STR)
    MEM_OP_2 = MemoryOperand(MEM_OP_2_STR)


###### Type aliases ######
OperandStr: TypeAlias = str
MemOperandStr: TypeAlias = str
LayerDimStr: TypeAlias = str
UnrollFactor: TypeAlias = int | float
UnrollFactorInt: TypeAlias = int

PrLoop: TypeAlias = dict[LayerDim, list[LayerDim]]
LoopList: TypeAlias = list[LayerDim]
PrScalingFactors: TypeAlias = dict[LayerDim, dict[LayerDim, int]]
