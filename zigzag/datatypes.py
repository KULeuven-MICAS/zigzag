import re
from abc import ABCMeta
from typing import Any, Literal, TypeAlias

import numpy as np

from zigzag.parser.accelerator_validator import AcceleratorValidator
from zigzag.utils import hash_sha512


class OperandABC(metaclass=ABCMeta):
    """! Abstract Base Class for all dimension- and operand-like classes"""

    def __init__(self, name: str):
        self.__name = name
        self.__hash = hash_sha512(name) ^ hash_sha512(type(self))

    @property
    def name(self):
        """Protect the class variable from reassignment (as this would invalidate the stored hash value)"""
        return self.__name

    def __eq__(self, other: "OperandABC"):  # type: ignore
        return self.__hash == other.__hash  # pylint: disable=W0212

    def __hash__(self):
        """Optimize performance by statically storing the hash"""
        return self.__hash

    def __str__(self):
        return self.__name

    def __repr__(self):
        return str(self)

    def __lt__(self, other: "OperandABC"):
        return self.__name < other.__name  # pylint: disable=W0212

    def __ge__(self, other: "OperandABC"):
        return self.__name >= other.__name  # pylint: disable=W0212

    def __jsonrepr__(self):
        return self.__name


class LayerOperand(OperandABC):
    """! Operand from the layer definition, e.g. `I`, `W`, `O`."""

    def is_output(self):
        return self == Constants.OUTPUT_LAYER_OP

    def is_final_output(self):
        return self == Constants.FINAL_OUTPUT_LAYER_OP


class MemoryOperand(OperandABC):
    """! Operand from the memory definition, e.g. `I1`, `I2`, `O`."""

    def is_output(self):
        return self == Constants.OUTPUT_MEM_OP

    def is_final_output(self):
        return self == Constants.FINAL_OUTPUT_MEM_OP


class LayerDim(OperandABC):
    """! (for-loop) dimension of a workload layer (e.g. `K`, `C`)"""

    def __init__(self, name: str):
        # assert name.isalpha(), "LayerDim name contains special characters or numbers"
        super().__init__(name.upper())

    def create_r_version(self) -> "LayerDim":
        """! Create a new LayerDim instance with is tagged `relevant` and can be distinguished from non-tagged
        LayerDims"""
        new_instance = LayerDim(self.name + "_r")
        return new_instance

    def create_ir_version(self) -> "LayerDim":
        """! Create a new LayerDim instance with is tagged `irrelevant` and can be distinguished from non-tagged
        LayerDims"""
        new_instance = LayerDim(self.name + "_ir")
        return new_instance


class OADimension(OperandABC):
    """! Operational Array Dimension"""

    def __init__(self, name: str):
        assert bool(re.match(AcceleratorValidator.DIMENSION_REGEX, name)), f"OADimension {name} does not resemble `D1`"
        super().__init__(name)


class Constants:
    """! Store constant objects used throughout ZigZag (instead of hardcoding them)"""

    LAYER_OP_I = LayerOperand("I")
    LAYER_OP_W = LayerOperand("W")
    OUTPUT_LAYER_OP = LayerOperand("O")
    FINAL_OUTPUT_LAYER_OP = LayerOperand("O_final")

    MEM_OP_1 = MemoryOperand("I1")
    MEM_OP_2 = MemoryOperand("I2")
    OUTPUT_MEM_OP = MemoryOperand("O")
    FINAL_OUTPUT_MEM_OP = MemoryOperand("O_final")

    UNKNOWN_DIM_OPERATOR = LayerDim("*")


###### Type aliases ######
UnknownUnrollFactor: TypeAlias = Literal["*"]
UnrollFactor: TypeAlias = int | float
UnrollFactorInt: TypeAlias = int | UnknownUnrollFactor

PrLoop: TypeAlias = dict[LayerDim, tuple[LayerDim, LayerDim]]
LoopList: TypeAlias = list[LayerDim]
# There can only be two factors
PrScalingFactors: TypeAlias = dict[LayerDim, tuple[tuple[LayerDim, int], tuple[LayerDim, int]]]
ArrayType = np.ndarray[Any, Any]  # pylint: disable=E1136
