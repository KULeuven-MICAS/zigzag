import math
import re
from typing import TypeAlias


from zigzag.workload.LayerAttribute import LayerAttribute
from zigzag.datatypes import (
    Constants,
    LayerOperand,
    LayerDim,
    MemoryOperand,
    LoopList,
    PrLoop,
    PrScalingFactors,
    UnrollFactor,
    UnrollFactorInt,
)

InputOperandSource: TypeAlias = dict[LayerOperand, int]


class LayerEquation(LayerAttribute):
    """ "!  core computation equation, e.g. `O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]`,
    `Y[i][j] += A[i][k] * B[k][j]`, `Y[i][j] += A[i][k][l] * B[k][j] * C[l][j]`, etc."""

    def __init__(self, data: str):
        self.data = data
        self.disassembly = self.__get_disassembly()

    def __get_disassembly(self) -> list[str]:
        """! Disassemble the equation, e.g. `['O', 'p', 'r', '=', 'I', 'p', 'q', '*', 'W', 'q', 'r']`"""
        return re.findall("[a-zA-Z,0-9,=,*,+]+", self.data)

    def __get_operand_start_indices(self) -> list[int]:
        end_indices = [i for (i, x) in enumerate(self.disassembly) if x in ["=", "*", "+"]]
        return [0] + [idx + 1 for idx in end_indices]

    def get_contained_operands(self) -> list[LayerOperand]:
        """! Return a list with all LayerOperands contained within this instance."""
        return [LayerOperand(self.disassembly[idx]) for idx in self.__get_operand_start_indices()]

    def get_r_layer_dims(self, layer_op: LayerOperand) -> list[LayerDim]:
        """! Return a list with all LayerDims that are `relevant` for the given LayerOperand"""
        layer_operands = self.get_contained_operands()
        assert layer_op in layer_operands, f"Given LayerOperand {layer_op} is not part of this equation"
        assert layer_op in layer_operands, f"Given LayerOperand {layer_op} is not part of this equation"
        layer_op_idx = layer_operands.index(layer_op)
        slice_indices = self.__get_operand_start_indices() + [len(self.disassembly) + 1]
        slice_indices = self.__get_operand_start_indices() + [len(self.disassembly) + 1]
        disassembly_start_idx = slice_indices[layer_op_idx] + 1
        disassembly_end_idx = slice_indices[layer_op_idx + 1] - 1
        equation_slice = self.disassembly[disassembly_start_idx:disassembly_end_idx]
        return [LayerDim(x.upper()) for x in equation_slice]


class LayerDimSizes(LayerAttribute):
    """! Contains the size of each computation loop as defined in the workload,
    e.g. `{'B': 1, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1`"""

    def __init__(self, data: dict[LayerDim, UnrollFactor]):
        self.data = data

    @property
    def layer_dims(self) -> list[LayerDim]:
        return list(self.data.keys())

    @property
    def total_size(self) -> UnrollFactor:
        return math.prod(self.data.values())

    def items(self):
        return self.data.items()

    def copy(self):
        return LayerDimSizes(self.data.copy())

    def __setitem__(self, key: LayerDim, value: int):
        self.data[key] = value

    def __delitem__(self, key: LayerDim):
        del self.data[key]

    def __add__(self, other: "LayerDimSizes"):
        return LayerDimSizes(self.data | other.data)


class LayerOperandPrecision(LayerAttribute):
    """! Contains the bit precision of each layer operand"""

    def __init__(self, data: dict[LayerOperand, int]):
        self.data = data

    @property
    def final_output_precision(self) -> int:
        """! Return the precision of either the final output (if defined by user) or the intermediate output"""
        if Constants.FINAL_OUTPUT_LAYER_OP in self.data:
            return self.data[Constants.FINAL_OUTPUT_LAYER_OP]
        return self.data[Constants.OUTPUT_LAYER_OP]


class MemoryOperandLinks(LayerAttribute):
    """! Links LayerOperand to MemoryOperand."""

    def __init__(self, data: dict[LayerOperand, MemoryOperand]):
        self.data = data

    def layer_to_mem_op(self, layer_op: LayerOperand) -> MemoryOperand:
        assert self.contains_layer_op(layer_op)
        return self.data[layer_op]

    def mem_to_layer_op(self, mem_op: MemoryOperand) -> LayerOperand:
        """! Given a MemoryOperand, return the linked LayerOperand or None if the MemoryOperand is not contained
        within"""
        assert self.contains_mem_op(mem_op)
        candidates = {k for k, v in self.data.items() if v == mem_op}
        assert len(candidates) <= 1, f"MemoryOperandLinks contains duplicate MemoryOperand {mem_op}"
        assert len(candidates) > 0, f"Memory operand {mem_op} is not present"
        return candidates.pop()

    def contains_layer_op(self, layer_op: LayerOperand) -> bool:
        return layer_op in self.layer_operands

    def contains_mem_op(self, mem_op: MemoryOperand) -> bool:
        return mem_op in self.mem_operands

    @property
    def layer_operands(self) -> set[LayerOperand]:
        return set(self.data.keys())

    @property
    def mem_operands(self) -> set[MemoryOperand]:
        return set(self.data.values())

    def copy(self):
        return MemoryOperandLinks(self.data.copy())

    def __str__(self):
        return str({str(k): str(v) for k, v in self.data.items()})


class LayerDimRelation(LayerAttribute):
    """! For the operand dimension that is not directly a loop dimension, a relation equations between them(operand
    dimension) and the loop dimension is required. e.g. `dim1 = coef2 * dim2 + coef3 * dim3`
    """

    def __init__(self, dim_1: LayerDim, dim_2: LayerDim, dim_3: LayerDim, coef_2: int, coef_3: int):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.coef_2 = coef_2
        self.coef_3 = coef_3
        self.data = f"{dim_1} = {coef_2}*{dim_2} + {coef_3}*{dim_3}"

    @staticmethod
    def extract_pr_loop_info(relations: list["LayerDimRelation"]) -> tuple[PrLoop, LoopList, PrScalingFactors]:
        """!
        # TODO requires cleanup and documentation
        """
        pr_loop: PrLoop = {}
        pr_loop_list: LoopList = []
        pr_scaling_factors: PrScalingFactors = {}

        for relation in relations:
            key = relation.dim_1
            val = [relation.dim_2, relation.dim_3]

        for relation in relations:
            key = relation.dim_1
            val = [relation.dim_2, relation.dim_3]
            pr_loop[key] = val
            pr_loop_list.extend([key] + val)
            scaling_factors = {relation.dim_2: relation.coef_2, relation.dim_3: relation.coef_3}
            scaling_factors = {relation.dim_2: relation.coef_2, relation.dim_3: relation.coef_3}
            pr_scaling_factors[key] = scaling_factors

        return pr_loop, pr_loop_list, pr_scaling_factors


class LayerTemporalOrdering(LayerAttribute):

    def __init__(self, data: dict[str, UnrollFactorInt]):
        self.data = [[LayerDim(loop[0]), loop[1]] for loop in data]

    @staticmethod
    def empty():
        return LayerTemporalOrdering({})

    def __delitem__(self, x: LayerDim):
        del self.data[x]


class LayerPadding(LayerAttribute):
    DEFAULT = (0, 0)

    def __init__(self, data: dict[LayerDim, tuple[int, int]]):
        self.data = data

    def __getitem__(self, key: LayerDim) -> tuple[int, int]:
        return self.data[key] if key in self.data else LayerPadding.DEFAULT

    @staticmethod
    def empty():
        return LayerPadding({})
