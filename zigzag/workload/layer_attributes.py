import logging
import re
from collections import defaultdict
from math import prod
from typing import TypeAlias

from zigzag.datatypes import (
    Constants,
    LayerDim,
    LayerOperand,
    LoopList,
    MemoryOperand,
    PrLoop,
    PrScalingFactors,
    UnrollFactor,
    UnrollFactorInt,
)
from zigzag.opt.loma.multipermute import (
    PermutationConstraint,
    StaticPositionsAndSizesConstraint,
    StaticPositionsConstraint,
)
from zigzag.workload.LayerAttribute import LayerAttribute

logger = logging.getLogger(__name__)

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
        layer_op_idx = layer_operands.index(layer_op)
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
        return prod(self.data.values())

    def items(self):
        return self.data.items()

    def copy(self):
        return LayerDimSizes(self.data.copy())

    def __setitem__(self, key: LayerDim, value: UnrollFactor):
        self.data[key] = value

    def __delitem__(self, key: LayerDim):
        del self.data[key]

    def __add__(self, other: "LayerDimSizes"):
        return LayerDimSizes(self.data | other.data)


class LayerOperandPrecision(LayerAttribute):
    """! Contains the bit precision of each layer operand"""

    def __init__(self, data: dict[LayerOperand, int]):
        self.data = data

    def __getitem__(self, layer_op: LayerOperand):
        return self.data[layer_op]

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
        # Class variables are computed and stored once to improve runtime performance
        self.layer_operands = list(self.data.keys())
        self.mem_operands = list(self.data.values())
        self.__mem_to_layer_op_dict = {self.data[layer_op]: layer_op for layer_op in self.layer_operands}
        assert len(self.__mem_to_layer_op_dict) == len(self.data), "MemoryOperandLinks contains duplicate MemoryOperand"

    def layer_to_mem_op(self, layer_op: LayerOperand) -> MemoryOperand:
        return self.data[layer_op]

    def mem_to_layer_op(self, mem_op: MemoryOperand) -> LayerOperand:
        """! Given a MemoryOperand, return the linked LayerOperand"""
        return self.__mem_to_layer_op_dict[mem_op]

    def layer_and_mem_ops(self):
        return self.data.items()

    def contains_layer_op(self, layer_op: LayerOperand) -> bool:
        return layer_op in self.layer_operands

    def contains_mem_op(self, mem_op: MemoryOperand) -> bool:
        return mem_op in self.mem_operands

    def copy(self):
        return MemoryOperandLinks(self.data.copy())

    def __str__(self):
        return str({str(k): str(v) for k, v in self.data.items()})


class LayerDimRelation(LayerAttribute):
    """! For the operand dimension that is not directly a loop dimension, a relation equations between them(operand
    dimension) and the loop dimension is required. e.g. `dim1 = coef2 * dim2 + coef3 * dim3`
    """

    def __init__(
        self,
        dim_1: LayerDim,
        dim_2: LayerDim,
        dim_3: LayerDim,
        coef_2: int,
        coef_3: int,
    ):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.coef_2 = coef_2
        self.coef_3 = coef_3
        self.data = f"{dim_1} = {coef_2}*{dim_2} + {coef_3}*{dim_3}"

    @staticmethod
    def extract_pr_loop_info(
        relations: list["LayerDimRelation"],
    ) -> tuple[PrLoop, LoopList, PrScalingFactors]:
        """!
        # TODO requires cleanup and documentation
        """
        pr_loop: PrLoop = {}
        pr_loop_list: LoopList = []
        pr_scaling_factors: PrScalingFactors = {}

        for relation in relations:
            pr_loop[relation.dim_1] = (relation.dim_2, relation.dim_3)
            pr_loop_list.extend([relation.dim_1, relation.dim_2, relation.dim_3])
            scaling_factors = (
                (relation.dim_2, relation.coef_2),
                (relation.dim_3, relation.coef_3),
            )

            pr_scaling_factors[relation.dim_1] = scaling_factors

        return pr_loop, pr_loop_list, pr_scaling_factors


class LayerTemporalOrdering(LayerAttribute):
    """Represents a user-defined temporal ordering"""

    def __init__(self, data: list[list[str | UnrollFactorInt]]):
        """data will look like:
        [['K', 12],
         ['C',  3]]
        """
        self.data = [(LayerDim(str(loop[0])), int(loop[1]) if isinstance(loop[1], int) else None) for loop in data]

    @staticmethod
    def empty():
        return LayerTemporalOrdering([])

    def is_empty(self):
        return len(self.data) == 0

    def is_complete(self, temporal_loop_sizes: dict[LayerDim, UnrollFactor]):
        """Return wether this temporal ordering matches the given, mandatory loop sizes"""
        all_loops: defaultdict[LayerDim, UnrollFactor] = defaultdict(lambda: 1)
        for layer_dim, factor in self.data:
            if not isinstance(factor, int):
                return False
            else:
                all_loops[layer_dim] *= factor

        for layer_dim in all_loops:
            if all_loops[layer_dim] == 1:
                del all_loops[layer_dim]

        return all_loops == temporal_loop_sizes

    def remove_invalid_layer_dims(self, layer_dim_sizes: LayerDimSizes, layer_name: str = ""):
        for i, mapping in list(enumerate(self.data))[::-1]:
            if mapping[0] not in layer_dim_sizes.layer_dims:
                logger.warning(
                    "Supplied temporal ordering %s%s thrown out because layer dimension is not present in " "the layer",
                    mapping,
                    "" if layer_name == "" else f" for layer {layer_name}",
                )
                del self.data[i]

    def to_legacy_format(self):
        return self.data

    def get_constraints(self) -> list[PermutationConstraint]:
        static_posistions_dict: dict[int, LayerDim] = {}
        static_posistions_and_sizes_dict: dict[int, tuple[LayerDim, int]] = {}
        outer_loop = False
        for count, (layer_dim, factor) in enumerate(self.data):
            if (layer_dim == Constants.UNKNOWN_DIM_OPERATOR) and (factor is None):
                outer_loop = True
            elif factor is None:
                if not outer_loop:
                    static_posistions_dict[count] = layer_dim
                else:
                    static_posistions_dict[count - len(self.data)] = layer_dim
            else:
                if not outer_loop:
                    static_posistions_and_sizes_dict[count] = (layer_dim, factor)
                else:
                    static_posistions_and_sizes_dict[count - len(self.data)] = (layer_dim, factor)
        static_positions = StaticPositionsConstraint(static_posistions_dict)
        static_posistions_and_sizes = StaticPositionsAndSizesConstraint(static_posistions_and_sizes_dict)
        return [static_positions, static_posistions_and_sizes]


class LayerPadding(LayerAttribute):
    DEFAULT = (0, 0)

    def __init__(self, data: dict[LayerDim, tuple[int, int]]):
        self.data = data

    def __getitem__(self, key: LayerDim) -> tuple[int, int]:
        return self.data[key] if key in self.data else LayerPadding.DEFAULT

    @staticmethod
    def empty():
        return LayerPadding({})
