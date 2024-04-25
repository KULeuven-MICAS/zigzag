import math
import re
from typing import Any, TypeAlias


from zigzag.mapping.spatial_mapping import SpatialMapping, SpatialMappingHint
from zigzag.workload.LayerAttribute import LayerAttribute
from zigzag.datatypes import (
    Constants,
    LayerOperand,
    LayerDim,
    MemoryOperand,
    LoopList,
    MemOperandStr,
    OperandStr,
    LayerDimStr,
    PrLoop,
    PrScalingFactors,
    UnrollFactor,
    UnrollFactorInt,
)

InputOperandSource: TypeAlias = dict[LayerOperand, list[int]]


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
        assert layer_op in layer_operands
        layer_op_idx = layer_operands.index(layer_op)
        slice_indices = self.__get_operand_start_indices() + [len(self.disassembly)]
        disassembly_start_idx = slice_indices[layer_op_idx] + 1
        disassembly_end_idx = slice_indices[layer_op_idx + 1] - 1
        equation_slice = self.disassembly[disassembly_start_idx:disassembly_end_idx]
        return [LayerDim(x.upper()) for x in equation_slice]

    @staticmethod
    def parse_user_input(x: str) -> "LayerEquation":
        assert isinstance(x, str)
        assert " " not in x, f"Please remove all spaces from `equation` {x}"
        x = x.replace("+=", "=")
        x = x.replace("++", "+")
        x = x.replace("*", " * ")
        x = x.replace("=", " = ")
        x = x.replace("+", " + ")

        return LayerEquation(x)


class LayerDimSizes(LayerAttribute):
    """! Contains the size of each computation loop as defined in the workload,
    e.g. `{'B': 1, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1`"""

    def __init__(self, data: dict[LayerDim, UnrollFactor]):
        self.data = data

    def layer_dims(self) -> list[LayerDim]:
        return list(self.data.keys())

    def get_total_size(self) -> UnrollFactor:
        return math.prod(self.data.values())

    def items(self):
        return self.data.items()

    def copy(self):
        return LayerDimSizes(self.data.copy())

    def __setitem__(self, key: LayerDim, value: int):
        self.data[key] = value

    def __delitem__(self, key: LayerDim):
        del self.data[key]

    @staticmethod
    def parse_user_input(x: dict[LayerDimStr, UnrollFactor]):
        assert isinstance(x, dict)
        assert all([isinstance(k, LayerDimStr) for k in x.keys()])
        assert all([isinstance(k, UnrollFactor) for k in x.values()])
        data = {LayerDim(layer_dim_str): size for layer_dim_str, size in x.items()}
        return LayerDimSizes(data)


class LayerOperandPrecision(LayerAttribute):
    """! Contains the bit precision of each layer operand"""

    def __init__(self, data: dict[LayerOperand, int]):
        self.data = data

    def get_final_output_precision(self) -> int:
        """! Return the precision of either the final output (if defined by user) or the intermediate output"""
        if Constants.FINAL_OUTPUT_LAYER_OP in self.data:
            return self.data[Constants.FINAL_OUTPUT_LAYER_OP]
        return self.data[Constants.OUTPUT_LAYER_OP]

    @staticmethod
    def parse_user_input(x: dict[OperandStr, int]):
        assert isinstance(x, dict)
        assert all([isinstance(k, OperandStr) for k in x.keys()])
        assert all([isinstance(k, int) for k in x.values()])
        assert (
            Constants.OUTPUT_OPERAND_STR in x or Constants.FINAL_OUTPUT_OPERAND_STR in x
        ), "Operand precision does not contain `O` or `O_final` as operand"
        data = {LayerOperand(operand_str): size for operand_str, size in x.items()}
        return LayerOperandPrecision(data)


class MemoryOperandLinks(LayerAttribute):
    """! Links LayerOperand to MemoryOperand."""

    def __init__(self, data: dict[LayerOperand, MemoryOperand]):
        self.data = data

    def layer_to_mem_op(self, layer_op: LayerOperand) -> MemoryOperand:
        assert layer_op in self.data
        return self.data[layer_op]

    def mem_to_layer_op(self, mem_op: MemoryOperand) -> LayerOperand | None:
        """! Given a MemoryOperand, return the linked LayerOperand or None if the MemoryOperand is not contained
        within"""
        assert mem_op in self.data.values()
        candidates = {k for k, v in self.items() if v == mem_op}
        assert len(candidates) <= 1, "MemoryOperandLinks contains duplicate MemoryOperand"
        if len(candidates) == 0:
            return None
        return candidates.pop()

    def items(self):
        return self.data.items()

    def copy(self):
        return MemoryOperandLinks(self.data.copy())

    def __str__(self):
        return str({str(k): str(v) for k, v in self.items()})

    @staticmethod
    def parse_user_input(x: dict[OperandStr, MemOperandStr]):
        assert isinstance(x, dict)
        assert all([isinstance(k, OperandStr) for k in x.keys()])
        assert all([isinstance(k, MemOperandStr) for k in x.values()])
        data = {LayerOperand(layer_op_str): MemoryOperand(mem_op_str) for layer_op_str, mem_op_str in x.items()}
        return MemoryOperandLinks(data)


class LayerDimRelations(LayerAttribute):
    """! For the operand dimension that is not directly a loop dimension, a set of specific relation equations between
    them (operand dimension and loop dimension) is required, e.g. ['ix=ox+fx-1', 'iy=oy+fy-1']
    """

    def __init__(self, data: list[str]):
        self.data = data

    def extract_pr_loop_info(self) -> tuple[PrLoop, LoopList, PrScalingFactors]:
        """!
        # TODO requires cleanup and documentation
        """
        pr_loop: PrLoop = {}
        pr_loop_list: LoopList = []
        pr_scaling_factors: PrScalingFactors = {}
        # Regex pattern to find dimensions and coefficients of form dim1 = coef_2*dim2 + coef_3*dim3
        pattern = r"(\w+)\s*=\s*(?:(\w+)\s*\*\s*)?(\w+)\s*\+\s*(?:(\w+)\s*\*\s*)?(\w+)"
        for relation in self.data:
            match = re.search(pattern, relation)
            if match:
                dim1, coef_2, dim2, coef_3, dim3 = match.groups()
                dim1, dim2, dim3 = LayerDim(dim1), LayerDim(dim2), LayerDim(dim3)
                coef_2 = int(coef_2) if coef_2 is not None else 1
                coef_3 = int(coef_3) if coef_3 is not None else 1
            else:
                raise ValueError(f"Please make sure {relation} is of the form 'dim1 = a*dim2 + b*dim3'")

            key = dim1
            val = [dim2, dim3]
            pr_loop[key] = val
            pr_loop_list.extend([key] + val)
            scaling_factors = {dim2: coef_2, dim3: coef_3}
            pr_scaling_factors[key] = scaling_factors

        return pr_loop, pr_loop_list, pr_scaling_factors

    @staticmethod
    def parse_user_input(x: list[str]):
        assert isinstance(x, list)
        assert all([isinstance(elem, str) for elem in x])
        return LayerDimRelations(x)


class LayerTemporalOrdering(LayerAttribute):
    def __init__(self, data: dict[LayerOperand, UnrollFactorInt]):
        self.data = data

    @staticmethod
    def parse_user_input(x: dict[OperandStr, int]):
        assert isinstance(x, dict)
        assert all([isinstance(k, OperandStr) for k in x.keys()])
        assert all([isinstance(v, int) for v in x.values()])
        data = {LayerOperand(layer_op_str): factor for layer_op_str, factor in x.items()}
        return LayerTemporalOrdering(data)


class LayerPadding(LayerAttribute):
    DEFAULT = (0, 0)

    def __init__(self, data: dict[LayerDim, tuple[int, int]]):
        self.data = data

    def __getitem__(self, key: LayerDim) -> tuple[int, int]:
        return self.data[key] if key in self.data else LayerPadding.DEFAULT

    @staticmethod
    def parse_user_input(x: dict[LayerDimStr, tuple[int, int]]):
        assert isinstance(x, dict)
        assert all([isinstance(k, LayerDimStr) for k in x.keys()])
        assert all(
            [isinstance(v, tuple) and len(v) == 2 and all([isinstance(elem, int) for elem in v]) for v in x.values()]
        )
        data = {LayerDim(layer_op_str): value for layer_op_str, value in x.items()}
        return LayerPadding(data)


class LayerConstantOperands(LayerAttribute):
    # TODO maybe this class is excessive and should just be list[LayerOperand] or empty list
    def __init__(self, data: list[LayerOperand]):
        self.data = data

    @staticmethod
    def parse_user_input(x: list[OperandStr]):
        # TODO should this check wether the list is empty?
        assert isinstance(x, list)
        assert all([isinstance(elem, OperandStr) for elem in x])
        data = [LayerOperand(layer_op_str) for layer_op_str in x]
        return LayerConstantOperands(data)


class LayerAttributes:
    """! Represents the layer attributes as given by the user and contains methods to parse each attribute.
    Rationale: only this class contains the (hard-coded) layer attribute strings from the user input format.
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def parse_equation(self) -> LayerEquation:
        key = "equation"
        assert key in self, f"Workload does not contain `{key}` definition"
        return LayerEquation.parse_user_input(self.data[key])

    def parse_layer_dim_sizes(self) -> LayerDimSizes:
        key = "loop_dim_size"
        assert key in self, f"Workload does not contain `{key}` definition"
        return LayerDimSizes.parse_user_input(self.data[key])

    def parse_pr_layer_dim_sizes(self) -> LayerDimSizes | None:
        key = "pr_loop_dim_size"
        # Fail soft
        if key not in self:
            return None
        return LayerDimSizes.parse_user_input(self.data[key])

    def parse_operand_precision(self) -> LayerOperandPrecision:
        key = "operand_precision"
        assert key in self, f"Workload does not contain `{key}` definition"
        return LayerOperandPrecision.parse_user_input(self.data[key])

    def parse_operand_source(self) -> InputOperandSource:
        key = "operand_source"
        assert key in self, f"Workload does not contain `{key}` definition"
        x = self.data[key]
        assert isinstance(x, dict)
        assert all([isinstance(key, str) for key in x.keys()])
        assert all([isinstance(value, list) for value in x.values()])
        assert all([all([isinstance(elem, int) for elem in value]) for value in x.values()])
        return {LayerOperand(key): [elem for elem in value] for key, value in x.items()}

    def parse_layer_dim_relations(self) -> LayerDimRelations | None:
        key = "dimension_relations"
        # Fail soft
        if key not in self:
            return None
        return LayerDimRelations.parse_user_input(self.data[key])

    def parse_spatial_mapping(self) -> SpatialMapping:
        key = "spatial_mapping"
        assert key in self, f"Workload does not contain `{key}` definition"
        return SpatialMapping.parse_user_input(self.data[key])

    def parse_spatial_mapping_hint(self) -> SpatialMappingHint:
        key = "spatial_mapping_hint"
        # Fail soft
        if key not in self:
            return SpatialMappingHint.empty()
        return SpatialMappingHint.parse_user_input(self.data[key])

    def parse_core_allocation(self) -> int:
        key = "core_allocation"
        assert key in self, f"Workload does not contain `{key}` definition"
        value = self.data[key]
        assert isinstance(value, int)
        return value

    def parse_mem_operand_links(self) -> MemoryOperandLinks:
        key = "memory_operand_links"
        assert key in self, f"Workload does not contain `{key}` definition"
        return MemoryOperandLinks.parse_user_input(self.data[key])

    def parse_temporal_ordering(self) -> LayerTemporalOrdering | None:
        key = "temporal_ordering"
        if key not in self:
            return None
        return LayerTemporalOrdering.parse_user_input(self.data[key])

    def parse_padding(self) -> LayerPadding | None:
        key = "padding"
        # Fail soft
        if key not in self:
            return None
        return LayerPadding.parse_user_input(self.data[key])

    def parse_constant_operands(self) -> list[LayerOperand]:
        key = "constant_operands"
        # Fail soft
        if key not in self:
            return list()
        x = self.data[key]
        assert isinstance(x, list)
        assert all([isinstance(elem, OperandStr) for elem in x])
        return [LayerOperand(layer_op_str) for layer_op_str in x]

    def parse_operator_type(self) -> str | None:
        key = "operator_type"
        if key not in self:
            return None
        return self.data[key]

    def __contains__(self, x):
        return x in self.data

    @staticmethod
    def parse_user_input(x: dict[str, Any]) -> "LayerAttributes":
        assert isinstance(x, dict)
        assert all([isinstance(elem, str) for elem in x])
        return LayerAttributes(x)
