import logging
from copy import deepcopy
from dataclasses import dataclass
from math import gcd, prod

from zigzag.datatypes import (
    LayerDim,
    LayerOperand,
    LoopList,
    PrLoop,
    PrScalingFactors,
    UnrollFactor,
)
from zigzag.mapping.spatial_mapping import SpatialMapping, SpatialMappingHint
from zigzag.utils import json_repr_handler
from zigzag.workload.layer_attributes import (
    InputOperandSource,
    LayerDimRelation,
    LayerDimSizes,
    LayerEquation,
    LayerOperandPrecision,
    LayerPadding,
    LayerTemporalOrdering,
    MemoryOperandLinks,
)
from zigzag.workload.LayerNodeABC import LayerNodeABC

logger = logging.getLogger(__name__)


class LoopRelevancyInfo:
    """! Per LayerOperand, store the Relevant, Irrelevant LayerDims, and which LayerDims are Partially Relevant
    to each other
    # TODO move to somewhere else
    """

    def __init__(self):
        self.r_dims: dict[LayerOperand, list[LayerDim]] = dict()
        self.ir_dims: dict[LayerOperand, list[LayerDim]] = dict()
        self.pr_dims: dict[LayerOperand, PrLoop] = dict()
        self.orig_pr_loop: PrLoop

    def get_r_layer_dims(self, layer_operand: LayerOperand) -> list[LayerDim]:
        return self.r_dims[layer_operand]

    def get_ir_layer_dims(self, layer_operand: LayerOperand) -> list[LayerDim]:
        return self.ir_dims[layer_operand]

    def get_pr_layer_dims(self, layer_operand: LayerOperand) -> PrLoop:
        return self.pr_dims[layer_operand]

    def get_r_or_pr_layer_dims(self, layer_operand: LayerOperand) -> list[LayerDim]:
        return self.r_dims[layer_operand] + list(self.pr_dims[layer_operand].keys())

    def create_pr_decoupled_relevancy_info(self) -> "LoopRelevancyInfo":
        """! remove the pr loop dict, and put the pr-related data dimension (e.g. IX and IY)
        to r and ir dict with "r" and "ir" tags
        # NOTE this method requires the unaltered pr_loop, before going through `extract_relevancy_info`. Kind of messy
        """
        new = LoopRelevancyInfo()
        new.r_dims = deepcopy(self.r_dims)
        new.ir_dims = deepcopy(self.ir_dims)
        for layer_op in self.pr_dims:
            new.r_dims[layer_op] += [pr_layer_dim.create_r_version() for pr_layer_dim in self.orig_pr_loop]
            new.ir_dims[layer_op] += [pr_layer_dim.create_ir_version() for pr_layer_dim in self.orig_pr_loop]
        return new

    @staticmethod
    def extract_relevancy_info(
        equation: LayerEquation,
        layer_dim_sizes: LayerDimSizes,
        pr_loop: PrLoop,
        pr_loop_list: LoopList,
    ) -> "LoopRelevancyInfo":
        """!
        # TODO requires cleanup and documentation
        """
        self = LoopRelevancyInfo()
        self.orig_pr_loop = pr_loop

        dimension_list = layer_dim_sizes.layer_dims
        for layer_op in equation.get_contained_operands():
            r_loop_list = equation.get_r_layer_dims(layer_op)
            ir_loop_list = list(set(dimension_list).difference(r_loop_list))

            pr_loop_remove_flag = any(layer_dim in pr_loop for layer_dim in r_loop_list)
            if pr_loop_remove_flag:
                self.r_dims[layer_op] = [layer_dim for layer_dim in r_loop_list if layer_dim not in pr_loop_list]
                self.ir_dims[layer_op] = [
                    layer_dim
                    for layer_dim in ir_loop_list
                    if layer_dim not in pr_loop_list and layer_dim_sizes[layer_dim] != 1
                ]
                self.pr_dims[layer_op] = pr_loop
            else:
                self.r_dims[layer_op] = [layer_dim for layer_dim in r_loop_list if layer_dim_sizes[layer_dim] != 1]
                self.ir_dims[layer_op] = [layer_dim for layer_dim in ir_loop_list if layer_dim_sizes[layer_dim] != 1]
                self.pr_dims[layer_op] = {}

        return self


@dataclass
class LayerNodeAttributes:
    """Packs the attributes needed to initialize a LayerNode"""

    layer_type: str
    equation: LayerEquation
    layer_dim_sizes: LayerDimSizes
    operand_precision: LayerOperandPrecision
    dimension_relations: list[LayerDimRelation]
    spatial_mapping: SpatialMapping
    spatial_mapping_hint: SpatialMappingHint
    core_allocation: list[int]
    core_allocation_is_fixed: bool
    memory_operand_links: MemoryOperandLinks
    temporal_ordering: LayerTemporalOrdering
    padding: LayerPadding
    constant_operands: list[LayerOperand]
    input_operand_source: InputOperandSource
    pr_layer_dim_sizes: LayerDimSizes | None


class LayerNode(LayerNodeABC):
    """! Represents a single layer in a workload."""

    def __init__(self, layer_id: int, node_name: str, node_attr: LayerNodeAttributes):
        """
        To construct each layer node, algorithm equation/dimension/indirect relation are parsed.
        This parser collects information of operand, loop dimension, and loop relevance.
        Equal-to-1 loop dimensions are eliminated.
        @param layer_id: The identifier (key) of the layer, as defined in the workload
        @param node_name: an optional name for the Node. E.g. the node's name from the onnx model.
        """
        LayerNodeABC.__init__(self, node_id=layer_id, node_name=node_name)

        # Unpack attributes
        self.type = node_attr.layer_type
        self.equation = node_attr.equation
        self.layer_dim_sizes = node_attr.layer_dim_sizes
        self.operand_precision = node_attr.operand_precision
        self.dimension_relations = node_attr.dimension_relations
        self.spatial_mapping = node_attr.spatial_mapping
        self.spatial_mapping_hint = node_attr.spatial_mapping_hint
        self.core_allocation = node_attr.core_allocation
        self.core_allocation_is_fixed = node_attr.core_allocation_is_fixed
        self.memory_operand_links = node_attr.memory_operand_links
        self.temporal_ordering = node_attr.temporal_ordering
        self.padding = node_attr.padding
        self.constant_operands = node_attr.constant_operands
        self.input_operand_source = node_attr.input_operand_source
        pr_layer_dim_sizes = node_attr.pr_layer_dim_sizes

        # Derived attributes
        self.layer_operands = self.equation.get_contained_operands()
        self.output_operand: LayerOperand = self.layer_operands[0]
        self.input_operands: list[LayerOperand] = self.layer_operands[1:]
        self.layer_dims = list(self.layer_dim_sizes.layer_dims)

        self.pr_loop, pr_loop_list, self.pr_scaling_factors = self.build_pr_funcs()
        self.pr_layer_dim_sizes = (
            LayerDimSizes({dim: self.calc_pr_dimension_size_total(dim) for dim in self.pr_loop})
            if (pr_layer_dim_sizes is None or len(pr_layer_dim_sizes) == 0)
            else pr_layer_dim_sizes
        )
        self.loop_relevancy_info = LoopRelevancyInfo.extract_relevancy_info(
            self.equation, self.layer_dim_sizes, self.pr_loop, pr_loop_list
        )
        self.pr_decoupled_relevancy_info = self.loop_relevancy_info.create_pr_decoupled_relevancy_info()

        # To compute
        self.operand_size_elem: dict[LayerOperand, UnrollFactor] = dict()
        self.extract_layer_info()

    def get_act_layer_op(self) -> LayerOperand:
        """Return the `I` LayerOperand: either the non-constant operand or one of the input operands if none are
        constant"""
        if len(self.constant_operands) == 1:
            # regular layers
            weight_layer_op = self.constant_operands[0]
            act_layer_op = [operand for operand in self.input_operands if operand != weight_layer_op].pop()
        elif len(self.constant_operands) == 0:
            # None constant: take the second one
            act_layer_op = self.input_operands[1]
        else:
            pr_loop_key = next(iter(self.pr_loop.keys()))
            related_loop_dict: dict[LayerOperand, list[LayerDim]] = {
                layer_op: self.equation.get_r_layer_dims(layer_op)
                for layer_op in self.equation.get_contained_operands()
            }
            act_layer_op = [
                operand_in_layer
                for operand_in_layer, related_loop in related_loop_dict.items()
                if pr_loop_key in related_loop
            ].pop()
        return act_layer_op

    def get_weight_layer_op(
        self,
    ) -> LayerOperand:
        """Return the `W` LayerOperand: either the constant operand or one of the input operands if none are
        constant"""
        if len(self.constant_operands) == 1:
            # regular layers
            weight_layer_op = self.constant_operands[0]
        elif len(self.constant_operands) == 0:
            # Adder layers
            weight_layer_op = self.input_operands[0]
        else:
            act_layer_op = self.get_act_layer_op()
            weight_layer_op = [x for x in self.constant_operands if x != act_layer_op].pop()
        return weight_layer_op

    def build_pr_funcs(self) -> tuple[PrLoop, LoopList, PrScalingFactors]:
        """!
        # TODO requires documentation
        """
        if len(self.dimension_relations) > 0:
            return LayerDimRelation.extract_pr_loop_info(self.dimension_relations)

        return {}, [], {}

    def __str__(self):
        return self.name

    def __jsonrepr__(self):
        """! JSON representation used for saving this object to a json file."""
        return json_repr_handler(
            {
                "name": self.name,
                "type": self.type,
                "equation": self.equation,
                "equation_relations": self.dimension_relations,
                "loop_dimensions": self.layer_dim_sizes,
                "operand_precision": self.operand_precision,
                "core_allocation": self.core_allocation,
                "core_allocation_is_fixed": self.core_allocation_is_fixed,
                "user_spatial_mapping": self.spatial_mapping,
                "memory_operand_links": self.memory_operand_links,
            }
        )

    def calc_tensor_size(self, layer_op: LayerOperand, layer_dim_sizes: LayerDimSizes) -> float:
        """! Calculates the tensor size (nb of elements) for the given operand layer_op with the given loop dimension
        sizes layer_dim_sizes."""
        all_dims = self.loop_relevancy_info.get_r_or_pr_layer_dims(layer_op)
        return prod(self.calc_tensor_dim(dim, layer_dim_sizes) for dim in all_dims)

    def calc_tensor_dim(self, dim: LayerDim, layer_dim_sizes: LayerDimSizes):
        if dim in layer_dim_sizes:
            return layer_dim_sizes[dim]
        elif dim in self.pr_loop:
            # TODO this int conversion is suspicious
            related_dimension_sizes = tuple(int(layer_dim_sizes[related_dim]) for related_dim in self.pr_loop[dim])
            scaling_factors = tuple(t[1] for t in self.pr_scaling_factors[dim])
            pr_dim_size = self.calc_pr_dimension_size(
                sa=scaling_factors[0], a=related_dimension_sizes[0], sb=scaling_factors[1], b=related_dimension_sizes[1]
            )
            # Clip this to the largest possible size for this partially relevant dimension (computed at initialization
            # based on padding)
            pr_dim_size = min(self.pr_layer_dim_sizes[dim], pr_dim_size)
            return pr_dim_size
        elif dim in self.layer_dim_sizes:
            return 1
        else:
            raise ValueError("Something went wrong in the initialization of the layer, or in the caller function.")

    def calc_tensor_dims(self, layer_op: LayerOperand, layer_dim_sizes: LayerDimSizes) -> dict[LayerDim, UnrollFactor]:
        """!
        # TODO requires documentation
        """
        all_dims = self.loop_relevancy_info.get_r_or_pr_layer_dims(layer_op)
        return {dim: self.calc_tensor_dim(dim, layer_dim_sizes) for dim in all_dims}

    def calc_pr_dimension_size_total(self, dim: LayerDim) -> int:
        """! Compute the total pr dimension size of this node, taking padding into account.
        @param dim (str): The partially relevant dimension, e.g. 'IX'.
        @return int: The total partially relevant dimension size
        """
        related_dimension_sizes = tuple(self.layer_dim_sizes[related_dim] for related_dim in self.pr_loop[dim])
        scaling_factors = tuple(t[1] for t in self.pr_scaling_factors[dim])
        total_pr_dim_size = self.calc_pr_dimension_size(
            sa=scaling_factors[0], a=related_dimension_sizes[0], sb=scaling_factors[1], b=related_dimension_sizes[1]
        )
        # Partially relevant loop dimensions can also have padding, so get the padding for this pr dimension and
        # subtract
        padding = LayerPadding.DEFAULT if dim not in self.padding else self.padding[dim]
        total_pr_dim_size_without_padding = int(total_pr_dim_size - sum(padding))
        return total_pr_dim_size_without_padding

    @staticmethod
    def calc_pr_dimension_size(sa: int, a: int, sb: int, b: int):
        """! Calculates the number of unique indices c generated by iterating through the indices
        a in range(0,A,1) and b in range(0,B,1) according to the equation c = sa * a + sb * b.
        sa and sb thus represent the scaling of a, resp. b.
        """
        gcd_value = gcd(sa, sb)
        return a * b - max(0, b - (sa // gcd_value)) * (a - (sb // gcd_value))

    def extract_layer_info(self):
        """! This function extract basic information for each layer node."""

        self.total_mac_count = self.layer_dim_sizes.total_size

        # each operand's size (Unit: # of data element)
        for layer_op in self.layer_operands:
            self.operand_size_elem[layer_op] = 1
            for r_layer_dim in self.loop_relevancy_info.get_r_layer_dims(layer_op):
                self.operand_size_elem[layer_op] *= self.layer_dim_sizes[r_layer_dim]
            for pr_layer_dim in self.loop_relevancy_info.get_pr_layer_dims(layer_op):
                self.operand_size_elem[layer_op] *= self.calc_tensor_dims(layer_op, self.layer_dim_sizes)[pr_layer_dim]

        # each operand's size (Unit: bit)
        operand_size_bit: dict[LayerOperand, int] = {}
        for layer_op, size_in_elem in self.operand_size_elem.items():
            operand_size_bit[layer_op] = int(size_in_elem * self.operand_precision[layer_op])
        self.operand_size_bit = operand_size_bit

        # each operand's total data reuse factor, which is total MAC Op/total operand size (in element),
        # i.e. each data element can be used to support how many MAC operation.
        operand_data_reuse: dict[LayerOperand, float] = {}
        for operand, size_in_elem in self.operand_size_elem.items():
            operand_data_reuse[operand] = self.total_mac_count / size_in_elem
        self.operand_data_reuse = operand_data_reuse

    def get_operand_irrelevant_layer_dims(self, layer_op: LayerOperand) -> list[LayerDim]:
        """! Return the irrelevant dimensions of layer operand 'layer_op'."""
        return self.loop_relevancy_info.get_ir_layer_dims(layer_op)

    def extract_node_attr(self) -> LayerNodeAttributes:
        """Pack this layer node's attributes in a LayerNodeAttributes instance. Useful for instantiating new layer nodes
        (used in Stream)"""
        attributes = LayerNodeAttributes(
            layer_type=self.type,
            equation=self.equation,
            layer_dim_sizes=self.layer_dim_sizes,
            operand_precision=self.operand_precision,
            dimension_relations=self.dimension_relations,
            spatial_mapping=self.spatial_mapping,
            spatial_mapping_hint=self.spatial_mapping_hint,
            core_allocation=self.core_allocation,
            core_allocation_is_fixed=self.core_allocation_is_fixed,
            memory_operand_links=self.memory_operand_links,
            temporal_ordering=self.temporal_ordering,
            padding=self.padding,
            constant_operands=self.constant_operands,
            input_operand_source=self.input_operand_source,
            pr_layer_dim_sizes=self.pr_layer_dim_sizes,
        )
        # Make sure the new attributes don't simply point to the old instances
        return deepcopy(attributes)
