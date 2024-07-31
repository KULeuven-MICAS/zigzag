import logging
import re
from typing import Any

from zigzag.datatypes import (
    LayerDim,
    LayerOperand,
    MemoryOperand,
    OADimension,
    UnrollFactor,
)
from zigzag.mapping.spatial_mapping import (
    MappingSingleOADim,
    SpatialMapping,
    SpatialMappingHint,
)
from zigzag.parser.WorkloadValidator import WorkloadValidator
from zigzag.utils import UniqueMessageFilter
from zigzag.workload.DNNWorkload import DNNWorkload
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
from zigzag.workload.layer_node import LayerNode, LayerNodeAttributes

logger = logging.getLogger(__name__)
logger.addFilter(UniqueMessageFilter())


class WorkloadFactory:
    """! Generates a `Workload` instance from the validated and normalized user-provided data."""

    def __init__(self, workload_data: list[dict[str, Any]], mapping_data: list[dict[str, Any]]):
        self.workload_data = workload_data
        self.mapping_data = mapping_data

    def create(self) -> DNNWorkload:
        node_list: list[LayerNode] = []

        for layer_data in self.workload_data:
            layer_node_factory = LayerNodeFactory(layer_data, self.mapping_data)
            layer_node = layer_node_factory.create()
            node_list.append(layer_node)

        return DNNWorkload(node_list)


class LayerNodeFactory:
    """Creates a LayerNode instance from a validated and normalized user definition of a single workload layer"""

    def __init__(self, node_data: dict[str, Any], mapping_data: list[dict[str, Any]]):
        """!
        @node_data validated and normalized user-defined data for a single workload layer
        @mapping_data validated and normalized user-defined data for all mappings
        """
        self.node_data = node_data
        self.mapping_data = mapping_data
        self.layer_id: int = self.node_data["id"]
        self.node_name: str = self.node_data["name"] if self.node_data["name"] is not None else f"Layer{self.layer_id}"

    def create(self) -> LayerNode:
        node_attr = self.create_node_attr()
        return LayerNode(layer_id=self.layer_id, node_name=self.node_name, node_attr=node_attr)

    def create_node_attr(self) -> LayerNodeAttributes:
        # From node data
        layer_type: str = self.node_data["operator_type"]
        equation = self.create_equation()
        layer_dim_sizes = self.create_layer_dim_sizes()
        operand_precision = self.create_operand_precision()
        dimension_relations = self.create_layer_dim_relations()
        constant_operands = self.create_constant_operands()
        input_operand_source = self.create_operand_source()
        padding = self.create_padding()
        pr_layer_dim_sizes = self.create_pr_layer_dim_sizes()

        # From mapping data
        mapping_factory = MappingFactory(self.node_name, layer_type, self.mapping_data)
        spatial_mapping = mapping_factory.create_spatial_mapping()
        spatial_mapping_hint = mapping_factory.create_spatial_mapping_hint()
        core_allocation = mapping_factory.get_core_allocation()
        core_allocation_is_fixed = mapping_factory.get_core_allocation_is_fixed()
        memory_operand_links = mapping_factory.create_memory_operand_links()
        temporal_ordering = mapping_factory.create_temporal_ordering()
        temporal_ordering.remove_invalid_layer_dims(layer_dim_sizes, self.node_name)

        return LayerNodeAttributes(
            layer_type=layer_type,
            equation=equation,
            layer_dim_sizes=layer_dim_sizes,
            operand_precision=operand_precision,
            dimension_relations=dimension_relations,
            constant_operands=constant_operands,
            input_operand_source=input_operand_source,
            spatial_mapping=spatial_mapping,
            spatial_mapping_hint=spatial_mapping_hint,
            core_allocation=core_allocation,
            core_allocation_is_fixed=core_allocation_is_fixed,
            memory_operand_links=memory_operand_links,
            temporal_ordering=temporal_ordering,
            padding=padding,
            pr_layer_dim_sizes=pr_layer_dim_sizes,
        )

    def create_equation(self) -> LayerEquation:
        equation: str = self.node_data["equation"]
        equation = equation.replace("+=", "=")
        equation = equation.replace("++", "+")
        equation = equation.replace("*", " * ")
        equation = equation.replace("=", " = ")
        equation = equation.replace("+", " + ")
        return LayerEquation(equation)

    def create_layer_dim_sizes(self) -> LayerDimSizes:
        loop_dims = [LayerDim(x) for x in self.node_data["loop_dims"]]
        loop_sizes: list[UnrollFactor] = self.node_data["loop_sizes"]

        data = {dim: size for dim, size in zip(loop_dims, loop_sizes)}
        return LayerDimSizes(data)

    def create_operand_precision(self) -> LayerOperandPrecision:
        precisions: dict[str, int] = self.node_data["operand_precision"]
        data: dict[LayerOperand, int] = {LayerOperand(operand_str): size for operand_str, size in precisions.items()}
        return LayerOperandPrecision(data)

    def create_layer_dim_relations(self) -> list[LayerDimRelation]:
        relations: list[LayerDimRelation] = []
        for relation_str in self.node_data["dimension_relations"]:
            match = re.search(WorkloadValidator.LAYER_DIM_RELATION_REGEX, relation_str)
            assert match is not None
            dim_1, coef_2, dim_2, coef_3, dim_3 = match.groups()
            layer_dim_relation = LayerDimRelation(
                dim_1=LayerDim(dim_1),
                dim_2=LayerDim(dim_2),
                dim_3=LayerDim(dim_3),
                coef_2=int(coef_2) if coef_2 is not None else 1,
                coef_3=int(coef_3) if coef_3 is not None else 1,
            )
            relations.append(layer_dim_relation)

        return relations

    def create_constant_operands(self) -> list[LayerOperand]:
        operand_sources: dict[str, int] = self.node_data["operand_source"]
        constant_operands: list[str] = [op for op, source in operand_sources.items() if source == self.node_data["id"]]
        return [LayerOperand(layer_op_str) for layer_op_str in constant_operands]

    def create_operand_source(self) -> InputOperandSource:
        operand_sources: dict[str, int] = self.node_data["operand_source"]
        return {
            LayerOperand(layer_dim_str): source
            for layer_dim_str, source in operand_sources.items()
            if source != self.node_data["id"]
        }

    def create_padding(self) -> LayerPadding:
        if "pr_loop_dims" not in self.node_data or self.node_data["pr_loop_dims"] is None:
            return LayerPadding.empty()
        if "padding" not in self.node_data or self.node_data["padding"] is None:
            return LayerPadding.empty()

        pr_layer_dims: list[LayerDim] = [LayerDim(x) for x in self.node_data["pr_loop_dims"]]
        # length of the inner list equals 2
        padding_data: list[list[int]] = self.node_data["padding"]
        padding_dict: dict[LayerDim, tuple[int, int]] = {
            layer_dim: (padding_data[i][0], padding_data[i][1]) for i, layer_dim in enumerate(pr_layer_dims)
        }
        return LayerPadding(padding_dict)

    def create_pr_layer_dim_sizes(self) -> LayerDimSizes | None:
        if "pr_loop_dims" not in self.node_data or self.node_data["pr_loop_dims"] is None:
            return None
        if "pr_loop_sizes" not in self.node_data or self.node_data["pr_loop_sizes"] is None:
            return None

        pr_layer_dims: list[LayerDim] = [LayerDim(x) for x in self.node_data["pr_loop_dims"]]
        pr_sizes: list[UnrollFactor] = self.node_data["pr_loop_sizes"]
        size_dict = {layer_dim: size for layer_dim, size in zip(pr_layer_dims, pr_sizes)}
        return LayerDimSizes(size_dict)


class MappingFactory:
    """Converts validated and normalized user-provided data into mapping-related instances.
    The mapping for this layer is chosen according to the following priority:
    1. The name of the layer
    2. The operation type of the layer (if the layer name is not defined in the mapping)
    3. The default mapping (if the operation type is not defined in the mapping)
    """

    def __init__(self, layer_name: str, operation_type: str, mapping_data: list[dict[str, Any]]):
        """
        @param Name of the layer for which the Mapping is being constructed.
        @param operation_type Name of the layer operation for which the Mapping is being constructed.
        @param mapping_data user-given, validated and normalized mapping data for all operation types.
        """
        if layer_name in map(lambda x: x["name"], mapping_data):
            self.mapping_data: dict[str, Any] = next(filter(lambda x: x["name"] == layer_name, mapping_data))
        elif operation_type in map(lambda x: x["name"], mapping_data):
            self.mapping_data: dict[str, Any] = next(filter(lambda x: x["name"] == operation_type, mapping_data))
        else:
            self.mapping_data = next(filter(lambda x: x["name"] == "default", mapping_data))
            logger.warning(
                "Operator %s not defined in mapping. Using default mapping instead.",
                operation_type,
            )

    def get_core_allocation(self) -> list[int]:
        return self.mapping_data["core_allocation"]

    def get_core_allocation_is_fixed(self) -> bool:
        return self.mapping_data["core_allocation_is_fixed"]

    def create_spatial_mapping(self) -> SpatialMapping:
        if self.mapping_data["spatial_mapping"] is None:
            return SpatialMapping.empty()

        user_data: dict[str, list[str]] = self.mapping_data["spatial_mapping"]
        spatial_mapping_dict: dict[OADimension, MappingSingleOADim] = {}

        for oa_dim_str, unrolling_list in user_data.items():
            oa_dim = OADimension(oa_dim_str)
            mapping_this_oa_dim = self.create_mapping_single_oa_dim(unrolling_list)
            spatial_mapping_dict[oa_dim] = mapping_this_oa_dim

        return SpatialMapping(spatial_mapping_dict)

    def create_mapping_single_oa_dim(self, mapping_data: list[str]) -> MappingSingleOADim:
        mapping_dict: dict[LayerDim, UnrollFactor] = {}

        for single_unrolling in mapping_data:
            layer_dim_str = single_unrolling.split(",")[0]
            unrolling = int(single_unrolling.split(",")[-1])
            layer_dim = LayerDim(layer_dim_str)
            mapping_dict[layer_dim] = unrolling

        return MappingSingleOADim(mapping_dict)

    def create_spatial_mapping_hint(self) -> SpatialMappingHint:
        if "spatial_mapping_hint" not in self.mapping_data or self.mapping_data["spatial_mapping_hint"] is None:
            return SpatialMappingHint.empty()

        user_data: dict[str, list[str]] = self.mapping_data["spatial_mapping_hint"]
        mapping_hint_dict: dict[OADimension, set[LayerDim]] = {
            OADimension(oa_dim_str): {LayerDim(layer_dim_str) for layer_dim_str in hint_list}
            for oa_dim_str, hint_list in user_data.items()
        }
        return SpatialMappingHint(mapping_hint_dict)

    def create_memory_operand_links(self) -> MemoryOperandLinks:
        user_data: dict[str, str] = self.mapping_data["memory_operand_links"]
        links_dict = {
            LayerOperand(layer_op_str): MemoryOperand(mem_op_str) for layer_op_str, mem_op_str in user_data.items()
        }
        return MemoryOperandLinks(links_dict)

    def create_temporal_ordering(self) -> LayerTemporalOrdering:
        """! This attribute lacks support within the MappingValidator. Returns an empty instance in case it is not
        provided (to be compatible with older code) or raises an error if it is present in the user-provided data.
        """
        if "temporal_ordering" not in self.mapping_data or not self.mapping_data["temporal_ordering"]:
            return LayerTemporalOrdering.empty()

        return LayerTemporalOrdering(self.mapping_data["temporal_ordering"])
