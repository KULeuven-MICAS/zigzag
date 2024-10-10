import logging
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
from zigzag.utils import UniqueMessageFilter
from zigzag.workload.layer_attributes import (
    LayerTemporalOrdering,
    MemoryOperandLinks,
)

logger = logging.getLogger(__name__)
logger.addFilter(UniqueMessageFilter())


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
