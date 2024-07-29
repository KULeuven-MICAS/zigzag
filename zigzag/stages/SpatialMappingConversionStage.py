import logging
import math
from typing import Any

from zigzag.datatypes import OADimension
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.mapping.spatial_mapping import (
    LayerDim,
    MappingSingleOADim,
    SpatialMapping,
    UnrollFactor,
)
from zigzag.mapping.SpatialMappingInternal import (
    SpatialMappingInternal,
    SpatialMappingPerMemLvl,
)
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.layer_attributes import LayerDimSizes
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class SpatialMappingConversionStage(Stage):
    """! Pipeline stage that converts the spatial mapping from a
    user-provided spatial mapping across operational array dimensions
    to the internal spatial mapping representation used in the cost model.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        **kwargs: Any,
    ):
        """
        Initialize the accelerator and layer attributes.
        """
        super().__init__(list_of_callables, **kwargs)
        self.layer = layer
        self.accelerator = accelerator
        self.memory_operand_links = layer.memory_operand_links
        self.user_spatial_mapping = self.layer.spatial_mapping

        assert (
            self.user_spatial_mapping.oa_dim_sizes is not None
        ), "SpatialMapping's OA Dimension sizes have not been initialized"
        self.oa_dim_sizes: dict[OADimension, int] = self.user_spatial_mapping.oa_dim_sizes

    def run(self):
        spatial_mapping, spatial_mapping_int = self.convert_user_spatial_mapping(self.user_spatial_mapping)

        kwargs = self.kwargs.copy()
        kwargs["spatial_mapping"] = spatial_mapping
        kwargs["spatial_mapping_int"] = spatial_mapping_int
        kwargs["accelerator"] = self.accelerator
        kwargs["layer"] = self.layer

        sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def convert_user_spatial_mapping(
        self, user_spatial_mapping: SpatialMapping
    ) -> tuple[SpatialMappingInternal, SpatialMappingInternal]:
        """! Convert the SpatialMapping instance in `user-defined` format (spatial mapping across operational array
        dimensions) to the  SpatialMappingInternal representation. For this conversion we need to know:
        - the user defined spatial mapping
        - the core (i.e. operational array) on which the unrolling happens,and the memory hierarchy that is connected
          to that operational array.
        @param user_spatial_mapping: The SpatialMapping in `user-defined` format to be converted.
        @returns: A SpatialMappingInternal object with the converted spatial mapping.
        """

        # Adjust the user defined spatial mapping size based on the operational array dimension and the layer dimension:
        # E.g. user-provided unrolling is 16 but operational array dimension size iso only 12: change unrolling to 12
        # E.g. user-provided unrolling is 16 but layer dimension is only 12: change unrolling to 12
        # E.g. user-provided unrolling is 16 but layer dimension is not a multiple of 16: change unrolling to fractional
        # number so that the temporal remainder is an integer.

        layer_dim_sizes = self.layer.layer_dim_sizes
        limited_usm: SpatialMapping = SpatialMapping.empty()
        limited_usm_int: SpatialMapping = SpatialMapping.empty()

        for oa_dim, mapping_this_oa_dim in user_spatial_mapping.items():
            limited_mix_usm_this_dim: list[tuple[LayerDim, int | float]] = []
            limited_mix_usm_this_dim_int: list[tuple[LayerDim, int | float]] = []
            for spatial_loop_element in mapping_this_oa_dim.items():
                limited_user_spatial_mapping_to_check = self.generate_limited_user_spatial_mapping(
                    layer_dim_sizes,
                    oa_dim,
                    spatial_loop_element,
                    user_spatial_mapping,
                    limited_usm,
                )
                limited_user_spatial_mapping_int_to_check = self.generate_limited_user_spatial_mapping(
                    layer_dim_sizes,
                    oa_dim,
                    spatial_loop_element,
                    user_spatial_mapping,
                    limited_usm_int,
                    False,
                )
                if (
                    limited_user_spatial_mapping_to_check is not None
                    and limited_user_spatial_mapping_int_to_check is not None
                ):
                    limited_mix_usm_this_dim.append(limited_user_spatial_mapping_to_check)
                    limited_mix_usm_this_dim_int.append(limited_user_spatial_mapping_int_to_check)

            # Skip this spatial dimension if the defined dims in sm don't exist in the layer
            if len(limited_mix_usm_this_dim) != 0:
                limited_usm[oa_dim] = MappingSingleOADim({})
                for layer_dim, unroll_factor in limited_mix_usm_this_dim:
                    limited_usm[oa_dim][layer_dim] = unroll_factor

            if len(limited_mix_usm_this_dim_int) != 0:
                limited_usm_int[oa_dim] = MappingSingleOADim({})
                for layer_dim, unroll_factor in limited_mix_usm_this_dim_int:
                    limited_usm_int[oa_dim][layer_dim] = unroll_factor

        logger.debug("User-provided spatial mapping converted to: %s", limited_usm)

        spatial_mapping_dict = self.generate_mapping_per_mem_lvl(user_spatial_mapping=limited_usm)
        # The next spatial_mapping_dict is used in cost model to calculate the interval between different data transfer.
        # Different with the one above, there are only integer numbers (corresponding to the real cases)
        spatial_mapping_dict_int = self.generate_mapping_per_mem_lvl(user_spatial_mapping=limited_usm_int)

        return SpatialMappingInternal(spatial_mapping_dict, self.layer), SpatialMappingInternal(
            spatial_mapping_dict_int, self.layer
        )

    def generate_limited_user_spatial_mapping(
        self,
        layer_dim_sizes: LayerDimSizes,
        oa_dim: OADimension,
        spatial_loop: tuple[LayerDim, UnrollFactor],
        user_spatial_mapping: SpatialMapping,
        limited_user_spatial_mapping: SpatialMapping,
        allow_decimal_sm_loop_size: bool = True,
    ) -> None | tuple[LayerDim, UnrollFactor]:
        layer_dim, unroll_factor = spatial_loop

        # Check 0: Skip this spatial dimension if it doesn't exist in the layer
        if layer_dim not in layer_dim_sizes:
            return None

        # Check 1: Limit unrolling if operational array dimension is smaller than provided unrolling
        oa_dim_size = self.oa_dim_sizes[oa_dim]
        if unroll_factor > oa_dim_size:
            logger.warning(
                "Provided unrolling (%s:%i) exceeds Operational Array size (%s:%i). Changing to (%s:%i).",
                layer_dim,
                unroll_factor,
                oa_dim,
                oa_dim_size,
                layer_dim,
                oa_dim_size,
            )
            unroll_factor = oa_dim_size

        # Check 2: Limit unrolling if layer dimension is smaller than provided unrolling or if the loop dim doesn't
        # exist (should be the case)
        layer_dim_size = layer_dim_sizes[layer_dim]
        assert unroll_factor <= layer_dim_size

        # Check 3: Adjust unrolling if it is not a divisor of the layer dimension size
        # and if there is no more mapping for this layer dimension
        no_more_mapping_for_current_layer_dim = self.check_if_oa_dim_mapping_is_first_max(
            oa_dim=oa_dim,
            loop_dim_unrolled=layer_dim,
            user_spatial_mapping=user_spatial_mapping,
        )
        if no_more_mapping_for_current_layer_dim:
            loop_size_unrolled_on_early_oa_dims = self.calc_unrolled_loop_size_on_early_oa_dims(
                oa_dim=oa_dim,
                loop_dim_unrolled=layer_dim,
                user_spatial_mapping=limited_user_spatial_mapping,
            )
            temporal_remainder = int(math.ceil(layer_dim_size / (unroll_factor * loop_size_unrolled_on_early_oa_dims)))
            unroll_factor_remainder = layer_dim_size / temporal_remainder / loop_size_unrolled_on_early_oa_dims
            unroll_factor_new: int | float = (
                unroll_factor_remainder if allow_decimal_sm_loop_size else int(unroll_factor_remainder)
            )

            return layer_dim, unroll_factor_new
        else:
            return layer_dim, unroll_factor

    def generate_mapping_per_mem_lvl(self, user_spatial_mapping: SpatialMapping) -> SpatialMappingPerMemLvl:
        """! This function is to convert spatial mapping to mapping_per_mem_lvl,
        which attaches spatial mapping to different memory levels.
        # TODO This should be a class
        """
        mapping_per_mem_lvl: SpatialMappingPerMemLvl = {}
        core_id = self.layer.core_allocation[0]
        mem_hierarchy = self.accelerator.get_core(core_id).memory_hierarchy
        for layer_op in self.memory_operand_links.layer_operands:
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            usm_copy = user_spatial_mapping.copy()
            mapping_per_mem_lvl[layer_op] = []
            memory_levels = mem_hierarchy.get_memory_levels(mem_op)

            for memory_level in memory_levels:
                spatial_mapping_lvl: list[tuple[LayerDim, UnrollFactor]] = []
                spatial_mapping_lvl_dict: dict[LayerDim, UnrollFactor] = {}
                served_dimensions = memory_level.served_dimensions
                for oa_dim in served_dimensions:
                    if oa_dim in usm_copy:
                        # The dimension name is present in the user defined spatial mapping
                        # Add the spatial loop of this dimension to the spatial mapping
                        spatial_loop = usm_copy[oa_dim]
                        for layer_dim, unrolling in spatial_loop.items():
                            if layer_dim in spatial_mapping_lvl_dict:
                                spatial_mapping_lvl_dict[layer_dim] *= unrolling
                            else:
                                spatial_mapping_lvl_dict[layer_dim] = unrolling

                        # Then remove this dim_name and spatial loop key value pair from the dict
                        # as the spatial mapping representation is a level-by-level one.
                        del usm_copy.data[oa_dim]
                for combination in spatial_mapping_lvl_dict.items():
                    spatial_mapping_lvl.append(combination)
                mapping_per_mem_lvl[layer_op].append(spatial_mapping_lvl)

            # After we have gone through the memory levels, if there are still user-defined dimensions
            # present, add them as the top level. Otherwise add an empty list to make arch levels correct:
            # because first list we added was the operational array level.

            # We will merge together if the top memory level is serving multiple oa dims
            # and there are layer dims existing on multiple oa dims.
            top_level_mapping_per_mem_lvl: dict[LayerDim, UnrollFactor | float] = {}

            for oa_dim, mapping_single_oa_dim in usm_copy.items():
                for layer_dim, unrolling in mapping_single_oa_dim.items():
                    if layer_dim not in top_level_mapping_per_mem_lvl:
                        top_level_mapping_per_mem_lvl[layer_dim] = unrolling
                    else:
                        top_level_mapping_per_mem_lvl[layer_dim] *= unrolling

            top_level_spatial_mapping: list[tuple[LayerDim, UnrollFactor | float]] = [
                combination for combination in top_level_mapping_per_mem_lvl.items()
            ]
            mapping_per_mem_lvl[layer_op].append(top_level_spatial_mapping)
        return mapping_per_mem_lvl

    def check_if_oa_dim_mapping_is_first_max(
        self,
        oa_dim: OADimension,
        loop_dim_unrolled: LayerDim,
        user_spatial_mapping: SpatialMapping,
    ):
        """! For the case when there is layer dimension that is mapped on multiple oa dimensions.
        We need to decide on which oa dimension to adjust the unrolling
        if the total unrolling size is not a multiple of the layer dimension size.
        In this case, we decide to only adjust the unrolling size of the first oa dimension with the largest unrolling.
        This function is to check if the given oa_dim has the largest unrolling for the given loop_dim_unrolled.
        """

        oa_dim_mapping_sizes: list[UnrollFactor] = []
        for mapping in user_spatial_mapping.values():
            layer_dim_mapping_size = mapping[loop_dim_unrolled] if loop_dim_unrolled in mapping.layer_dims else 0
            oa_dim_mapping_sizes.append(layer_dim_mapping_size)
        max_mapping_size = max(oa_dim_mapping_sizes)
        assert max_mapping_size > 0, f"Given {oa_dim=} is not present in {user_spatial_mapping=}"
        first_oa_dim_with_max_mapping = next(
            curr_oa_dim
            for curr_oa_dim, mapping in user_spatial_mapping.items()
            if loop_dim_unrolled in mapping.layer_dims and mapping[loop_dim_unrolled] == max_mapping_size
        )
        should_be_limited = oa_dim == first_oa_dim_with_max_mapping
        return should_be_limited

    def calc_unrolled_loop_size_on_early_oa_dims(
        self,
        oa_dim: OADimension,
        loop_dim_unrolled: LayerDim,
        user_spatial_mapping: SpatialMapping,
    ) -> UnrollFactor:
        # calculate the unrolled loop size for the specific layer dim on oa dims earlier than current oa dim
        loop_unrolled_size_already = 1
        for curr_oa_dim, mapping_this_oa_dim in user_spatial_mapping.items():
            if oa_dim == curr_oa_dim:
                break

            for layer_dim, unroll_factor in mapping_this_oa_dim.items():
                if loop_dim_unrolled == layer_dim:
                    loop_unrolled_size_already *= unroll_factor

        return loop_unrolled_size_already
