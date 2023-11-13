import logging

import numpy as np

from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.stages.Stage import Stage

logger = logging.getLogger(__name__)


## Pipeline stage that converts the spatial mapping from a
# user-provided spatial mapping across operational array dimensions
# to the internal spatial mapping representation used in the cost model.
class SpatialMappingConversionStage(Stage):
    ## The class constructor
    # Initialize the accelerator and layer attributes.
    def __init__(self, list_of_callables, *, accelerator, layer, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.check_layer(layer)  # raise ValueError in case anything is wrong
        self.layer = layer
        self.accelerator = accelerator

    @staticmethod
    ## Check the layer attribute of the main_inputs:
    #
    # check that the layer includes:
    # - the core which it is allocated to
    # - the user-defined spatial mapping
    #
    # If not, a ValueError is raised.
    # @return: True
    def check_layer(layer):
        if not isinstance(layer.core_allocation, int):
            logger.critical(f"Layer {layer} has no core allocation.")
            raise ValueError(f"Missing core allocation for {layer}.")
        if not layer.user_spatial_mapping:
            logger.critical(f"Layer {layer} has no user-defined spatial mapping.")
            raise ValueError(
                "Missing spatial mapping for {layer}. Please provide 'spatial_mapping' for {layer}."
            )

        return True

    @staticmethod
    def is_nested_tuple(obj):
        if isinstance(obj, tuple):
            for item in obj:
                if isinstance(item, tuple):
                    # If any item within the tuple is itself a tuple, it's a nested tuple
                    return True
        return False

    def run(self):
        user_spatial_mapping = self.layer.user_spatial_mapping
        spatial_mapping, spatial_mapping_int = self.convert_user_spatial_mapping(
            user_spatial_mapping
        )
        # Since the spatial_mapping may be modified in the previous step,
        # we have to update this change to self.layer
        updated_user_spatial_mapping = {}
        for oa_dim, sm_loop in user_spatial_mapping.items():
            if self.is_nested_tuple(sm_loop):  # a mix sm loop
                sm_comb = []
                for sub_sm_loop in sm_loop:
                    sm_layer_dim = sub_sm_loop[0]
                    for sm_element in spatial_mapping.spatial_loop_dim_size:
                        if sm_element[0] == sm_layer_dim:
                            sm_comb.append(sm_element)
                sm_comb = tuple(sm_comb)
                updated_user_spatial_mapping[oa_dim] = sm_comb
            else:
                sm_layer_dim = sm_loop[0]
                for sm_element in spatial_mapping.spatial_loop_dim_size:
                    if sm_element[0] == sm_layer_dim:
                        updated_user_spatial_mapping[oa_dim] = sm_element
        self.layer.user_spatial_mapping = updated_user_spatial_mapping

        kwargs = self.kwargs.copy()
        kwargs["spatial_mapping"] = spatial_mapping
        kwargs["spatial_mapping_int"] = spatial_mapping_int
        kwargs["accelerator"] = self.accelerator
        kwargs["layer"] = self.layer

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    ## Convert the user-defined spatial mapping across operational array dimensions
    # to the internal SpatialMapping representation.
    ""

    # For this conversion we need to know:
    # - the user defined spatial mapping
    # - the core (i.e. operational array) on which the unrolling happens,
    #     and the memory hierarchy that is connected to that operational array.
    # @param user_spatial_mapping: The user-defined spatial mapping to be converted.
    # @returns: A SpatialMapping object with the converted spatial mapping.
    def convert_user_spatial_mapping(self, user_spatial_mapping):
        # Adjust the user defined spatial mapping size based on the operational array dimension and the layer dimension:
        # E.g. user-provided unrolling is 16 but operational array dimension size iso only 12: change unrolling to 12
        # E.g. user-provided unrolling is 16 but layer dimension is only 12: change unrolling to 12
        # E.g. user-provided unrolling is 16 but layer dimension is not a multiple of 16: change unrolling to fractional number
        # so that the temporal remainder is an integer.
        core_id = self.layer.core_allocation
        core = self.accelerator.get_core(core_id)
        mem_hierarchy = core.memory_hierarchy
        oa_dims = core.operational_array.dimensions
        layer_dim_sizes = self.layer.loop_dim_size.copy()
        limited_user_spatial_mapping = {}  # init dict we will be filling
        limited_user_spatial_mapping_int = {}  # init dict int we will be filling
        for oa_dim_name, spatial_loop in user_spatial_mapping.items():
            if self.is_nested_tuple(spatial_loop):  # mix sm loop
                limited_mix_user_spatial_mapping_on_dim = []
                limited_mix_user_spatial_mapping_int_on_dim = []
                for spatial_loop_element in spatial_loop:
                    limited_user_spatial_mapping_to_check = (
                        self.generate_limited_user_spatial_mapping(
                            layer_dim_sizes,
                            oa_dims,
                            oa_dim_name,
                            spatial_loop_element,
                            user_spatial_mapping,
                            limited_user_spatial_mapping,
                        )
                    )
                    limited_user_spatial_mapping_int_to_check = (
                        self.generate_limited_user_spatial_mapping(
                            layer_dim_sizes,
                            oa_dims,
                            oa_dim_name,
                            spatial_loop_element,
                            user_spatial_mapping,
                            limited_user_spatial_mapping,
                            False,
                        )
                    )
                    if limited_user_spatial_mapping_to_check == None:
                        continue  # Same to Check 0: Skip this spatial dimension if it doesn't exist in the layer
                    else:
                        limited_mix_user_spatial_mapping_on_dim.append(
                            limited_user_spatial_mapping_to_check
                        )
                        limited_mix_user_spatial_mapping_int_on_dim.append(
                            limited_user_spatial_mapping_int_to_check
                        )
                if len(limited_mix_user_spatial_mapping_on_dim) == 0:
                    continue  # Skip this spatial dimension if the defined dims in sm don't exist in the layer
                else:
                    limited_mix_user_spatial_mapping_on_dim = tuple(
                        limited_mix_user_spatial_mapping_on_dim
                    )
                    limited_mix_user_spatial_mapping_int_on_dim = tuple(
                        limited_mix_user_spatial_mapping_int_on_dim
                    )
                    limited_user_spatial_mapping[
                        oa_dim_name
                    ] = limited_mix_user_spatial_mapping_on_dim
                    limited_user_spatial_mapping_int[
                        oa_dim_name
                    ] = limited_mix_user_spatial_mapping_int_on_dim
            else:  # single-dim sm loop
                limited_user_spatial_mapping_to_check = (
                    self.generate_limited_user_spatial_mapping(
                        layer_dim_sizes,
                        oa_dims,
                        oa_dim_name,
                        spatial_loop,
                        user_spatial_mapping,
                        limited_user_spatial_mapping,
                    )
                )
                limited_user_spatial_mapping_int_to_check = (
                    self.generate_limited_user_spatial_mapping(
                        layer_dim_sizes,
                        oa_dims,
                        oa_dim_name,
                        spatial_loop,
                        user_spatial_mapping,
                        limited_user_spatial_mapping,
                        False,
                    )
                )
                if limited_user_spatial_mapping_to_check == None:
                    continue  # Skip this spatial dimension if the defined dims in sm don't exist in the layer
                else:
                    limited_user_spatial_mapping[
                        oa_dim_name
                    ] = limited_user_spatial_mapping_to_check
                    limited_user_spatial_mapping_int[
                        oa_dim_name
                    ] = limited_user_spatial_mapping_int_to_check
            # Update the layer_dim_size to support multiple oa dims unrolling the same loop dim but not unrolling it more than the total layer dim
            # if (
            #     temporal_remainder == 1
            # ):  # Remove it from the dict if we have unrolled the entire layer dim onto the array dimension(s)
            #     del layer_dim_sizes[loop_dim_unrolled]
            # else:  # Update the dict if we have some layer dims left to potentially unroll onto the next oa dims
            #     layer_dim_sizes[loop_dim_unrolled] = temporal_remainder

        user_spatial_mapping_for_log = {
            array_dim: loop_comb
            for (
                array_dim,
                loop_comb,
            ) in limited_user_spatial_mapping.items()
        }
        logger.debug(
            f"User-provided spatial mapping converted to: {user_spatial_mapping_for_log}"
        )

        spatial_mapping_dict = self.generate_spatial_mapping_dict(
            user_spatial_mapping=limited_user_spatial_mapping,
            layer=self.layer,
            accelerator=self.accelerator,
        )
        # The next spatial_mapping_dict is used in cost model to calculate the interval between different data transfer.
        # Different with the one above, there are only integer numbers (corresponding to the real cases)
        spatial_mapping_dict_int = self.generate_spatial_mapping_dict(
            user_spatial_mapping=limited_user_spatial_mapping_int,
            layer=self.layer,
            accelerator=self.accelerator,
        )

        return SpatialMapping(
            spatial_mapping_dict=spatial_mapping_dict, layer_node=self.layer
        ), SpatialMapping(
            spatial_mapping_dict=spatial_mapping_dict_int, layer_node=self.layer
        )

    def generate_limited_user_spatial_mapping(
        self,
        layer_dim_sizes,
        oa_dims,
        oa_dim_name,
        spatial_loop,
        user_spatial_mapping,
        limited_user_spatial_mapping,
        allow_decimal_sm_loop_size=True,
    ):
        ## Do check on spatial mapping, and convert the mapping to a tuple
        (loop_dim_unrolled, loop_size_unrolled) = spatial_loop
        # Check 0: Skip this spatial dimension if it doesn't exist in the layer
        if loop_dim_unrolled not in layer_dim_sizes.keys():
            return None
        # Check 1: Limit unrolling if operational array dimension is smaller than provided unrolling
        oa_dim_size = next(
            (oa_dim for oa_dim in oa_dims if oa_dim.name == oa_dim_name)
        ).size
        loop_size_unrolled = min(oa_dim_size, loop_size_unrolled)
        # Check 2: Limit unrolling if layer dimension is smaller than provided unrolling or if the loop dim doesn't exist
        layer_dim_size = layer_dim_sizes.get(loop_dim_unrolled, 1)
        loop_size_unrolled = min(layer_dim_size, loop_size_unrolled)
        # Check 3: Adjust unrolling if it is not a multiple of the layer dimension size
        # and if there is no more mapping for this layer dimension
        no_more_mapping_for_current_layer_dim = (
            self.check_if_there_is_further_oa_mapping_for_current_layer_dim(
                oa_dim_name=oa_dim_name,
                loop_dim_unrolled=loop_dim_unrolled,
                user_spatial_mapping=user_spatial_mapping,
            )
        )
        if no_more_mapping_for_current_layer_dim:
            loop_size_unrolled_on_early_oa_dims = (
                self.calc_unrolled_loop_size_on_early_oa_dims(
                    oa_dim_name=oa_dim_name,
                    loop_dim_unrolled=loop_dim_unrolled,
                    user_spatial_mapping=limited_user_spatial_mapping,
                )
            )
            temporal_remainder = int(
                np.ceil(
                    layer_dim_size
                    / (loop_size_unrolled * loop_size_unrolled_on_early_oa_dims)
                )
            )
            if allow_decimal_sm_loop_size:
                loop_size_unrolled = (
                    layer_dim_size
                    / temporal_remainder
                    / loop_size_unrolled_on_early_oa_dims
                )
            else:
                loop_size_unrolled = int(
                    np.ceil(
                        layer_dim_size
                        / temporal_remainder
                        / loop_size_unrolled_on_early_oa_dims
                    )
                )
        return (
            loop_dim_unrolled,
            loop_size_unrolled,
        )

    def generate_spatial_mapping_dict(self, user_spatial_mapping, layer, accelerator):
        # This function is to convert spatial mapping to spatial_mapping_dict,
        # which attaches spatial mapping to different memory levels.
        spatial_mapping_dict = {}
        layer_to_mem_op = layer.memory_operand_links
        mem_to_layer_op = {
            mem_op: layer_op for (layer_op, mem_op) in layer_to_mem_op.items()
        }
        core_id = layer.core_allocation
        mem_hierarchy = accelerator.get_core(core_id).memory_hierarchy
        for mem_op, layer_op in mem_to_layer_op.items():
            user_sm_copy = user_spatial_mapping.copy()
            # layer_op = mem_to_layer_op[mem_op]
            spatial_mapping_dict[layer_op] = []
            memory_levels = mem_hierarchy.get_memory_levels(
                mem_op,
            )

            for memory_level in memory_levels:
                spatial_mapping_lvl = []
                spatial_mapping_lvl_dict = {}
                served_dimensions = memory_level.served_dimensions
                for dimension in served_dimensions:
                    dim_name = dimension.name
                    if dim_name in user_sm_copy:
                        # The dimension name is present in the user defined spatial mapping
                        # Add the spatial loop of this dimension to the spatial mapping
                        spatial_loop = user_sm_copy[dim_name]
                        if self.is_nested_tuple(spatial_loop):  # mix sm loop
                            # Reformat the spatial_loop to the original format
                            for sub_spatial_loop in spatial_loop:
                                (
                                    spatial_mapping_dim,
                                    spatial_mapping_size,
                                ) = sub_spatial_loop
                                if (
                                    spatial_mapping_dim
                                    in spatial_mapping_lvl_dict.keys()
                                ):
                                    spatial_mapping_lvl_dict[
                                        spatial_mapping_dim
                                    ] *= spatial_mapping_size
                                else:
                                    spatial_mapping_lvl_dict[
                                        spatial_mapping_dim
                                    ] = spatial_mapping_size
                        else:  # single-dim sm loop
                            (spatial_mapping_dim, spatial_mapping_size) = spatial_loop
                            if spatial_mapping_dim in spatial_mapping_lvl_dict.keys():
                                spatial_mapping_lvl_dict[
                                    spatial_mapping_dim
                                ] *= spatial_mapping_size
                            else:
                                spatial_mapping_lvl_dict[
                                    spatial_mapping_dim
                                ] = spatial_mapping_size
                        # Then remove this dim_name and spatial loop key value pair from the dict
                        # as the spatial mapping representation is a level-by-level one.
                        del user_sm_copy[dim_name]
                for (
                    spatial_mapping_lvl_dict_dim,
                    spatial_mapping_lvl_dict_size,
                ) in spatial_mapping_lvl_dict.items():
                    spatial_mapping_lvl.append(
                        (spatial_mapping_lvl_dict_dim, spatial_mapping_lvl_dict_size)
                    )
                spatial_mapping_dict[layer_op].append(spatial_mapping_lvl)

            # After we have gone through the memory levels, if there are still user-defined dimensions
            # present, add them as the top level. Otherwise add an empty list to make arch levels correct:
            # because first list we added was the operational array level.

            # We will merge together if the top memory level is serving multiple oa dims
            # and there are layer dims existing on multiple oa dims.
            top_level_spatial_mapping_dict = {}
            for (dim_name, spatial_loop) in user_sm_copy.items():
                if self.is_nested_tuple(spatial_loop):  # mix sm loop
                    for sub_spatial_loop in spatial_loop:
                        spatial_loop_dim = sub_spatial_loop[0]
                        spatial_loop_size = sub_spatial_loop[1]
                        if spatial_loop_dim not in top_level_spatial_mapping_dict.keys():
                            top_level_spatial_mapping_dict[spatial_loop_dim] = spatial_loop_size
                        else:
                            top_level_spatial_mapping_dict[spatial_loop_dim] *= spatial_loop_size
                else:
                    spatial_loop_dim = spatial_loop[0]
                    spatial_loop_size = spatial_loop[1]
                    if spatial_loop_dim not in top_level_spatial_mapping_dict.keys():
                        top_level_spatial_mapping_dict[spatial_loop_dim] = spatial_loop_size
                    else:
                        top_level_spatial_mapping_dict[spatial_loop_dim] *= spatial_loop_size
            top_level_spatial_mapping = [
                (layer_dim, layer_size) for (layer_dim, layer_size) in top_level_spatial_mapping_dict.items()
            ]
            spatial_mapping_dict[layer_op].append(top_level_spatial_mapping)
        return spatial_mapping_dict

    def check_if_there_is_further_oa_mapping_for_current_layer_dim(
        self, oa_dim_name, loop_dim_unrolled, user_spatial_mapping
    ):
        # For the case when there is layer dimension that is mapped on multiple oa dimensions.
        # We need to decide on which oa dimension to adjust the unrolling
        # if the total unrolling size is not a multiple of the layer dimension size.
        # In this case, we decide to only adjust the unrolling size on the last oa dimension,
        # This function is to check if the current oa dimension is the last oa dimension for the current layer dim.
        start_check_on_layer_dim_mapping = False
        no_more_mapping_for_current_layer_dim = True
        for oa_dim_name_private, spatial_loop_private in user_spatial_mapping.items():
            if oa_dim_name == oa_dim_name_private:
                start_check_on_layer_dim_mapping = True
                continue
            if start_check_on_layer_dim_mapping:
                if self.is_nested_tuple(spatial_loop_private):  # mix sm loop
                    for spatial_loop_element in spatial_loop_private:
                        loop_dim_unrolled_private = spatial_loop_element[0]
                        if loop_dim_unrolled == loop_dim_unrolled_private:
                            no_more_mapping_for_current_layer_dim = False
                            break
                else:
                    loop_dim_unrolled_private = spatial_loop_private[0]
                    if loop_dim_unrolled == loop_dim_unrolled_private:
                        no_more_mapping_for_current_layer_dim = False
            if (
                not no_more_mapping_for_current_layer_dim
            ):  # early exit if the flag is already False
                break
        return no_more_mapping_for_current_layer_dim

    def calc_unrolled_loop_size_on_early_oa_dims(
        self, oa_dim_name, loop_dim_unrolled, user_spatial_mapping
    ):
        # calculate the unrolled loop size for the specific layer dim on oa dims earlier than current oa dim
        loop_unrolled_size_already = 1
        for oa_dim_name_private, spatial_loop_private in user_spatial_mapping.items():
            if oa_dim_name == oa_dim_name_private:
                break
            if self.is_nested_tuple(spatial_loop_private):  # mix sm loop
                for spatial_loop_element in spatial_loop_private:
                    (
                        loop_dim_unrolled_private,
                        loop_size_unrolled_private,
                    ) = spatial_loop_element
                    if loop_dim_unrolled == loop_dim_unrolled_private:
                        loop_unrolled_size_already *= loop_size_unrolled_private
            else:
                (
                    loop_dim_unrolled_private,
                    loop_size_unrolled_private,
                ) = spatial_loop_private
                if loop_dim_unrolled == loop_dim_unrolled_private:
                    loop_unrolled_size_already *= loop_size_unrolled_private
        return loop_unrolled_size_already
