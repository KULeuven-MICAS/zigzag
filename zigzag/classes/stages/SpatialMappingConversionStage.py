import logging

import numpy as np

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class SpatialMappingConversionStage(Stage):
    """
    Pipeline stage that converts the spatial mapping from a
    user-provided spatial mapping across operational array dimensions
    to the internal spatial mapping representation used in the cost model.
    """

    def __init__(self, list_of_callables, *, accelerator, layer, **kwargs):
        """
        Initialize the accelerator and layer attributes.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.check_layer(layer)  # raise ValueError in case anything is wrong
        self.layer = layer
        self.accelerator = accelerator

    @staticmethod
    def check_layer(layer):
        """
        Check the layer attribute of the main_inputs:
        check that the layer includes:
        - the core which it is allocated to
        - the user-defined spatial mapping
        If not, a ValueError is raised.
        :return: True
        """
        if not isinstance(layer.core_allocation, int):
            logger.critical(f"Layer {layer} has no core allocation.")
            raise ValueError(f"Missing core allocation for {layer}.")
        if not layer.user_spatial_mapping:
            logger.critical(f"Layer {layer} has no user-defined spatial mapping.")
            raise ValueError("Missing spatial mapping for {layer}. Please provide 'spatial_mapping' for {layer}.")

        return True

    def run(self):

        user_spatial_mapping = self.layer.user_spatial_mapping
        spatial_mapping = self.convert_user_spatial_mapping(user_spatial_mapping)
        kwargs = self.kwargs.copy()
        kwargs['spatial_mapping'] = spatial_mapping
        kwargs['accelerator'] = self.accelerator
        kwargs['layer'] = self.layer

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def convert_user_spatial_mapping(self, user_spatial_mapping):
        """
        Convert the user-defined spatial mapping across operational array dimensions
        to the internal SpatialMapping representation.
        For this conversion we need to know:
        - the user defined spatial mapping
        - the core (i.e. operational array) on which the unrolling happens,
          and the memory hierarchy that is connected to that operational array.
        :param user_spatial_mapping: The user-defined spatial mapping to be converted.
        Returns: A SpatialMapping object with the converted spatial mapping.
        """
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
        for oa_dim_name, spatial_loop in user_spatial_mapping.items():
            (loop_dim_unrolled, loop_size_unrolled) = spatial_loop
            # Check 0: Skip this spatial dimension if it doesn't exist in the layer
            if loop_dim_unrolled not in layer_dim_sizes.keys():
                continue
            # Check 1: Limit unrolling if operational array dimension is smaller than provided unrolling
            oa_dim_size = next((oa_dim for oa_dim in oa_dims if oa_dim.name == oa_dim_name)).size
            loop_size_unrolled = min(oa_dim_size, loop_size_unrolled)
            # Check 2: Limit unrolling if layer dimension is smaller than provided unrolling or if the loop dim doesn't exist
            layer_dim_size = layer_dim_sizes.get(loop_dim_unrolled, 1)
            loop_size_unrolled = min(layer_dim_size, loop_size_unrolled)
            # Check 3: Adjust unrolling if it is not a multiple of the layer dimension size
            temporal_remainder = int(np.ceil(layer_dim_size/loop_size_unrolled))
            loop_size_unrolled = layer_dim_size / temporal_remainder
            # Set the adjusted unrolling size in the original user_spatial_mapping dict if it is greater than 1
            limited_user_spatial_mapping[oa_dim_name] = (loop_dim_unrolled, loop_size_unrolled)
            # Update the layer_dim_size to support multiple oa dims unrolling the same loop dim but not unrolling it more than the total layer dim
            if temporal_remainder == 1:  # Remove it from the dict if we have unrolled the entirely layer dim onto the array dimension(s)
                del layer_dim_sizes[loop_dim_unrolled]
            else: # Update the dict if we have some layer dims left to potentially unroll onto the next oa dims
                layer_dim_sizes[loop_dim_unrolled] = temporal_remainder

        user_spatial_mapping_for_log = {array_dim: (loop_dim, f"{loop_size:.2f}") for (array_dim, (loop_dim, loop_size)) in limited_user_spatial_mapping.items()}
        logger.debug(f"User-provided spatial mapping converted to: {user_spatial_mapping_for_log}")

        spatial_mapping_dict = {}
        layer_to_mem_op = self.layer.memory_operand_links
        mem_to_layer_op = {mem_op: layer_op for (layer_op, mem_op) in layer_to_mem_op.items()}
        core_id = self.layer.core_allocation
        mem_hierarchy = self.accelerator.get_core(core_id).memory_hierarchy
        for mem_op, layer_op in mem_to_layer_op.items():
            user_sm_copy = limited_user_spatial_mapping.copy()
            # layer_op = mem_to_layer_op[mem_op]
            spatial_mapping_dict[layer_op] = []
            memory_levels = mem_hierarchy.get_memory_levels(mem_op, )

            for memory_level in memory_levels:
                spatial_mapping_lvl = []
                served_dimensions = memory_level.served_dimensions
                for dimension in served_dimensions:
                    dim_name = dimension.name
                    if dim_name in user_sm_copy:
                        # The dimension name is present in the user defined spatial mapping
                        # Add the spatial loop of this dimension to the spatial mapping
                        spatial_loop = user_sm_copy[dim_name]
                        spatial_mapping_lvl.append(spatial_loop)
                        # Then remove this dim_name and spatial loop key value pair from the dict
                        # as the spatial mapping representation is a level-by-level one.
                        del user_sm_copy[dim_name]
                spatial_mapping_dict[layer_op].append(spatial_mapping_lvl)

            # After we have gone through the memory levels, if there are still user-defined dimensions
            # present, add them as the top level. Otherwise add an empty list to make arch levels correct:
            # because first list we added was the operational array level.
            top_level_spatial_mapping = [spatial_loop for (dim_name, spatial_loop) in user_sm_copy.items()]
            spatial_mapping_dict[layer_op].append(top_level_spatial_mapping)

        return SpatialMapping(spatial_mapping_dict=spatial_mapping_dict,
                              layer_node=self.layer)
