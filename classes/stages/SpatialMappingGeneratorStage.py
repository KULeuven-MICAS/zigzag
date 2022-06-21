import itertools
import logging
from typing import Set
import bisect

from sympy import factorint

from classes.hardware.architecture.accelerator import Accelerator
from classes.hardware.architecture.core import Core
from classes.hardware.architecture.dimension import Dimension
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.operational_array import OperationalArray
from classes.mapping.spatial.spatial_mapping import SpatialMapping
from classes.stages.Stage import Stage
from classes.stages.SpatialMappingConversionStage import SpatialMappingConversionStage
from classes.workload.layer_node import LayerNode
import classes.io.input_config as inputs

logger = logging.getLogger(__name__)


class SpatialMappingGeneratorStage(Stage):
    """
    Pipeline stage that finds spatial mappings given a:
    - accelerator
    - core allocation
    - interconnection pattern on the allocated core
    - layer
    The spatial mappings are found using the interconnection pattern present on the core.
    The inner-most memory level served dimensions is used,
    as this is how the memories connect to the operational array.

    :param main_inputs: MainInputs, NOT copied
    """
    def __init__(self, list_of_callables, *, accelerator, layer, **kwargs):
        """
        Note: list_of_callables does NOT need to include SpatialMappingConversionStage. Although this is used,
        this usage is done automatically.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.check_layer(layer)
        self.layer = layer


    @staticmethod
    def check_layer(layer):
        """
        Check that the layer includes:
        - the core which it is allocated to
        If not, a ValueError is raised.
        If the layer in main_inputs is not set, False is returned
        :return: True if layer is set correctly
        """
        if layer is None:
            return False
        if not layer.core_allocation:
            logger.critical(f"Layer {layer} has no core allocation.")
            raise ValueError()
        return True

    def run(self):
        """
        Run this stage by generating user-formatted spatial mappings which are converted
        to the memory-level based spatial mapping representation.
        """
        user_spatial_mappings = list(self.generate_user_spatial_mappings())
        nb_user_spatial_mappings = len(user_spatial_mappings)
        logger.info(f"Generated {nb_user_spatial_mappings} spatial mappings.")

        for i, user_spatial_mapping in enumerate(user_spatial_mappings):
            logger.info(f"Launching spatial mapping {i+1}/{nb_user_spatial_mappings}: {user_spatial_mapping}.")
            # Set the user_spatial_mapping in the layer, as this is required by SpatialMappingConversionStage
            self.layer.user_spatial_mapping = user_spatial_mapping
            # Note: manual instantiation of spatial mapping conversion stage here. We let that class deal with
            # everything else, including instantion of the actual substages
            spatial_mapping_conversion_stage = SpatialMappingConversionStage(self.list_of_callables,
                                                                             accelerator=self.accelerator,
                                                                             layer=self.layer,
                                                                             **self.kwargs)
            for cme, extra_info in spatial_mapping_conversion_stage.run():
                yield cme, (user_spatial_mapping, extra_info)


    def generate_user_spatial_mappings(self):
        """
        Generator that yields user-defined spatial mappings.
        User-defined means across operational array dimensions.
        For example, this might yield {'D1': (C, 16), 'D2': (K,16)}
        In essence it works as follows:
        for each operational array dimension oa_dim (D1, D2, ...):
          for each layer operand layer_op (W, I, O, ...):
            if oa_dim not in served_dimensions(layer_op):
                continue
            else:
                for layer dimensions layer_dim (B, K, ...) in the layer:
                    if layer_dim is irrelevant for layer_op:
                            layer_dim can be unrolled maximally
                    if layer_dim is not irrelevant for layer_op:
                      layer_dim can be unrolled if the BW allows it (assumes flexible "bus" reads)
        """
        core_id = self.layer.core_allocation
        core: Core = self.accelerator.get_core(core_id=core_id)
        operational_array: OperationalArray = core.operational_array
        oa_dims = operational_array.dimensions
        memory_hierarchy: MemoryHierarchy = core.memory_hierarchy
        innermost_levels = memory_hierarchy.get_inner_memories()

        # For every operational array dimension, we initialize it by maximally unrolling all layer dimensions.
        # Later these will be restricted if the memory structure doesn't allow for this unrolling
        oa_dim_unrolling = {oa_dim: {layer_dim: int(min(layer_size, oa_dim.size)) for layer_dim, layer_size in
                                     self.layer.loop_dim_size.items()} for oa_dim in oa_dims}

        for memory_level in innermost_levels:
            served_dimensions: Set[Dimension] = memory_level.served_dimensions
            mem_ops = memory_level.operands
            for mem_op in mem_ops:
                layer_op = self.layer.get_layer_operand(mem_op=mem_op)  # get the layer operand
                if layer_op == 'O':
                    mem_bandwidth = memory_level.write_bw  # partial outputs are written to the memory
                else:
                    mem_bandwidth = memory_level.read_bw  # inputs are read from the memory
                precision = self.layer.operand_precision[layer_op]  # bit precision of layer operand
                irrelevant_dimensions = self.layer.get_operand_irrelevant_dimensions(layer_op)
                for oa_dim in oa_dims:
                    if oa_dim not in served_dimensions:
                        continue
                    # If the operational array dimension is a served dimension of the lowest memory level,
                    # we ought to limit the unrolling for the relevant and partially relevant loop dimensions
                    for (layer_dim, unrolling_size) in oa_dim_unrolling[oa_dim].items():
                        if layer_dim in irrelevant_dimensions:
                            continue
                        # If not irrelevant, it is (partially) relevant. Limit based on BW and operand precision.
                        max_multicast_elements = mem_bandwidth // precision
                        oa_dim_unrolling[oa_dim][layer_dim] = min(max_multicast_elements, unrolling_size)

        # At this point the unrolled layer dimensions are maximal (wrt the served dimensions and bandwidth of the lowest memory level).
        # The unrolling size might not be a factor of the layer dimension size, which is required (for non greedy mapping).
        # Convert the unrolling size to be a factor of the layer dimension size. At the same time convert them to a list.
        unrollings = []
        for oa_dim in oa_dims:
            oa_dim_unrollings = []
            for (layer_dim, unrolling_size) in oa_dim_unrolling[oa_dim].items():
                layer_dim_size = self.layer.loop_dim_size[layer_dim]
                # If e.g. the unrolling size is 10 (because operational array dimension size is 10)
                # but the layer dimension size is 14, this would result in a temporal remainder of 14/10.
                # In that case we change the unrolling size to 7 (to be a factor of 14).
                # We have to make sure the unrolling size is a divisor of the layer dimension size:
                while layer_dim_size % unrolling_size != 0:
                    unrolling_size -= 1  # decrement the unrolling by 1

                # If the unrolling_size is not 1, don't add it to the unrollings for this oa_dim
                if unrolling_size != 1:
                    oa_dim_unrollings.append((layer_dim, unrolling_size))

            # In case there are no unrollings (of size > 1) possible, add a single unrolling of size 1.
            # The loop dimension we pick is randomly chosen as the first loop dimension in the layer.
            # The loop dimension chosen shouldn't matter as the size of unrolling is 1 anyway.
            if len(oa_dim_unrollings) == 0:
                # oa_dim_unrollings.append((self.layer.loop_dim_list[0], 1))
                oa_dim_unrollings.append(None)

            unrollings.append(oa_dim_unrollings)

        # Now we have for each operational array dimension the layer dimensions and size they can be unrolled without fractional remainder.
        # Now we have to combine them into user-defined spatial mappings.
        for combination in itertools.product(*unrollings):
            # If the combination has two oa dimensions that unroll the same layer dimension, skip it as this is impossible.
            if not self.all_unique([loop_dimension for (loop_dimension, loop_size) in combination]):
                continue
            # Zip the combination (which is a (layer_dim, layer_size) for each oa_dim with the oa_dim names.
            oa_dim_names = [oa_dim.name for oa_dim in oa_dims]
            user_spatial_mapping = {oa_dim_name: unrolling for (oa_dim_name, unrolling) in zip(oa_dim_names, combination) if unrolling is not None}
            yield user_spatial_mapping

    @staticmethod
    def all_unique(items):
        return len(set(items)) == len(items)