import logging

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.opt.spatial.autogenerator import UserSpatialMappingAutoGenerator
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.stages.SpatialMappingMixConversionStage import (
    SpatialMappingMixConversionStage,
)
import copy
from zigzag.utils import pickle_deepcopy

logger = logging.getLogger(__name__)


## Pipeline stage that finds spatial mappings given a:
# - accelerator
# - core allocation
# - interconnection pattern on the allocated core
# - layer
#
# The spatial mappings are found using the interconnection pattern present on the core.
#
# The inner-most memory level served dimensions is used,
# as this is how the memories connect to the operational array.
class SpatialMappingAutoGeneratorStage(Stage):
    ## The class constructor
    # Note: list_of_callables does NOT need to include SpatialMappingConversionStage. Although this is used,
    # this usage is done automatically.
    def __init__(
        self,
        list_of_callables,
        *,
        accelerator,
        layer,
        enable_mix_sm,
        enable_speedup,
        enable_ox_unroll,
        **kwargs,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.check_layer(layer)
        self.layer = layer
        self.enable_mix_sm = enable_mix_sm  # True: enable generating mix sm
        self.enable_speedup = enable_speedup  # True: only keep 3 sm with the highest hardware utilization to speedup simulation time
        self.enable_ox_unroll = enable_ox_unroll  # True: enable OX/OY unrolling when automatically generating sm

    @staticmethod
    # Check that the layer includes:
    # - the core which it is allocated to
    #
    # If not, a ValueError is raised.
    #
    # If the layer in main_inputs is not set, False is returned
    #
    # @return: True if layer is set correctly
    def check_layer(layer):
        if layer is None:
            raise ValueError()
        if layer.core_allocation is None:
            logger.critical(f"Layer {layer} has no core allocation.")
            raise ValueError()
        return True

    ## Run this stage by generating user-formatted spatial mappings which are converted
    # to the memory-level based spatial mapping representation.
    def run(self, enable_ox_unroll=True):
        # @ param enable_ox_unroll: True - will adjust input mem size if there is OX / OY mapping in the spatial mapping.
        # Note: this param should be True if @param enable_ox_unroll in autogenerator.py is True
        user_provided_spatial_mappings = self.layer.user_spatial_mapping
        user_provided_spatial_mapping_hint = self.layer.user_spatial_mapping_hint
        core_id = self.layer.core_allocation
        oa_dims = self.accelerator.get_core(
            core_id=core_id
        ).operational_array.dimensions

        if isinstance(
            user_provided_spatial_mappings, dict
        ):  # There is a single USM provided
            user_spatial_mappings = [user_provided_spatial_mappings]
        elif isinstance(
            user_provided_spatial_mappings, list
        ):  # There are multiple USMs provided
            user_spatial_mappings = user_provided_spatial_mappings
        else:  # There is no USM provided
            # Initialize user_provided_spatial_mapping_hint
            if user_provided_spatial_mapping_hint is None:
                logger.warning(
                    f"No user-provided spatial mappings or hint found. Auto-generating.."
                )
                user_provided_spatial_mapping_hint = {}
                for oa_dim in oa_dims:
                    user_provided_spatial_mapping_hint[oa_dim.name] = [
                        layer_dim for layer_dim in self.layer.loop_dim_list
                    ]
                self.layer.user_spatial_mapping_hint = (
                    user_provided_spatial_mapping_hint
                )
            else:
                # Check if every oa_dim is in user_provided_spatial_mapping_hint. Completion for not-existed hint.
                for oa_dim in oa_dims:
                    if oa_dim.name not in user_provided_spatial_mapping_hint.keys():
                        user_provided_spatial_mapping_hint[oa_dim.name] = [
                            layer_dim for layer_dim in self.layer.loop_dim_list
                        ]
                logger.debug(
                    f"No user-provided spatial mappings found, but hint found. Auto-generating.."
                )
            # Initialize the UserSpatialMappingGenerator which will automatically generate SMs
            user_spatial_mapping_generator = UserSpatialMappingAutoGenerator(
                self.layer,
                self.accelerator,
                self.enable_mix_sm,
                self.enable_speedup,
                self.enable_ox_unroll,
            )
            # Get all the USMs by running the generator
            user_spatial_mappings = list(
                (usm for usm in user_spatial_mapping_generator.run())
            )

        nb_user_spatial_mappings = len(user_spatial_mappings)

        for i, user_spatial_mapping in enumerate(user_spatial_mappings):
            logger.info(f"Launching spatial mapping {i+1}/{nb_user_spatial_mappings}:")
            # Set the user_spatial_mapping in the layer, as this is required by SpatialMappingConversionStage
            self.layer.user_spatial_mapping = user_spatial_mapping
            # Note: manual instantiation of spatial mapping conversion stage here. We let that class deal with
            # everything else, including instantion of the actual substages

            # TODO: [jiacong] [ADD] modify the size of lower input mem to support OX, OY spatial unrolling
            # enable_ox_unroll: True - will adjust input mem size if there is OX / OY mapping in the spatial mapping.
            if enable_ox_unroll:
                # if True, get the new accelerator and the flag of telling if input mem size will be scaled
                # @param update_input_mem_size: True - input mem scaling is required, so accelerator will be modified.
                (
                    update_input_mem_size,
                    new_accelerator,
                ) = self.modify_innermost_input_mem_size(core_id, user_spatial_mapping)
            if enable_ox_unroll and update_input_mem_size:
                original_accelerator = self.accelerator
                spatial_mapping_conversion_stage = SpatialMappingMixConversionStage(
                    self.list_of_callables,
                    accelerator=new_accelerator,
                    layer=copy.copy(self.layer),
                    **self.kwargs,
                )
            else:
                # TODO: [jiacong] [FINISH]

                spatial_mapping_conversion_stage = SpatialMappingMixConversionStage(
                    self.list_of_callables,
                    accelerator=self.accelerator,
                    layer=copy.copy(self.layer),
                    **self.kwargs,
                )

            for cme, extra_info in spatial_mapping_conversion_stage.run():
                # TODO: [jiacong] [ADD] recover accelerator if its mem size is adjusted before
                if enable_ox_unroll and update_input_mem_size:
                    cme.accelerator = original_accelerator
                # TODO: [jiacong] [FINISH]
                yield cme, (user_spatial_mapping, extra_info)

    ## Modify memory size of the innermost input mem to support OX, OY unrolling
    def modify_innermost_input_mem_size(self, core_id, user_spatial_mapping):
        # To support OX, OY unrolling, we will scale the lowest input mem size by OXu*OYu
        # to avoid the MemoryTooSmallException in loma stage.
        core = self.accelerator.get_core(core_id=core_id)
        operational_array = core.operational_array
        oa_dims = operational_array.dimensions
        memory_hierarchy = copy.deepcopy(core.memory_hierarchy)
        innermost_levels = memory_hierarchy.get_inner_memories()
        # get the link from layer op to mem op
        layer_op_to_mem_op: dict = self.layer.memory_operand_links
        # get weight operand name
        const_operand = self.layer.constant_operands[0]  # weight representation
        # get activation operand name
        act_operand = [
            operand for operand in self.layer.input_operands if operand != const_operand
        ][0]
        # get name of OX, OY (weight ir layer dims)
        weight_ir_layer_dims: list = self.layer.operand_loop_dim[const_operand]["ir"]
        # get the oa_dim name served by input innermost memory level
        for memory_level in innermost_levels:
            mem_ops = memory_level.operands
            if layer_op_to_mem_op[act_operand] in mem_ops:
                act_innermost_mem_level = memory_level
                act_served_oa_dim: set = memory_level.served_dimensions
                act_served_oa_dim_name = list(act_served_oa_dim)[0].name
        # get the mem scaling factor if OX, OY exist
        mem_scaling_factor = 1
        try:
            if (
                act_served_oa_dim_name not in user_spatial_mapping.keys()
            ):  # there is no sm loop
                pass
            else:  # there is sm loop on act served oa dim
                act_served_oa_mapping = user_spatial_mapping[act_served_oa_dim_name]
                if isinstance(
                    act_served_oa_mapping[0], str
                ):  # a single layer dim mapping
                    layer_dim = act_served_oa_mapping[0]
                    if layer_dim in weight_ir_layer_dims:
                        layer_size = act_served_oa_mapping[1]
                        mem_scaling_factor *= layer_size
                else:  # a mix sm mapping, e.g. (("K", 2), ("OX", 5))
                    for element in act_served_oa_mapping:
                        layer_dim = element[0]
                        if layer_dim in weight_ir_layer_dims:
                            layer_size = element[1]
                            mem_scaling_factor *= layer_size
        except (
            UnboundLocalError
        ):  # except when act_layer_dim is not served in the innermost mems
            pass
        # scale the mem size
        if mem_scaling_factor == 1:
            # No need to change the input mem size
            update_input_mem_size = False
            return update_input_mem_size, self.accelerator
        else:
            update_input_mem_size = True
            # Initialize the new memory hierarchy
            mh_name = memory_hierarchy.name
            new_mh_name = mh_name + "-supporting-diagonal-map"
            new_memory_hierarchy = MemoryHierarchy(operational_array, new_mh_name)
            # Add memories to the new memory hierarchy with the correct attributes
            for curr_mem_level, memory_level in enumerate(
                memory_hierarchy.mem_level_list
            ):
                memory_instance = memory_level.memory_instance
                if memory_level == act_innermost_mem_level:
                    memory_instance.size *= mem_scaling_factor  # scale here. For others, keep them unchanged.
                operands = tuple(memory_level.operands)
                port_alloc = memory_level.port_alloc_raw
                served_dimensions_vec = memory_level.served_dimensions_vec
                assert len(served_dimensions_vec) >= 1
                served_dimensions = served_dimensions_vec[0]

                new_memory_instance = pickle_deepcopy(memory_instance)
                new_operands = pickle_deepcopy(operands)
                new_port_alloc = pickle_deepcopy(port_alloc)
                new_served_dimensions = pickle_deepcopy(served_dimensions)
                new_memory_hierarchy.add_memory(
                    memory_instance=new_memory_instance,
                    operands=new_operands,
                    port_alloc=new_port_alloc,
                    served_dimensions=new_served_dimensions,
                )
            # Create the new core
            id = core.id
            dataflows = core.dataflows
            new_id = id
            new_dataflows = pickle_deepcopy(dataflows)

            new_core = Core(
                id=new_id,
                operational_array=operational_array,
                memory_hierarchy=new_memory_hierarchy,
                dataflows=new_dataflows,
            )

            # Create the new accelerator
            name = self.accelerator.name
            new_name = name + "-scaled"
            new_cores = {new_core}
            new_accelerator = Accelerator(
                name=new_name,
                core_set=new_cores,
            )
            return update_input_mem_size, new_accelerator
