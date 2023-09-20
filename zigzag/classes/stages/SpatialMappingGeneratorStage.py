import logging

from zigzag.classes.opt.spatial.generator import UserSpatialMappingGenerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.stages.SpatialMappingConversionStage import (
    SpatialMappingConversionStage,
)
import copy

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
class SpatialMappingGeneratorStage(Stage):

    ## The class constructor
    # Note: list_of_callables does NOT need to include SpatialMappingConversionStage. Although this is used,
    # this usage is done automatically.
    def __init__(self, list_of_callables, *, accelerator, layer, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.check_layer(layer)
        self.layer = layer

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
    def run(self):
        user_provided_spatial_mappings = self.layer.user_spatial_mapping
        core_id = self.layer.core_allocation
        core: Core = self.accelerator.get_core(core_id=core_id)
        oa_dims = core.operational_array.dimensions

        if isinstance(
            user_provided_spatial_mappings, dict
        ):  # There is a single USM provided
            if len(user_provided_spatial_mappings) < len(oa_dims):
                user_spatial_mapping_generator = UserSpatialMappingGenerator(
                    self.layer, self.accelerator, user_provided_spatial_mappings
                )
                # Get all the USMs by running the generator
                user_spatial_mappings = list(
                    (usm for usm in user_spatial_mapping_generator.run())
                )
            else:
                user_spatial_mappings = [user_provided_spatial_mappings]
        elif isinstance(
            user_provided_spatial_mappings, list
        ):  # There are multiple USMs provided
            user_spatial_mappings = user_provided_spatial_mappings
        else:  # There is no USM provided
            # Initialize the UserSpatialMappingGenerator which will automatically generate SMs
            user_spatial_mapping_generator = UserSpatialMappingGenerator(
                self.layer, self.accelerator
            )
            # Get all the USMs by running the generator
            user_spatial_mappings = list(
                (usm for usm in user_spatial_mapping_generator.run())
            )
            logger.debug(f"No user-provided spatial mappings found. Auto-generating..")
        nb_user_spatial_mappings = len(user_spatial_mappings)

        for i, user_spatial_mapping in enumerate(user_spatial_mappings):
            logger.info(
                f"Launching spatial mapping {i+1}/{nb_user_spatial_mappings}: {user_spatial_mapping}."
            )
            # Set the user_spatial_mapping in the layer, as this is required by SpatialMappingConversionStage
            self.layer.user_spatial_mapping = user_spatial_mapping
            # Note: manual instantiation of spatial mapping conversion stage here. We let that class deal with
            # everything else, including instantion of the actual substages
            spatial_mapping_conversion_stage = SpatialMappingConversionStage(
                self.list_of_callables,
                accelerator=self.accelerator,
                layer=copy.copy(self.layer),
                **self.kwargs,
            )
            for cme, extra_info in spatial_mapping_conversion_stage.run():
                yield cme, (user_spatial_mapping, extra_info)
