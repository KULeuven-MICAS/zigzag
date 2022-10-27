import logging

import numpy as np

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.opt.temporal.loma.memory_allocator import MemoryAllocator
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.layer_node import LayerNode


logger = logging.getLogger(__name__)


class TemporalOrderingConversionStage(Stage):
    def __init__(self, list_of_callables, *, accelerator, layer, spatial_mapping, **kwargs):
        """
        Initialize the accelerator and layer attributes.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.check_layer(layer)
        self.layer = layer
        self.spatial_mapping = spatial_mapping
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
        if not layer.core_allocation:
            logger.critical(f"Layer {layer} has no core allocation.")
            raise ValueError()
        if not layer.user_temporal_ordering:
            logger.critical(f"Layer {layer} has no user-defined temporal ordering.")
            raise ValueError(f"Layer {layer} has no user-defined temporal ordering. Use LomaStage to generate automatically.")

        return True

    def run(self):
        """
        Run this stage by converting the user-defined temporal loop ordering
        to the memory-level based temporal mapping representation.

        :param user_spatial_mapping: The user-defined spatial mapping to be converted. If not provided, self.layer.user_spatial_mapping is used.
        """
        temporal_mapping = self.convert_user_temporal_mapping(self.layer.user_temporal_ordering)
        kwargs = self.kwargs.copy()
        kwargs['temporal_mapping'] = temporal_mapping
        kwargs['spatial_mapping'] = self.spatial_mapping
        kwargs['layer'] = self.layer
        kwargs['accelerator'] = self.accelerator
        substage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in substage.run():
            yield cme, extra_info

    def convert_user_temporal_mapping(self, user_temporal_mapping):
        spatial_mapping = self.spatial_mapping
        layer = self.layer
        layer_dim_sizes = layer.loop_dim_size
        for i, utm in list(enumerate(user_temporal_mapping))[::-1]:
            if utm[0] not in layer_dim_sizes:
                logger.warning(f"Supplied temporal ordering {utm} for layer {layer} thrown out because loop not present in the layer")
                del user_temporal_mapping[i]

        # I don't think this is actually necessary to check:
        # If a dimension is fully unrolled spatially it doesn't have to be present in temporal ordering.
        # for d in layer_dim_sizes:
        #     if d not in [utm[0] for utm in user_temporal_mapping]:
        #         logger.error(f"Supplied temporal ordering for layer {layer} is missing dimension {d}")
        #         raise ValueError(f"Supplied temporal ordering for layer {layer} is missing dimension {d}")

        converted_mapping = []
        for dim, size in user_temporal_mapping:
            if size == 'all':
                size = layer_dim_sizes[dim]
                size_already = 1
                for dim_already, size_already_sub in converted_mapping + spatial_mapping.spatial_loop_dim_size:
                    if dim_already == dim:
                        size_already *= size_already_sub
                size //= size_already
            converted_mapping.append((dim, size))
        allocator = MemoryAllocator(self.accelerator, layer, spatial_mapping, converted_mapping)

        temporal_mapping = allocator.run()  # allocate this ordering to the memories
        return temporal_mapping