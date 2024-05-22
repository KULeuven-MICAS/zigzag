"""
# TODO this file isn't used. Remove?
"""

import logging
from typing import Any


from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping
from zigzag.opt.loma.MemoryAllocator import MemoryAllocator
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.layer_attributes import LayerTemporalOrdering
from zigzag.workload.layer_node import LayerNode


logger = logging.getLogger(__name__)


class TemporalOrderingConversionStage(Stage):

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.accelerator = accelerator

    def run(self):
        """! Run this stage by converting the user-defined temporal loop ordering
        to the memory-level based temporal mapping representation.
        """
        temporal_mapping = self.convert_user_temporal_mapping(self.layer.temporal_ordering)
        kwargs = self.kwargs.copy()
        kwargs["temporal_mapping"] = temporal_mapping
        kwargs["spatial_mapping"] = self.spatial_mapping
        kwargs["layer"] = self.layer
        kwargs["accelerator"] = self.accelerator
        substage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in substage.run():
            yield cme, extra_info

    def convert_user_temporal_mapping(self, user_temporal_mapping: LayerTemporalOrdering) -> TemporalMapping:
        """!
        # TODO move to `LayerTemporalOrdering`, fix types.
        """
        spatial_mapping = self.spatial_mapping
        layer = self.layer
        layer_dim_sizes = layer.layer_dim_sizes
        for i, utm in list(enumerate(user_temporal_mapping.data))[::-1]:
            if utm[0] not in layer_dim_sizes.layer_dims:
                logger.warning(
                    f"Supplied temporal ordering {utm} for layer {layer} thrown out because loop not present in the layer"
                )
                del user_temporal_mapping[i]

        converted_mapping = []
        for dim, size in user_temporal_mapping:
            if size == "all":
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
