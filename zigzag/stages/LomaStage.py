from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.opt.loma.LomaEngine import LomaEngine
from zigzag.workload.layer_node import LayerNode
from typing import Any
from zigzag.stages.Stage import Stage, StageCallable


class LomaStage(Stage):
    """! Class that iterates through the different temporal mappings generated through
    the loop order based memory allocation (loma) engine
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        **kwargs: Any,
    ):
        """
        Initialize the LomaStage by setting the accelerator, layer, and spatial mapping.
        @param list_of_callables (List[Callable]): List of substages to call with each generated temporal mapping.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping

    def run(self):
        engine = LomaEngine(
            accelerator=self.accelerator,
            layer=self.layer,
            spatial_mapping=self.spatial_mapping,
            **self.kwargs,
        )

        for temporal_mapping in engine.run():

            kwargs = self.kwargs.copy()
            kwargs["accelerator"] = self.accelerator
            kwargs["layer"] = self.layer
            kwargs["spatial_mapping"] = self.spatial_mapping
            kwargs["temporal_mapping"] = temporal_mapping
            sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (temporal_mapping, extra_info)
