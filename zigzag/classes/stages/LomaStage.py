from typeguard import typechecked
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.mapping.spatial.SpatialMappingInternal import SpatialMappingInternal
from zigzag.classes.opt.temporal.loma.engine import LomaEngine
from zigzag.classes.workload.layer_node import LayerNode
from typing import Callable
from zigzag.classes.stages.Stage import Stage


@typechecked
class LomaStage(Stage):
    """!  Class that iterates through the different temporal mappings generated through
    the loop order based memory allocation (loma) engine
    """

    def __init__(
        self,
        list_of_callables: list[Callable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        **kwargs,
    ):
        """!  The class constructor
        Initialize the LomaStage by setting the accelerator, layer, and spatial mapping.
        @param list_of_callables (List[Callable]): List of substages to call with each generated temporal mapping.
        @param accelerator (Accelerator): The accelerator object.
        @param layer (Layer): The layer object.
        @param spatial_mapping (SpatialMappingInternal): The spatial mapping object.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping = (
            accelerator,
            layer,
            spatial_mapping,
        )
        self.engine = None

    def run(self):
        self.engine = LomaEngine(
            accelerator=self.accelerator,
            layer=self.layer,
            spatial_mapping=self.spatial_mapping,
            **self.kwargs,
        )

        for tm in self.engine.run():

            kwargs = self.kwargs.copy()
            kwargs["accelerator"] = self.accelerator
            kwargs["layer"] = self.layer
            kwargs["spatial_mapping"] = self.spatial_mapping
            kwargs["temporal_mapping"] = tm
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (tm, extra_info)
