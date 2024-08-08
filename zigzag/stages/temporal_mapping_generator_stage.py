import logging
from typing import Any, Generator

from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping
from zigzag.opt.loma.LomaEngine import LomaEngine
from zigzag.opt.loma.MemoryAllocator import MemoryAllocator
from zigzag.opt.loma.multipermute import PermutationConstraint
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class TemporalMappingGeneratorStage(Stage):
    """! Class that iterates through the different temporal mappings generated through the loop order based memory
    allocation (loma) engine or defined by the user as mapping input.
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
        @param list_of_callables (List[Callable]): List of substages to call with each generated temporal mapping.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping

    def run(self):
        for temporal_mapping in self.generate_temporal_mappings():
            kwargs = self.kwargs.copy()
            kwargs["accelerator"] = self.accelerator
            kwargs["layer"] = self.layer
            kwargs["spatial_mapping"] = self.spatial_mapping
            kwargs["temporal_mapping"] = temporal_mapping
            sub_stage: Stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (temporal_mapping, extra_info)

    def generate_temporal_mappings(self) -> Generator[TemporalMapping, None, None]:
        engine = LomaEngine(
            accelerator=self.accelerator,
            layer=self.layer,
            spatial_mapping=self.spatial_mapping,
            **self.kwargs,
        )

        # Return the full, user-provided temporal mapping
        provided_ordering = self.layer.temporal_ordering
        all_temporal_loops = engine.get_temporal_loops()
        if provided_ordering.is_complete(all_temporal_loops):
            allocator = MemoryAllocator(
                self.accelerator, self.layer, self.spatial_mapping, provided_ordering.to_legacy_format()  # type: ignore
            )
            temporal_mapping = allocator.run()
            yield temporal_mapping
            return
        else:
            constraints: list[PermutationConstraint] = provided_ordering.get_constraints()
            if any(not constr.is_empty() for constr in constraints):
                engine.set_constraints(constraints)

            # Generate from scratch
            for mapping in engine.run():
                yield mapping
