from typing import Generator, Callable, Any

from typeguard import typechecked

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.cost_model.cost_model_for_sram_imc import CostModelEvaluationForIMC
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.mapping.spatial.SpatialMappingInternal import SpatialMappingInternal
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)


@typechecked
class CostModelStage(Stage):
    """!   Pipeline stage that calls a cost model to evaluate a mapping on a HW config."""

    def __init__(
        self,
        list_of_callables: list[Callable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        spatial_mapping_int: SpatialMappingInternal,
        temporal_mapping: TemporalMapping,
        access_same_data_considered_as_no_access=True,
        **kwargs,
    ):
        super().__init__(list_of_callables, **kwargs)

        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = spatial_mapping_int
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = access_same_data_considered_as_no_access
        self.cme: CostModelEvaluation | CostModelEvaluationForIMC

    def run(self) -> Generator[tuple[CostModelEvaluation, Any], None, None]:
        """!  Run the cost model stage by calling the internal zigzag cost model with the correct inputs."""
        core_id = self.layer.core_allocation
        core = self.accelerator.get_core(core_id)
        operational_array = core.operational_array
        pe_type = getattr(operational_array, "pe_type", None)  # return None if it does not exist
        if pe_type is not None and pe_type in ["in_sram_computing"]:  # if pe_type exists and in the list
            self.cme = CostModelEvaluationForIMC(
                accelerator=self.accelerator,
                layer=self.layer,
                spatial_mapping=self.spatial_mapping,
                spatial_mapping_int=self.spatial_mapping_int,
                temporal_mapping=self.temporal_mapping,
                # the below parameter is optional
                access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
            )
        else:
            self.cme = CostModelEvaluation(
                accelerator=self.accelerator,
                layer=self.layer,
                spatial_mapping=self.spatial_mapping,
                spatial_mapping_int=self.spatial_mapping_int,
                temporal_mapping=self.temporal_mapping,
                # the below parameter is optional
                access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
            )
        yield (self.cme, None)

    def is_leaf(self) -> bool:
        return True
