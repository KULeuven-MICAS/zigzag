import logging
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.cost_model.cost_model_imc import CostModelEvaluationForIMC
from zigzag.hardware.architecture.accelerator import Accelerator
from zigzag.hardware.architecture.imc_array import ImcArray
from zigzag.mapping.spatial_mapping_internal import SpatialMappingInternal
from zigzag.mapping.temporal_mapping import TemporalMapping
from zigzag.stages.stage import Stage, StageCallable
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class CostModelStage(Stage):
    """!  Pipeline stage that calls a cost model to evaluate a mapping on a HW config."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        spatial_mapping_int: SpatialMappingInternal,
        temporal_mapping: TemporalMapping,
        access_same_data_considered_as_no_access: bool = True,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)

        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = spatial_mapping_int
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = access_same_data_considered_as_no_access

    def run(self):
        """! Run the cost model stage by calling the internal zigzag cost model with the correct inputs."""
        operational_array = self.accelerator.operational_array
        if isinstance(operational_array, ImcArray):
            cme = CostModelEvaluationForIMC(
                accelerator=self.accelerator,
                layer=self.layer,
                spatial_mapping=self.spatial_mapping,
                spatial_mapping_int=self.spatial_mapping_int,
                temporal_mapping=self.temporal_mapping,
                access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
            )
        else:
            cme = CostModelEvaluation(
                accelerator=self.accelerator,
                layer=self.layer,
                spatial_mapping=self.spatial_mapping,
                spatial_mapping_int=self.spatial_mapping_int,
                temporal_mapping=self.temporal_mapping,
                access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
            )
        yield cme, None

    def is_leaf(self) -> bool:
        return True
