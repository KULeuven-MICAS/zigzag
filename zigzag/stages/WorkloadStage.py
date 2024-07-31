import logging
from typing import Any

from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.ImcArray import ImcArray
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.layer_node import LayerNode
from zigzag.workload.LayerNodeABC import LayerNodeABC
from zigzag.workload.Workload import WorkloadABC

logger = logging.getLogger(__name__)


class WorkloadStage(Stage):
    """! Class that iterates through the nodes in a given workload graph."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: WorkloadABC[LayerNodeABC],
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        """
        Initialization of self.workload.
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator

    def run(self):
        for layer in self.workload.topological_sort():
            # skip the DummyNodes
            if not isinstance(layer, LayerNode):
                continue
            # Skip Pooling, Add layers for imc. This happens only when the workload is manually defined.
            # No skipping if the workload is from onnx.
            core_id = layer.core_allocation[0]
            core = self.accelerator.get_core(core_id)
            operational_array = core.operational_array
            if isinstance(operational_array, ImcArray) and layer.type in [
                "Pooling",
                "Add",
            ]:
                continue

            kwargs = self.kwargs.copy()
            kwargs["layer"] = layer
            kwargs["accelerator"] = self.accelerator

            logger.info("Processing  %s...", layer.name)
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (layer, extra_info)
