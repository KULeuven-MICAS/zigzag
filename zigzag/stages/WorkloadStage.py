import logging
from typing import Any


from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.Workload import Workload
from zigzag.workload.DummyNode import DummyNode


logger = logging.getLogger(__name__)


class WorkloadStage(Stage):
    """! Class that iterates through the nodes in a given workload graph."""

    def __init__(
        self, list_of_callables: list[StageCallable], *, workload: Workload, accelerator: Accelerator, **kwargs: Any
    ):
        """
        Initialization of self.workload.
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.accelerator = accelerator

    def run(self):
        for id, layer in enumerate(self.workload.topological_sort()):
            # skip the DummyNodes
            if isinstance(layer, DummyNode):
                continue
            # Skip a layer if the layer type is "Pooling" and the hardware template is an IMC core.
            # This wil have impact when the workload is defined manually.
            # If the workload is from onnx, no skipping will be done.
            core_id: int = layer.core_allocation
            core = self.accelerator.get_core(core_id)
            operational_array = core.operational_array
            pe_type = getattr(operational_array, "pe_type", None)  # return None if it does not exist
            layer_type: str | None = layer.layer_attrs.parse_operator_type()

            if (pe_type in ["in_sram_computing"]) and (layer_type in ["Pooling", "Add"]):
                continue

            kwargs = self.kwargs.copy()
            kwargs["layer"] = layer
            kwargs["accelerator"] = self.accelerator
            layer_name = layer.name if layer.name is not None else id

            logger.info(f"Processing  {layer_name}...")
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (layer, extra_info)
