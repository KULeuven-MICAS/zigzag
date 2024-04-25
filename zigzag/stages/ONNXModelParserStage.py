from typing import Any, Generator

from typeguard import typechecked

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.io.onnx.ONNXModelParser import ONNXModelParser
from zigzag.stages.Stage import Stage

import logging

logger = logging.getLogger(__name__)


@typechecked
class ONNXModelParserStage(Stage):

    def __init__(self, list_of_callables, *, workload, mapping, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.onnx_model_parser = ONNXModelParser(workload, mapping)

    def run(self) -> Generator[tuple[CostModelEvaluation, Any], None, None]:
        self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.get_onnx_model()
        workload = self.onnx_model_parser.get_workload()

        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
