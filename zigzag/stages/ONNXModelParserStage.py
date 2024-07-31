import logging
from typing import Any

from zigzag.parser.onnx.ONNXModelParser import ONNXModelParser
from zigzag.stages.Stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    """Stage to parse ONNX model to internal representation"""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: str,
        mapping: str,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.onnx_model_parser = ONNXModelParser(workload, mapping)

    def run(self):
        workload = self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.onnx_model

        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
