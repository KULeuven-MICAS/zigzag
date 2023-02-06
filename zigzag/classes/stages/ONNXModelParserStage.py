from typing import Generator

from zigzag.classes.io.onnx.model import ONNXModelParser
from zigzag.classes.stages.Stage import Stage

import logging
logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    def __init__(self, list_of_callables, *, workload, mapping, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.onnx_model_parser = ONNXModelParser(workload, mapping)
    
    def run(self) -> Generator:
        self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.get_onnx_model()
        workload = self.onnx_model_parser.get_workload()

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], onnx_model=onnx_model, workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    # # For testing purposes
    # def is_leaf(self) -> bool:
    #     return True
