"""
# TODO remove this file
"""

import logging
import yaml

from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.parser.accelerator_factory import AcceleratorFactory
from zigzag.parser.AcceleratorValidator import AcceleratorValidator

logger = logging.getLogger(__name__)


# class AcceleratorParser:
#     """! Parse an accelerator module path into an accelerator object"""

#     # known_accelerator_paths = {
#     #     "ascend": "zigzag.inputs.examples.hardware.Ascend_like",
#     #     "edge-tpu": "zigzag.inputs.examples.hardware.Edge_TPU_like",
#     #     "eyeriss": "zigzag.inputs.examples.hardware.Eyeriss_like",
#     #     "meta-prototype": "zigzag.inputs.examples.hardware.Meta_prototype",
#     #     "tesla-npu": "zigzag.inputs.examples.hardware.Tesla_NPU_like",
#     #     "tpu": "zigzag.inputs.examples.hardware.TPU_like",
#     # }
#     # known_accelerators = list(known_accelerator_paths.keys())

#     def __init__(self, accelerator_path: str) -> None:
#         """
#         Initialize the parser by checking if the provided argument is a module path or accelerator object
#         @param accelerator_path (str or Accelerator): The accelerator path or accelerator object
#         """
#         self.accelerator_path = accelerator_path

#     def run(self) -> Accelerator:
#         accelerator_data = self.open_yaml(self.accelerator_path)

#         validator = AcceleratorValidator(accelerator_data)
#         validate_success = validator.validate()
#         if not validate_success:
#             raise ValueError("Failed to validate user provided accelerator.")

#         factory = AcceleratorFactory(accelerator_data)
#         return factory.create()

#     def open_yaml(self, path: str):
#         with open(path) as f:
#             data = yaml.safe_load(f)
#         return data

#     # @staticmethod
#     # def parse_accelerator_from_path(accelerator_path: str):
#     #     """! Parse the input accelerator residing in accelerator_path
#     #     @param accelerator_path
#     #     """
#     #     global module
#     #     module = importlib.import_module(accelerator_path)
#     #     accelerator = module.accelerator
#     #     logger.info(f"Parsed accelerator with cores {[core.id for core in accelerator.cores]}.")
#     #     return accelerator

#     # @staticmethod
#     # def parse_known_accelerator(standard_accelerator: str) -> Accelerator:
#     #     accelerator_path = AcceleratorParser.known_accelerator_paths[standard_accelerator]
#     #     parser = AcceleratorParser(accelerator_path)
#     #     return parser.run()
