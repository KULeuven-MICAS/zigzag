import importlib

from zigzag.classes.hardware.architecture.accelerator import Accelerator

import logging
logger = logging.getLogger(__name__)


class AcceleratorParser:
    """Parse an accelerator module path into an accelerator object.
    """
    def __init__(self, accelerator) -> None:
        """Initialize the parser by checking if the provided argument is a module path or accelerator object.

        Args:
            accelerator_path (str or Accelerator): The accelerator path or accelerator object.
        """
        if isinstance(accelerator, str):
            self.accelerator_path = accelerator
            self.accelerator = None
        elif isinstance(accelerator, Accelerator):
            self.accelerator_path = None
            self.accelerator = accelerator
        else:
            raise TypeError("Given accelerator is nor a module path string or an Accelerator object.")
    
        self.supported_accelerators = {
            "ascend": "zigzag.inputs.examples.hardware.Ascend_like",
            "edge-tpu": "zigzag.inputs.examples.hardware.Edge_TPU_like",
            "eyeriss": "zigzag.inputs.examples.hardware.Eyeriss_like",
            "meta-prototype": "zigzag.inputs.examples.hardware.Meta_prototype",
            "tesla-npu": "zigzag.inputs.examples.hardware.Tesla_NPU_like",
            "tpu": "zigzag.inputs.examples.hardware.TPU_like"
        }

    def run(self):
        if not self.accelerator:
            try:
                accelerator = self.parse_accelerator_from_path(self.accelerator_path)
            except ModuleNotFoundError:
                try:
                    accelerator = self.parse_supported_accelerator(self.accelerator_path)
                except KeyError:
                    raise ValueError(f"Provided accelerator path ({self.accelerator_path}) is not a valid module path, nor a supported standard accelerator. \
                        Supported standard accelerators = {self.get_supported_accelerators()}")
            self.accelerator = accelerator

    @staticmethod
    def parse_accelerator_from_path(accelerator_path):
        """
        Parse the input accelerator residing in accelerator_path.
        """
        global module
        module = importlib.import_module(accelerator_path)
        accelerator = module.accelerator
        logger.info(f"Parsed accelerator with cores {[core.id for core in accelerator.cores]}.")
        return accelerator

    def parse_supported_accelerator(self, standard_accelerator):
        accelerator_path = self.supported_accelerators[standard_accelerator]
        return self.parse_accelerator_from_path(accelerator_path)

    def get_accelerator(self):
        return self.accelerator

    def get_supported_accelerators(self):
        return list(self.supported_accelerators.keys())