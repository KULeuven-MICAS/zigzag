import logging
from typing import Any

from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.parser.accelerator_factory import AcceleratorFactory
from zigzag.parser.AcceleratorValidator import AcceleratorValidator
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.utils import open_yaml

logger = logging.getLogger(__name__)


class AcceleratorParserStage(Stage):
    """! Parse to parse an accelerator from a user-defined yaml file."""

    def __init__(self, list_of_callables: list[StageCallable], *, accelerator: str, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        assert accelerator.split(".")[-1] == "yaml", "Expected a yaml file as accelerator input"
        self.accelerator_yaml_path = accelerator

    def run(self):
        accelerator = self.parse_accelerator(self.accelerator_yaml_path)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    @staticmethod
    def parse_accelerator(accelerator_yaml_path: str) -> Accelerator:
        accelerator_data = open_yaml(accelerator_yaml_path)

        validator = AcceleratorValidator(accelerator_data)
        accelerator_data = validator.normalized_data
        validate_success = validator.validate()
        if not validate_success:
            raise ValueError("Failed to validate user provided accelerator.")

        factory = AcceleratorFactory(accelerator_data)
        return factory.create()
