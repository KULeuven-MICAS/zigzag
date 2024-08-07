import logging
from typing import Any

from zigzag.parser.MappingValidator import MappingValidator
from zigzag.parser.workload_factory import WorkloadFactory
from zigzag.parser.WorkloadValidator import WorkloadValidator
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.utils import open_yaml
from zigzag.workload.DNNWorkload import DNNWorkload

logger = logging.getLogger(__name__)


class WorkloadParserStage(Stage):
    """! Parses a user-provided workload from a yaml file."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: str,
        mapping: str,
        **kwargs: Any,
    ):
        assert mapping.endswith(".yaml"), "Mapping is not a yaml file path"
        assert workload.endswith(".yaml"), "Workload is not a yaml file path"
        super().__init__(list_of_callables, **kwargs)
        self.workload_yaml_path = workload
        self.mapping_yaml_path = mapping

    def run(self):
        workload = self.parse_workload()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload(self) -> DNNWorkload:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactory(workload_data, mapping_data)
        return factory.create()

    def _parse_workload_data(self) -> list[dict[str, Any]]:
        """! Parse, validate and normalize workload"""
        workload_data = open_yaml(self.workload_yaml_path)
        workload_validator = WorkloadValidator(workload_data)
        workload_data = workload_validator.normalized_data
        workload_validate_succes = workload_validator.validate()
        if not workload_validate_succes:
            raise ValueError("Failed to validate user provided workload.")
        return workload_data

    def _parse_mapping_data(self) -> list[dict[str, Any]]:
        return self.parse_mapping_data(self.mapping_yaml_path)

    @staticmethod
    def parse_mapping_data(mapping_yaml_path: str) -> list[dict[str, Any]]:
        """Parse, validate and normalize workload mapping from a given yaml file path"""
        mapping_data = open_yaml(mapping_yaml_path)
        mapping_validator = MappingValidator(mapping_data)
        mapping_data = mapping_validator.normalized_data
        mapping_validate_success = mapping_validator.validate()
        if not mapping_validate_success:
            raise ValueError("Failed to validate user provided mapping.")
        return mapping_data
