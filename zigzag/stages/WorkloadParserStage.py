from typing import Any

from zigzag.parser.MappingValidator import MappingValidator
from zigzag.parser.WorkloadValidator import WorkloadValidator
from zigzag.parser.workload_factory import WorkloadFactory
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.utils import open_yaml
from zigzag.workload.DNNWorkload import DNNWorkload

import logging

logger = logging.getLogger(__name__)


class WorkloadParserStage(Stage):
    """! Parses a user-provided workload from a yaml file."""

    def __init__(self, list_of_callables: list[StageCallable], *, workload: str, mapping: str, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.workload_yaml_path = workload
        self.mapping_yaml_path = mapping

    def run(self):
        workload = self.parse_workload()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload(self) -> DNNWorkload:
        workload_data = self.__parse_workload_data()
        mapping_data = self.__parse_mapping_data()
        factory = WorkloadFactory(workload_data, mapping_data)
        return factory.create()

    def __parse_workload_data(self) -> list[dict[str, Any]]:
        """! Parse, validate and normalize workload"""
        workload_data = open_yaml(self.workload_yaml_path)
        workload_validator = WorkloadValidator(workload_data)
        workload_data = workload_validator.normalized_data
        workload_validate_succes = workload_validator.validate()
        if not workload_validate_succes:
            raise ValueError("Failed to validate user provided workload.")
        return workload_data

    def __parse_mapping_data(self) -> list[dict[str, Any]]:
        return self.parse_mapping_data(self.mapping_yaml_path)

    @staticmethod
    def parse_mapping_data(mapping_yaml_path: str) -> list[dict[str, Any]]:
        """Parse, validate and normalize workload mapping from a given yaml file path"""
        mapping_data = open_yaml(mapping_yaml_path)
        mapping_validator = MappingValidator(mapping_data)
        mapping_data = mapping_validator.normalized_data
        mapping_validate_succes = mapping_validator.validate()
        if not mapping_validate_succes:
            raise ValueError("Failed to validate user provided mapping.")
        return mapping_data

    # def parse_workload_from_path_or_from_module(
    #     self, workload: str | dict[int, dict[str, Any]], mapping: str | dict[str, dict[str, Any]]
    # ) -> DNNWorkload:
    #     """! Parse the input workload residing in workload_path.
    #     The "workload" dict is converted to a NetworkX graph.
    #     """
    #     if isinstance(workload, str):  # load from path
    #         module = importlib.import_module(workload)
    #         workload_dict: dict[int, dict[str, Any]] = module.workload
    #     else:
    #         workload_dict = workload

    #     if isinstance(mapping, str):  # load from path
    #         module = importlib.import_module(mapping)
    #         mapping_dict: dict[str, dict[str, Any]] = module.mapping
    #     else:
    #         mapping_dict = mapping

    #     # make a copy here to prevent later it is being changed in the following stages
    #     workload_copy: dict[int, dict[str, Any]] = pickle_deepcopy(workload_dict)
    #     workload_converted = DNNWorkload(workload_copy, mapping_dict)
    #     logger.info(
    #         f"Created workload graph with {workload_converted.number_of_nodes()} nodes and "
    #         f"{workload_converted.number_of_edges()} edges."
    #     )

    #     return workload_converted
