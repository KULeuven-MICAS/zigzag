import importlib
from typing import Any


from zigzag.io.AcceleratorParser import AcceleratorParser
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.DNNWorkload import DNNWorkload
from zigzag.utils import pickle_deepcopy

import logging

logger = logging.getLogger(__name__)


class AcceleratorParserStage(Stage):

    def __init__(self, list_of_callables: list[StageCallable], *, accelerator: str, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator_parser = AcceleratorParser(accelerator)

    def run(self):
        self.accelerator_parser.run()
        accelerator = self.accelerator_parser.get_accelerator()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


class WorkloadParserStage(Stage):

    def __init__(self, list_of_callables: list[StageCallable], *, workload: str, mapping: str, **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.mapping = mapping

    def run(self):
        workload = self.parse_workload_from_path_or_from_module(self.workload, self.mapping)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload_from_path_or_from_module(
        self, workload: str | dict[int, dict[str, Any]], mapping: str | dict[str, dict[str, Any]]
    ) -> DNNWorkload:
        """! Parse the input workload residing in workload_path.
        The "workload" dict is converted to a NetworkX graph.
        """
        if isinstance(workload, str):  # load from path
            module = importlib.import_module(workload)
            workload_dict: dict[int, dict[str, Any]] = module.workload
        else:
            workload_dict = workload

        if isinstance(mapping, str):  # load from path
            module = importlib.import_module(mapping)
            mapping_dict: dict[str, dict[str, Any]] = module.mapping
        else:
            mapping_dict = mapping

        # make a copy here to prevent later it is being changed in the following stages
        workload_copy: dict[int, dict[str, Any]] = pickle_deepcopy(workload_dict)
        workload_converted = DNNWorkload(workload_copy, mapping_dict)
        logger.info(
            f"""Created workload graph with {workload_converted.number_of_nodes()} nodes and 
            {workload_converted.number_of_edges()} edges."""
        )

        return workload_converted
