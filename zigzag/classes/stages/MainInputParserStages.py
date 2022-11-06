import importlib

from zigzag.classes.io.accelerator.parser import AcceleratorParser
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.dnn_workload import DNNWorkload

import logging
logger = logging.getLogger(__name__)


class AcceleratorParserStage(Stage):
    def __init__(self, list_of_callables, *, accelerator, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator_parser = AcceleratorParser(accelerator)

    def run(self):
        self.accelerator_parser.run()
        accelerator = self.accelerator_parser.get_accelerator()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


def parse_workload_from_path(workload_path):
    """
    Parse the input workload residing in workload_path.
    The "workload" dict is converted to a NetworkX graph.
    """
    module = importlib.import_module(workload_path)
    workload = module.workload
    # Take only first dict element to start simple
    # workload = {1: workload[1]}

    workload = DNNWorkload(workload)
    logger.info(
        f"Created workload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges.")

    return workload


class WorkloadParserStage(Stage):
    def __init__(self, list_of_callables, *, workload_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload_path

    def run(self):
        workload = parse_workload_from_path(self.workload)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


class WorkloadAndAcceleratorParserStage(Stage):
    """
    Convenience class to parse both the workload and accelerator
    """
    def __init__(self, list_of_callables, *, workload_path, accelerator_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload_path = workload_path
        self.accelerator_path = accelerator_path

    def run(self):
        workload = parse_workload_from_path(self.workload_path)
        accelerator_parser = AcceleratorParser(self.accelerator_path)
        accelerator_parser.run()
        accelerator = accelerator_parser.get_accelerator()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
