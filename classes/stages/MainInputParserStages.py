from typing import Generator, Callable, List, Tuple, Any
from classes.stages.Stage import Stage
import importlib
from classes.workload.dnn_workload import DNNWorkload
import logging
logger = logging.getLogger(__name__)


def parse_accelerator_from_path(accelerator_path):
    """
    Parse the input accelerator residing in accelerator_path.
    """
    global module
    module = importlib.import_module(accelerator_path)
    accelerator = module.accelerator
    logger.info(f"Parsed accelerator with cores {[core.id for core in accelerator.cores]}.")
    return accelerator

class AcceleratorParserStage(Stage):
    def __init__(self, list_of_callables, *, accelerator_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator_path = accelerator_path

    def run(self):
        accelerator = parse_accelerator_from_path(self.accelerator_path)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info




def parse_workload_from_path(workload_path):
    """
    Parse the input workload residing in accelerator_path.
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
        accelerator = parse_accelerator_from_path(self.accelerator_path)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=accelerator, workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info
