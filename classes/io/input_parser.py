import importlib
import copy
import logging
logger = logging.getLogger(__name__)
import copy

import classes.io.input_config as inputs
from classes.workload.dnn_workload import DNNWorkload


def parse_inputs_from_path(accelerator_path, workload_path, settings_path):
    """
    Get all the settings defined in settings_path. [Do this first for logger init]
    Parse the input accelerator residing in accelerator_path.
    Parse the input workload residing in workload_path.
    """
    settings = parse_settings_from_path(settings_path)
    workload = parse_workload_from_path(workload_path)
    accelerator = parse_accelerator_from_path(accelerator_path)

    inputs.init(input_workload=workload,
                input_accelerator=accelerator,
                input_settings=settings)


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


def parse_accelerator_from_path(accelerator_path):
    """
    Parse the input accelerator residing in accelerator_path.
    """
    global module
    module = importlib.import_module(accelerator_path)
    accelerator = module.accelerator
    logger.info(f"Parsed accelerator with cores {[core.id for core in accelerator.cores]}.")
    return accelerator


def parse_settings_from_path(settings_path):
    settings_module = importlib.import_module(settings_path)
    settings = get_variables_from_module(settings_module)
    from argparse import Namespace
    for k in list(settings.keys()):
        try:
            copy.deepcopy(settings[k])
        except:
            del settings[k]

    settings = Namespace(**settings)
    logger.info(f"Parsed settings with attributes {settings}.")
    return settings


def get_variables_from_module(module):
    '''https://stackoverflow.com/questions/9759820/how-to-get-a-list-of-variables-in-specific-python-module'''
    book = {}
    if module:
        book = {key: value for key, value in module.__dict__.items() if not key.startswith('_')}
    return book