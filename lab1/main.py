import argparse
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.main import MainStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.parser.accelerator_parser import AcceleratorParserStage
from zigzag.stages.parser.onnx_model_parser import ONNXModelParserStage
from zigzag.stages.results.reduce_stages import MinimalLatencyStage
from zigzag.stages.results.save import CompleteSaveStage
from zigzag.stages.workload_iterator import WorkloadStage
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

# Get the onnx model, the mapping and accelerator arguments
parser = argparse.ArgumentParser(description="Setup zigzag inputs")
parser.add_argument(
    "--model",
    metavar="path",
    required=False,
    default="lab1/resnet18_first_layer.onnx",
    help="path to onnx model, e.g. zigzag/inputs/workload/resnet18.onnx",
)
parser.add_argument(
    "--mapping",
    metavar="path",
    required=False,
    default="lab1/mapping.yaml",
    help="path to mapping file, e.g., zigzag/inputs/mapping/resnet18.yaml",
)
parser.add_argument(
    "--accelerator",
    metavar="path",
    required=False,
    default="zigzag/inputs/hardware/tpu_like.yaml",
    help="module path to the accelerator, e.g. zigzag/inputs/hardware/tpu_like.yaml",
)
args = parser.parse_args()

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
# _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.model)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", args.model)[-2]
experiment_id = f"{hw_name}-{wl_name}"

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage(
    [  # Initializes the MainStage as entry point
        ONNXModelParserStage,  # Parses the ONNX Model into the workload
        AcceleratorParserStage,  # Parses the accelerator
        CompleteSaveStage,  # Saves all received CMEs information to a json
        WorkloadStage,  # Iterates through the different layers in the workload
        SpatialMappingGeneratorStage,  # Converts the provided spatial mapping to ZigZag's internal representation
        MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
        TemporalMappingGeneratorStage,  # Converts defined temporal_ordering to temporal mapping
        CostModelStage,  # Evaluates generated SM and TM through cost model
    ],
    accelerator=args.accelerator,  # required by AcceleratorParserStage
    workload=args.model,  # required by ONNXModelParserStage
    mapping=args.mapping,  # required by ONNXModelParserStage
    dump_folder="lab1/outputs",  # output file save pattern, ? will be replaced
    loma_lpf_limit=6,  # required by LomaStage
    loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
)

# Launch the MainStage
answers = mainstage.run()
# Plot the energy and latency breakdown of our cost model evaluation
cme = answers[0][0]
save_path = "lab1/outputs/breakdown.png"
bar_plot_cost_model_evaluations_breakdown([cme], save_path=save_path)
