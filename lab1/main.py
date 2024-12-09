import os
import sys
import argparse
import re

sys.path.insert(0, os.getcwd())
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.stages.ONNXModelParserStage import ONNXModelParserStage
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.TemporalOrderingConversionStage import TemporalOrderingConversionStage
from zigzag.stages.save_stages import CompleteSaveStage
from zigzag.stages.reduce_stages import MinimalLatencyStage


from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

# Get the onnx model, the mapping and accelerator arguments
parser = argparse.ArgumentParser(description="Setup zigzag inputs")
parser.add_argument(
    "--model",
    metavar="path",
    required=True,
    help="path to onnx model, e.g. inputs/examples/my_onnx_model.onnx",
)
parser.add_argument(
    "--mapping",
    metavar="path",
    required=True,
    help="path to mapping file, e.g., inputs.examples.my_mapping",
)
parser.add_argument(
    "--accelerator",
    metavar="path",
    required=True,
    help="module path to the accelerator, e.g. inputs.examples.accelerator1",
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
        TemporalOrderingConversionStage,  # Converts defined temporal_ordering to temporal mapping
        CostModelStage,  # Evaluates generated SM and TM through cost model
    ],
    accelerator=args.accelerator,  # required by AcceleratorParserStage
    workload=args.model,  # required by ONNXModelParserStage
    mapping=args.mapping,  # required by ONNXModelParserStage
    dump_filename_pattern=f"lab1/outputs/{experiment_id}-?.json",  # output file save pattern, ? will be replaced
    loma_lpf_limit=6,  # required by LomaStage
    loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
)

# Launch the MainStage
answers = mainstage.run()
# Plot the energy and latency breakdown of our cost model evaluation
cme = answers[0][0]
save_path = "lab1/outputs/breakdown.png"
bar_plot_cost_model_evaluations_breakdown([cme], save_path=save_path, xtick_rotation=0)
