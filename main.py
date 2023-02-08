from zigzag.classes.stages import *
import argparse
import re

# Parse the workload and accelerator arguments
parser = argparse.ArgumentParser(description="Setup zigzag-v2 inputs")
parser.add_argument('--model', metavar='path', required=True, help='module path to workload, e.g. inputs.examples.workloads.resnet18')
parser.add_argument('--mapping', metavar='path', required=True, help='path to mapping file, e.g., inputs.examples.mapping.tpu_like')
parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.hardware.TPU_like')
args = parser.parse_args()

# Initialize the logger
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.model)[-1]
if wl_name == 'onnx':
    wl_name = re.split(r"/|\.", args.model)[-2]
experiment_id = f"{hw_name}-{wl_name}"
pkl_name = f'{experiment_id}-saved_list_of_cmes'

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage([
    WorkloadParserStage,
    AcceleratorParserStage,
    SimpleSaveStage,
    PickleSaveStage,
    SumStage,
    CompleteSaveStage,
    WorkloadStage,
    SpatialMappingGeneratorStage,
    MinimalLatencyStage,
    LomaStage,
    CostModelStage,
],
    accelerator=args.accelerator,
    workload=args.model,
    mapping=args.mapping,
    dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",
    pickle_filename=f"outputs/{pkl_name}.pickle",
    loma_lpf_limit=6,
    loma_show_progress_bar=True,
)

# Launch the MainStage
mainstage.run()