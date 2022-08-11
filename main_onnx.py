from classes.stages import *
import argparse

# Get the onnx model, the mapping and accelerator arguments
parser = argparse.ArgumentParser(description="Setup zigzag inputs")
parser.add_argument('--model', metavar='path', required=True, help='path to onnx model, e.g. inputs/examples/my_onnx_model.onnx')
parser.add_argument('--mapping', metavar='path', required=True, help='path to mapping file, e.g., inputs.examples.my_mapping.py')
parser.add_argument('--accelerator', metavar='path', required=True, help='module path to the accelerator, e.g. inputs.examples.accelerator1')
args = parser.parse_args()

# Initialize the logger
import logging as _logging
_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format)

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage([  # Initialization of the MainStage which starts off stages (1st arg) with parameters (2nd arg)
    ONNXModelParserStage,  # Parses the ONNX Model into the workload
    AcceleratorParserStage,  # Parses the Accelerator
    CompleteSaveStage,  # Saves all CostModelEvaluation information to a json
    WorkloadStage,  # Iterates through the different layers in the workload
    # MinimalEnergyStage,  # Reduces all CostModelEvaluations it receives, keeping the minimal energy one
    MinimalLatencyStage,  # Reduces all CostModelEvaluations it receives, keeping the minimal latency one
    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
    # MinimalEnergyStage,  # Reduces all CostModelEvaluations it receives, keeping the minimal energy one
    MinimalLatencyStage,  # Reduces all CostModelEvaluations it receives, keeping the minimal latency one
    LomaStage,  # Generates multiple temporal mappings (TM)
    CostModelStage  # Evaluates every SM + TM combination for all layers and returns the CostModelEvaluation
],
    accelerator_path=args.accelerator,  # accelerator argument, req by AcceleratorParserStage
    onnx_model_path=args.model,  # onnx model argumentm, req by ONNXModelParserStage
    mapping_path=args.mapping,  # mapping argument, req by ONNXModelParserStage
    dump_filename_pattern="outputs_workshop/{datetime}.json",  # output file save pattern
    loma_lpf_limit=6  # max number of lpfs, req by LomaStage
)

# Launch the MainStage
mainstage.run()