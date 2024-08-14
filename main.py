import logging
import re

from zigzag.api import get_hardware_performance_zigzag
from zigzag.parser.arguments import get_arg_parser

parser = get_arg_parser()
args = parser.parse_args()

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format)

hw_name = args.accelerator.split(".")[-1]
workload_name = re.split(r"/|\.", args.model)[-1]
if workload_name == "onnx":
    workload_name = re.split(r"/|\.", args.model)[-2]
experiment_id = f"{hw_name}-{workload_name}"
pickle_name = f"{experiment_id}-saved_list_of_cmes"


dump_folder = f"outputs/{experiment_id}"
pickle_filename = f"outputs/{pickle_name}.pickle"


get_hardware_performance_zigzag(
    accelerator=args.accelerator,
    workload=args.model,
    mapping=args.mapping,
    opt="latency",
    dump_folder=f"outputs/{experiment_id}",
    pickle_filename=f"outputs/{pickle_name}.pickle",
)
