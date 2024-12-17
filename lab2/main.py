import logging
import os
import sys

sys.path.insert(0, os.getcwd())  # For importing zigzag
from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format)

# Define the experiment id and pickle name
hw_name = "accelerator1"
workload_name = "resnet18_first_layer"
experiment_id = f"{hw_name}-{workload_name}"
pickle_name = f"{experiment_id}-saved_list_of_cmes"

# Define main input paths
accelerator = "lab2/inputs/hardware/accelerator1.yaml"
workload = "lab2/inputs/workload/resnet18_first_layer.onnx"
mapping = "lab2/inputs/mapping/mapping.yaml"
assert os.path.exists(mapping), "Copy mapping.yaml from lab1 and modify it for lab2."

# Define other inputs of api call
temporal_mapping_search_engine = "loma"
optimization_criterion = "latency"
dump_folder = f"lab2/outputs/{experiment_id}"
pickle_filename = f"lab2/outputs/{pickle_name}.pickle"

# Get the hardware performance through api call
energy, latency, results = get_hardware_performance_zigzag(
    accelerator=accelerator,
    workload=workload,
    mapping=mapping,
    temporal_mapping_search_engine=temporal_mapping_search_engine,
    opt=optimization_criterion,
    dump_folder=dump_folder,
    pickle_filename=pickle_filename,
)

# Save a bar plot of the cost model evaluations breakdown
cmes = [result[0] for result in results[0][1]]
save_path = "lab2/outputs/breakdown.png"
bar_plot_cost_model_evaluations_breakdown(cmes, save_path=save_path)
print_mapping(cmes[0])
