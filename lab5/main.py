import logging as _logging
import os
import sys

sys.path.insert(0, os.getcwd())
from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping

# Initialize the logger
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

# Define the experiment id and pickle name
hw_name = "accelerator1"
workload_name = "resnet18_first_layer"
experiment_id = f"{hw_name}-{workload_name}"
pickle_name = f"{experiment_id}-saved_list_of_cmes"

# Define main input paths
accelerator = "lab5/inputs/hardware/accelerator.yaml"
workload = "lab5/inputs/workload/resnet18_first_layer.onnx"
mapping = "lab5/inputs/mapping/mapping.yaml"

# Define other inputs of api call
temporal_mapping_search_engine = "loma"
optimization_criterion = "latency"
dump_folder = f"lab5/outputs/{experiment_id}"
pickle_filename = f"lab5/outputs/{pickle_name}.pickle"


# Get the hardware performance through api call
energy, latency, tclk, area, results = get_hardware_performance_zigzag(
    accelerator=accelerator,
    workload=workload,
    mapping=mapping,
    temporal_mapping_search_engine=temporal_mapping_search_engine,
    opt=optimization_criterion,
    dump_folder=dump_folder,
    pickle_filename=pickle_filename,
    in_memory_compute=True,
)

# Save a bar plot of the cost model evaluations breakdown
cmes = [result[0] for result in results[0][1]]
save_path = "lab5/outputs/breakdown.png"
bar_plot_cost_model_evaluations_breakdown(cmes, save_path=save_path)
print_mapping(cmes[0])

# Calculate system-level performance metrics
total_mac_count = cmes[0].layer.total_mac_count
delay_in_ns = energy * cmes[0].tclk  # unit: ns
tops_system = total_mac_count * 2 / delay_in_ns / 1000
topsw_system = total_mac_count * 2 / energy
topsmm2_system = tops_system / cmes[0].area_total

# Calculate macro-level performance metrics
imc_macro = cmes[0].accelerator.operational_array
tops_peak, topsw_peak, topsmm2_peak = imc_macro.get_macro_level_peak_performance()

# Extract area and latency details
area_breakdown_info = f"memory area: {round(cmes[0].mem_area, 4)}, imc area: {round(cmes[0].imc_area, 4)}"
energy_breakdown_info = f"computation: {round(cmes[0].mac_energy, 2)}, memory: {round(cmes[0].mem_energy, 2)}"
cycles_breakdown_info = f"computation: {cmes[0].ideal_temporal_cycle}, memory stalling: {cmes[0].stall_slack_comb}, \
data loading: {cmes[0].data_onloading_cycle}, data offloading: {cmes[0].data_offloading_cycle}"

# Print out information in the terminal
macro_performance_info = f"TOP/s: {round(tops_peak, 4)}, TOP/s/W: {round(topsw_peak, 4)}, \
TOP/s/mm^2: {round(topsmm2_peak, 4)}"
system_performance_info = f"TOP/s: {round(tops_system, 4)}, TOP/s/W: {round(topsw_system, 4)}, \
TOP/s/mm^2: {round(topsmm2_system, 4)}"

logger.info("spatial mapping: %s", cmes[0].layer.spatial_mapping)
logger.info("energy (pJ): %s [%s]", round(energy, 2), energy_breakdown_info)
logger.info("#cycles: %s [%s]", latency, cycles_breakdown_info)
logger.info("Tclk (ns): %s", round(cmes[0].tclk, 4))
logger.info("system area (mm^2): %s [%s]", round(cmes[0].area_total, 4), area_breakdown_info)
logger.info("macro-level performance: [%s]", macro_performance_info)
logger.info("system-level performance: [%s]", system_performance_info)
exit()
