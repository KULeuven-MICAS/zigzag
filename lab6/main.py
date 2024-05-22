import os
import sys
import logging as _logging
sys.path.insert(0, os.getcwd())
from zigzag.api import get_hardware_performance_zigzag_imc
from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown, bar_plot_cost_model_evaluations_total

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)

# Path to the workload onnx model
# onnx_model_path = "zigzag/inputs/examples/workload/resnet18.onnx"
onnx_model_path = "lab6/resnet18_first_layer.onnx"

# List of accelerators architectures we run our experiment for
hardware = "lab6/accelerator.yaml"
# List of mappings for each accelerator
mapping = "lab6/mapping.yaml"

# Pickle filename to save list of cmes
pickle_filename = "lab6/outputs/list_of_cmes.pickle"
json_filename = "lab6/outputs/accelerator_layer_?.json"
# Call the zigzag api, using a provided accelerator and mapping
energy, latency, tclk, area, results = get_hardware_performance_zigzag_imc(
    onnx_model_path,
    hardware,
    mapping,
    opt="latency",
    dump_filename_pattern=json_filename,
    pickle_filename=pickle_filename,
)

cme = results[0][1][0][0]
imc_macro = cme.accelerator.cores[0].operational_array

x_labels = ["imc accelerator"]
bar_plot_cost_model_evaluations_total(
    [cme],
    labels=x_labels,
    save_path="lab6/outputs/plot_total.png",
)
bar_plot_cost_model_evaluations_breakdown(
    [cme],
    save_path="lab6/outputs/plot_breakdown.png",
    xtick_rotation=0
)
total_mac_count = cme.layer.total_MAC_count
delay_in_ns = energy * cme.tclk  # unit: ns
tops_system = total_mac_count * 2 / delay_in_ns / 1000
topsw_system = total_mac_count * 2 / energy
topsmm2_system = tops_system / cme.area_total
tops_peak, topsw_peak, topsmm2_peak = imc_macro.get_macro_level_peak_performance()

area_breakdown_info = f"memory area: {round(cme.mem_area, 4)}, imc area: {round(cme.imc_area, 4)}"
energy_breakdown_info = f"computation: {round(cme.mac_energy, 2)}, memory: {round(cme.mem_energy, 2)}"
cycles_breakdown_info = f"computation: {cme.ideal_temporal_cycle}, memory stalling: {cme.SS_comb}, \
data loading: {cme.data_loading_cycle}, data offloading: {cme.data_offloading_cycle}"

macro_performance_info = f"TOP/s: {round(tops_peak, 4)}, TOP/s/W: {round(topsw_peak, 4)}, \
TOP/s/mm^2: {round(topsmm2_peak, 4)}"
system_performance_info = f"TOP/s: {round(tops_system, 4)}, TOP/s/W: {round(topsw_system, 4)}, \
TOP/s/mm^2: {round(topsmm2_system, 4)}"

logger.info("spatial mapping: %s", cme.layer.spatial_mapping)
logger.info("energy (pJ): %s [%s]", round(energy, 2), energy_breakdown_info)
logger.info("#cycles: %s [%s]", latency, cycles_breakdown_info)
logger.info("Tclk (ns): %s", round(cme.tclk, 4))
logger.info("system area (mm^2): %s [%s]", round(cme.area_total, 4), area_breakdown_info)
logger.info("macro-level performance: [%s]", macro_performance_info)
logger.info("system-level performance: [%s]", system_performance_info)
exit()
