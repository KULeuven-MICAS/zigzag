import os
import sys
import argparse
import re

from zigzag.stages.MainStage import MainStage
from zigzag.stages.WorkloadParserStage import WorkloadParserStage
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.save_stages import CompleteSaveStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.reduce_stages import MinimalLatencyStage
from zigzag.stages.LomaStage import LomaStage
from zigzag.stages.TemporalOrderingConversionStage import TemporalOrderingConversionStage
from zigzag.stages.CostModelStage import CostModelStage
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.print_mapping import print_mapping


model = "gemm"
workload_path = "zigzag/inputs/workload/gemm_layer.yaml"  # or "zigzag/inputs/workload/resnet18.yaml"
accelerator_path = "zigzag/inputs/hardware/gemm.yaml"
mapping_path = "zigzag/inputs/mapping/gemm.yaml"
pickle_filename = f"outputs/{model}-saved_list_of_cmes.pickle"

# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
# _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


energy, latency, answer = get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="latency",
    pickle_filename=pickle_filename,
)


# Launch the MainStage
# save_path = "outputs/breakdown.png"
# bar_plot_cost_model_evaluations_breakdown([cme], save_path=save_path, xtick_rotation=0)
# from zigzag.visualization.results.print_mapping import print_mapping

# mem_names = [ml.memory_instance.name for ml in cme.mem_level_list]
# stall_slacks = cme.SS_comb_collect
# print("Stall and slack per port of each memory instance:")
# for mem_name, ports_ss in zip(mem_names, stall_slacks):
#     print(f"  {mem_name}: {ports_ss}")
cme = answer[0][1][0][0]
print_mapping(cme)
# print(f"SM: {cme.spatial_mapping}")
# print(f"TM: {cme.temporal_mapping}")
print(f"Latency: {cme.latency_total2:.3e} (bd: ideal -> {cme.ideal_temporal_cycle}, stall -> {cme.latency_total0 - cme.ideal_temporal_cycle} onload -> {cme.latency_total1 - cme.latency_total0}, offload -> {cme.latency_total2 - cme.latency_total1})")
