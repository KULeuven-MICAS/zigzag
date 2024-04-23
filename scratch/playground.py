import pickle

# import os
import sys


# import onnx

sys.path.append("../zigzag")
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph


model = "transformer"
workload = "scratch.transformer_workload"
mapping = "scratch.transformer_mapping"
accelerator = "zigzag.inputs.examples.hardware.TPU_like"


dump_filename_pattern = f"outputs/TPU-{model}-layer_?.json"
pickle_filename = f"outputs/TPU-{model}-saved_list_of_cmes.pickle"

if True:
    energy, latency, cme = api.get_hardware_performance_zigzag(
        workload=workload,
        accelerator=accelerator,
        mapping=mapping,
        opt="energy",
        dump_filename_pattern=dump_filename_pattern,
        pickle_filename=pickle_filename,
    )

    print(f"Total network energy = {energy:.2e} pJ")
    print(f"Total network latency = {latency:.2e} cycles")


with open(pickle_filename, "rb") as fp:
    cmes: list[CostModelEvaluation] = pickle.load(fp)


bar_plot_cost_model_evaluations_breakdown(cmes, save_path="outputs/plot_breakdown.png")
# bar_plot_cost_model_evaluations_breakdown(cme_for_all_layers, save_path="plot_breakdown.png")
# uncomment this line to plot for all the layers

visualize_memory_hierarchy_graph(
    cmes[0].accelerator.cores[0].memory_hierarchy,
    save_path="outputs/mem_hierarchy.png",
)

for cme in cmes:
    print_mapping(cme)
