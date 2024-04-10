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


model = "custom"
# workload = f"zigzag/inputs/examples/workload/{model}_inf.onnx"
# # create inferred ONNX model
# if not os.path.exists(workload):
#     onnx.shape_inference.infer_shapes_path(
#         f"zigzag/inputs/examples/workload/{model}.onnx",
#         workload,
#     )
# mapping = "zigzag.inputs.examples.mapping.default"
workload = "zigzag.inputs.examples.workload.transformer"
mapping = "zigzag.inputs.examples.mapping.transformer_custom"
accelerator = "zigzag.inputs.examples.hardware.Edge_TPU_like"


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
    cme_for_all_layers: list[CostModelEvaluation] = pickle.load(fp)


show_layers = [1, 4, 6, 7, 8, 9]
cmes = [cme_for_all_layers[i] for i in show_layers]

bar_plot_cost_model_evaluations_breakdown(cmes, save_path="outputs/plot_breakdown.png")
# bar_plot_cost_model_evaluations_breakdown(cme_for_all_layers, save_path="plot_breakdown.png")
# uncomment this line to plot for all the layers

visualize_memory_hierarchy_graph(
    cme_for_all_layers[0].accelerator.cores[0].memory_hierarchy,
    save_path="outputs/mem_hierarchy.png",
)

for cme in cmes:
    print_mapping(cme)
    print(cme.spatial_mapping_dict_int)
    print("O")
    for i in cme.memory_word_access["O"]:
        print(i)
    print("I")
    for i in cme.memory_word_access["I"]:
        print(i)
