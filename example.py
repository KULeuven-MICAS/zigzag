from datetime import datetime
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

model = "resnet"
workload_path = "zigzag/inputs/workload/resnet18.onnx"
accelerator_path = "zigzag/inputs/hardware/tpu_like.yaml"
mapping_path = "zigzag/inputs/mapping/tpu_like.yaml"
pickle_filename = f"outputs/{model}-saved_list_of_cmes.pickle"


energy, latency, cmes = api.get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="energy",
    dump_folder=f"outputs/{datetime.now()}",
    pickle_filename=pickle_filename,
)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")

with open(pickle_filename, "rb") as fp:
    cmes = pickle.load(fp)


bar_plot_cost_model_evaluations_breakdown(cmes[0:5], save_path="outputs/plot_breakdown.png")
