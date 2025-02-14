import pickle
from datetime import datetime

from zigzag import api
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping

workload_path = "zigzag/inputs/workload/resnet18.onnx"  # or "zigzag/inputs/workload/resnet18.yaml"
accelerator_path = "zigzag/inputs/hardware/tpu_like.yaml"
mapping_path = "zigzag/inputs/mapping/tpu_like.yaml"
experiment_id = datetime.now()
dump_folder = f"outputs/{experiment_id}"
pickle_filename = f"outputs/{experiment_id}/cmes.pickle"

energy, latency, cmes = api.get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="energy",
    dump_folder=dump_folder,
    pickle_filename=pickle_filename,
)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")

with open(pickle_filename, "rb") as fp:
    cmes = pickle.load(fp)
cme: CostModelEvaluation = cmes[0]

bar_plot_cost_model_evaluations_breakdown(cmes[0:5], save_path=f"{dump_folder}/breakdown.png")
print_mapping(cme)
mem_names = [ml.memory_instance.name for ml in cme.mem_level_list]
stall_slacks = cme.stall_slack_comb_collect
print("Stall and slack per port of each memory instance:")
for mem_name, ports_ss in zip(mem_names, stall_slacks):
    print(f"  {mem_name}: {ports_ss}")
print(
    f"Latency: {cme.latency_total2:.3e} (bd: ideal -> {cme.ideal_temporal_cycle}, stall -> {cme.latency_total0 - cme.ideal_temporal_cycle} onload -> {cme.latency_total1 - cme.latency_total0}, offload -> {cme.latency_total2 - cme.latency_total1})"
)
