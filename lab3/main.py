import os
import sys

sys.path.insert(0, os.getcwd())

from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_total


# Path to the workload onnx model
# onnx_model_path = "zigzag/inputs/examples/workload/resnet18.onnx"
onnx_model_path = "lab1/resnet18_first_layer.onnx"

# List of accelerators architectures we run our experiment for
hardwares = [
    "lab3/inputs/hardware/accelerator1.yaml",
    "lab3/inputs/hardware/accelerator2.yaml",
    "lab3/inputs/hardware/accelerator3.yaml",
]
# List of mappings for each accelerator
mappings = [
    "lab3/inputs/mapping/accelerator1.yaml",
    "lab3/inputs/mapping/accelerator2.yaml",
    "lab3/inputs/mapping/accelerator3.yaml",
]

cmes = []
for i, (hardware, mapping) in enumerate(zip(hardwares, mappings)):
    # Pickle filename to save list of cmes
    pickle_filename = f"lab3/outputs/list_of_cmes_{i}.pickle"
    # Call the zigzag api, using a provided accelerator and mapping
    energy, latency, results = get_hardware_performance_zigzag(
        onnx_model_path,
        hardware,
        mapping,
        opt="latency",
        dump_filename_pattern=f"lab3/outputs/accelerator{i}.json",
        pickle_filename=pickle_filename,
    )
    cmes.append(results[0][0])

x_labels = [f"accelerator{i}" for i in range(len(hardwares))]
bar_plot_cost_model_evaluations_total(
    cmes,
    labels=x_labels,
    save_path="lab3/outputs/plot_total.png",
)
