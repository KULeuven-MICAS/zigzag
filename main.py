import pickle
from datetime import datetime

from zigzag import api
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.visualization import bar_plot_cost_model_evaluations_breakdown, print_mapping
import os
import json
import pandas as pd

workload_path = "zigzag/inputs/workload/early_late_conv.yaml"  # or "zigzag/inputs/workload/resnet18.yaml"
#accelerator_path = "zigzag/inputs/hardware/gemm_l1_l3.yaml"
#accelerator_path = "zigzag/inputs/hardware/gemm_lop_equal_l3.yaml"
accelerator_path = "zigzag/inputs/hardware/gemm_lop_input_l3.yaml"
mapping_path = "zigzag/inputs/mapping/gemm_l1_l3.yaml"
experiment_id = datetime.now()
dump_folder = f"outputs/{experiment_id}"
pickle_filename = f"outputs/{experiment_id}/cmes.pickle"

# Define the accelerator paths
accelerator_paths = [
    "zigzag/inputs/hardware/gemm_l1_l3.yaml",
    "zigzag/inputs/hardware/gemm_lop_equal_l3.yaml",
    "zigzag/inputs/hardware/gemm_lop_input_l3.yaml",
]

# Initialize a list to store results
results = []

# Loop through each accelerator path
for accelerator in accelerator_paths:
    # Update the dump folder and pickle filename for each run
    experiment_id = datetime.now()
    dump_folder = f"outputs/{experiment_id}"
    pickle_filename = f"outputs/{experiment_id}/cmes.pickle"

    # Run the hardware performance evaluation
    energy, latency, cmes = api.get_hardware_performance_zigzag(
        workload=workload_path,
        accelerator=accelerator,
        mapping=mapping_path,
        opt="EDP",
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
    )

    # Process the JSON files in the dump folder
    for layer_idx in range(4):  # Assuming Layer0 to Layer3
        json_path = os.path.join(dump_folder, f"Layer{layer_idx}_complete.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                memory_energy_breakdown = data["outputs"]["energy"]["memory_energy_breakdown_per_level"]
                total_energy = 0
                total_latency = 0
                if "latency" in data["outputs"]:
                    total_latency = sum(data["outputs"]["latency"].values())
                for key, values in memory_energy_breakdown.items():
                    if values:  # Ensure the list is not empty
                        total_energy += values[-1] * 512 / 8000 / 1000
                results.append({
                    "Accelerator": accelerator,
                    "Layer": f"Layer{layer_idx}",
                    "Total kB transfer": total_energy,
                    "Total Latency (cycles)": total_latency,
                })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Print the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv("energy_results.csv", index=False)
