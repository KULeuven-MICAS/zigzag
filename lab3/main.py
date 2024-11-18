from zigzag.api import get_hardware_performance_zigzag

# Path to the workload onnx model
workload = "lab3/inputs/workload/resnet18_first_layer.onnx"
workload_name = "resnet18_first_layer"

# List of accelerators architectures we run our experiment for
accelerators = [
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

temporal_mapping_search_engine = "loma"
optimization_criterion = "latency"
energies = []
latencies = []
for i, (accelerator, mapping) in enumerate(zip(accelerators, mappings)):
    # Inputs for api call
    hw_name = f"accelerator{i+1}"
    experiment_id = f"{hw_name}-{workload_name}"
    dump_folder = f"lab3/outputs/{experiment_id}"
    pickle_filename = f"lab3/outputs/{experiment_id}_cmes.pickle"
    # Call the zigzag api, using a provided accelerator and mapping
    energy, latency, results = get_hardware_performance_zigzag(
        workload,
        accelerator,
        mapping,
        temporal_mapping_search_engine=temporal_mapping_search_engine,
        opt=optimization_criterion,
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
    )
    energies.append(energy)
    latencies.append(latency)

for i, (energy, latency) in enumerate(zip(energies, latencies)):
    print(f"Accelerator {i+1}: Energy =  {energy:.2e}, Latency = {latency:.2e}")
