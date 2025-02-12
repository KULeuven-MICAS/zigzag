from zigzag.api import get_hardware_performance_zigzag
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.visualization.results.print_mapping import print_mapping

if __name__ == "__main__":
    model = "gemm"
    workload_path = "zigzag/inputs/workload/gemm_layer.yaml"  # or "zigzag/inputs/workload/resnet18.yaml"
    accelerator_path = "zigzag/inputs/hardware/gemm_l1.yaml"
    mapping_path = "zigzag/inputs/mapping/gemm_l1.yaml"
    pickle_filename = f"outputs/gemm_l1/{model}-saved_list_of_cmes.pickle"
    dump_folder = "outputs/gemm_l1/"

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
        dump_folder=dump_folder,
        lpf_limit=3,
    )

cme: CostModelEvaluation = answer[0][1][0][0]
print_mapping(cme)
mem_names = [ml.memory_instance.name for ml in cme.mem_level_list]
stall_slacks = cme.stall_slack_comb_collect
print("Stall and slack per port of each memory instance:")
for mem_name, ports_ss in zip(mem_names, stall_slacks):
    print(f"  {mem_name}: {ports_ss}")
print(
    f"Latency: {cme.latency_total2:.3e} (bd: ideal -> {cme.ideal_temporal_cycle}, stall -> {cme.latency_total0 - cme.ideal_temporal_cycle} onload -> {cme.latency_total1 - cme.latency_total0}, offload -> {cme.latency_total2 - cme.latency_total1})"
)
