"""! This file models a trivial workload on an accelerator with trivial MAC array and a memory
hierarchy similar to Edge_TPU_like but with unit costs. The goal is to be able to manually track
data movements based on the results.
"""

import pickle
import sys
from extended_int import int_inf


sys.path.append("../zigzag")
from zigzag.stages import *  #  pylint: disable=C0413
from zigzag.cost_model.cost_model import CostModelEvaluation  # pylint: disable=C0413
from zigzag.hardware.architecture.Core import Core  # pylint: disable=C0413
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy  # pylint: disable=C0413
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance  # pylint: disable=C0413
from zigzag.hardware.architecture.operational_array import MultiplierArray  # pylint: disable=C0413
from zigzag.hardware.architecture.operational_unit import Multiplier  # pylint: disable=C0413
from zigzag.hardware.architecture.Accelerator import Accelerator  # pylint: disable=C0413


from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)  # pylint: disable=C0413
from zigzag.visualization.results.print_mapping import print_mapping  # pylint: disable=C0413
from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph  # pylint: disable=C0413

workload = {
    0: {  # fully connected
        "operator_type": "matmul",
        "equation": "O[p][r]+=I[p][q]*W[q][r]",
        "loop_dim_size": {
            "P": 1024,
            "Q": 1024,
            "R": 1024,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [], "W": []},
        "constant_operands": ["W"],
    },
}

# Define accelerator
multiplier = Multiplier(input_precision=[8, 8], energy_cost=1, area=1)
multiplier_array = MultiplierArray(multiplier, dimensions={"D1": 8, "D2": 8, "D3": 4, "D4": 4})

reg_IW1 = MemoryInstance(
    name="rf_1B",
    size=8,
    r_bw=8,
    w_bw=8,
    r_cost=1,
    w_cost=1,
    area=0,
    r_port=1,
    w_port=1,
    rw_port=0,
    latency=1,
)
reg_O1 = MemoryInstance(
    name="rf_2B",
    size=16,
    r_bw=16,
    w_bw=16,
    r_cost=1,
    w_cost=1,
    area=0,
    r_port=2,
    w_port=2,
    rw_port=0,
    latency=1,
)
x = 8
sram_32KB_512_1r_1w = MemoryInstance(
    name="sram_32KB",
    size=1000 * 8,
    r_bw=8 * x,
    w_bw=8 * x,
    r_cost=1 * x,
    w_cost=1 * x,
    area=0,
    r_port=1,
    w_port=1,
    rw_port=0,
    latency=1,
)
x = 8
sram_2M_with_16_128K_bank_128_1r_1w = MemoryInstance(
    name="sram_2MB",
    size=131072 * 16 * 8,
    r_bw=8 * x,
    w_bw=8 * x,
    r_cost=1 * x,
    w_cost=1 * x,
    area=0,
    r_port=1,
    w_port=1,
    rw_port=0,
    latency=1,
)
dram = MemoryInstance(
    name="dram",
    size=10000000000,
    r_bw=8,
    w_bw=8,
    r_cost=1,
    w_cost=1,
    area=0,
    r_port=0,
    w_port=0,
    rw_port=1,
    latency=1,
)

memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)
memory_hierarchy_graph.add_memory(
    memory_instance=reg_IW1,
    operands=("I2",),
    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
    # served_dimensions=("D1",),
    served_dimensions=(),
)
memory_hierarchy_graph.add_memory(
    memory_instance=reg_O1,
    operands=("O",),
    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
    served_dimensions=(),
)
memory_hierarchy_graph.add_memory(
    memory_instance=sram_32KB_512_1r_1w,
    operands=("I2",),
    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
    served_dimensions=("D1", "D2", "D3", "D4"),
)
memory_hierarchy_graph.add_memory(
    memory_instance=sram_2M_with_16_128K_bank_128_1r_1w,
    operands=("I1", "O"),
    port_alloc=(
        {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
        {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_1", "th": "r_port_1"},
    ),
    served_dimensions=("D1", "D2", "D3", "D4"),
)
memory_hierarchy_graph.add_memory(
    memory_instance=dram,
    operands=("I1", "I2", "O"),
    port_alloc=(
        {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
        {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
        {
            "fh": "rw_port_1",
            "tl": "rw_port_1",
            "fl": "rw_port_1",
            "th": "rw_port_1",
        },
    ),
    served_dimensions=("D1", "D2", "D3", "D4"),
)


mem_hierarchy = memory_hierarchy_graph
core = Core(1, multiplier_array, mem_hierarchy)
accelerator = Accelerator("trivial_accel", {core})
mapping = {
    "matmul": {
        "core_allocation": 1,
        "spatial_mapping": {
            # "D1": ("P", 8),
            # "D2": ("P", 8),
            # "D3": ("P", 4),
            # "D4": ("Q", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
}


model = "trivial"

dump_filename_pattern = f"outputs/TPU-{model}-layer_?.json"
pickle_filename = f"outputs/TPU-{model}-saved_list_of_cmes.pickle"


###### Run stages ######


# opt_stage = MinimalEDPStage
# workload_parser_stage = WorkloadParserStage

# mainstage = MainStage(
#     [  # Initialize the MainStage as entry point
#         workload_parser_stage,  # Parse the ONNX Model into the workload
#         AcceleratorParserStage,  # Parse the accelerator module/passthrough given accelerator
#         SimpleSaveStage,  # Save the summed CME energy and latency to a json
#         PickleSaveStage,  # Save all received CMEs in a list to a pickle file
#         SumStage,  # Sum up the received best CME across all layers of the workload
#         # SearchUnusedMemoryStage,  # Search for unused memory instance
#         WorkloadStage,  # Iterate through the different layers in the workload
#         # RemoveUnusedMemoryStage,  # Remove unused memory instance
#         CompleteSaveStage,  # Save each processed layer to a json
#         opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
#         SpatialMappingGeneratorStage,  # Generate multiple spatial mappings (SM)
#         opt_stage,  # Reduce all CMEs, returning minimal energy/latency one
#         LomaStage,  # Generate multiple temporal mappings (TM)
#         # TemporalOrderingConversionStage,  # Based on the fixed temporal mapping order, generate one temporal mapping (TM)
#         CostModelStage,  # Evaluate generated SM and TM through cost model
#     ],
#     accelerator=accelerator,  # required by AcceleratorParserStage
#     workload=workload,  # required by workload_parser_stage
#     mapping=mapping,  # required by workload_parser_stage
#     dump_filename_pattern=dump_filename_pattern,  # output file save pattern
#     pickle_filename=pickle_filename,  # filename for pickled list of cmes
#     loma_lpf_limit=6,  # required by LomaStage
#     loma_show_progress_bar=True,
#     # If we need access the same input data multiple times from the innermost memory level and the data size is smaller than the memory read bw,
#     # take into account only one-time access cost (assume the data can stay at the output pins of the memory as long as it is needed).
#     # By default, if the parameter is not defined, it will be set as False internally.
#     access_same_data_considered_as_no_access=True,
#     enable_mix_spatial_mapping_generation=False,  # enable auto-generation of mix spatial mapping
#     enable_weight_diagonal_mapping=True,
#     nb_mappings_generated=3,
# )
# Launch the MainStage
# cmes = mainstage.run()
# energy, latency = cmes[0][0].energy_total, cmes[0][0].latency_total2


energy, latency, _ = api.get_hardware_performance_zigzag(
    workload=workload,
    accelerator=accelerator,
    mapping=mapping,
    opt="latency",
    dump_filename_pattern=dump_filename_pattern,
    pickle_filename=pickle_filename,
)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")


with open(pickle_filename, "rb") as fp:
    cme_for_all_layers: list[CostModelEvaluation] = pickle.load(fp)


bar_plot_cost_model_evaluations_breakdown(cme_for_all_layers, save_path="outputs/plot_breakdown.png")

visualize_memory_hierarchy_graph(
    cme_for_all_layers[0].accelerator.cores[0].memory_hierarchy,
    save_path="outputs/mem_hierarchy.png",
)

for cme in cme_for_all_layers:
    print_mapping(cme)
    print(cme.spatial_mapping_dict_int)
