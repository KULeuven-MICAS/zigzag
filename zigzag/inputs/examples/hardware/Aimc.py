"""
Analog In-Memory Computing (AIMC) core definition
This example will define an AIMC core with a single macro, sized 32 rows x 32 columns.
Supported operand precision: 8 bit
Technology node: 28 nm
The architecture hierarchy looks like:
                   ------- dram (I, W, O) ----------
                   |                               |
                  sram (I, O)                 cell_group (W)
                   |-> reg_I1 (I) --> imc_array <--|
                   |                       |
                   | <---> reg_O1 (O) <--> |
"""

import os, math
import random

from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.ImcArray import ImcArray
from zigzag.hardware.architecture.get_cacti_cost import (
    get_w_cost_per_weight_from_cacti,
)
from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost


def memory_hierarchy_dut(imc_array, visualize=False):
    """[OPTIONAL] Get w_cost of imc cell group from CACTI if required"""
    cacti_path = "zigzag/classes/cacti/cacti_master"
    tech_param = imc_array.unit.logic_unit.tech_param
    hd_param = imc_array.unit.hd_param
    dimensions = imc_array.unit.dimensions
    output_precision = hd_param["input_precision"] + hd_param["weight_precision"]
    if hd_param["enable_cacti"]:
        # unit: pJ/weight writing
        w_cost_per_weight_writing = get_w_cost_per_weight_from_cacti(cacti_path, tech_param, hd_param, dimensions)
    else:
        w_cost_per_weight_writing = hd_param["w_cost_per_weight_writing"]  # user-provided value (unit: pJ/weight)

    # Memory hierarchy variables
    #   size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy)
    cell_group = MemoryInstance(
        name="cell_group",
        size=hd_param["weight_precision"] * hd_param["group_depth"],
        r_bw=hd_param["weight_precision"],
        w_bw=hd_param["weight_precision"],
        r_cost=0,
        w_cost=w_cost_per_weight_writing,  # unit: pJ/weight
        area=0,  # this area is already included in imc_array
        r_port=0,  # no standalone read port
        w_port=0,  # no standalone write port
        rw_port=1,  # 1 port for both reading and writing
        latency=0,  # no extra clock cycle required
    )
    reg_I1 = MemoryInstance(
        name="rf_I1",
        size=hd_param["input_precision"],
        r_bw=hd_param["input_precision"],
        w_bw=hd_param["input_precision"],
        r_cost=0,
        w_cost=tech_param["dff_cap"] * (tech_param["vdd"] ** 2) * hd_param["input_precision"],  # pJ/access
        area=tech_param["dff_area"] * hd_param["input_precision"],  # mm^2
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O1 = MemoryInstance(
        name="rf_O1",
        size=output_precision,
        r_bw=output_precision,
        w_bw=output_precision,
        r_cost=0,
        w_cost=tech_param["dff_cap"] * (tech_param["vdd"] ** 2) * output_precision,  # pJ/access
        area=tech_param["dff_area"] * output_precision,  # mm^2
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    sram_size = 256 * 1024  # unit: byte
    sram_bw = max(
        imc_array.unit.bl_dim_size * hd_param["input_precision"] * imc_array.unit.nb_of_banks,
        imc_array.unit.wl_dim_size * output_precision * imc_array.unit.nb_of_banks,
    )
    ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(
        cacti_path,
        tech_param["tech_node"],
        "sram",
        sram_size,
        sram_bw,
        hd_hash=str(hash((sram_size, sram_bw, random.randbytes(8)))),
    )
    sram_256KB_256_3r_3w = MemoryInstance(
        name="sram_256KB",
        size=sram_size * 8,  # byte -> bit
        r_bw=sram_bw,
        w_bw=sram_bw,
        r_cost=sram_r_cost,
        w_cost=sram_w_cost,
        area=sram_area,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=sram_bw // 16,  # assume there are 16 sub-banks
    )

    #######################################################################################################################

    dram_size = 1 * 1024 * 1024 * 1024  # unit: byte
    dram_ac_cost_per_bit = 3.7  # unit: pJ/bit
    dram_bw = imc_array.unit.wl_dim_size * hd_param["weight_precision"] * imc_array.unit.nb_of_banks
    dram_100MB_32_3r_3w = MemoryInstance(
        name="dram_1GB",
        size=dram_size * 8,  # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit * dram_bw,  # pJ/access
        w_cost=dram_ac_cost_per_bit * dram_bw,  # pJ/access
        area=0,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=imc_array)

    # fh: from high = wr_in_by_high
    # fl: from low = wr_in_by_low
    # th: to high = rd_out_to_high
    # tl: to low = rd_out_to_low
    memory_hierarchy_graph.add_memory(
        memory_instance=cell_group,
        operands=("I2",),
        port_alloc=({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_I1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions=("D1",),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O1,
        operands=("O",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
        served_dimensions=("D2",),
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_256_3r_3w,
        operands=("I1", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions=("D1", "D2", "D3"),
    )

    ####################################################################################################################

    memory_hierarchy_graph.add_memory(
        memory_instance=dram_100MB_32_3r_3w,
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions=("D1", "D2", "D3"),
    )

    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def imc_array_dut():
    """Multiplier array variables"""
    tech_param = {  # 28nm
        "tech_node": 0.028,  # unit: um
        "vdd": 0.9,  # unit: V
        "nd2_cap": 0.7 / 1e3,  # unit: pF
        "xor2_cap": 0.7 * 1.5 / 1e3,  # unit: pF
        "dff_cap": 0.7 * 3 / 1e3,  # unit: pF
        "nd2_area": 0.614 / 1e6,  # unit: mm^2
        "xor2_area": 0.614 * 2.4 / 1e6,  # unit: mm^2
        "dff_area": 0.614 * 6 / 1e6,  # unit: mm^2
        "nd2_dly": 0.0478,  # unit: ns
        "xor2_dly": 0.0478 * 2.4,  # unit: ns
        # "dff_dly":  0.0478*3.4,        # unit: ns
    }
    hd_param = {
        "pe_type": "in_sram_computing",  # for in-memory-computing. Digital core for different values.
        "imc_type": "analog",  # "digital" or "analog"
        "input_precision": 8,  # activation precision
        "weight_precision": 8,  # weight precision
        "input_bit_per_cycle": 2,  # nb_bits of input/cycle (treated as DAC resolution)
        "group_depth": 1,  # #cells/multiplier
        "adc_resolution": 8,  # ADC resolution
        "wordline_dimension": "D1",  # hardware dimension where wordline is (corresponds to the served dimension of input regs)
        "bitline_dimension": "D2",  # hardware dimension where bitline is (corresponds to the served dimension of output regs)
        "enable_cacti": True,  # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
        # Energy of writing weight. Required when enable_cacti is False.
        # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
    }

    dimensions = {
        "D1": 4,  # wordline dimension
        "D2": 32,  # bitline dimension
        "D3": 1,  # nb_macros (nb_arrays)
    }  # {"D1": ("K", 4), "D2": ("C", 32),}
    hd_param["adc_resolution"] = hd_param["input_bit_per_cycle"] + 0.5 * int(math.log2(dimensions["D2"]))

    aimc_array = ImcArray(tech_param, hd_param, dimensions)

    return aimc_array


def cores_dut():
    imc_array1 = imc_array_dut()
    memory_hierarchy1 = memory_hierarchy_dut(imc_array1)

    core1 = Core(1, imc_array1, memory_hierarchy1)

    return {core1}


cores = cores_dut()
acc_name = os.path.basename(__file__)[:-3]
accelerator = Accelerator(acc_name, cores)
