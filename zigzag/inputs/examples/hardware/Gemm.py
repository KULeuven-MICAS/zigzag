import os
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core


def memory_hierarchy_dut(multiplier_array, visualize=False):
    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """


    reg_O = MemoryInstance(
        name="rf_32b_O",
        size=32,
        r_bw=32,
        w_bw=32,
        r_cost=0.02,
        w_cost=0.02,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )
    l1 = MemoryInstance(
        name="l1",
        size=32*128*64, # 32 banks of 128 words at 64 bit/word
        r_bw=2048,
        w_bw=2048,
        r_cost=22.9,  # TODO
        w_cost=52.01,  # TODO
        area=0,
        r_port=0,
        w_port=0,
        rw_port=1,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )
    l3 = MemoryInstance(
        name="l3",
        size=10000000000,
        r_bw=512,
        w_bw=512,
        r_cost=700,
        w_cost=750,
        area=0,
        r_port=0,
        w_port=0,
        rw_port=1,
        latency=1,
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    ################################## O buffer ###################################
    # There are 8x8 reg_O1, serving across the dim which unrolls K which is D3
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O,
        operands=("O",),
        port_alloc=(
            {
                "fh": "w_port_1",
                "tl": "r_port_1",
                "fl": "w_port_2",
                "th": "r_port_2",
            },
        ),
        served_dimensions={(0, 0, 1,)},
    )

    ################################## l1 ###################################
    memory_hierarchy_graph.add_memory(
        memory_instance=l1,
        operands=("I1", "I2", "O",),
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
        served_dimensions="all",
    )

    ################################## l3 ###################################
    memory_hierarchy_graph.add_memory(
        memory_instance=l3,
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
        served_dimensions="all",
    )
    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def multiplier_array_dut():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.04
    multiplier_area = 1
    dimensions = {
        "D1": 8,
        "D2": 8,
        "D3": 8,
    }  # {'D1': ('M', 8), 'D2': ('N', 8), 'D3': ('K', 4)}

    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def cores_dut():
    multiplier_array1 = multiplier_array_dut()
    memory_hierarchy1 = memory_hierarchy_dut(multiplier_array1)

    core1 = Core(1, multiplier_array1, memory_hierarchy1)

    return {core1}


cores = cores_dut()
acc_name = "Gemm Accelerator"
accelerator = Accelerator(acc_name, cores)
