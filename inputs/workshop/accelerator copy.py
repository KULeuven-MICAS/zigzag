import os
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.memory_level import MemoryLevel
from classes.hardware.architecture.operational_unit import Multiplier
from classes.hardware.architecture.operational_array import MultiplierArray
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.accelerator import Accelerator
from classes.hardware.architecture.core import Core


def multiplier_array():
    """ Multiplier array variables """
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.04
    multiplier_area = 1
    dimensions = {'D1': 4}

    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    ''' size = #bit '''

    gb = MemoryInstance(name="sram_32B", size=32*8, r_bw=32, w_bw=64, r_cost=20, w_cost=25, area=0)
    dram = MemoryInstance(name="dram", size=10000000000, r_bw=64, w_bw=64, r_cost=700, w_cost=750, area=0)

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)
    memory_hierarchy_graph.add_memory(memory_instance=gb, operands=('I1', 'I2', 'O'), served_dimensions='all')
    memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'), served_dimensions='all')


    from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def cores():
    multiplier_array1 = multiplier_array()
    memory_hierarchy1 = memory_hierarchy(multiplier_array1)

    core1 = Core(1, multiplier_array1, memory_hierarchy1)

    return {core1}


cores = cores()
global_buffer = None
acc_name = "MyAccelerator"
accelerator = Accelerator(acc_name, cores, global_buffer)

a = 1
