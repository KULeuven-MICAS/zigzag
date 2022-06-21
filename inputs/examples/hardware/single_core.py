from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.memory_level import MemoryLevel
from classes.hardware.architecture.operational_unit import Multiplier
from classes.hardware.architecture.operational_array import MultiplierArray
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.accelerator import Accelerator
from classes.hardware.architecture.core import Core
from visualization.graph.dnn import visualize_dnn_graph


def create_memory_hierarchy(multiplier_array):
    """Memory hierarchy variables"""
    rf1 = MemoryInstance(name="rf_64B_BW_16b", size=512, r_bw=8, w_bw=8, r_cost=1.5, w_cost=1.8, area=0.3, bank=1,
                         random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    rf2 = MemoryInstance(name="rf_16B_BW_16b", size=128, r_bw=16, w_bw=16, w_cost=3.5, r_cost=4.8, area=0.95, bank=1,
                         random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    gb = MemoryInstance(name="sram_4KB_BW_8b", size=32768, r_bw=8, w_bw=8, r_cost=5, w_cost=8, area=0.7, bank=1,
                        random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    dram = MemoryInstance(name="dram", size=10000000000, r_bw=24, w_bw=24, r_cost=100, w_cost=150, area=25, bank=1,
                          random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)

    # I1_Reg = MemoryLevel(memory_instance=rf2, operands=('I1',), served_dimensions=(set(),),
    #                      operational_array=multiplier_array)
    # I2_Reg = MemoryLevel(memory_instance=rf2, operands=('I2',), served_dimensions=(set(),),
    #                      operational_array=multiplier_array)
    # O_Reg = MemoryLevel(memory_instance=rf1, operands=('O',), served_dimensions=(set(),),
    #                     operational_array=multiplier_array)
    #
    # GB = MemoryLevel(memory_instance=gb, operands=('I1', 'O'), served_dimensions=('all', 'all'),
    #                  operational_array=multiplier_array)
    # DRAM = MemoryLevel(memory_instance=dram, operands=('I1', 'I2', 'O'), served_dimensions=('all', 'all', 'all'),
    #                    operational_array=multiplier_array)
    #
    # nodes = [I1_Reg, I2_Reg, O_Reg, GB, DRAM]
    #
    # mem_hierarchy_dict = {'I1': (I1_Reg, GB, DRAM),
    #                          'I2': (I2_Reg, None, DRAM),
    #                          'O': (O_Reg, GB, DRAM)}
    #
    # memory_hierarchy = MemoryHierarchy(name="Eyeriss-like 2D", operational_array=multiplier_array, nodes=nodes,
    #                                    hierarchy_dict=mem_hierarchy_dict)

    memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)
    memory_hierarchy_graph.add_memory(memory_instance=rf1, operands=('I1',), served_dimensions=set())  # {(1, 0)})
    memory_hierarchy_graph.add_memory(memory_instance=rf1, operands=('I2',), served_dimensions=set())  # {(0, 1)})
    memory_hierarchy_graph.add_memory(memory_instance=rf1, operands=('O',), served_dimensions=set())
    memory_hierarchy_graph.add_memory(memory_instance=gb, operands=('I1', 'O'), served_dimensions='all')
    memory_hierarchy_graph.add_memory(memory_instance=dram, operands=('I1', 'I2', 'O'), served_dimensions='all')
    from visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
    # visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph


def create_multiplier_array():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 4}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array


def cores():
    multiplier_array = create_multiplier_array()
    memory_hierarchy = create_memory_hierarchy(multiplier_array)

    core = Core(1, multiplier_array, memory_hierarchy)

    return {core}


cores = cores()
global_buffer = None
accelerator = Accelerator("accelerator_single_core", cores, global_buffer)
a=1
