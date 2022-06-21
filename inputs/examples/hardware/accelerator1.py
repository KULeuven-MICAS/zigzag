from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.memory_level import MemoryLevel
from classes.hardware.architecture.operational_unit import Multiplier
from classes.hardware.architecture.operational_array import MultiplierArray
from classes.hardware.architecture.memory_instance import MemoryInstance
from classes.hardware.architecture.accelerator import Accelerator
from classes.hardware.architecture.core import Core


def memory_hierarchy_1(multiplier_array):
    """Memory hierarchy variables"""
    rf1 = MemoryInstance(name="rf_64B_BW_16b", size=512, bw=(8, 8), cost=(1.5, 1.8), area=0.3, bank=1,
                         random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    rf2 = MemoryInstance(name="rf_16B_BW_16b", size=128, bw=(16, 16), cost=(3.5, 4.8), area=0.95, bank=1,
                         random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    lb1 = MemoryInstance(name="sram_64KB_BW_112b", size=524288, bw=(112, 112), cost=(15, 18), area=3, bank=1,
                         random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    lb2 = MemoryInstance(name="sram_8KB_BW_8b", size=65536, bw=(8, 8), cost=(5, 8), area=0.7, bank=1,
                         random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)
    lb3 = MemoryInstance(name="sram_24KB_BW_24b", size=65536, bw=(24, 24), cost=(5, 8), area=2.2, bank=3,
                         random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1)
    gb = MemoryInstance(name="sram_256KB_BW_384b", size=2097152, bw=(384, 384), cost=(10, 15), area=25, bank=4,
                        random_bank_access=True, r_port=1, w_port=1, rw_port=0, latency=1)
    dram = MemoryInstance(name="dram", size=10000000000, bw=(24, 24), cost=(100, 150), area=25, bank=1,
                          random_bank_access=False, r_port=1, w_port=1, rw_port=0, latency=1)

    W_Reg = MemoryLevel(memory_instance=rf2, operands=('I1',), served_dimensions=(set(),),
                        operational_array=multiplier_array)
    I_Reg = MemoryLevel(memory_instance=rf2, operands=('I2',), served_dimensions=(set(),),
                        operational_array=multiplier_array)
    O_Reg = MemoryLevel(memory_instance=rf1, operands=('O',), served_dimensions=(set(),),
                        operational_array=multiplier_array)

    W_LB = MemoryLevel(memory_instance=lb3, operands=('I1',), served_dimensions=({(1, 0, 0), (0, 1, 0)},),
                       operational_array=multiplier_array)
    I_LB = MemoryLevel(memory_instance=lb2, operands=('I2',), served_dimensions=({(1, 1, 0), (0, 0, 1)},),
                       operational_array=multiplier_array)
    O_LB = MemoryLevel(memory_instance=lb1, operands=('O',), served_dimensions=({(1, 0, 0), (0, 1, 0)},),
                       operational_array=multiplier_array)

    IO_GB = MemoryLevel(memory_instance=gb, operands=('I2', 'O'), served_dimensions=('all', 'all'),
                        operational_array=multiplier_array)
    DRAM = MemoryLevel(memory_instance=dram, operands=('I1', 'I2', 'O'), served_dimensions=('all', 'all', 'all'),
                       operational_array=multiplier_array)

    nodes = [W_Reg, I_Reg, O_Reg, W_LB, I_LB, O_LB, IO_GB, DRAM]

    memory_hierarchy_dict = {'I1': (W_Reg, W_LB, None, DRAM),
                             'I2': (I_Reg, I_LB, IO_GB, DRAM),
                             'O': (O_Reg, O_LB, IO_GB, DRAM)}

    memory_hierarchy = MemoryHierarchy(name="Eyeriss-like 3D", operational_array=multiplier_array, nodes=nodes,
                                       hierarchy_dict=memory_hierarchy_dict)
    return memory_hierarchy


def memory_hierarchy_2(multiplier_array):
    """Memory hierarchy variables"""
    rf1 = MemoryInstance(name="rf_64B_BW_16b", size=512, bw=(8, 8), cost=(1.5, 1.8), area=0.3, bank=1,
                         random_bank_access=False, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    rf2 = MemoryInstance(name="rf_16B_BW_16b", size=128, bw=(16, 16), cost=(3.5, 4.8), area=0.95, bank=1,
                         random_bank_access=False, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    lb1 = MemoryInstance(name="sram_64KB_BW_112b", size=524288, bw=(112, 112), cost=(15, 18), area=3, bank=1,
                         random_bank_access=False, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    lb2 = MemoryInstance(name="sram_8KB_BW_8b", size=65536, bw=(8, 8), cost=(5, 8), area=0.7, bank=1,
                         random_bank_access=False, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    lb3 = MemoryInstance(name="sram_24KB_BW_24b", size=65536, bw=(24, 24), cost=(5, 8), area=2.2, bank=3,
                         random_bank_access=True, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    gb = MemoryInstance(name="sram_256KB_BW_384b", size=2097152, bw=(384, 384), cost=(10, 15), area=25, bank=4,
                        random_bank_access=True, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    dram = MemoryInstance(name="dram", size=10000000000, bw=(24, 24), cost=(100, 150), area=25, bank=1,
                          random_bank_access=False, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)

    W_Reg = MemoryLevel(memory_instance=rf2, operands=('I1',), served_dimensions=(set(),),
                        operational_array=multiplier_array)
    I_Reg = MemoryLevel(memory_instance=rf2, operands=('I2',), served_dimensions=(set(),),
                        operational_array=multiplier_array)
    O_Reg = MemoryLevel(memory_instance=rf1, operands=('O',), served_dimensions=(set(),),
                        operational_array=multiplier_array)

    W_LB = MemoryLevel(memory_instance=lb3, operands=('I1',), served_dimensions=({(1, 0)},),
                       operational_array=multiplier_array)
    I_LB = MemoryLevel(memory_instance=lb2, operands=('I2',), served_dimensions=({(1, 1)},),
                       operational_array=multiplier_array)
    O_LB = MemoryLevel(memory_instance=lb1, operands=('O',), served_dimensions=({(0, 1)},),
                       operational_array=multiplier_array)

    IO_GB = MemoryLevel(memory_instance=gb, operands=('I2', 'O'), served_dimensions=('all', 'all'),
                        operational_array=multiplier_array)
    DRAM = MemoryLevel(memory_instance=dram, operands=('I1', 'I2', 'O'), served_dimensions=('all', 'all', 'all'),
                       operational_array=multiplier_array)

    nodes = [W_Reg, I_Reg, O_Reg, W_LB, I_LB, O_LB, IO_GB, DRAM]

    memory_hierarchy_dict = {'I1': (W_Reg, W_LB, None, DRAM),
                             'I2': (I_Reg, I_LB, IO_GB, DRAM),
                             'O': (O_Reg, O_LB, IO_GB, DRAM)}

    memory_hierarchy = MemoryHierarchy(name="Eyeriss-like 2D", operational_array=multiplier_array, nodes=nodes,
                                       hierarchy_dict=memory_hierarchy_dict)
    return memory_hierarchy


def multiplier_array_1():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 3, 'D3': 4}
    operand_spatial_sharing = {'I1': {(1, 0, 0)},
                       'O': {(0, 1, 0)},
                       'I2': {(0, 0, 1), (1, 1, 0)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


def multiplier_array_2():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 12}
    operand_spatial_sharing = {'I1': {(1, 0)},
                       'O': {(0, 1)},
                       'I2': {(1, 1)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


def cores():
    multiplier_array1 = multiplier_array_1()
    memory_hierarchy1 = memory_hierarchy_1(multiplier_array1)

    multiplier_array2 = multiplier_array_2()
    memory_hierarchy2 = memory_hierarchy_2(multiplier_array2)

    core1 = Core(1, multiplier_array1, memory_hierarchy1)
    core2 = Core(2, multiplier_array2, memory_hierarchy2)

    return {core1, core2}


if __name__ == "__main__":
    cores = cores()
    global_buffer = MemoryInstance(name="sram_256KB_BW_384b", size=2097152, bw=(384, 384), cost=(10, 15), area=25,
                                   bank=4,
                                   random_bank_access=True, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    accelerator = Accelerator("accelerator1", cores, global_buffer)
    a=1

