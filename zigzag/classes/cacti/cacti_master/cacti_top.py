import yaml
import os
import argparse

from zigzag.classes.cacti.cacti_master.cacti_config_creator import CactiConfig


parser = argparse.ArgumentParser()
parser.add_argument('--mem_type')
parser.add_argument('--cache_size')
parser.add_argument('--IO_bus_width')
parser.add_argument('--ex_rd_port')
parser.add_argument('--ex_wr_port')
parser.add_argument('--rd_wr_port')
parser.add_argument('--bank_count')
parser.add_argument('--mem_pool_path')
args = parser.parse_args()


mem_pool_path = args.mem_pool_path
cacti_master_path = os.path.dirname(mem_pool_path)
print(f"{cacti_master_path=}")

self_gen_folder_name = 'self_gen'
self_gen_path = os.path.join(cacti_master_path, self_gen_folder_name)
if not os.path.isdir(self_gen_path):
    os.mkdir(self_gen_path)

os.system(f'rm -rf {self_gen_path}/*')
C = CactiConfig()

'''Function 1: set default value'''
# C.change_default_value(['technology'], [0.090])

'''Function 2: use default values to run CACTI'''
# C.cacti_auto(['default'], file_path + '/cache.cfg')

'''Function 3: use user-defined + default values to run CACTI'''
# C.cacti_auto(['single', [['technology', 'cache_size'],[0.022, 524288]]], file_path+'/cache.cfg')

'''Function 4: sweep any one variable using the default list & other default value'''
# C.cacti_auto(['sweep', ['IO_bus_width']], file_path+'/cache.cfg')

''' Combining Function 1 & 4 to do multi-variable sweep '''

mem_type = args.mem_type

if mem_type == 'sram':
    mem_type = '"ram"'
else:
    mem_type == '"main memory"'

cache_size = args.cache_size
IO_bus_width = args.IO_bus_width
ex_rd_port = args.ex_rd_port
ex_wr_port = args.ex_wr_port
rd_wr_port = args.rd_wr_port
bank_count = args.bank_count

technology = 0.090


C.cacti_auto(['single', [['mem_type', 'cache_size', 'IO_bus_width', 'ex_rd_port', 'ex_wr_port', 'rd_wr_port', 'technology'],[mem_type, cache_size, IO_bus_width, ex_rd_port, ex_wr_port, rd_wr_port, technology]]], cacti_master_path, f'{self_gen_path}/cache.cfg')

result = {}
with open('%s/cache.cfg.out' % self_gen_path, 'r') as fp:
    raw_result = fp.readlines()
    for ii, each_line in enumerate(raw_result):
        if ii == 0:
            attribute_list = each_line.split(',')
            for each_attribute in attribute_list:
                result[each_attribute] = []
        else:
            for jj, each_value in enumerate(each_line.split(',')):
                try:
                    result[attribute_list[jj]].append(float(each_value))
                except:
                    pass


for i in range(len(result[' Capacity (bytes)'])):
    size_byte = result[' Capacity (bytes)'][i]
    area = result[' Area (mm2)'][i]
    read_word = result[' Dynamic read energy (nJ)'][i]
    write_word = result[' Dynamic write energy (nJ)'][i]
    mem_bw = result[' Output width (bits)'][i]
    utilization_rate = 0.7

    if mem_type == '"ram"':
        mem_type = 'sram'
    else:
        mem_type = 'dram'

    mem_name = str(int(size_byte)) + '_Byte_' + str(int(mem_bw)) + '_BW_' + str(ex_rd_port) + '_' + str(ex_wr_port) + '_' + str(rd_wr_port)

    new_result = {'%s' % mem_name: {
        'size_byte': int(size_byte),
        'size_bit': int(size_byte * 8),
        'area': area*2,
        'cost': {'read_word': read_word, 'write_word': write_word},
        'IO_bus_width': int(mem_bw),
        'ex_rd_port': ex_rd_port,
        'ex_wr_port': ex_wr_port,
        'rd_wr_port': rd_wr_port,
        'bank_count': 1, 
        'memory_type': mem_type
    }}
    with open(mem_pool_path, 'a+') as fp:
        yaml.dump(new_result, fp)
        fp.write('\n')

