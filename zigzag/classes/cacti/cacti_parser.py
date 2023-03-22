import yaml
import os
import subprocess

import logging
logger = logging.getLogger(__name__)

CACTI_PATH = './zigzag/classes/cacti'

class CactiParser:
    def __init__(self):
        pass

    def item_exists(self, mem_type, size, r_bw, r_port, w_port, rw_port, bank):
        memory_pool_file = open(f'{CACTI_PATH}/cacti-master/example_mem_pool.yaml')
        memory_pool = yaml.full_load(memory_pool_file)
        
        if memory_pool != None:
            for instance in memory_pool:

                IO_bus_width = int(memory_pool[instance]['IO_bus_width'])
                area = memory_pool[instance]['area']
                bank_count = int(memory_pool[instance]['bank_count'])
                read_cost = memory_pool[instance]['cost']['read_word'] * 1000
                write_cost = memory_pool[instance]['cost']['write_word'] * 1000
                ex_rd_port = int(memory_pool[instance]['ex_rd_port'])
                ex_wr_port = int(memory_pool[instance]['ex_wr_port'])
                rd_wr_port = int(memory_pool[instance]['rd_wr_port'])
                cache_size = int(memory_pool[instance]['size_bit'])

                if (size == cache_size) and (IO_bus_width == r_bw) and (r_port == ex_rd_port) and (w_port == ex_wr_port) and (rw_port == rd_wr_port):
                    return True

        return False

    def create_item(self, mem_type, size, r_bw, r_port, w_port, rw_port, bank):
        # print("No match in Cacti memory pool found!", size, r_bw, r_port, w_port, rw_port, bank)
        os.chdir(f'{CACTI_PATH}/cacti-master/')

        p = subprocess.call(['python', 'cacti_top.py', 
                            '--mem_type', str(mem_type),
                            '--cache_size', str(int(size/8)),
                            '--IO_bus_width', str(r_bw), 
                            '--ex_rd_port', str(r_port), 
                            '--ex_wr_port', str(w_port),
                            '--rd_wr_port', str(rw_port),
                            '--bank_count', str(bank)])

        os.chdir("../../../..")

    def get_item(self, mem_type, size, r_bw, r_port, w_port, rw_port, bank):  
        logger.info(f"Extracting memory costs with CACTI for size = {size} and r_bw = {r_bw}")

        if mem_type == 'rf':
            mem_type = 'sram'
            size = int(size * 128)
            r_bw = int(r_bw)

            print("mem_type is register file! Changed to 'sram' and size", size, "instead of", int(size/128))

        if not self.item_exists(mem_type, size, r_bw, r_port, w_port, rw_port, bank):
            self.create_item(mem_type, size, r_bw, r_port, w_port, rw_port, bank) 

        memory_pool_file = open(f'{CACTI_PATH}/cacti-master/example_mem_pool.yaml')
        memory_pool = yaml.full_load(memory_pool_file)
        
        if memory_pool != None:
            for instance in memory_pool:

                IO_bus_width = int(memory_pool[instance]['IO_bus_width'])
                area = memory_pool[instance]['area']
                bank_count = int(memory_pool[instance]['bank_count'])
                read_cost = memory_pool[instance]['cost']['read_word'] * 1000
                write_cost = memory_pool[instance]['cost']['write_word'] * 1000
                ex_rd_port = int(memory_pool[instance]['ex_rd_port'])
                ex_wr_port = int(memory_pool[instance]['ex_wr_port'])
                rd_wr_port = int(memory_pool[instance]['rd_wr_port'])
                cache_size = int(memory_pool[instance]['size_bit'])
                memory_type = memory_pool[instance]['memory_type']

                if (mem_type == memory_type) and (size == cache_size) and (IO_bus_width == r_bw) and (r_port == ex_rd_port) and (w_port == ex_wr_port) and (rw_port == rd_wr_port):
                    # print("Memory instance found in Cacti memory pool!", cache_size, IO_bus_width, ex_rd_port, ex_wr_port, rd_wr_port, bank_count, read_cost, write_cost)
                    return cache_size, IO_bus_width, IO_bus_width, read_cost, write_cost, area, bank_count, ex_rd_port, ex_wr_port, rd_wr_port

        # should be never reached
        raise ModuleNotFoundError(f"No match in Cacti memory pool found {size=}, {r_bw=}, {r_port=}, {w_port=}, {rw_port=}, {bank=}")
