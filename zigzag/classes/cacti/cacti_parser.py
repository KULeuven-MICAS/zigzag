import yaml
import os
import subprocess

import logging

logger = logging.getLogger(__name__)

##  Class that provides the interface between ZigZag and CACTI.
class CactiParser:

    ## Path of current directory
    cacti_path = os.path.dirname(os.path.realpath(__file__))
    ## Path to cached cacti simulated memories
    MEM_POOL_PATH = f"{cacti_path}/cacti_master/example_mem_pool.yaml"
    ## Path to cacti python script to extract costs
    CACTI_TOP_PATH = f"{cacti_path}/cacti_master/cacti_top.py"

    ## The class constructor
    def __init__(self):
        pass
    
    ## This function checks if the provided memory configuration was already used in the past.
    # @param mem_type
    # @param size
    # @param r_bw
    # @param r_port
    # @param w_port
    # @param rw_port
    # @param bank
    # @param mem_pool_path  Path to cached cacti simulated memories
    # @return True          The requested memory item has been simulated once.
    # @return False         The requested memory item has not been simualted so far.
    def item_exists(
        self,
        mem_type,
        size,
        r_bw,
        r_port,
        w_port,
        rw_port,
        bank,
        mem_pool_path=MEM_POOL_PATH,
    ):
        with open(mem_pool_path, "r") as fp:
            memory_pool = yaml.full_load(fp)

        if memory_pool != None:
            for instance in memory_pool:

                IO_bus_width = int(memory_pool[instance]["IO_bus_width"])
                area = memory_pool[instance]["area"]
                bank_count = int(memory_pool[instance]["bank_count"])
                read_cost = memory_pool[instance]["cost"]["read_word"] * 1000
                write_cost = memory_pool[instance]["cost"]["write_word"] * 1000
                ex_rd_port = int(memory_pool[instance]["ex_rd_port"])
                ex_wr_port = int(memory_pool[instance]["ex_wr_port"])
                rd_wr_port = int(memory_pool[instance]["rd_wr_port"])
                cache_size = int(memory_pool[instance]["size_bit"])

                if (
                    (size == cache_size)
                    and (IO_bus_width == r_bw)
                    and (r_port == ex_rd_port)
                    and (w_port == ex_wr_port)
                    and (rw_port == rd_wr_port)
                ):
                    return True

        return False

    ## This function simulates a new item by calling CACTI7 based on the provided parameters
    # @param mem_type
    # @param size
    # @param r_bw
    # @param r_port
    # @param w_port
    # @param rw_port
    # @param bank
    # @param mem_pool_path  Path to cached cacti simulated memories
    # @param cacti_top_path Path to cacti python script to extract costs
    def create_item(
        self,
        mem_type,
        size,
        r_bw,
        r_port,
        w_port,
        rw_port,
        bank,
        mem_pool_path=MEM_POOL_PATH,
        cacti_top_path=CACTI_TOP_PATH,
    ):
        # print("No match in Cacti memory pool found!", size, r_bw, r_port, w_port, rw_port, bank)
        # os.chdir(f'{CACTI_PATH}/cacti-master/')

        p = subprocess.call(
            [
                "python",
                cacti_top_path,
                "--mem_type",
                str(mem_type),
                "--cache_size",
                str(int(size / 8)),
                "--IO_bus_width",
                str(r_bw),
                "--ex_rd_port",
                str(r_port),
                "--ex_wr_port",
                str(w_port),
                "--rd_wr_port",
                str(rw_port),
                "--bank_count",
                str(bank),
                "--mem_pool_path",
                str(mem_pool_path),
            ]
        )

        if p != 0:
            raise ChildProcessError(
                f"Cacti subprocess call failed with return value {p}."
            )

    ## This functions checks first if the memory with the provided parameters was already simulated once.
    # In case it hasn't been simulated, then it will create a new memory item based on the provided parameters.
    # @param mem_type
    # @param size
    # @param r_bw
    # @param r_port
    # @param w_port
    # @param rw_port
    # @param bank
    # @param mem_pool_path  Path to cached cacti simulated memories
    # @param cacti_top_path Path to cacti python script to extract costs
    def get_item(
        self,
        mem_type,
        size,
        r_bw,
        r_port,
        w_port,
        rw_port,
        bank,
        mem_pool_path=MEM_POOL_PATH,
        cacti_top_path=CACTI_TOP_PATH,
    ):
        if not os.path.exists(cacti_top_path):
            raise FileNotFoundError(f"Cacti top file doesn't exist: {cacti_top_path}.")

        logger.info(
            f"Extracting memory costs with CACTI for size = {size} and r_bw = {r_bw}."
        )

        if mem_type == "rf":
            new_mem_type = "sram"
            new_size = int(size * 128)
            new_r_bw = int(r_bw)
            logger.warning(
                f"Type {mem_type} -> {new_mem_type}. Size {size} -> {new_size}. BW {r_bw} -> {new_r_bw}."
            )
            mem_type = new_mem_type
            size = new_size
            r_bw = new_r_bw

        if not self.item_exists(
            mem_type, size, r_bw, r_port, w_port, rw_port, bank, mem_pool_path
        ):
            self.create_item(
                mem_type,
                size,
                r_bw,
                r_port,
                w_port,
                rw_port,
                bank,
                mem_pool_path,
                cacti_top_path,
            )

        with open(mem_pool_path, "r") as fp:
            memory_pool = yaml.full_load(fp)

        if memory_pool != None:
            for instance in memory_pool:

                IO_bus_width = int(memory_pool[instance]["IO_bus_width"])
                area = memory_pool[instance]["area"]
                bank_count = int(memory_pool[instance]["bank_count"])
                read_cost = memory_pool[instance]["cost"]["read_word"] * 1000
                write_cost = memory_pool[instance]["cost"]["write_word"] * 1000
                ex_rd_port = int(memory_pool[instance]["ex_rd_port"])
                ex_wr_port = int(memory_pool[instance]["ex_wr_port"])
                rd_wr_port = int(memory_pool[instance]["rd_wr_port"])
                cache_size = int(memory_pool[instance]["size_bit"])
                memory_type = memory_pool[instance]["memory_type"]

                if (
                    (mem_type == memory_type)
                    and (size == cache_size)
                    and (IO_bus_width == r_bw)
                    and (r_port == ex_rd_port)
                    and (w_port == ex_wr_port)
                    and (rw_port == rd_wr_port)
                ):
                    # print("Memory instance found in Cacti memory pool!", cache_size, IO_bus_width, ex_rd_port, ex_wr_port, rd_wr_port, bank_count, read_cost, write_cost)
                    return (
                        cache_size,
                        IO_bus_width,
                        IO_bus_width,
                        read_cost,
                        write_cost,
                        area,
                        bank_count,
                        ex_rd_port,
                        ex_wr_port,
                        rd_wr_port,
                    )

        # should be never reached
        raise ModuleNotFoundError(
            f"No match in Cacti memory pool found {size=}, {r_bw=}, {r_port=}, {w_port=}, {rw_port=}, {bank=}"
        )
