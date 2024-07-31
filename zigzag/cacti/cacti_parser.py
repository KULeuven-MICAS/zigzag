import logging
import os
import subprocess
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class CactiParser:
    """!  Class that provides the interface between ZigZag and CACTI."""

    ## Path of current directory
    cacti_path = os.path.dirname(os.path.realpath(__file__))
    ## Path to cached cacti simulated memories
    MEM_POOL_PATH = f"{cacti_path}/cacti_master/example_mem_pool.yaml"
    ## Path to cacti python script to extract costs
    CACTI_TOP_PATH = f"{cacti_path}/cacti_master/cacti_top.py"

    def __init__(self):
        """"""
        pass

    def item_exists(
        self,
        size: int,
        r_bw: int,
        r_port: int,
        w_port: int,
        rw_port: int,
        bank: int,
        technology: float,
        mem_pool_path: str = MEM_POOL_PATH,
    ) -> bool:
        """! This function checks whether the provided memory configuration was already used in the past.
        @param mem_pool_path  Path to cached cacti simulated memories
        @return Return wether the requested memory item has been simulated before.
        """
        with open(mem_pool_path, "r") as fp:  # pylint: disable=W1514
            memory_pool: None | dict[str, dict[str, Any]] = yaml.full_load(fp)

        if memory_pool is not None:
            for instance in memory_pool:
                io_bus_width = int(memory_pool[instance]["IO_bus_width"])
                ex_rd_port = int(memory_pool[instance]["ex_rd_port"])
                ex_wr_port = int(memory_pool[instance]["ex_wr_port"])
                rd_wr_port = int(memory_pool[instance]["rd_wr_port"])
                cache_size = int(memory_pool[instance]["size_bit"])
                bank_count = int(memory_pool[instance].get("bank_count", -1))
                tech = float(memory_pool[instance].get("technology", -1))

                if (
                    (size == cache_size)
                    and (io_bus_width == r_bw)
                    and (r_port == ex_rd_port)
                    and (w_port == ex_wr_port)
                    and (rw_port == rd_wr_port)
                    and (bank_count == bank)
                    and (tech == technology)
                ):
                    return True

        return False

    def create_item(
        self,
        mem_type: str,
        size: int,
        r_bw: int,
        r_port: int,
        w_port: int,
        rw_port: int,
        bank: int,
        technology: float = 0.022,
        mem_pool_path: str = MEM_POOL_PATH,
        cacti_top_path: str = CACTI_TOP_PATH,
    ) -> None:
        """! This function simulates a new item by calling CACTI7 based on the provided parameters
        @param mem_pool_path  Path to cached cacti simulated memories
        @param cacti_top_path Path to cacti python script to extract costs
        """

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
                "--technology",
                str(technology),
            ],
        )

        if p != 0:
            raise ChildProcessError(f"Cacti subprocess call failed with return value {p}.")

    def get_item(
        self,
        *,
        mem_name: str,
        mem_type: str,
        size: int,
        r_bw: int,
        r_port: int,
        w_port: int,
        rw_port: int,
        bank: int,
        technology: float = 0.022,
        mem_pool_path: str = MEM_POOL_PATH,
        cacti_top_path: str = CACTI_TOP_PATH,
    ) -> tuple[float, float, float]:
        """! This functions checks first if the memory with the provided parameters was already simulated once.
        In case it hasn't been simulated, then it will create a new memory item based on the provided parameters.
        @param mem_pool_path  Path to cached cacti simulated memories
        @param cacti_top_path Path to cacti python script to extract costs
        """
        if not os.path.exists(cacti_top_path):
            raise FileNotFoundError(f"Cacti top file doesn't exist: {cacti_top_path}.")

        logger.info(
            "Extracting memory costs with CACTI for %s with size = %i and r_bw = %i.",
            mem_name,
            size,
            r_bw,
        )

        if mem_type == "rf":
            new_mem_type = "sram"
            new_size = int(size * 128)
            new_r_bw = int(r_bw)
            logger.warning(
                "%s: Type %s -> %s. Size %i -> %i. BW %i -> %i.",
                mem_name,
                mem_type,
                new_mem_type,
                size,
                new_size,
                r_bw,
                new_r_bw,
            )
            mem_type = new_mem_type
            size = new_size
            r_bw = new_r_bw

        if not self.item_exists(
            size,
            r_bw,
            r_port,
            w_port,
            rw_port,
            bank,
            technology,
            mem_pool_path,
        ):
            self.create_item(
                mem_type,
                size,
                r_bw,
                r_port,
                w_port,
                rw_port,
                bank,
                technology,
                mem_pool_path,
                cacti_top_path,
            )

        with open(mem_pool_path, "r", encoding="UTF-8") as fp:
            memory_pool: None | dict[str, dict[str, Any]] = yaml.full_load(fp)

        if memory_pool is not None:
            for instance in memory_pool:
                io_bus_width = int(memory_pool[instance]["IO_bus_width"])
                area = memory_pool[instance]["area"]
                bank_count = int(memory_pool[instance]["bank_count"])
                read_cost = memory_pool[instance]["cost"]["read_word"] * 1000
                write_cost = memory_pool[instance]["cost"]["write_word"] * 1000
                ex_rd_port = int(memory_pool[instance]["ex_rd_port"])
                ex_wr_port = int(memory_pool[instance]["ex_wr_port"])
                rd_wr_port = int(memory_pool[instance]["rd_wr_port"])
                cache_size = int(memory_pool[instance]["size_bit"])
                memory_type = memory_pool[instance]["memory_type"]
                tech = float(memory_pool[instance].get("technology", -1))

                if (
                    (mem_type == memory_type)
                    and (size == cache_size)
                    and (io_bus_width == r_bw)
                    and (r_port == ex_rd_port)
                    and (w_port == ex_wr_port)
                    and (rw_port == rd_wr_port)
                    and (tech == technology)
                    and (bank_count == bank)
                ):
                    logger.info(
                        "Extracted memory costs with CACTI for %s: r_cost = %f, w_cost = %f, area = %f.",
                        mem_name,
                        read_cost,
                        write_cost,
                        area,
                    )
                    return read_cost, write_cost, area

        # should be never reached
        raise ModuleNotFoundError(
            f"No match in Cacti memory pool found {size=}, {r_bw=}, {r_port=}, {w_port=}, {rw_port=}, {bank=}"
        )
