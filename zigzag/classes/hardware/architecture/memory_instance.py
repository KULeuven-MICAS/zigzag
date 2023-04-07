from zigzag.classes.cacti.cacti_parser import CactiParser


class MemoryInstance:
    def __init__(
        self,
        name: str,
        size: int,
        r_bw: int,
        w_bw: int = 0,
        r_cost: float = 0,
        w_cost: float = 0,
        area: float = 0,
        r_port: int = 1,
        w_port: int = 1,
        rw_port: int = 0,
        latency: int = 1,
        min_r_granularity=None,
        min_w_granularity=None,
        mem_type: str = "sram",
        auto_cost_extraction: bool = False,
    ):
        """
        Collect all the basic information of a physical memory module.

        :param name: memory module name, e.g. 'SRAM_512KB_BW_16b', 'I_RF'.
        :param size: total memory capacity (unit: bit).
        :param r_bw/w_bw: memory bandwidth (or wordlength) (unit: bit/cycle).
        :param r_cost/w_cost: memory unit data access energy.
        :param area: memory area (unit can be whatever user-defined unit).
        :param r_port: number of memory read port.
        :param w_port: number of memory write port (rd_port and wr_port can work in parallel).
        :param rw_port: number of memory port for both read and write (read and write cannot happen in parallel).
        :param latency: memory access latency (unit: number of cycles).
        min_r_granularity (int): The minimal number of bits than can be read in a clock cycle (can be a less than r_bw)
        min_w_granularity (int): The minimal number of bits that can be written in a clock cycle (can be less than w_bw)
        mem_type (str): The type of memory. Used for CACTI cost extraction.
        auto_cost_extraction (bool): Automatically extract the read cost, write cost and area using CACTI.
        """

        if auto_cost_extraction:
            # Size must be a multiple of 8 when using CACTI
            assert (
                size % 8 == 0
            ), "Memory size must be a multiple of 8 when automatically extracting costs using CACTI."
            cacti_parser = CactiParser()
            (
                _,
                r_bw,
                w_bw,
                r_cost,
                w_cost,
                area,
                bank,
                r_port,
                w_port,
                rw_port,
            ) = cacti_parser.get_item(
                mem_type=mem_type,
                size=size,
                r_bw=r_bw,
                r_port=r_port,
                w_port=w_port,
                rw_port=rw_port,
                bank=1,
            )

        self.name = name
        self.size = size
        self.r_bw = r_bw
        self.w_bw = w_bw
        self.r_cost = r_cost
        self.w_cost = w_cost
        self.area = area
        self.r_port = r_port
        self.w_port = w_port
        self.rw_port = rw_port
        self.latency = latency
        if not min_r_granularity:
            self.r_bw_min = r_bw
        else:
            self.r_bw_min = min_r_granularity
        if not min_w_granularity:
            self.w_bw_min = w_bw
        else:
            self.w_bw_min = min_w_granularity

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return self.__dict__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryInstance) and self.__dict__ == other.__dict__
