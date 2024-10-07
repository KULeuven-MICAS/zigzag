from zigzag.cacti.cacti_parser import CactiParser
from zigzag.utils import json_repr_handler


class MemoryInstance:
    """A single instance within the memory hierarchy, without information about connectivity."""

    name: str
    size: int
    r_bw: int
    w_bw: int
    r_cost: float
    w_cost: float
    area: float
    r_port: int
    w_port: int
    rw_port: int
    latency: int
    min_r_granularity: int
    min_w_granularity: int
    mem_type: str
    auto_cost_extraction: bool
    double_buffering_support: bool
    shared_memory_group_id: int

    name: str
    size: int
    r_bw: int
    w_bw: int
    r_cost: float
    w_cost: float
    area: float
    r_port: int
    w_port: int
    rw_port: int
    latency: int
    min_r_granularity: int
    min_w_granularity: int
    mem_type: str
    auto_cost_extraction: bool
    double_buffering_support: bool
    shared_memory_group_id: int

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
        min_r_granularity: int | None = None,
        min_w_granularity: int | None = None,
        mem_type: str = "sram",
        auto_cost_extraction: bool = False,
        double_buffering_support: bool = False,
        shared_memory_group_id: int = -1,
    ):
        """
        Collect all the basic information of a physical memory module.
        @param name: memory module name, e.g. 'SRAM_512KB_BW_16b', 'I_RF'.
        @param size: total memory capacity (unit: bit).
        @param r_bw/w_bw: memory bandwidth (or word length) (unit: bit/cycle).
        @param r_cost/w_cost: memory unit data access energy (unit: pJ/access).
        @param area: memory area (unit can be whatever user-defined unit).
        @param r_port: number of memory read port.
        @param w_port: number of memory write port (rd_port and wr_port can work in parallel).
        @param rw_port: number of memory port for both read and write (read and write cannot happen in parallel).
        @param latency: memory access latency (unit: number of cycles).
        @param min_r_granularity (int): The minimal number of bits than can be read in a clock cycle (can be a less than
          r_bw)
        @param min_w_granularity (int): The minimal number of bits that can be written in a clock cycle (can be less
        than w_bw)
        @param mem_type (str): The type of memory. Used for CACTI cost extraction.
        @param auto_cost_extraction (bool): Automatically extract the read cost, write cost and area using CACTI.
        @param double_buffering_support (bool): Support for double buffering on this memory instance.
        @param shared_memory_group_id: used to indicate whether two MemoryInstance instances represent the same, shared
            memory between two cores (feature used in Stream).
        """
        if auto_cost_extraction:
            cacti_parser = CactiParser()
            r_cost, w_cost, area = cacti_parser.get_item(
                mem_name=name,
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
        self.r_port_nb = r_port
        self.w_port_nb = w_port
        self.rw_port_nb = rw_port
        self.latency = latency
        self.double_buffering_support = double_buffering_support
        self.shared_memory_group_id = shared_memory_group_id

        self.r_bw_min: int = min_r_granularity if min_r_granularity is not None else r_bw
        self.w_bw_min: int = min_w_granularity if min_w_granularity is not None else w_bw

    def update_size(self, new_size: int) -> None:
        """! Update the memory size of this instance."""
        self.size = new_size

    def __jsonrepr__(self):
        """! JSON Representation of this class to save it to a json file."""
        return json_repr_handler(self.__dict__)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryInstance) and self.__dict__ == other.__dict__

    def has_same_performance(self, other: "MemoryInstance"):
        """Wether using this instance will result in the same estimations as using the other instance. This method
        differs from __eq__ since it does not consider e.g. the shared_memory_group_id"""
        return (
            self.size == other.size
            and self.r_bw == other.r_bw
            and self.w_bw == other.w_bw
            and self.r_cost == other.r_cost
            and self.w_cost == other.w_cost
            and self.r_port_nb == other.r_port_nb
            and self.w_port_nb == other.w_port_nb
            and self.rw_port_nb == other.rw_port_nb
            and self.latency == other.latency
            and self.double_buffering_support == other.double_buffering_support
            and self.r_bw_min == other.r_bw_min
            and self.w_bw_min == other.w_bw_min
        )

    def __hash__(self):
        return hash(frozenset(self.__dict__.values()))

    def __str__(self):
        return f"MemoryInstance({self.name})"

    def __repr__(self):
        return str(self)
