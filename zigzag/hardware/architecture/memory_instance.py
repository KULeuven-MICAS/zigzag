from zigzag.cacti.cacti_parser import CactiParser
from zigzag.hardware.architecture.memory_port import MemoryPort, MemoryPortType
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
    ports: tuple[MemoryPort, ...]
    mem_type: str
    auto_cost_extraction: bool
    double_buffering_support: bool
    shared_memory_group_id: int

    def __init__(
        self,
        name: str,
        size: int,
        r_cost: float = 0,
        w_cost: float = 0,
        area: float = 0,
        r_port: int = 1,
        w_port: int = 1,
        rw_port: int = 0,
        latency: int = 1,
        ports: tuple[MemoryPort, ...] = tuple(),
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
        @param latency: memory access latency (unit: number of cycles).
        @param ports: tuple of MemoryPort instances.
        @param mem_type (str): The type of memory. Used for CACTI cost extraction.
        @param auto_cost_extraction (bool): Automatically extract the read cost, write cost and area using CACTI.
        @param double_buffering_support (bool): Support for double buffering on this memory instance.
        @param shared_memory_group_id: used to indicate whether two MemoryInstance instances represent the same, shared
            memory between two cores (feature used in Stream).
        """
        if auto_cost_extraction:
            try:
                r_bw = next(port.bw_max for port in ports if port.type == MemoryPortType.READ)
            except StopIteration:
                try:
                    r_bw = next(port.bw_max for port in ports if port.type == MemoryPortType.READ_WRITE)
                except StopIteration:
                    raise ValueError(f"MemoryInstance {name} does not have a read or read_write port.")
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
        self.r_cost = r_cost
        self.w_cost = w_cost
        self.area = area
        self.latency = latency
        self.ports = ports
        self.double_buffering_support = double_buffering_support
        self.shared_memory_group_id = shared_memory_group_id

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
            and self.r_cost == other.r_cost
            and self.w_cost == other.w_cost
            and self.latency == other.latency
            and self.ports == other.ports
            and self.double_buffering_support == other.double_buffering_support
        )

    def __hash__(self):
        return hash(frozenset(self.__dict__.values()))

    def __str__(self):
        return f"MemoryInstance({self.name})"

    def __repr__(self):
        return str(self)
