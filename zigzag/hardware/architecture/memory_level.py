import math
from typing import Any

from zigzag.datatypes import MemoryOperand, OADimension
from zigzag.hardware.architecture.memory_instance import MemoryInstance
from zigzag.hardware.architecture.memory_port import (
    DataDirection,
    MemoryPort,
    PortAllocation,
)
from zigzag.hardware.architecture.operational_array import OperationalArrayABC
from zigzag.utils import hash_sha512, pickle_deepcopy


class ServedMemDimensions:
    """! Represents a collection of Operational Array Dimensions (served by some Memory Instance)"""

    def __init__(self, data: set[OADimension]):
        assert isinstance(data, set)
        assert all([isinstance(x, OADimension) for x in data])
        self.data = data

    @property
    def nb_dims(self):
        return len(self.data)

    def __eq__(self, other: Any):
        return (
            isinstance(other, ServedMemDimensions)
            and len(self.data) == len(other.data)
            and all([x in other.data for x in self.data])
        )

    def __str__(self):
        return str(self.data)

    def __contains__(self, other: OADimension):
        return other in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class MemoryLevel:
    """Represents a single memory in the memory hierarchy, consisting of a memory instance and connectivity
    information"""

    memory_instance: MemoryInstance
    operands: list[MemoryOperand]
    mem_level_of_operands: dict[MemoryOperand, int]
    oa_dim_sizes: dict[OADimension, int]
    port_alloc: PortAllocation
    served_dimensions: ServedMemDimensions
    id: int
    name: str
    port_alloc_raw: PortAllocation
    read_energy: float
    write_energy: float
    read_bw: float
    write_bw: float
    ports: tuple[MemoryPort, ...]

    def __init__(
        self,
        memory_instance: MemoryInstance,
        operands: list[MemoryOperand],
        mem_level_of_operands: dict[MemoryOperand, int],
        port_alloc: PortAllocation,
        served_dimensions: ServedMemDimensions,
        operational_array: OperationalArrayABC,
        identifier: int,
    ):
        """! Initialize the memory level in the hierarchy with the physical memory instance
        @param port_alloc: memory port allocation (physical memory port -> functional memory port)
        @param id: an identifier used for reference check.
        """
        self.memory_instance = memory_instance
        self.operands = operands
        self.mem_level_of_operands = mem_level_of_operands
        self.oa_dim_sizes = operational_array.dimension_sizes
        self.id = identifier
        self.served_dimensions = served_dimensions
        self.name = self.memory_instance.name

        # for each operand that current memory level holds, allocate physical memory ports to its 4 potential data
        # movement
        self.ports = pickle_deepcopy(memory_instance.ports)
        self.port_alloc_raw = port_alloc
        self.__allocate_ports()

        #  memory access bandwidth and energy extraction
        self.read_energy = memory_instance.r_cost
        self.write_energy = memory_instance.w_cost

    def __allocate_ports(self):
        port_names = [port.name for port in self.ports]
        self.bandwidths_min: dict[MemoryOperand, dict[DataDirection, int]]
        self.bandwidths_max: dict[MemoryOperand, dict[DataDirection, int]]
        self.bandwidths_min = {op: {data_dir: None for data_dir in DataDirection} for op in self.operands}
        self.bandwidths_max = {op: {data_dir: None for data_dir in DataDirection} for op in self.operands}
        ports_used = [False] * len(self.ports)
        for mem_op, mem_lvl in self.mem_level_of_operands.items():
            allocation_this_mem_op = self.port_alloc_raw.get_alloc_for_mem_op(mem_op)
            for direction, port_name in allocation_this_mem_op.items():
                # Add operand, memory level, and served data movement direction for each port.
                port_idx = port_names.index(port_name)
                mem_port = self.ports[port_idx]
                mem_port.add_port_function((mem_op, mem_lvl, direction))
                # Save bandwidth for this mem_op and direction for faster access later.
                self.bandwidths_min[mem_op][direction] = mem_port.bw_min
                self.bandwidths_max[mem_op][direction] = mem_port.bw_max
                ports_used[port_idx] = True
        # Remove all ports from self.ports that are not used for this MemoryLevel (e.g. due to removal of some operands)
        self.ports = tuple([port for port, used in zip(self.ports, ports_used) if used])

    def get_min_bandwidth(self, operand: MemoryOperand, data_dir: DataDirection) -> int | None:
        """! Get the minimum memory bandwidth for a specific memory operand and data movement direction"""
        return self.bandwidths_min[operand][data_dir]

    def get_max_bandwidth(self, operand: MemoryOperand, data_dir: DataDirection) -> int | None:
        """! Get the maximum memory bandwidth for a specific memory operand and data movement direction"""
        return self.bandwidths_max[operand][data_dir]

    @property
    def unroll_count(self) -> int:
        """! Calculate how many times this memory instance is unrolled (duplicated) on the Operational Array"""
        return math.prod(
            self.oa_dim_sizes[oa_dim] for oa_dim in self.oa_dim_sizes if oa_dim not in self.served_dimensions
        )

    def __jsonrepr__(self):
        """! JSON Representation of this class to save it to a json file."""
        return str(self)

    def __update_formatted_string(self):
        self.formatted_string = (
            f"MemoryLevel(instance={self.memory_instance.name},operands={self.operands}, "
            f"served_dimensions={self.served_dimensions})"
        )
        self.formatted_string = (
            f"MemoryLevel(instance={self.memory_instance.name},operands={self.operands}, "
            f"served_dimensions={self.served_dimensions})"
        )

    def __str__(self):
        self.__update_formatted_string()
        return self.formatted_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MemoryLevel)
            and self.memory_instance == other.memory_instance
            and self.operands == other.operands
            and self.mem_level_of_operands == other.mem_level_of_operands
            and self.ports == other.ports
            and self.served_dimensions == other.served_dimensions
        )

    def has_same_performance(self, other: "MemoryLevel"):
        return (
            self.memory_instance.has_same_performance(other.memory_instance)
            and self.operands == other.operands
            and self.mem_level_of_operands == other.mem_level_of_operands
            and self.ports == other.ports
            and self.served_dimensions == other.served_dimensions
        )

    def __hash__(self) -> int:
        return hash_sha512(self.id)
