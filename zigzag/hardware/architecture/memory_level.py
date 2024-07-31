import math
from typing import Any

from zigzag.datatypes import MemoryOperand, OADimension
from zigzag.hardware.architecture.memory_port import (
    MemoryPort,
    MemoryPortType,
    PortAllocation,
)
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.operational_array import OperationalArrayABC


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
        self.operands = operands
        self.mem_level_of_operands = mem_level_of_operands
        self.oa_dim_sizes = operational_array.dimension_sizes
        self.id: int = identifier
        self.served_dimensions = served_dimensions
        self.name = self.memory_instance.name

        # for each operand that current memory level holds, allocate physical memory ports to its 4 potential data
        # movement
        self.port_alloc_raw = port_alloc
        self.__allocate_ports()

        #  memory access bandwidth and energy extraction
        self.read_energy = memory_instance.r_cost
        self.write_energy = memory_instance.w_cost
        self.read_bw = memory_instance.r_bw
        self.write_bw = memory_instance.w_bw

    def __allocate_ports(self):
        # Step 1: according to the port count of the memory instance, initialize the physical port object
        # (so far, we don't know what the port will be used for. But we do know the port's id/bw/attribute)
        port_list: list[MemoryPort] = []
        r_port_nb = self.memory_instance.r_port_nb
        w_port_nb = self.memory_instance.w_port_nb
        rw_port_nb = self.memory_instance.rw_port_nb
        for i in range(1, r_port_nb + 1):
            port_name = "r_port_" + str(i)
            port_bw = self.memory_instance.r_bw
            port_bw_min = self.memory_instance.r_bw_min
            port_attr = MemoryPortType.READ
            new_port = MemoryPort(port_name, port_bw, port_bw_min, port_attr)
            port_list.append(new_port)
        for i in range(1, w_port_nb + 1):
            port_name = "w_port_" + str(i)
            port_bw = self.memory_instance.w_bw
            port_bw_min = self.memory_instance.w_bw_min
            port_attr = MemoryPortType.WRITE
            new_port = MemoryPort(port_name, port_bw, port_bw_min, port_attr)
            port_list.append(new_port)
        for i in range(1, rw_port_nb + 1):
            port_name = "rw_port_" + str(i)
            # we assume the read-write port has the same bw for read and write
            port_bw = self.memory_instance.r_bw
            # we assume the read-write port has the same bw for read and write
            port_bw = self.memory_instance.r_bw
            port_bw_min = self.memory_instance.r_bw_min
            port_attr = MemoryPortType.READ_WRITE
            new_port = MemoryPort(port_name, port_bw, port_bw_min, port_attr)
            port_list.append(new_port)
        port_names = [port.name for port in port_list]

        # Step 2: add operand, memory level, and served data movement direction for each port.
        for mem_op, mem_lvl in self.mem_level_of_operands.items():
            allocation_this_mem_op = self.port_alloc_raw.get_alloc_for_mem_op(mem_op)
            for direction, port_name in allocation_this_mem_op.items():
                port_idx = port_names.index(port_name)
                mem_port = port_list[port_idx]
                mem_port.add_port_function((mem_op, mem_lvl, direction))

        self.port_list = port_list

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
            and self.port_list == other.port_list
            and self.served_dimensions == other.served_dimensions
        )

    def __hash__(self) -> int:
        return hash(self.id)
