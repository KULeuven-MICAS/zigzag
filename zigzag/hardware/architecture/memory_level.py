from typing import Any, TypeAlias
import math

from zigzag.datatypes import Dimension, MemoryOperand
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.memory_port import MemoryPort, MemoryPortType, PortAllocation
from zigzag.hardware.architecture.operational_array import OperationalArray


ServedMemDimsUserFormat: TypeAlias = tuple[str, ...]


class ServedMemDimensions:
    """! Represents a collection of Operational Array Dimensions (served by some Memory Instance)
    # TODO let this inherit from some `Attribute` ABC
    """

    def __init__(self, data: set[Dimension]):
        assert isinstance(data, set)
        assert all([isinstance(x, Dimension) for x in data])
        self.data = data

    def to_vec_format(self, nb_oa_dims: int, nb_operands: int) -> tuple[set[tuple[int, ...]], ...]:
        """! Convert the instance to the one-hot encoded `vector` format used in legacy code,
        e.g. ({(1,0), (1,1)}, {(1,0), (1,1)}, {(1,0), (1,1)}) - identical copy for each memory operand
        # TODO replace the legacy code parts with the new representation
        """
        assert all([x.id < nb_oa_dims for x in self.data])
        default = [0] * nb_oa_dims
        vec_single_operand: set[tuple[int, ...]] = set()
        if len(self.data) == 0:
            vec_single_operand = {tuple(default)}
        for oa_dim in self.data:
            encoding = default.copy()
            encoding[oa_dim.id] = 1
            vec_single_operand.add(tuple(encoding))

        return tuple(vec_single_operand for _ in range(nb_operands))

    def nb_dims(self):
        return len(self.data)

    def assert_valid(self, oa_dims: list[Dimension]) -> None:
        """! Return True iff:
        - all served dimensions are contained within the given Operational Array Dimensions
        (Not the other way around: the served dimensions are a subset of the Dimensions of the Operational Array)
        @param oa_dims a list with OA Dimensions to compare to
        """
        assert all(
            [served_dim in oa_dims for served_dim in self]
        ), f"""User-specified served dimensions {self.data} contains element not part of the Operational
        Array Dimensions {oa_dims}"""

    def to_user_format(self) -> ServedMemDimsUserFormat:
        return tuple(oa_dim.name for oa_dim in self)

    def __eq__(self, other: Any):
        return (
            isinstance(other, ServedMemDimensions)
            and len(self.data) == len(other.data)
            and all([x in other.data for x in self.data])
        )

    def __str__(self):
        return str(self.data)

    def __contains__(self, other: Dimension):
        return other in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def parse_user_format(x: ServedMemDimsUserFormat) -> "ServedMemDimensions":
        """! Initialize an instance from the given data in user format"""
        assert isinstance(x, tuple), "User provided served memory dimensions must be a tuple"
        assert all([isinstance(x, str) for x in x])

        data = {Dimension.parse_user_input(oa_dim) for oa_dim in x}
        return ServedMemDimensions(data)


class MemoryLevel:

    def __init__(
        self,
        memory_instance: MemoryInstance,
        operands: tuple[str, ...],
        mem_level_of_operands: dict[MemoryOperand, int],
        port_alloc: PortAllocation,
        served_dimensions: ServedMemDimensions,
        operational_array: OperationalArray,
        id: int,
    ):
        """! Initialize the memory level in the hierarchy with the physical memory instance
        @param port_alloc: memory port allocation (physical memory port -> functional memory port)
        @param id: an identifier used for reference check.
        """
        self.memory_instance = memory_instance
        self.name = self.memory_instance.name
        # TODO encapsulate
        self.operands = [MemoryOperand(x) for x in operands]
        self.mem_level_of_operands = mem_level_of_operands
        self.oa_dims: list[Dimension] = operational_array.dimensions
        self.id: int = id
        self.served_dimensions: ServedMemDimensions = served_dimensions
        self.served_dimensions.assert_valid(self.oa_dims)

        # To be compatible with legacy code
        self.served_dimensions_vec = served_dimensions.to_vec_format(len(self.oa_dims), len(self.operands))

        #  for each operand that current memory level holds, allocate
        # physical memory ports to its 4 potential data movement
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
            port_bw = self.memory_instance.r_bw  # we assume the read-write port has the same bw for read and write
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
        return math.prod(oa_dim.size for oa_dim in self.oa_dims if oa_dim in self.served_dimensions)

    def __jsonrepr__(self):
        """! JSON Representation of this class to save it to a json file."""
        return str(self)

    def __update_formatted_string(self):
        self.formatted_string = f"""MemoryLevel(instance={self.memory_instance.name},operands={self.operands},
        served_dimensions={self.served_dimensions})"""

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
