import re
from typing import TypeAlias
from typeguard import typechecked

from zigzag.classes.hardware.architecture.Dimension import Dimension
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
from math import prod
import numpy as np

ServedMemDimsUserFormat: TypeAlias = tuple[str, ...]


@typechecked
class ServedMemDimensions:
    """! Represents a collection of Operational Array Dimensions (served by some Memory Instance)"""

    def __init__(self, data: set[Dimension]):
        assert isinstance(data, set)
        assert all([isinstance(x, Dimension) for x in data])
        self.data = data

    def to_vec_format(self, nb_oa_dims: int, nb_operands: int) -> tuple[set[tuple], ...]:
        """! Convert the instance to the one-hot encoded `vector` format used in legacy code,
        e.g. ({(1,0), (1,1)}, {(1,0), (1,1)}, {(1,0), (1,1)}) - identical copy for each memory operand"""
        assert all([x.id < nb_oa_dims for x in self.data])
        default = [0] * nb_oa_dims
        vec_single_operand: set[tuple] = set()
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
        ), f"User-specified served dimensions {self.data} contains element not part of the Operational Array Dimensions {oa_dims}"

    def to_user_format(self) -> ServedMemDimsUserFormat:
        return tuple(oa_dim.name for oa_dim in self)

    def __eq__(self, other):
        return isinstance(other, ServedMemDimensions) and self.data == other.data

    def __str__(self):
        return str(self.data)

    def __contains__(self, other):
        return isinstance(other, Dimension) and other in self.data

    def __iter__(self):
        return iter(self.data)

    @staticmethod
    def parse_user_format(x: ServedMemDimsUserFormat) -> "ServedMemDimensions":
        """! Initialize an instance from the given data in user format"""
        assert isinstance(x, tuple), "User provided served memory dimensions must be a tuple"
        assert all([isinstance(x, str) for x in x])

        data = {Dimension.parse_user_input(oa_dim) for oa_dim in x}
        return ServedMemDimensions(data)


class MemoryPort:
    """!  Description missing"""

    port_id_counter = 0

    def __init__(
        self,
        port_name: str,
        port_bw: int,
        port_bw_min: int,
        port_attr: str,
        port_id=None,
    ):
        """!  The class constructor
        Collect all the physical memory port related information here.
        @param port_name:
        @param port_bw: bit/cc
        @param port_bw_in:
        @param port_attr: read_only (r), write_only (w), read_write (rw)
        @param port_id: port index per memory
        """
        self.name = port_name
        self.bw = port_bw
        self.bw_min = port_bw_min
        self.attr = port_attr
        self.served_op_lv_dir: list = []

        """ to give each port a unique id number """
        if port_id is None:
            self.port_id = MemoryPort.port_id_counter
            MemoryPort.port_id_counter += 1
        else:
            self.port_id = port_id
            MemoryPort.port_id_counter = port_id + 1

    def add_port_function(self, operand_level_direction: tuple[str, int, str]):
        self.served_op_lv_dir.append(operand_level_direction)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MemoryPort)
            and self.bw == other.bw
            and self.bw_min == other.bw_min
            and self.attr == other.attr
        )

    def __hash__(self):
        return self.port_id


@typechecked
class MemoryLevel:
    """!  Description missing"""

    def __init__(
        self,
        memory_instance: MemoryInstance,
        operands: tuple[str, ...],
        mem_level_of_operands: dict,
        port_alloc: tuple[dict, ...],
        served_dimensions: ServedMemDimensions,
        operational_array: OperationalArray,
        id: int,
    ):
        """!  The class constructor
        Initialize the memory level in the hierarchy with the physical memory instance
        @param memory_instance:
        @param operands:
        @param mem_level_of_operands:
        @param port_alloc: memory port allocation (physical memory port -> functional memory port)
        @param served_dimensions:
        @param operational_array:
        @param id: an identifier used for reference check.
        """
        self.memory_instance = memory_instance
        self.name = self.memory_instance.name
        self.operands = list(operands)
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
        self.port_allocation()

        #  memory access bandwidth and energy extraction
        self.read_energy = memory_instance.r_cost
        self.write_energy = memory_instance.w_cost
        self.read_bw = memory_instance.r_bw
        self.write_bw = memory_instance.w_bw

        #  calculate memory unrolling count
        # Todo: for memory level using diagonal dimension, only allow it to have an unrolling count of '1'.
        self.calc_unroll_count()

        #  calculate in ideal case memory's total fanout and per-data fanout
        # Todo: not consider systolic array for now.
        self.calc_fanout()

    def __update_formatted_string(self):
        self.formatted_string = f"MemoryLevel(instance={self.memory_instance.name},operands={self.operands},served_dimensions={self.served_dimensions})"

    def __str__(self):
        self.__update_formatted_string()
        return self.formatted_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other) -> bool:
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

    def port_allocation(self):
        """!  Create port object"""
        # Step 1: according to the port count of the memory instance, initialize the physical port object
        # (so far, we don't know what the port will be used for. But we do know the port's id/bw/attribute)
        port_list = []
        r_port = self.memory_instance.r_port
        w_port = self.memory_instance.w_port
        rw_port = self.memory_instance.rw_port
        for i in range(1, r_port + 1):
            port_name = "r_port_" + str(i)
            port_bw = self.memory_instance.r_bw
            port_bw_min = self.memory_instance.r_bw_min
            port_attr = "r"
            new_port = MemoryPort(port_name, port_bw, port_bw_min, port_attr)
            port_list.append(new_port)
        for i in range(1, w_port + 1):
            port_name = "w_port_" + str(i)
            port_bw = self.memory_instance.w_bw
            port_bw_min = self.memory_instance.w_bw_min
            port_attr = "w"
            new_port = MemoryPort(port_name, port_bw, port_bw_min, port_attr)
            port_list.append(new_port)
        for i in range(1, rw_port + 1):
            port_name = "rw_port_" + str(i)
            port_bw = self.memory_instance.r_bw  # we assume the read-write port has the same bw for read and write
            port_bw_min = self.memory_instance.r_bw_min
            port_attr = "rw"
            new_port = MemoryPort(port_name, port_bw, port_bw_min, port_attr)
            port_list.append(new_port)
        port_names = [port.name for port in port_list]

        # Step 2: add operand, memory level, and served data movement direction for each port.
        mov_LUT = {
            "fh": "wr_in_by_high",
            "fl": "wr_in_by_low",
            "th": "rd_out_to_high",
            "tl": "rd_out_to_low",
        }
        for idx, (op, lv) in enumerate(list(self.mem_level_of_operands.items())):
            for mov, port in self.port_alloc_raw[idx].items():
                if port is None:
                    continue
                port_idx = port_names.index(port)
                port_list[port_idx].add_port_function((op, lv, mov_LUT[mov]))

        self.port_list = port_list

    def get_port_list(self):
        return self.port_list

    def __jsonrepr__(self):
        """!  JSON Representation of this class to save it to a json file."""
        return str(self)
        # return {"memory_instance": self.memory_instance,
        #         "served_dimensions": self.served_dimensions,
        #         "served_dimensions": self.served_dimensions}

    # def assert_valid(self):
    #     """!  Assert if the served_dimension of this MemoryLevel is valid.
    #     - in served_dimension tuple set, each dimension should only show up once, e.g. {(1,0), (1,1)} is not valid since the '1' in the first position showed up twice.
    #     """
    #     sum_served_dims = []
    #     for op_served_dimensions in self.served_dimensions_vec:
    #         sum_op_served_dimensions = [sum(x) for x in zip(*op_served_dimensions)]
    #         assert not any(
    #             dim > 1 for dim in sum_op_served_dimensions
    #         ), f"Invalid served dimensions for MemoryLevel of Memory {self}"
    #         sum_served_dims.append(sum_op_served_dimensions)
    #     self.sum_served_dims = sum_served_dims

    def calc_unroll_count(self) -> int:
        """! Calculate how many times this memory instance is unrolled (duplicated) on the Operational Array"""
        return prod(oa_dim.size for oa_dim in self.oa_dims if oa_dim in self.served_dimensions)

    def calc_fanout(self) -> int:
        """!  Calculates the total fanout of this MemoryLevel.
        This equals the total amount of multipliers all instances in this level combined serve.
        To calculate the number of lower-level instances a single instance of this level serves,
        this number should be divided by the total_fanouts of all lower levels.
        """
        return int(np.prod([oa_dim.size for oa_dim in self.oa_dims if oa_dim in self.served_dimensions]))

    # def check_served_dimensions(self) -> None:
    #     """!  Function that modifies the served_dimensions for this MemoryLevel if it is an empty set or 'all'.
    #     Empty set signals that the Memory Level has no dimensions served to the level below, thus a fanout of 1.
    #     'all' signals that the MemoryLevel's served_dimensions are all dimensions, thus there is only one instance of the MemoryNode at this level.
    #     """
    #     operands = self.operands
    #     # Modify served_dimensions to list to be able to change it if empty set or None.
    #     served_dimensions: list[tuple] = list(self.served_dimensions_vec)
    #     for op_idx, (op, op_served_dimensions) in enumerate(zip(operands, served_dimensions)):
    #         # If served_dimensions is an empty set, it means this memory level is fully unrolled wrt operational_array
    #         # We then convert it to be consistent with used notation
    #         if op_served_dimensions == set():
    #             served_dimensions[op_idx] = (0,) * self.nb_dimensions
    #         # If served_dimensions is 'all', it means this memory level is not unrolled
    #         # We then convert it to a set containing all base dimensions of the operational_array (corresponds to a flat identity matrix)
    #         if op_served_dimensions == "all":
    #             identity_array = np.eye(self.nb_dimensions, dtype=int)
    #             flat_identity_tuple = tuple([tuple(row) for row in identity_array])
    #             op_served_dimensions = set(flat_identity_tuple)
    #             served_dimensions[op_idx] = tuple(op_served_dimensions)
    #     self.served_dimensions_vec = tuple(served_dimensions)

    #     # Based on the vector representation of the served dimensions,
    #     # we also save all the dimension objects this memory level serves.

    #     for op_served_dimensions_vec in self.served_dimensions_vec:
    #         for served_dimension_vec in op_served_dimensions_vec:
    #             # vector indices that are non-zero
    #             non_zero_idxs = [idx for idx, elem in enumerate(served_dimension_vec) if elem != 0]
    #             self.served_dimensions.update({self.find_dimension_with_idx(idx) for idx in non_zero_idxs})

    # def find_dimension_with_idx(self, idx: int) -> Dimension:
    #     """!  Find the dimension object with idx 'idx'.
    #     @param idx
    #     """
    #     for dim in self.dimensions:
    #         if dim.id == idx:
    #             return dim

    #     raise ValueError("idx passed to function is not a valid dimension id.")
