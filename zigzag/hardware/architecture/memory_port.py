from enum import StrEnum
import re
from typing import Any, TypeAlias

from zigzag.datatypes import Constants, MemoryOperand


class MemoryPortType(StrEnum):
    READ = "r"
    WRITE = "w"
    READ_WRITE = "rw"


class DataDirection(StrEnum):
    RD_OUT_TO_LOW = "rd_out_to_low"
    WR_IN_BY_LOW = "wr_in_by_low"
    RD_OUT_TO_HIGH = "rd_out_to_high"
    WR_IN_BY_HIGH = "wr_in_by_high"


OperandDirection: TypeAlias = tuple[MemoryOperand, int, DataDirection]
PortAllocUserFormat: TypeAlias = tuple[dict[str, str], ...]


class MemoryPort:

    port_id_counter = 0

    def __init__(
        self,
        port_name: str,
        port_bw: int,
        port_bw_min: int,
        port_attr: MemoryPortType,
        port_id: int | None = None,
    ):
        """
        Collect all the physical memory port related information here.
        @param port_name:
        @param port_bw: bit/cc
        @param port_attr: read_only (r), write_only (w), read_write (rw)
        @param port_id: port index per memory
        """
        self.name = port_name
        self.bw = port_bw
        self.bw_min = port_bw_min
        self.attr = port_attr
        self.served_op_lv_dir: list[OperandDirection] = []

        #  to give each port a unique id number
        if port_id is None:
            self.port_id = MemoryPort.port_id_counter
            MemoryPort.port_id_counter += 1
        else:
            self.port_id = port_id
            MemoryPort.port_id_counter = port_id + 1

    def add_port_function(self, operand_level_direction: OperandDirection):
        self.served_op_lv_dir.append(operand_level_direction)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MemoryPort)
            and self.bw == other.bw
            and self.bw_min == other.bw_min
            and self.attr == other.attr
        )

    def __hash__(self):
        return self.port_id


class PortAllocation:
    def __init__(self, data: dict[MemoryOperand, dict[DataDirection, str]]):
        assert all(
            [
                all([isinstance(v, str) and re.match(r"^[r]?[w]?_port_\d+$", v) for v in d.values()])
                for d in data.values()
            ]
        )
        self.data = data

    def get_alloc_for_mem_op(self, mem_op: MemoryOperand):
        return self.data[mem_op]

    @staticmethod
    def get_default(mem_operands: list[MemoryOperand]) -> "PortAllocation":
        data: dict[MemoryOperand, dict[DataDirection, str]] = dict()
        for mem_op in mem_operands:
            if mem_op == Constants.OUTPUT_MEM_OP:
                data[mem_op] = {
                    DataDirection.WR_IN_BY_HIGH: "w_port_1",
                    DataDirection.WR_IN_BY_LOW: "w_port_1",
                    DataDirection.RD_OUT_TO_HIGH: "r_port_1",
                    DataDirection.RD_OUT_TO_LOW: "r_port_1",
                }
            else:
                data[mem_op] = {
                    DataDirection.WR_IN_BY_HIGH: "w_port_1",
                    DataDirection.RD_OUT_TO_LOW: "r_port_1",
                }
        return PortAllocation(data)

    @staticmethod
    def parse_user_input(x: PortAllocUserFormat, mem_operands: list[MemoryOperand]) -> "PortAllocation":
        """!
        The order of the port allocations matches the order of the MemoryOperands from the given list.
        """

        def translate_to_data_direction(x: str) -> DataDirection:
            match x:
                case "fh":
                    return DataDirection.WR_IN_BY_HIGH
                case "fl":
                    return DataDirection.WR_IN_BY_LOW
                case "th":
                    return DataDirection.RD_OUT_TO_HIGH
                case "tl":
                    return DataDirection.RD_OUT_TO_LOW
                case _:
                    raise ValueError(f"Data direction must be either `fh`, `th`, `fl`, or `tl`. Not {x}")

        assert isinstance(x, tuple)
        assert all([isinstance(d, dict) for d in x])
        assert all([isinstance(d, dict) for d in x])
        assert all([all([isinstance(k, str) for k in d.keys()]) for d in x])
        assert all([all([isinstance(v, str) for v in d.values()]) for d in x])
        assert all(
            [all([re.match(r"^[r]?[w]?_port_\d+$", v) for v in d.values()]) for d in x]
        ), "Port name should follow the pattern `r_`, `w_` or `rw_port_1`" + str(x)
        assert len(x) == len(mem_operands)

        data: dict[MemoryOperand, dict[DataDirection, str]] = {
            mem_op: {translate_to_data_direction(k): v for k, v in x[idx].items()}
            for idx, mem_op in enumerate(mem_operands)
        }
        return PortAllocation(data)
