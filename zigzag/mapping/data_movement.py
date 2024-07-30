from abc import ABCMeta
from typing import Generic, TypeVar

from zigzag.datatypes import LayerOperand
from zigzag.hardware.architecture.memory_port import DataDirection

T = TypeVar("T", bound=int | float)


class FourWayDataMoving(Generic[T], metaclass=ABCMeta):
    """! The standard four-way data moving attribute of a memory interface."""

    def __init__(
        self,
        rd_out_to_low: T,
        wr_in_by_low: T,
        rd_out_to_high: T,
        wr_in_by_high: T,
    ):
        self.rd_out_to_low = rd_out_to_low
        self.wr_in_by_low = wr_in_by_low
        self.rd_out_to_high = rd_out_to_high
        self.wr_in_by_high = wr_in_by_high

    @property
    def info_list(self):
        """! Format used in the original ZigZag version"""
        return [
            (self.rd_out_to_low, self.wr_in_by_low),
            (self.rd_out_to_high, self.wr_in_by_high),
        ]

    def get_single_dir_data(self, direction: DataDirection) -> T:
        match direction:
            case DataDirection.RD_OUT_TO_LOW:
                return self.rd_out_to_low
            case DataDirection.WR_IN_BY_LOW:
                return self.wr_in_by_low
            case DataDirection.RD_OUT_TO_HIGH:
                return self.rd_out_to_high
            case DataDirection.WR_IN_BY_HIGH:
                return self.wr_in_by_high

    def update_single_dir_data(self, direction: DataDirection, new_value: T):
        match direction:
            case DataDirection.RD_OUT_TO_LOW:
                self.rd_out_to_low = new_value
            case DataDirection.WR_IN_BY_LOW:
                self.wr_in_by_low = new_value
            case DataDirection.RD_OUT_TO_HIGH:
                self.rd_out_to_high = new_value
            case DataDirection.WR_IN_BY_HIGH:
                self.wr_in_by_high = new_value

    def get_total_read_outs_to_above(self) -> T:
        """! Return the total amount of times this memory interface is read from to the level above."""
        return self.rd_out_to_high

    def get_total_read_outs_to_below(self) -> T:
        """! Return the total amount of times this memory interface is read from to the level below."""
        return self.rd_out_to_low

    def get_total_write_ins_from_above(self) -> T:
        """! Return the total amount of times this memory interface is written to from the level above."""
        return self.wr_in_by_high

    def get_total_write_ins_from_below(self) -> T:
        """! Return the total amount of times this memory interface is written to from the level below."""
        return self.wr_in_by_low

    def _add_with_type(self, other: "FourWayDataMoving[T]", return_type: type):
        return return_type(
            self.rd_out_to_low + other.rd_out_to_low,
            self.wr_in_by_low + other.wr_in_by_low,
            self.rd_out_to_high + other.rd_out_to_high,
            self.wr_in_by_high + other.wr_in_by_high,
        )

    def _mul_with_type(self, other: T, return_type: type):
        return return_type(
            self.rd_out_to_low * other,
            self.wr_in_by_low * other,
            self.rd_out_to_high * other,
            self.wr_in_by_high * other,
        )

    def __repr__(self):
        return (
            f"4waydatamoving (rd ^: {self.rd_out_to_high}, wr v: {self.wr_in_by_high}, "
            f"rd v: {self.rd_out_to_low}, wr ^: {self.wr_in_by_low})"
        )

    def __jsonrepr__(self):
        return {
            "rd ^": self.rd_out_to_high,
            "wr v": self.wr_in_by_high,
            "rd v": self.rd_out_to_low,
            "wr ^": self.wr_in_by_low,
        }


class MemoryAccesses(FourWayDataMoving[int]):
    """Represents the number of memory accesses in four directions"""

    def __add__(self, other: "MemoryAccesses"):
        return self._add_with_type(other, type(self))

    def __mul__(self, other: int):
        return self._mul_with_type(other, type(self))


class AccessEnergy(FourWayDataMoving[float]):
    """Represents the memory access energy in four directions"""

    def __add__(self, other: "AccessEnergy"):
        return self._add_with_type(other, type(self))

    def __mul__(self, other: float):
        return self._mul_with_type(other, type(self))


class DataMovePattern:
    """! Collect the memory access pattern for each unit memory (memory that only hold one operand at one level)."""

    def __init__(self, operand: LayerOperand, mem_level: int):
        self.name = operand.name + str(mem_level)
        self.data_elem_move_count = MemoryAccesses(0, 0, 0, 0)
        self.data_precision = MemoryAccesses(0, 0, 0, 0)
        self.req_mem_bw_aver = MemoryAccesses(0, 0, 0, 0)
        self.req_mem_bw_inst = MemoryAccesses(0, 0, 0, 0)
        self.data_trans_period = MemoryAccesses(0, 0, 0, 0)
        self.data_trans_period_count = MemoryAccesses(0, 0, 0, 0)
        self.data_trans_amount_per_period = MemoryAccesses(0, 0, 0, 0)
        self.inst_data_trans_window = MemoryAccesses(0, 0, 0, 0)

    def set_data_elem_move_count(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        self.data_elem_move_count = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_data_precision(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        self.data_precision = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_req_mem_bw_aver(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        self.req_mem_bw_aver = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_req_mem_bw_inst(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        self.req_mem_bw_inst = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_data_trans_period(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        # data_trans_period: every how many cycle, the memory link need to be activated for a certain duration
        self.data_trans_period = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_data_trans_period_count(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        # data_trans_period_count: to finish all the for-loop computation, how many such ideal_period is required
        self.data_trans_period_count = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_data_trans_amount_per_period(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        # data_trans_amount_per_period: data amount that being transferred for each single period
        self.data_trans_amount_per_period = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def set_inst_data_trans_window(
        self,
        rd_out_to_low: int,
        wr_in_by_low: int,
        rd_out_to_high: int,
        wr_in_by_high: int,
    ):
        # inst_data_trans_window: the allowed memory updating window, assuming the served memory level
        # is non-double buffered (thus need to avoid the data overwriting issue
        self.inst_data_trans_window = MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

    def update_single_dir_data(self, direction: DataDirection, new_value: int):
        """! update a single direction value for all data move attributes"""
        self.data_elem_move_count.update_single_dir_data(direction, new_value)
        self.data_precision.update_single_dir_data(direction, new_value)
        self.req_mem_bw_aver.update_single_dir_data(direction, new_value)
        self.req_mem_bw_inst.update_single_dir_data(direction, new_value)
        self.data_trans_period.update_single_dir_data(direction, new_value)
        self.data_trans_period_count.update_single_dir_data(direction, new_value)
        self.data_trans_amount_per_period.update_single_dir_data(direction, new_value)
        self.inst_data_trans_window.update_single_dir_data(direction, new_value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
