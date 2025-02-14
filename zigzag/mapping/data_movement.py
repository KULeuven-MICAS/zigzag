from enum import StrEnum
from typing import Generic, TypeVar

from zigzag.datatypes import LayerOperand
from zigzag.hardware.architecture.memory_port import DataDirection

T = TypeVar("T", bound=int | float)


class FourWayDataMoving(Generic[T]):
    """Represents a standard four-way data moving attribute of a memory interface."""

    def __init__(self, data: dict[DataDirection, T] | None = None):
        """Initialize with a dictionary containing all four DataDirection values, defaulting to zero."""
        self.data: dict[DataDirection, T] = (
            {direction: 0 for direction in DataDirection} if data is None else data.copy()
        )

        # Ensure all required keys exist
        missing_keys = set(DataDirection) - set(self.data)
        if missing_keys:
            raise ValueError(f"Missing keys in input dictionary: {missing_keys}")

    def get(self, direction: DataDirection) -> T:
        """Retrieve the value associated with a specific data direction."""
        return self.data[direction]

    def set(self, direction: DataDirection, value: T):
        """Update the value of a specific data direction."""
        self.data[direction] = value

    def __add__(self, other: "FourWayDataMoving[T]") -> "FourWayDataMoving[T]":
        """Element-wise addition of two FourWayDataMoving instances."""
        return FourWayDataMoving({key: self.data[key] + other.data[key] for key in DataDirection})  # type: ignore

    def __mul__(self, scalar: T) -> "FourWayDataMoving[T]":
        """Element-wise multiplication by a scalar."""
        return FourWayDataMoving({key: self.data[key] * scalar for key in DataDirection})  # type: ignore

    def __repr__(self):
        """Readable string representation of the class."""
        return (
            f"4WayDataMoving("
            f"rd ↑: {self.data[DataDirection.RD_OUT_TO_HIGH]}, "
            f"wr ↓: {self.data[DataDirection.WR_IN_BY_HIGH]}, "
            f"rd ↓: {self.data[DataDirection.RD_OUT_TO_LOW]}, "
            f"wr ↑: {self.data[DataDirection.WR_IN_BY_LOW]})"
        )

    def __jsonrepr__(self):
        """JSON-friendly representation."""
        return {key.value: self.data[key] for key in DataDirection}


class MemoryAccesses(FourWayDataMoving[int]):
    """Represents the number of memory accesses in four directions."""

    def __add__(self, other: "FourWayDataMoving[int]") -> "MemoryAccesses":
        """Element-wise addition of two AccessEnergy instances."""
        return MemoryAccesses({key: self.data[key] + other.data[key] for key in DataDirection})

    def __mul__(self, scalar: int) -> "MemoryAccesses":
        """Element-wise multiplication by a scalar."""
        return MemoryAccesses({key: self.data[key] * scalar for key in DataDirection})


class AccessEnergy(FourWayDataMoving[float]):
    """Represents the memory access energy in four directions."""

    def __add__(self, other: "FourWayDataMoving[float]") -> "AccessEnergy":
        """Element-wise addition of two AccessEnergy instances."""
        return AccessEnergy({key: self.data[key] + other.data[key] for key in DataDirection})

    def __mul__(self, scalar: float) -> "AccessEnergy":
        """Element-wise multiplication by a scalar."""
        return AccessEnergy({key: self.data[key] * scalar for key in DataDirection})


class DataMoveAttr(StrEnum):
    DATA_ELEM_MOVE_COUNT = "data_elem_move_count"
    DATA_PRECISION = "data_precision"
    REQ_MEM_BW_AVER = "req_mem_bw_aver"
    REQ_MEM_BW_INST = "req_mem_bw_inst"
    DATA_TRANS_PERIOD = "data_trans_period"
    DATA_TRANS_PERIOD_COUNT = "data_trans_period_count"
    DATA_TRANS_AMOUNT_PER_PERIOD = "data_trans_amount_per_period"
    INST_DATA_TRANS_WINDOW = "inst_data_trans_window"


class DataMovePattern:
    """Collects the memory access pattern for each unit memory (memory holding one operand at one level)."""

    def __init__(self, operand: LayerOperand, mem_level: int):
        self.name = operand.name + str(mem_level)

        # Use a dictionary to store all attributes as FourWayDataMoving instances
        self.attributes: dict[DataMoveAttr, FourWayDataMoving[int]] = {
            attr: FourWayDataMoving() for attr in DataMoveAttr
        }

    def set_attribute(self, attr: DataMoveAttr, values: dict[DataDirection, int]):
        """Set a given attribute using a dictionary of DataDirection values."""
        if attr not in DataMoveAttr:
            raise ValueError(f"Invalid attribute name: {attr}")
        self.attributes[attr] = FourWayDataMoving(values)

    def get_attribute(self, attr: DataMoveAttr) -> FourWayDataMoving[int]:
        """Retrieve a specific attribute."""
        return self.attributes[attr]

    def update_single_dir_data(self, direction: DataDirection, new_value: int):
        """Update a single direction value across all attributes."""
        for attr in self.attributes.values():
            attr.set(direction, new_value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"DataMovePattern(name={self.name}, attributes={self.attributes})"
