import math
from abc import ABCMeta
from typing import Any

from zigzag.datatypes import OADimension
from zigzag.hardware.architecture.operational_unit import (
    Multiplier,
    OperationalUnit,
)
from zigzag.utils import json_repr_handler


class OperationalArrayABC(metaclass=ABCMeta):
    """Abstract base class for every concept that can do computations.
    Defines all properties and methods that subclasses should have"""

    def __init__(self, dimension_sizes: dict[OADimension, int]):
        self.dimension_sizes = dimension_sizes
        self.total_unit_count: int


class OperationalArray(OperationalArrayABC):
    """! This class captures multi-dimensional operational array size."""

    def __init__(self, operational_unit: OperationalUnit, dimension_sizes: dict[OADimension, int]):
        """
        @param operational_unit: an OperationalUnit object including precision and single operation energy, later we
        can add idle energy also (e.g. for situations that one or two of the input operands is zero).
        @param dimensions: define the name and size of each multiplier array dimensions, e.g. {'D1': 3, 'D2': 5}.
        """
        OperationalArrayABC.__init__(self, dimension_sizes=dimension_sizes)
        self.unit: OperationalUnit = operational_unit
        self.total_unit_count = int(math.prod(list(dimension_sizes.values())))
        self.total_area = operational_unit.area * self.total_unit_count

    def __jsonrepr__(self):
        return json_repr_handler({"operational_unit": self.unit, "dimensions": self.dimension_sizes})

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, OperationalArray)
            and self.unit == other.unit
            and self.dimension_sizes == other.dimension_sizes
        )


class MultiplierArray(OperationalArray):
    def __init__(
        self,
        multiplier: Multiplier,
        dimensions: dict[OADimension, int],
        operand_spatial_sharing: dict[str, set[tuple[int, ...]]] | None = None,
    ):
        super(MultiplierArray, self).__init__(multiplier, dimensions)
        self.multiplier = self.unit
        self.operand_spatial_sharing = operand_spatial_sharing
