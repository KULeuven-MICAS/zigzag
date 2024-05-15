from typing import Any
import numpy as np
from zigzag.datatypes import OADimension
from zigzag.hardware.architecture.operational_unit import (
    OperationalUnit,
    Multiplier,
)
from zigzag.utils import json_repr_handler


class OperationalArray:
    """! This class captures multi-dimensional operational array size."""

    def __init__(self, operational_unit: OperationalUnit, dimensions: dict[OADimension, int]):
        """
        @param operational_unit: an OperationalUnit object including precision and single operation energy, later we
        can add idle energy also (e.g. for situations that one or two of the input operands is zero).
        @param dimensions: define the name and size of each multiplier array dimensions, e.g. {'D1': 3, 'D2': 5}.
        """
        self.unit: OperationalUnit = operational_unit
        self.total_unit_count = int(np.prod(list(dimensions.values())))
        self.oa_dim_sizes = dimensions
        self.total_area = operational_unit.area * self.total_unit_count

    def __jsonrepr__(self):
        return json_repr_handler({"operational_unit": self.unit, "dimensions": self.oa_dim_sizes})

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, OperationalArray) and self.unit == other.unit and self.oa_dim_sizes == other.oa_dim_sizes
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


# def multiplier_array_example1():
#     """Multiplier array variables"""
#     multiplier_input_precision = [8, 8]
#     multiplier_energy = 0.5
#     multiplier_area = 0.1
#     dimensions = {"D1": 14, "D2": 3, "D3": 4}
#     operand_spatial_sharing = {
#         "I1": {(1, 0, 0)},
#         "O": {(0, 1, 0)},
#         "I2": {(0, 0, 1), (1, 1, 0)},
#     }
#     multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
#     multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

#     return multiplier_array


# def multiplier_array_example2():
#     """Multiplier array variables"""
#     multiplier_input_precision = [8, 8]
#     multiplier_energy = 0.5
#     multiplier_area = 0.1
#     dimensions = {"D1": 14, "D2": 12}
#     operand_spatial_sharing = {"I1": {(1, 0)}, "O": {(0, 1)}, "I2": {(1, 1)}}
#     multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
#     multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

#     return multiplier_array
