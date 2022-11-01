from typing import Dict, Set
import numpy as np
from zigzag.classes.hardware.architecture.dimension import Dimension
from zigzag.classes.hardware.architecture.operand_spatial_sharing import OperandSpatialSharing
from zigzag.classes.hardware.architecture.operational_unit import OperationalUnit, Multiplier

class OperationalArray:
    def __init__(self, operational_unit: OperationalUnit, dimensions: Dict[str, int]):
        """
        This class captures multi-dimensional operational array size.

        :param operational_unit: an OperationalUnit object including precision and single operation energy, later we
                           can add idle energy also (e.g. for situations that one or two of the input operands is zero).

        :param dimensions: define the name and size of each multiplier array dimensions, e.g. {'D1': 3, 'D2': 5}.
        """
        self.unit = operational_unit
        self.total_unit_count = int(np.prod(list(dimensions.values())))
        self.total_area = operational_unit.area * self.total_unit_count

        base_dims = [Dimension(idx, name, size) for idx, (name, size) in enumerate(dimensions.items())]
        self.dimensions = base_dims
        self.dimension_sizes = [dim.size for dim in base_dims]
        self.nb_dimensions = len(base_dims)

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return {"operational_unit": self.unit, "dimensions": self.dimensions}

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, OperationalArray):
            return False
        return self.unit == __o.unit and self.dimensions == __o.dimensions


class MultiplierArray(OperationalArray):
    pass


def multiplier_array_example1():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 3, 'D3': 4}
    operand_spatial_sharing = {'I1': {(1, 0, 0)},
                       'O': {(0, 1, 0)},
                       'I2': {(0, 0, 1), (1, 1, 0)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


def multiplier_array_example2():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 12}
    operand_spatial_sharing = {'I1': {(1, 0)},
                             'O': {(0, 1)},
                             'I2': {(1, 1)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


if __name__ == "__main__":
    multiplier_array = multiplier_array_example1()
    for os in multiplier_array.operand_spatial_sharing:
        print(f'{os}\tdirection: {os.direction} operand: {os.operand} instances: {os.instances}')