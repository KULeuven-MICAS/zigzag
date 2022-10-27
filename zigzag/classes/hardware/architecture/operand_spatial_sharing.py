import math
import numpy as np
from typing import List
from zigzag.classes.hardware.architecture.dimension import Dimension


class OperandSpatialSharing:
    def __init__(self, index: int, name: str, direction: tuple, operand: str, base_dims: List[Dimension]):
        """
        Initialize the OperandSpatialSharing object. This object is used to denote the
        sharing of an 'operand' along a 'direction' of the MultiplierArray.

        :param index: The index of this interconnect.
        :param name: The name of this interconnect.
        :param direction: The defined direction of the interconnect in the base coordinate space.
        :param operand: The operand that is interconnected.
        :param base_dims: The base dimensions of the MultiplierArray.
        """
        self.id = index
        self.name = name
        self.direction = direction
        self.operand = operand

        '''Sanity checks on the direction'''
        assert_msg = f'Invalid interconnects definition: {direction}. '
        assert all([type(d) == int for d in direction]), assert_msg + 'All elements must be integers.'
        non_zero = [d for d in direction if d != 0]
        assert len(non_zero) <= 2, assert_msg + 'Number of non-zero elements must be 0, 1 or 2.'
        if len(non_zero) > 0:
            #assert math.gcd(*non_zero) == 1, assert_msg + 'Greatest common divisor (GCD) of non-zero elements must be 1.'
            assert min(non_zero) == 1, assert_msg + 'Minimal non-zero value must be 1. Otherwise instance calculation is incorrect.'

        '''Sanity checks on the operand (must be defined)'''
        # TODO

        '''Save the number of non-zero direction elements'''
        self.nb_non_zero_dims = len(non_zero)

        '''
        Calculate the number of instances of this interconnect.
        This is done through multiplying together all orthogonal dimensions on the given direction.
        '''
        self.instances = self.calc_interconnect_instances(base_dims)

        '''
        Calculate the size of this interconnect.
        For dual-dimension interconnects, this can be a fractional number.
        This is calculated by dividing the total number of multipliers by the number of instances.
        '''
        self.size = float(np.prod([d.size for d in base_dims]) / self.instances)

    def __str__(self):
        return self.operand + ": " + str(self.direction)

    def __repr__(self):
        return str(self)

    def calc_interconnect_instances(self, base_dims: List[Dimension]) -> int:
        """
        TODO: Should this function be inside of MultiplierArray instead of Interconnect?
        Calculate the number of instances there exist for this interconnect
        within the MultiplierArray.

        :param base_dims: List containing all base dimensions of the MultiplierArray.
        :return: The number of instances for this interconnect.
        """
        dir = self.direction
        '''Start off by looking at all orthogonal dimensions (0 in dir)'''
        nb_instances = int(np.prod([dim.size for (d, dim) in zip(dir, base_dims) if d == 0]))
        '''
        If dual dimension interconnect, use formula:
        D1 = size of dimension 1        
        D2 = size of dimension 2          
        d1 = direction for dimension 1  
        d2 = direction for dimension 2  
        nb_instances *= d2*(D1-1) + d1*(D2-1) + 1
        '''
        if self.nb_non_zero_dims == 2:
            non_zero_zipped = [(dim.size, d) for (dim, d) in zip(base_dims, dir) if d != 0]
            (D1, d1) = non_zero_zipped[0]
            (D2, d2) = non_zero_zipped[1]
            nb_instances *= d2 * (D1 - 1) + d1 * (D2 - 1) + 1

        return nb_instances