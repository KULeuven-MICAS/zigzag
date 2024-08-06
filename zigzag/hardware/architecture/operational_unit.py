from zigzag.utils import json_repr_handler


class OperationalUnit:
    """! General class for a unit that performs a certain operation. For example: a multiplier unit."""

    def __init__(
        self,
        energy_cost: float,
        area: float,
    ):
        """
        @param energy_cost: The energy cost of performing a single operation.
        @param area: The area of a single operational unit.
        """
        self.energy_cost = energy_cost
        self.area = area

    def __jsonrepr__(self):
        """! JSON Representation of this class to save it to a json file."""
        return json_repr_handler(self.__dict__)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OperationalUnit) and self.energy_cost == other.energy_cost and self.area == other.area


class Multiplier(OperationalUnit):
    """Renames OperationalUnit to Multiplier, same functionality"""
