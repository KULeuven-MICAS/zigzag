from zigzag.utils import json_repr_handler


class OperationalUnit:
    """! General class for a unit that performs a certain operation. For example: a multiplier unit."""

    def __init__(
        self,
        input_precision: list[int],
        output_precision: int,
        unit_cost: float,
        unit_area: float,
    ):
        """
        @param input_precision: The bit precision of the operation inputs.
        @param output_precision: The bit precision of the operation outputs.
        @param unit_cost: The energy cost of performing a single operation.
        @param unit_area: The area of a single operational unit.
        """
        self.input_precision = input_precision
        self.output_precision = output_precision
        self.precision = input_precision + [output_precision]
        self.cost = unit_cost
        self.area = unit_area

    def __jsonrepr__(self):
        """! JSON Representation of this class to save it to a json file."""
        return json_repr_handler(self.__dict__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperationalUnit):
            return False
        return self.precision == other.precision and self.cost == other.cost and self.area == other.area


class Multiplier(OperationalUnit):

    def __init__(self, input_precision: list[int], energy_cost: float, area: float):
        """
        @param input_precision: The bit precision of the multiplication inputs.
        @param energy_cost: The energy cost of performing a single multiplication.
        @param area: The area of a single multiplier.
        """
        output_precision = sum(input_precision)
        super().__init__(input_precision, output_precision, energy_cost, area)
