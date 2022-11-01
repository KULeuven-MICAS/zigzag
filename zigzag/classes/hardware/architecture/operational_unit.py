from typing import List


class OperationalUnit:
    def __init__(self, input_precision: List[int], output_precision: int, unit_cost: float, unit_area: float):
        """
        General class for a unit that performs a certain operation. For example: a multiplier unit.

        :param input_precision: The bit precision of the operation inputs.
        :param output_precision: The bit precision of the operation outputs.
        :param unit_cost: The energy cost of performing a single operation.
        :param unit_area: The area of a single operational unit.
        """
        self.input_precision = input_precision
        self.output_precision = output_precision
        self.precision = input_precision + [output_precision]
        self.cost = unit_cost
        self.area = unit_area

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return self.__dict__

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, OperationalUnit):
            return False
        return self.precision == __o.precision and self.cost == __o.cost and self.area == __o.area


class Multiplier(OperationalUnit):
    def __init__(self, input_precision: List[int], energy_cost: float, area: float):
        """
        Initialize the Multiplier object.

        :param input_precision: The bit precision of the multiplication inputs.
        :param output_precision: The bit precision of the multiplication outputs.
        :param energy_cost: The energy cost of performing a single multiplication.
        :param area: The area of a single multiplier.
        """
        output_precision = sum(input_precision)
        super().__init__(input_precision, output_precision, energy_cost, area)