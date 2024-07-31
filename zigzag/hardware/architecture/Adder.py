class Adder:
    """! This class represents one single adder."""

    def __init__(
        self,
        fan_in: int,
        unit_cost: float,
        unit_area: float,
        input_precision: list[int] | int,
        output_precision: int,
    ):
        """
        @param fan_in: the number of input data to be added together.
        @param unit_cost: one addition energy.
        @param unit_area: one adder area.
        @param input_precision: input data precision. If it is 'int' format, it means the same precision is applied to
        all input data; if it is 'list' format, it allows to define for different input data the different precision.
        @param output_precision: output data precision.
        """
        self.fan_in = fan_in
        self.cost = unit_cost
        self.area = unit_area
        self.input_precision = input_precision
        self.output_precision = output_precision
