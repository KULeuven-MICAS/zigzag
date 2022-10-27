from typing import Dict, List, Tuple
from math import log2, ceil, prod
from zigzag.classes.hardware.architecture.operational_array import Multiplier, MultiplierArray


class Adder:
    def __init__(self, fan_in: int, unit_cost: float, unit_area: float, input_precision: List[int] or int, output_precision: int):
        """
        Initiate Adder class. The adder can be used in aggregation (AG) adder level or accumulation (AC) adder level.
        :param fan_in: the number of input data to be added together.
        :param unit_cost: one addition energy.
        :param unit_area: one adder area.
        :param input_precision: input data precision. If it is 'int' format, it means the same precision is applied to all input data;
        if it is 'list' format, it allows to define for different input data the different precision.
        :param output_precision: output data precision.
        """
        self.fan_in = fan_in
        self.cost = unit_cost
        self.area = unit_area
        self.input_precision = input_precision
        self.output_precision = output_precision


class AdderLevel:
    def __init__(self, index: int, name: str, details: Dict[str, str or int]):
        """
        Adder Level is the basic building block for Adder Hierarchy. 
        It can be an array of aggregators (AG, addition over space) or accumulators (AC, addition over time).
        :param index: Adder Level index.
        :param name: Adder Level name's default format: 'ALi' (i = 1,2,3,...).
        :param details: Adder Level's type, fan-in, and so on.
        """
        self.id = index
        self.name = name
        self.type = details['type']
        self.unit = Adder(details['fan_in'], details['unit_cost'], details['unit_area'], details['input_precision'], details['output_precision'])
        self.one_instance_unit_count = details['one_instance_unit_count']
        self.total_unit_count = details['total_unit_count']

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class AdderHierarchy:
    def __init__(self, adder_hierarchy: Dict[str, Dict[str, str or int]], multiplier_array: MultiplierArray):
        """
        Construct AdderHierarchy class based on user-defined adder hierarchy. It will check if users' definition is valid,
        and extract all the related info, e.g. unit count for each adder level and total area.
        :param adder_hierarchy: user-defined adder hierarchy.
        For aggregation level (AG), it should contain 'type', 'fan_in', 'unit_cost', 'unit_area';
        for accumulation level (AC), it should contain 'type', output_precision', 'unit_cost', 'unit_area'.
        :param multiplier_array: MultiplierArray object, check in "architecture/operational_array.py" for more info.
        """
        self.calc_output_reduction_size(multiplier_array)
        self.assert_valid(adder_hierarchy)
        multiplier_output_precision = multiplier_array.unit.output_precision
        self.construct_adder_levels(multiplier_output_precision, adder_hierarchy)
        self.total_area = prod([adder_level.unit.area * adder_level.total_unit_count for adder_level in self.adder_levels])

    def calc_output_reduction_size(self, multiplier_array: MultiplierArray):
        """
        From dimensions and operand_spatial_sharing defined by user, calculate total output-sharing dimension size.
        :param multiplier_array: MultiplierArray object, check in "architecture/operational_array.py" for more info.
        :return: update self.output_reduction_size and self.output_non_reduction_size
        """
        total_dimension_size = multiplier_array.total_unit_count
        output_reduction_size = 1
        for os in multiplier_array.operand_spatial_sharing:
            if os.operand == 'O':
                output_reduction_size *= int(os.size)
        self.output_reduction_size = output_reduction_size
        self.output_non_reduction_size = total_dimension_size // output_reduction_size

    def assert_valid(self, adder_hierarchy: Dict[str, Dict[str, str or int]]):
        """
        A valid adder hierarchy need to match operand_spatial_sharing (especially the output reduction dimension).
        :param adder_hierarchy: user-defined adder hierarchy.
        :return: none
        """
        assert all([adder_level['type'] in ['AG', 'AC'] for adder_level in adder_hierarchy.values()]), \
            "Some adder type not recognized. Adder type can only be 'AG' or 'AC'."

        total_fan_in = 1
        fan_in_list = []
        acc_flag = False

        for adder_level in adder_hierarchy.values():
            if adder_level['type'] == 'AG':
                total_fan_in *= adder_level['fan_in']
                fan_in_list.append(adder_level['fan_in'])
            else:
                acc_flag = True

        num = self.output_reduction_size
        output_reduction_size_factors = [n for n in range(1, num + 1) if num % n == 0]

        assert set(fan_in_list).issubset(set(output_reduction_size_factors)), \
            f"Invalid adder hierarchy due to at least 1 element in adder tree's fan-in ({fan_in_list}) " \
            f"is not in the factor list of total output-reduction size ({output_reduction_size_factors})."
        assert (total_fan_in in output_reduction_size_factors), \
            f"Invalid adder hierarchy due to adder tree's total fan-in ({total_fan_in}) is not a factor " \
            f"of output reduction size ({output_reduction_size_factors})."
        assert (self.output_reduction_size == total_fan_in or acc_flag), \
            f"Invalid adder hierarchy due to adder tree's total fan-in ({total_fan_in}) < total output " \
            f"reduction size ({self.output_reduction_size}) and no accumulator found."

    def construct_adder_levels(self, multiplier_output_precision: int, adder_hierarchy: Dict[str, Dict[str, str or int]]):
        """
        Construct adder level from the innermost level (close to multiplier) to the outermost.
        Calculate adder count and precision at each adder level.
        :param multiplier_output_precision: treated as the innermost-level adder's input precision.
        :param adder_hierarchy: user-defined adder hierarchy.
        :return: update self.adder_levels
        """
        precision_counter = multiplier_output_precision
        unit_counter = self.output_reduction_size

        for name, adder_details in adder_hierarchy.items():
            if adder_details['type'] == 'AG':
                adder_details['input_precision'] = precision_counter
                adder_details['output_precision'] = precision_counter + ceil(log2(adder_details['fan_in']))
                adder_details['one_instance_unit_count'] = unit_counter//adder_details['fan_in']
                adder_details['total_unit_count'] = adder_details['one_instance_unit_count'] * self.output_non_reduction_size

                ''' update precision and unit count when encounter aggregation (AG) adder level '''
                precision_counter = adder_details['output_precision']
                unit_counter = adder_details['one_instance_unit_count']
            else:
                adder_details['fan_in'] = 2
                adder_details['input_precision'] = [precision_counter, adder_details['output_precision']]
                adder_details['one_instance_unit_count'] = unit_counter
                adder_details['total_unit_count'] = adder_details['one_instance_unit_count'] * self.output_non_reduction_size

                ''' only update precision when encounter accumulation (AC) adder level '''
                precision_counter = adder_details['output_precision']

        adder_levels_obj = [AdderLevel(idx, name, details) for idx, (name, details) in enumerate(adder_hierarchy.items())]
        self.adder_levels = adder_levels_obj


if __name__ == "__main__":

    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 8, 'D2': 3, 'D3': 2}
    operand_spatial_sharing = {'OS1': ((1, 0, 0), 'O'),
                       'OS2': ((0, 1, 0), 'I1'),
                       'OS3': ((0, 0, 1), 'I1'),
                       'OS4': ((1, 1, 0), 'I2')}

    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    user_defined_adder_hierarchy = {'AL1': {'type': 'AG', 'fan_in': 4, 'unit_cost': 0.08, 'unit_area': 0.03},
                                    'AL2': {'type': 'AC', 'output_precision': 24, 'unit_cost': 0.1, 'unit_area': 0.05},
                                    'AL3': {'type': 'AG', 'fan_in': 2, 'unit_cost': 0.13, 'unit_area': 0.07}}

    ah = AdderHierarchy(user_defined_adder_hierarchy, multiplier_array)
    a=1

