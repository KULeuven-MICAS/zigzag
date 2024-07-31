import copy
import math

from zigzag.datatypes import LayerDim, LayerOperand, OADimension, UnrollFactor
from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost
from zigzag.hardware.architecture.operational_array import OperationalArrayABC
from zigzag.mapping.Mapping import Mapping
from zigzag.mapping.spatial_mapping import MappingSingleOADim
from zigzag.workload.layer_node import LayerNode


class ImcUnit(OperationalArrayABC):
    """definition of general initialization function for D/AIMC"""

    TECH_PARAM_28NM = {
        "tech_node": 0.028,  # unit: um
        "vdd": 0.9,  # unit: V
        "nd2_cap": 0.7 / 1e3,  # unit: pF
        "wl_cap": 0.7 / 2 / 1e3,  # unit: pF (wordline cap of each SRAM cell is treated as NAND2_cap/2)
        "bl_cap": 0.7 / 2 / 1e3,  # unit: pF (bitline cap of each SRAM cell is treated as NAND2_cap/2)
        "xor2_cap": 0.7 * 1.5 / 1e3,  # unit: pF
        "dff_cap": 0.7 * 3 / 1e3,  # unit: pF
        "nd2_area": 0.614 / 1e6,  # unit: mm^2
        "xor2_area": 0.614 * 2.4 / 1e6,  # unit: mm^2
        "dff_area": 0.614 * 6 / 1e6,  # unit: mm^2
        "nd2_dly": 0.0478,  # unit: ns
        "xor2_dly": 0.0478 * 2.4,  # unit: ns
    }

    def __init__(
        self,
        is_analog_imc: bool,
        bit_serial_precision: int,
        input_precision: list[int],
        adc_resolution: int,
        cells_size: int,
        cells_area: float | None,
        dimension_sizes: dict[OADimension, int],
        auto_cost_extraction: bool = False,
    ):
        OperationalArrayABC.__init__(self, dimension_sizes=dimension_sizes)
        # initialization
        self.tech_param = ImcUnit.TECH_PARAM_28NM
        self.is_aimc = is_analog_imc
        self.bit_serial_precision = bit_serial_precision
        self.adc_resolution = adc_resolution
        self.cells_size = cells_size
        self.cells_area = cells_area
        self.auto_cost_extraction = auto_cost_extraction

        self.activation_precision = input_precision[0]
        self.weight_precision = input_precision[1]
        self.total_unit_count = math.prod(dimension_sizes.values())

        # wl_dim should be the same with the dimension served by input_reg.
        self.wl_dim = sorted(dimension_sizes.keys())[0]
        self.wordline_dim_size = dimension_sizes[self.wl_dim]
        # bl_dim should be the same with the dimension served by output_reg.
        self.bl_dim = sorted(dimension_sizes.keys())[1]
        self.bitline_dim_size = dimension_sizes[self.bl_dim]

        self.nb_of_banks = math.prod([dimension_sizes[oa_dim] for oa_dim in sorted(dimension_sizes.keys())[2:]])

        # parameters to be updated in function
        self.energy = None
        self.energy_breakdown = None
        self.area = None
        self.area_breakdown = None
        self.delay = None
        self.delay_breakdown = None
        self.mapped_rows_total_per_macro = None
        self.mapped_group_depth = None
        self.cells_w_cost = None

    def get_1b_adder_energy(self) -> float:
        """energy of 1b full adder
        Assume a 1b adder has 3 ND2 gate and 2 XOR2 gate"""
        adder_cap = 3 * self.tech_param["nd2_cap"] + 2 * self.tech_param["xor2_cap"]
        return adder_cap * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_adder_energy_half_activated(self) -> float:
        """energy of 1b full adder when 1 input is 0"""
        adder_cap = 2 * self.tech_param["xor2_cap"]
        return adder_cap * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_multiplier_energy(self) -> float:
        """energy of 1b multiplier
        1b mult includes 1 NOR gate, which is assumed as the same cost of ND2 gate
        why 0.5: considering weight stays constant during multiplication"""
        return 0.5 * self.tech_param["nd2_cap"] * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_reg_energy(self) -> float:
        """energy of 1b DFF"""
        return self.tech_param["dff_cap"] * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_adder_area(self) -> float:
        """area of 1b full adder
        Assume a 1b adder has 3 ND2 gate and 2 XOR2 gate"""
        adder_area = 3 * self.tech_param["nd2_area"] + 2 * self.tech_param["xor2_area"]
        return adder_area

    def get_1b_multiplier_area(self) -> float:
        """area of 1b multiplier
        1b mult includes 1 NOR gate, which is assumed as the same cost of ND2 gate"""
        return self.tech_param["nd2_area"]

    def get_1b_reg_area(self) -> float:
        """area of 1b DFF"""
        return self.tech_param["dff_area"]

    def get_1b_adder_dly_in2sum(self) -> float:
        """delay of 1b adder: input to sum-out"""
        adder_dly = 2 * self.tech_param["xor2_dly"]
        return adder_dly

    def get_1b_adder_dly_in2cout(self) -> float:
        """delay of 1b adder: input to carry-out"""
        adder_dly = self.tech_param["xor2_dly"] + 2 * self.tech_param["nd2_dly"]
        return adder_dly

    def get_1b_adder_dly_cin2cout(self) -> float:
        """delay of 1b adder: carry-in to carry-out"""
        adder_dly = 2 * self.tech_param["nd2_dly"]
        return adder_dly

    def get_1b_multiplier_dly(self) -> float:
        """delay of 1b multiplier
        1b mult includes 1 NOR gate, which is assumed as the same cost of ND2 gate"""
        return self.tech_param["nd2_dly"]

    @staticmethod
    def get_single_cell_array_cost_from_cacti(
        tech_node: float,
        wordline_dim_size: float,
        bitline_dim_size: float,
        cells_size: float,
        weight_precision: int,
    ) -> tuple[float, float, float, float]:
        """get the area, energy cost of a single macro (cell array) using CACTI
        this function is called when cacti is required for cost estimation
        @param tech_node:   the technology node (e.g. 0.028, 0.032, 0.022 ... unit: um)
        @param wordline_dim_size: the size of dimension where wordline is.
        @param bitline_dim_size: the size of dimension where bitline (adder tree) is.
        @param cells_size: the size of each cell group (unit: bit)
        @param weight_precision: weight precision (number of SRAM cells required to store a operand)
        """
        cell_array_size = wordline_dim_size * bitline_dim_size * cells_size / 8  # array size. unit: byte
        array_bw = wordline_dim_size * weight_precision  # imc array bandwidth. unit: bit

        # we will call cacti to get the area (mm^2), access_time (ns), r_cost (nJ/access), w_cost (nJ/access)
        if __name__ == "imc_unit":
            cacti_path = "../../cacti/cacti_master"
        else:
            cacti_path = "zigzag/cacti/cacti_master"
        access_time, area, r_cost, w_cost = get_cacti_cost(
            cacti_path=cacti_path,
            tech_node=tech_node,
            mem_type="sram",
            mem_size_in_byte=cell_array_size,
            bw=array_bw,
        )
        return access_time, area, r_cost, w_cost

    def get_mapped_oa_dim(
        self, layer: LayerNode, wl_dim: OADimension, bl_dim: OADimension
    ) -> tuple[float, float, float, float]:
        """
        get the mapped oa_dim in current mapping. The energy of unmapped oa_dim will be set to 0.
        """

        # activation/weight representation in layer
        layer_act_operand = layer.get_act_layer_op()
        layer_const_operand = layer.get_weight_layer_op()
        assert layer_const_operand is not None

        spatial_mapping = copy.deepcopy(layer.spatial_mapping)

        # Figure out the spatial mapping in a single macro
        spatial_mapping_size_in_macro = 1
        for oa_dim, loop in spatial_mapping.items():
            if oa_dim in [wl_dim, bl_dim]:  # serve the dimension inside the macro
                spatial_unroll_sizes: list[UnrollFactor] = loop.unroll_sizes
                spatial_mapping_size_in_macro *= math.prod(spatial_unroll_sizes)

        # We will derive how many number of PE columns and rows are mapping.
        # Energy of unmapped rows and columns will be set to 0.
        if wl_dim not in spatial_mapping.keys():
            mapped_cols_per_macro = 1
            weight_ir_loop_on_wl_dim = False  # if there is OX / OY mapped on wl dims
        else:
            spatial_mapping_on_wordline_dim: MappingSingleOADim = spatial_mapping[wl_dim]
            wl_dim_unroll_dims: list[LayerDim] = list(spatial_mapping_on_wordline_dim.layer_dims)
            wl_dim_unroll_sizes: list[UnrollFactor] = [
                spatial_mapping_on_wordline_dim[layer_dim] for layer_dim in wl_dim_unroll_dims
            ]
            mapped_cols_per_macro = math.prod(wl_dim_unroll_sizes)

            # Calculate the number of mapped rows in each macro.
            # As there might be OX / OY unrolling, resulting in a diagonal mapping, we will have a special check on that
            # Firstly check if there is OX / OY unrolling
            weight_ir_layer_dims: list[LayerDim] = layer.loop_relevancy_info.ir_dims[layer_const_operand]
            weight_ir_loop_on_wl_dim = False  # set default value
            if len(wl_dim_unroll_dims) == 1:  # single layer mapping (e.g. {"K", 2})
                weight_ir_loop_on_wl_dim = True if wl_dim_unroll_dims[0] in weight_ir_layer_dims else False
            else:  # mix layer_dim mapping (e.g. (("K",2), ("OX",2)) )
                for unroll_dim in wl_dim_unroll_dims:
                    if unroll_dim in weight_ir_layer_dims:
                        weight_ir_loop_on_wl_dim = True
                        break

        # Calculate total mapped number of rows
        if bl_dim in spatial_mapping:
            spatial_mapping_on_bitline_dim: MappingSingleOADim = spatial_mapping[bl_dim]
            bl_dim_unroll_sizes: list[UnrollFactor] = list(spatial_mapping_on_bitline_dim.unroll_sizes)
            # if False: mean there is no OX / OY unrolling on wl_dim, so no diagonal unrolling required
            if not weight_ir_loop_on_wl_dim:
                mapped_rows_total_per_macro = math.ceil(math.prod(bl_dim_unroll_sizes))
                mapped_rows_for_adder_per_macro = mapped_rows_total_per_macro
            else:
                spatial_mapping_on_wordline_dim: MappingSingleOADim = spatial_mapping[wl_dim]
                (
                    mapped_rows_total_per_macro,
                    mapped_rows_for_adder_per_macro,
                ) = ImcUnit.calculate_mapped_rows_when_diagonal_mapping_found(
                    layer,
                    layer_const_operand,
                    layer_act_operand,
                    spatial_mapping_on_wordline_dim,
                    spatial_mapping_on_bitline_dim,
                )
        else:  # there is no sm loop on bl_dim
            mapped_rows_total_per_macro = 1
            mapped_rows_for_adder_per_macro = 1

        # Get the number of time of activating macro
        # Note: it is normalized to a hardware that has only one macro (see equation below)
        # Equation = total MAC number of a layer/spatial mapping on a single macro
        macro_activation_times = layer.total_mac_count / float(spatial_mapping_size_in_macro)
        return (
            mapped_rows_total_per_macro,
            mapped_rows_for_adder_per_macro,
            mapped_cols_per_macro,
            macro_activation_times,
        )

    @staticmethod
    def calculate_mapped_rows_when_diagonal_mapping_found(
        layer: LayerNode,
        layer_const_operand: LayerOperand,
        layer_act_operand: LayerOperand,
        spatial_mapping_on_wordline_dim: MappingSingleOADim,
        spatial_mapping_on_bitline_dim: MappingSingleOADim,
    ) -> tuple[float, float]:
        """This function is used for calculating the total mapped number of rows when OX, OY unroll is found,
        which requires a diagonal data mapping.
        If OX, OY unroll does not exist, you can also use this function to calculate the total mapped number of rows.
        The only drawback is the simulation time is longer."""

        # First, fetch the dimension name of OX / OY (they are weight ir loops)
        weight_ir_layer_dims = layer.loop_relevancy_info.get_ir_layer_dims(layer_const_operand)
        # Second, we will find out what pr loops they pair with. Create a dict to record them down for later use.
        # For neural network, OX pairs with FX, OY with FY. So, it is assumed the pair size is in 2.
        act_pr_layer_dims = layer.loop_relevancy_info.get_pr_layer_dims(layer_act_operand)
        pr_sm: dict[LayerDim, dict[LayerDim, UnrollFactor]] = {}
        pr_sm_link: dict[LayerDim, LayerDim] = {}
        for [layer_dim1, layer_dim2] in act_pr_layer_dims.values():
            # for weight_ir_layer_dim in weight_ir_layer_dims:
            if layer_dim1 in weight_ir_layer_dims:
                pr_sm[layer_dim2] = {layer_dim1: 1}  # 1 by default, which means no mapping found
                pr_sm_link[layer_dim1] = layer_dim2
            else:  # layer_dim2 in weight_ir_layer_dims
                pr_sm[layer_dim1] = {layer_dim2: 1}  # 1 by default, which means no mapping found
                pr_sm_link[layer_dim2] = layer_dim1
        # Third, check if they are mapped on wl_dim and record down the mapped value if exist
        for weight_ir_layer_dim in weight_ir_layer_dims:
            pr_sm_key = pr_sm_link[weight_ir_layer_dim]
            wl_dim_unroll_dims: list[LayerDim] = list(spatial_mapping_on_wordline_dim.layer_dims)
            if len(wl_dim_unroll_dims) == 1:  # single layer mapping (e.g. ("K", 2))
                if weight_ir_layer_dim == wl_dim_unroll_dims[0]:
                    pr_sm[pr_sm_key][weight_ir_layer_dim] = spatial_mapping_on_wordline_dim[wl_dim_unroll_dims[0]]
            else:  # mix layer_dim mapping (e.g. (("K",2), ("OX",2)) )
                for unroll_dim in wl_dim_unroll_dims:
                    if weight_ir_layer_dim == unroll_dim:
                        # use *= in case there are multiple OX / OY in a mix sm loop
                        pr_sm[pr_sm_key][weight_ir_layer_dim] *= spatial_mapping_on_wordline_dim[unroll_dim]
        # Then, we calculate the total mapped number of rows
        # mapped_rows_total_per_macro: used for energy estimation of wordline and multipliers
        # mapped_rows_for_adder_per_macro: number of activated inputs of an adder tree,
        # used for energy estimation of adder trees
        bl_dim_unroll_dims = list(spatial_mapping_on_bitline_dim.layer_dims)
        bl_dim_unroll_sizes = spatial_mapping_on_bitline_dim.unroll_sizes
        if len(bl_dim_unroll_dims) == 1:  # single layer mapping
            layer_dim = bl_dim_unroll_dims[0]
            layer_dim_size = bl_dim_unroll_sizes[1]
            # pr_sm.keys() include FX, FY
            if layer_dim not in pr_sm:  # e.g. ("C", 2)
                additional_diag_rows = 0
            else:  # e.g. ("FX", 2)
                additional_diag_rows = list(pr_sm[layer_dim].values())[0] - 1
            mapped_rows_total_per_macro = layer_dim_size + additional_diag_rows
            mapped_rows_for_adder_per_macro = layer_dim_size
        else:  # mix layer_dim mapping (e.g. (("C",2), ("FX",2)) )
            # mapped_rows_total_per_macro = Cu * (OYu + FYu - 1) * (OXu + FXu - 1)
            # mapped_rows_for_adder_per_macro = Cu * FYu * FXu
            # In reality, OXu, OYu will not both exist. But the function still support this by the equation above.
            mapped_rows_total_per_macro = 1
            mapped_rows_for_adder_per_macro = 1
            for bl_dim_idx in range(len(bl_dim_unroll_dims)):
                layer_dim = bl_dim_unroll_dims[bl_dim_idx]
                layer_dim_size = bl_dim_unroll_sizes[bl_dim_idx]
                if layer_dim not in pr_sm:
                    additional_diag_rows = 0
                else:
                    additional_diag_rows = list(pr_sm[layer_dim].values())[0] - 1
                mapped_rows_total_per_macro *= layer_dim_size + additional_diag_rows
                mapped_rows_for_adder_per_macro *= layer_dim_size
        # Lastly, ceil to an upper integer, as required in the adder-trees model.
        mapped_rows_total_per_macro = math.ceil(mapped_rows_total_per_macro)
        mapped_rows_for_adder_per_macro = math.ceil(mapped_rows_for_adder_per_macro)
        return mapped_rows_total_per_macro, mapped_rows_for_adder_per_macro

    def get_precharge_energy(
        self, tech_param: dict[str, float], layer: LayerNode, mapping: Mapping
    ) -> tuple[float, float]:
        # calculate pre-charging energy on local bitlines for specific layer and mapping
        # also calculate mapped group depth (number of weights stored in a cell group)
        group_depth = self.cells_size / self.weight_precision
        if group_depth > 1:
            # Pre-charge operation is required on local bitline if group_depth > 1
            # The final pre-charge energy = energy/PE * nb_of_precharge_times
            # nb_of_precharge_times is normalized to single PE.

            # activation/weight representation in layer
            layer_const_operand = layer.get_weight_layer_op()
            assert layer_const_operand is not None
            # Get the precharge interval between two precharge operations
            precharge_interval = 1  # 1: precharge every cycle
            tm_loops_in_cell_group = mapping.temporal_mapping.mapping_dic_origin[layer_const_operand][0]
            # As loops close to the beginning will be executed firstly, we will count how many weight ir loops there are
            # until we reach a weight r loop
            weight_r_layer_dims = layer.loop_relevancy_info.get_r_layer_dims(layer_const_operand)
            weight_ir_layer_dims = layer.loop_relevancy_info.get_ir_layer_dims(layer_const_operand)

            for loop_name, loop_size in tm_loops_in_cell_group:
                if loop_name in weight_ir_layer_dims:
                    precharge_interval *= loop_size
                else:
                    break  # break when we meet the first ir loop of weight
            # Equation: nb_of_precharge_times = rd_out_to_low_count_of_lowest_weight_mem / precharge_intervals
            nb_of_precharge_times = (
                mapping.unit_mem_data_movement[layer_const_operand][0].data_elem_move_count.rd_out_to_low
                / precharge_interval
            )
            single_pe_precharge_energy = (
                (tech_param["wl_cap"] * (tech_param["vdd"] ** 2))
                + (tech_param["bl_cap"] * (tech_param["vdd"] ** 2) * group_depth)
            ) * (self.weight_precision)
            energy_precharging = single_pe_precharge_energy * nb_of_precharge_times
            # Calculate mapped_group_depth
            mapped_group_depth = 1
            for loop_name, loop_size in tm_loops_in_cell_group:
                if loop_name in weight_r_layer_dims:
                    mapped_group_depth *= loop_size
        else:
            energy_precharging = 0
            mapped_group_depth = 1
        return energy_precharging, mapped_group_depth

    def get_regular_adder_trees_energy(
        self,
        adder_input_precision: int,
        active_inputs_number: float,
        physical_inputs_number: float,
    ) -> float:
        """
        get the energy spent on regular RCA adder trees without place values
        """
        adder_depth = math.log2(physical_inputs_number)
        adder_depth = int(adder_depth)  # float -> int for simplicity
        mapped_inputs = active_inputs_number  # number of used inputs for an adder tree

        # nb of 1b adders in a single adder tree
        nb_of_1b_adder = physical_inputs_number * (adder_input_precision + 1) - (
            adder_input_precision + adder_depth + 1
        )

        # In the adders' model, we classify the basic FA (1-b full adder) as two types:
        # 1. fully activated FA: two of its inputs having data comes in. (higher energy cost)
        # 2. half activated FA: only one of its inputs having data comes in.
        # The 2nd type has lower energy cost, because no carry will be generated and the carry path stays unchanged.
        # Below we figure out how many there are of fully activated FA and half activated FA
        if mapped_inputs >= 1:
            if mapped_inputs >= physical_inputs_number:
                # @param fully_activated_number_of_1b_adder: fully activated 1b adder, probably will produce a carry
                # @param half_activated_number_of_1b_adder: only 1 input is activate and the other port is 0, so carry
                # path is activated.
                fully_activated_number_of_1b_adder = nb_of_1b_adder
                half_activated_number_of_1b_adder = 0
            else:
                # find out fully_activated_number_of_1b_adder and half_activated_number_of_1b_adder when inputs are not
                # fully mapped.
                # method: iteratively check if left_input is bigger or smaller than baseline, which will /2 each time,
                # until left_input == 1
                # @param left_input: the number of inputs waiting for processing
                # @param baseline: serves as references for left_input
                fully_activated_number_of_1b_adder = 0
                half_activated_number_of_1b_adder = 0
                left_input = mapped_inputs
                baseline = physical_inputs_number
                while left_input != 0:
                    baseline = baseline / 2
                    activated_depth = int(math.log2(baseline))
                    if left_input <= 1 and baseline == 1:  # special case
                        fully_activated_number_of_1b_adder += 0
                        half_activated_number_of_1b_adder += adder_input_precision
                        left_input = 0
                    elif left_input > baseline:
                        fully_activated_number_of_1b_adder += (
                            baseline * (adder_input_precision + 1)
                            - (adder_input_precision + activated_depth + 1)
                            + (adder_input_precision + activated_depth)
                        )
                        half_activated_number_of_1b_adder += 0
                        left_input = left_input - baseline
                    elif left_input < baseline:
                        half_activated_number_of_1b_adder += adder_input_precision + activated_depth
                    else:  # left_input == baseline
                        fully_activated_number_of_1b_adder += baseline * (adder_input_precision + 1) - (
                            adder_input_precision + activated_depth + 1
                        )
                        half_activated_number_of_1b_adder += adder_input_precision + activated_depth
                        left_input = left_input - baseline

            single_adder_tree_energy = (
                fully_activated_number_of_1b_adder * self.get_1b_adder_energy()
                + half_activated_number_of_1b_adder * self.get_1b_adder_energy_half_activated()
            )
        else:
            single_adder_tree_energy = 0
        return single_adder_tree_energy
