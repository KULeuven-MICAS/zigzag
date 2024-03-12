import numpy as np
import math
import copy
if __name__ == "__main__":
    from imc_unit import ImcUnit
    import logging as _logging
    _logging_level = _logging.INFO
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)
else:
    import logging as _logging
    from zigzag.classes.hardware.architecture.imc_unit import ImcUnit

###############################################################################################################
# README
#   . class DimcArrayUnit (defines the energy/area/delay cost of a DIMC array)
# How to use this file?
#   . This file is internally called in ZigZag-IMC framework.
#   . It can also be run independently, for mainly debugging. An example is given at the end of the file.
###############################################################################################################

class DimcArrayUnit(ImcUnit):
    """definition of a Digtal In-SRAM-Computing (DIMC) array"""
    """
    constraint:
        -- activation precision must be in the power of 2.
        -- input_bit_per_cycle must be in the power of 2.
        -- 
    assumption:
    """
    def __init__(self,tech_param:dict, hd_param:dict, dimensions:dict):
        super().__init__(tech_param, hd_param, dimensions)

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        # not implemented
        #return {"operational_unit": self.unit, "dimensions": self.dimensions}
        pass

    ## area of imc macros (cells, mults, adders, adders_pv, accumulators. Not include input/output regs)
    def get_area(self):
        # area of cell array
        tech_node = self.logic_unit.tech_param["tech_node"]
        group_depth = self.hd_param["group_depth"]
        w_pres = self.hd_param["weight_precision"]
        if self.hd_param["enable_cacti"] == True:
            single_cell_array_area = self.get_single_cell_array_cost_from_cacti(tech_node,
                                                                                self.wl_dim_size,
                                                                                self.bl_dim_size,
                                                                                group_depth,
                                                                                w_pres)[1]
            # at this point, we have the area of single cell array. Then multiply it with the number of banks.
            area_cells = single_cell_array_area * self.nb_of_banks # total cell array area in the core
        else:
            # TODO: [jiacong] [TO BE SUPPORTED OR YOU CAN MODIFY YOURSELF]
            area_cells = None  # user-provided cell array area (from somewhere?)
            raise Exception(f"User-provided cell area is not supported yet.")

        """area of multiplier array"""
        area_mults = self.logic_unit.get_1b_multiplier_area() * self.hd_param["input_bit_per_cycle"] * \
                     w_pres * self.wl_dim_size * self.bl_dim_size * self.nb_of_banks

        """area of adder trees (type: RCA)"""
        adder_input_pres = w_pres # input precision of the adder tree
        nb_inputs_of_adder = self.bl_dim_size # the number of inputs of the adder tree
        adder_depth = math.log2(nb_inputs_of_adder)
        assert adder_depth%1==0, \
            f"[DimcArray] The number of inputs [{nb_inputs_of_adder}] for the adder tree is not in the power of 2."
        adder_depth = int(adder_depth) # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth # output precision of the adder tree
        nb_of_1b_adder_in_single_adder_tree = nb_inputs_of_adder * (adder_input_pres+1) - (adder_input_pres+adder_depth+1) # nb of 1b adders in a single adder tree
        nb_of_adder_trees = self.hd_param["input_bit_per_cycle"] * self.wl_dim_size * self.nb_of_banks
        area_adders = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_in_single_adder_tree * nb_of_adder_trees

        """area of extra adders with place values (pv) when input_bit_per_cycle>1 (type: RCA)"""
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        if nb_inputs_of_adder_pv == 1:
            nb_of_1b_adder_pv = 0 # number of 1b adder in an pv_adder tree
            nb_of_adder_trees_pv = 0 # number of pv_adder trees
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            input_precision_pv = adder_output_pres
            assert adder_depth_pv%1==0, \
                f"[DimcArray] The value [{nb_inputs_of_adder_pv}] of [input_bit_per_cycle] is not in the power of 2."
            adder_depth_pv = int(adder_depth_pv) # float -> int for simplicity
            nb_of_1b_adder_pv = input_precision_pv * (nb_inputs_of_adder_pv-1) + nb_inputs_of_adder_pv * (adder_depth_pv-0.5) # nb of 1b adders in a single place-value adder tree
            nb_of_adder_trees_pv = self.wl_dim_size * self.nb_of_banks
        area_adders_pv = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_pv * nb_of_adder_trees_pv

        """area of accumulators (adder type: RCA)"""
        if self.hd_param["input_bit_per_cycle"] == self.hd_param["input_precision"]:
            area_accumulators = 0
        else:
            accumulator_output_pres = self.hd_param["input_precision"]+self.hd_param["weight_precision"]+math.log2(self.bl_dim_size)
            nb_of_1b_adder_accumulator = accumulator_output_pres * self.wl_dim_size * self.nb_of_banks # number of 1b adder in all accumulators
            nb_of_1b_reg_accumulator = nb_of_1b_adder_accumulator # number of regs in all accumulators
            area_accumulators = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_accumulator + \
                                self.logic_unit.get_1b_reg_area() * nb_of_1b_reg_accumulator
        """total area of imc"""
        self.area_breakdown = { # unit: same with in input hd file
            "cells":    area_cells,
            "mults":    area_mults,
            "adders":   area_adders,
            "adders_pv":area_adders_pv,
            "accumulators": area_accumulators
        }
        self.area = sum([v for v in self.area_breakdown.values()])
        # return self.area_breakdown

    def get_delay(self):
        """delay of imc arrays (worst path: mults -> adders -> adders_pv -> accumulators) """
        """ unit: ns (if CACTI is used). whatever it can be otherwise. """
        dly_mults = self.logic_unit.get_1b_multiplier_dly()

        """delay of adders (tree) (type: RCA)"""
        adder_input_pres = self.hd_param["weight_precision"]
        nb_inputs_of_adder = self.bl_dim_size
        adder_depth = math.log2(nb_inputs_of_adder)
        assert adder_depth%1==0, \
            f"[DimcArray] The number of inputs [{nb_inputs_of_adder}] for the adder tree is not in the power of 2."
        adder_depth = int(adder_depth)  # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth
        dly_adders = (adder_depth-1) * self.logic_unit.get_1b_adder_dly_in2sum() + \
                     self.logic_unit.get_1b_adder_dly_in2cout() + \
                     (adder_output_pres-1-1) * self.logic_unit.get_1b_adder_dly_cin2cout()

        """delay of adders_pv (type: RCA)"""
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        if nb_inputs_of_adder_pv == 1:
            dly_adders_pv = 0
            accumulator_input_pres = adder_output_pres
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            assert adder_depth_pv%1==0, \
                f"[DimcArray] The value [{nb_inputs_of_adder_pv}] of [input_bit_per_cycle] is not in the power of 2."
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            adder_pv_input_precision = adder_output_pres
            adder_pv_output_precision = nb_inputs_of_adder_pv + adder_output_pres  # output precision from adders_pv (depth + input_precision)
            accumulator_input_pres = adder_pv_output_precision
            dly_adders_pv = (adder_depth_pv - 1) * self.logic_unit.get_1b_adder_dly_in2sum() + self.logic_unit.get_1b_adder_dly_in2cout() + (adder_pv_output_precision - adder_pv_input_precision-1) * self.logic_unit.get_1b_adder_dly_cin2cout()

        """delay of accumulators (adder type: RCA)"""
        accumulator_output_pres = self.hd_param["input_precision"] + self.hd_param["weight_precision"] + math.log2(self.bl_dim_size)
        accumulator_output_pres = int(accumulator_output_pres) # float -> int for simplicity
        if accumulator_output_pres == accumulator_input_pres: # no accumulator
            dly_accumulators = 0
        else:
            dly_accumulators = self.logic_unit.get_1b_adder_dly_in2cout() + \
                               (accumulator_output_pres - accumulator_input_pres) * self.logic_unit.get_1b_adder_dly_cin2cout()

        """total delay of imc"""
        self.delay_breakdown = {
            "mults":    dly_mults,
            "adders":   dly_adders,
            "adders_pv":dly_adders_pv,
            "accumulators": dly_accumulators
        }
        self.delay = sum([v for v in self.delay_breakdown.values()])
        # return self.delay_breakdown

    def get_peak_energy_single_cycle(self):
        """
        macro-level one-cycle energy of imc arrays (fully utilization, no weight updating)
        (components: cells, mults, adders, adders_pv, accumulators. Not include input/output regs)
        """
        w_pres = self.hd_param["weight_precision"]
        """energy of precharging"""
        energy_precharging = 0

        """energy of multiplier array"""
        nb_of_mults = self.hd_param["input_bit_per_cycle"] * \
                     w_pres * self.wl_dim_size * self.bl_dim_size * self.nb_of_banks
        energy_mults = self.logic_unit.get_1b_multiplier_energy() * nb_of_mults

        """energy of adder trees (type: RCA)"""
        adder_input_pres = w_pres # input precision of the adder tree
        nb_inputs_of_adder = self.bl_dim_size # the number of inputs of the adder tree
        adder_depth = math.log2(nb_inputs_of_adder)
        assert adder_depth%1==0, \
            f"[DimcArray] The number of inputs [{nb_inputs_of_adder}] for the adder tree is not in the power of 2."
        adder_depth = int(adder_depth) # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth # output precision of the adder tree
        nb_of_1b_adder_in_single_adder_tree = nb_inputs_of_adder * (adder_input_pres+1) - (adder_input_pres+adder_depth+1) # nb of 1b adders in a single adder tree
        nb_of_adder_trees = self.hd_param["input_bit_per_cycle"] * self.wl_dim_size * self.nb_of_banks
        energy_adders = self.logic_unit.get_1b_adder_energy() * nb_of_1b_adder_in_single_adder_tree * nb_of_adder_trees

        """energy of adders_pv (type: RCA)"""
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = adder_output_pres
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (math.log2(nb_inputs_of_adder_pv) - 0.5)
            nb_of_adder_trees_pv = self.wl_dim_size * self.nb_of_banks
            energy_adders_pv = self.logic_unit.get_1b_adder_energy() * nb_of_1b_adder_pv * nb_of_adder_trees_pv

        """energy of accumulators (adder type: RCA)"""
        if self.hd_param["input_bit_per_cycle"] == self.hd_param["input_precision"]:
            energy_accumulators = 0
        else:
            accumulator_output_pres = self.hd_param["input_precision"]+self.hd_param["weight_precision"]+math.log2(self.bl_dim_size)
            nb_of_1b_adder_accumulator = accumulator_output_pres * self.wl_dim_size * self.nb_of_banks # number of 1b adder in all accumulators
            nb_of_1b_reg_accumulator = nb_of_1b_adder_accumulator # number of regs in all accumulators
            energy_accumulators = self.logic_unit.get_1b_adder_energy() * nb_of_1b_adder_accumulator + \
                                self.logic_unit.get_1b_reg_energy() * nb_of_1b_reg_accumulator

        peak_energy_breakdown = {  # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "mults": energy_mults,
            "adders": energy_adders,
            "adders_pv": energy_adders_pv,
            "accumulators": energy_accumulators
        }
        # peak_energy = sum([v for v in peak_energy_breakdown.values()])
        return peak_energy_breakdown

    def get_macro_level_peak_performance(self):
        """
        macro-level peak performance of imc arrays (fully utilization, no weight updating)
        """
        nb_of_macs_per_cycle = self.wl_dim_size * self.bl_dim_size / \
                               (self.hd_param["input_precision"] / self.hd_param["input_bit_per_cycle"]) * \
                               self.nb_of_banks

        self.get_area()
        self.get_delay()

        clock_cycle_period = self.delay # unit: ns
        peak_energy_per_cycle = sum([v for v in self.get_peak_energy_single_cycle().values()]) # unit: pJ
        imc_area = self.area # unit: mm^2

        tops_peak = nb_of_macs_per_cycle * 2 / clock_cycle_period / 1000
        topsw_peak = nb_of_macs_per_cycle * 2 / peak_energy_per_cycle
        topsmm2_peak = tops_peak / imc_area

        logger = _logging.getLogger(__name__)
        logger.info(f"Current macro-level peak performance:")
        logger.info(f"TOP/s: {tops_peak}, TOP/s/W: {topsw_peak}, TOP/s/mm^2: {topsmm2_peak}")

        return tops_peak, topsw_peak, topsmm2_peak

    @staticmethod
    def calculate_mapped_rows_total_when_diagonal_mapping_found(layer, layer_const_operand, layer_act_operand, sm_on_wl_dim, sm_on_bl_dim):
        # This function is used for calcualting the total mapped number of rows when OX, OY unroll is found,
        # which requires a diagonal data mapping.
        # If OX, OY unroll does not exist, you can also use this function to calculate the total mapped number of rows.
        # The only drawback is the simulation time is longer.
        # First, fetch the dimension name of OX / OY (they are weight ir loops)
        weight_ir_layer_dims: list = layer.operand_loop_dim[layer_const_operand]["ir"]
        # Second, we will find out what pr loops they pair with. Create a dict to record them down for later use.
        # For neural network, OX pairs with FX, OY with FY. So, it is assumed the pair size is in 2.
        act_pr_layer_dims: dict = layer.operand_loop_dim[layer_act_operand]["pr"]
        pr_sm: dict = {}
        pr_sm_link: dict = {}
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
            if isinstance(sm_on_wl_dim[0], str):  # single layer mapping (e.g. ("K", 2))
                if weight_ir_layer_dim == sm_on_wl_dim[0]:
                    pr_sm[pr_sm_key][weight_ir_layer_dim] = sm_on_wl_dim[1]
            else:  # mix layer_dim mapping (e.g. (("K",2), ("OX",2)) )
                for element in sm_on_wl_dim:
                    if weight_ir_layer_dim == element[0]:
                        # use *= in case there are multiple OX / OY in a mix sm loop
                        pr_sm[pr_sm_key][weight_ir_layer_dim] *= element[1]
        # Then, we calculate the total mapped number of rows
        # mapped_rows_total: used for energy estimation of wordline and multipliers
        # mapped_rows_for_adder: number of activated inputs of an adder tree, used for energy estimation of adder trees
        if isinstance(sm_on_bl_dim[0], str):  # single layer mapping
            layer_dim = sm_on_bl_dim[0]
            layer_dim_size = sm_on_bl_dim[1]
            # pr_sm.keys() include FX, FY
            if layer_dim not in pr_sm.keys(): # e.g. ("C", 2)
                additional_diag_rows = 0
            else: # e.g. ("FX", 2)
                additional_diag_rows = list(pr_sm[layer_dim].values())[0] - 1
            mapped_rows_total = layer_dim_size + additional_diag_rows
            mapped_rows_for_adder = layer_dim_size
        else:  # mix layer_dim mapping (e.g. (("C",2), ("FX",2)) )
            # mapped_rows_total = Cu * (OYu + FYu - 1) * (OXu + FXu - 1)
            # mapped_rows_for_adder = Cu * FYu * FXu
            # In reality, OXu, OYu will not both exist. But the function still support this by the equation above.
            mapped_rows_total = 1
            mapped_rows_for_adder = 1
            for element in sm_on_bl_dim:
                layer_dim = element[0]
                layer_dim_size = element[1]
                if layer_dim not in pr_sm.keys():
                    additional_diag_rows = 0
                else:
                    additional_diag_rows = list(pr_sm[layer_dim].values())[0] - 1
                mapped_rows_total *= (layer_dim_size + additional_diag_rows)
                mapped_rows_for_adder *= layer_dim_size
        # Lastly, ceil to an upper integer, as required in the adder-trees model.
        mapped_rows_total = math.ceil(mapped_rows_total)
        mapped_rows_for_adder = math.ceil(mapped_rows_for_adder)
        return mapped_rows_total, mapped_rows_for_adder

    @staticmethod
    def get_mapped_oa_dim(layer, wl_dim, bl_dim):
        """
        get the mapped oa_dim in current mapping. The energy of unmapped oa_dim will be set to 0.
        """

        layer_const_operand = layer.constant_operands[0]  # weight representation
        layer_act_operand = [operand for operand in layer.input_operands if operand != layer_const_operand][0]  # activation representation

        spatial_mapping = copy.deepcopy(layer.user_spatial_mapping)

        # Figure out the spatial mapping in a single macro
        spatial_mapping_in_macro = []
        for layer_dim, loop in spatial_mapping.items():
            if layer_dim in [wl_dim, bl_dim]:  # serve the dimension inside the macro
                if isinstance(loop[0], str):  # single layer_dim unrolling
                    spatial_mapping_in_macro.append(loop)
                else:  # mix layer_dim unrolling
                    for element in loop:
                        spatial_mapping_in_macro.append(element)

        # We will firstly derive how many number of PE columns and rows are mapping.
        # Later, energy of unmapped rows and columns will be set to 0.
        # We start from deriving the number of mapped columns in each macro.
        # the sm loop would do not exist if did not find any
        if wl_dim not in spatial_mapping.keys():
            mapped_cols = 1 # mapped number of wl dims
            weight_ir_loop_on_wl_dim = False  # if there is OX / OY mapped on wl dims
        else:
            sm_on_wl_dim = spatial_mapping[wl_dim]  # spatial mapping on wl_dimension
            if isinstance(sm_on_wl_dim[0], str):  # single layer mapping (e.g. ("K", 2))
                mapped_cols = sm_on_wl_dim[1]  # floating number is also supported for calculation
            else:  # mix layer_dim mapping (e.g. (("K",2), ("OX",2)) )
                mapped_cols = math.prod([v[1] for v in sm_on_wl_dim])
            # We then calculate the number of mapped rows in each macro.
            # As there might be OX / OY unrolling, which results in a diagonal mapping, we will have a special check on that
            # Firstly check if there is OX / OY unrolling
            weight_ir_layer_dims: list = layer.operand_loop_dim[layer_const_operand]["ir"]
            weight_ir_loop_on_wl_dim = False  # set default value
            if isinstance(sm_on_wl_dim[0], str):  # single layer mapping (e.g. ("K", 2))
                weight_ir_loop_on_wl_dim = True if sm_on_wl_dim[0] in weight_ir_layer_dims else False
            else:  # mix layer_dim mapping (e.g. (("K",2), ("OX",2)) )
                for element in sm_on_wl_dim:
                    layer_dim = element[0]
                    if layer_dim in weight_ir_layer_dims:
                        weight_ir_loop_on_wl_dim = True
                        break

        # Calculate total mapped number of rows
        if bl_dim in spatial_mapping.keys():
            sm_on_bl_dim = spatial_mapping[bl_dim]  # spatial mapping on bl_dimension
            if not weight_ir_loop_on_wl_dim:  # if False: mean there is no OX / OY unrolling on wl_dim, so no diagonal unrolling required
                if isinstance(sm_on_bl_dim[0], str):  # single layer mapping (e.g. ("FX", 2))
                    mapped_rows_total = sm_on_bl_dim[1]  # floating number is also supported for calculation
                else:  # mix layer_dim mapping (e.g. (("C",2), ("FX",2)) )
                    mapped_rows_total = math.prod([v[1] for v in sm_on_bl_dim])
                mapped_rows_total = math.ceil(mapped_rows_total)  # must be an integer, as it is used for adder trees.
                mapped_rows_for_adder = mapped_rows_total
            else:
                mapped_rows_total, mapped_rows_for_adder = DimcArrayUnit.calculate_mapped_rows_total_when_diagonal_mapping_found(
                    layer,
                    layer_const_operand,
                    layer_act_operand,
                    sm_on_wl_dim,
                    sm_on_bl_dim)
        else: # there is no sm loop on bl_dim
            mapped_rows_total = 1
            mapped_rows_for_adder = 1

        # Get the number of time of activating macro
        # Note: it is normalized to a hardware that has only one macro (see equation below)
        # Equation = total MAC number of a layer/spatial mapping on a single macro
        macro_activation_times = layer.total_MAC_count / np.prod([x[1] for x in spatial_mapping_in_macro])
        return mapped_rows_total, mapped_rows_for_adder, mapped_cols, macro_activation_times

    @staticmethod
    def get_precharge_energy(hd_param, tech_param, layer, mapping):
        # calculate pre-charging energy on local bitlines for specific layer and mapping
        # also calculate mapped group depth (number of weights stored in a cell group)
        group_depth = hd_param["group_depth"]
        if group_depth > 1:
            # Pre-charge operation is required on local bitline if group_depth > 1
            # The final pre-charge energy = energy/PE * nb_of_precharge_times
            # nb_of_precharge_times is normalized to single PE.
            layer_const_operand = layer.constant_operands[0]  # weight representation
            # Get the precharge interval between two precharge operations
            precharge_interval = 1  # 1: precharge every cycle
            tm_loops_in_cell_group: list = mapping.temporal_mapping.mapping_dic_origin[layer_const_operand][0]
            # As loops close to the beginning will be executed firstly, we will count how many weight ir loops there are
            # until we reach a weight r loop
            weight_r_layer_dims: list = layer.operand_loop_dim[layer_const_operand]["r"]
            weight_ir_layer_dims: list = layer.operand_loop_dim[layer_const_operand]["ir"]
            for (loop_name, loop_size) in tm_loops_in_cell_group:
                if loop_name in weight_ir_layer_dims:
                    precharge_interval *= loop_size
                else:
                    break  # break when we meet the first ir loop of weight
            # Equation: nb_of_precharge_times = rd_out_to_low_count_of_lowest_weight_mem / precharge_intervals
            nb_of_precharge_times = mapping.unit_mem_data_movement[layer_const_operand][0].data_elem_move_count.rd_out_to_low / precharge_interval
            single_pe_precharge_energy = ((tech_param["wl_cap"] * (tech_param["vdd"] ** 2)) + \
                                          (tech_param["bl_cap"] * (tech_param["vdd"] ** 2) * group_depth)) * \
                                         (hd_param["weight_precision"])
            energy_precharging = single_pe_precharge_energy * nb_of_precharge_times
            # Calculate mapped_group_depth
            mapped_group_depth = 1
            for (loop_name, loop_size) in tm_loops_in_cell_group:
                if loop_name in weight_r_layer_dims:
                    mapped_group_depth *= loop_size
        else:
            energy_precharging = 0
            mapped_group_depth = 1
        return energy_precharging, mapped_group_depth

    def get_mults_energy(self, hd_param, logic_unit, layer, mapped_rows_total, wl_dim_size, macro_activation_times) -> float:
        """
        calculate energy spent on multipliers for specific layer and mapping
        """
        layer_const_operand = layer.constant_operands[0]  # weight representation
        layer_act_operand = [operand for operand in layer.input_operands if operand != layer_const_operand][0]  # activation representation
        layer_act_operand_pres = layer.operand_precision[layer_act_operand]
        nb_of_mapped_mults_in_macro = hd_param["weight_precision"] * hd_param["input_bit_per_cycle"] * \
                                      mapped_rows_total * wl_dim_size
        nb_of_activation_times = macro_activation_times * \
                                 (layer_act_operand_pres / hd_param["input_bit_per_cycle"])
        energy_mults = logic_unit.get_1b_multiplier_energy() * nb_of_mapped_mults_in_macro * nb_of_activation_times
        return energy_mults

    def get_adder_trees_energy(self, layer, logic_unit, mapped_rows_for_adder, bl_dim_size, mapped_cols, layer_act_operand_pres, macro_activation_times):
        """
        get the energy spent on RCA adder trees for specific layer and mapping
        """
        layer_const_operand = layer.constant_operands[0]  # weight representation
        layer_const_operand_pres = layer.operand_precision[layer_const_operand]
        nb_inputs_of_adder = bl_dim_size  # physical number of inputs in a single adder tree
        adder_depth = math.log2(nb_inputs_of_adder)
        assert nb_inputs_of_adder % 1 == 0, \
            f"The number of inputs for an adder tree [{nb_inputs_of_adder}] is not in the power of 2."
        adder_depth = int(adder_depth)  # float -> int for simplicity
        mapped_inputs = mapped_rows_for_adder  # number of used inputs for an adder tree
        adder_input_pres = layer_const_operand_pres  # input precision for a single adder tree
        adder_output_pres = adder_input_pres + adder_depth
        nb_of_1b_adder = nb_inputs_of_adder * (adder_input_pres + 1) - (adder_input_pres + adder_depth + 1)  # nb of 1b adders in a single adder tree

        # In the adders' model, we classify the basic FA (1-b full adder) as two types:
        # 1. fully activated FA: two of its inputs having data comes in. (higher energy cost)
        # 2. half activated FA: only one of its inputs having data comes in.
        # The 2nd type has lower energy cost, because no carry will be generated and the carry path stays unchanged.
        # Below we figure out how many there are of fully activated FA and half activated FA
        if mapped_inputs >= 1:
            if mapped_inputs >= nb_inputs_of_adder:
                """
                :param fully_activated_number_of_1b_adder: fully activated 1b adder, probably will produce a carry
                :param half_activated_number_of_1b_adder: only 1 input is activate and the other port is 0, so carry path is activated.
                """
                fully_activated_number_of_1b_adder = nb_of_1b_adder
                half_activated_number_of_1b_adder = 0
            else:
                """
                find out fully_activated_number_of_1b_adder and half_activated_number_of_1b_adder when inputs are not fully mapped.
                method: iteratively check if left_input is bigger or smaller than baseline, which will /2 each time, until left_input == 1
                :param left_input: the number of inputs waiting for processing
                :param baseline: serves as references for left_input
                """
                fully_activated_number_of_1b_adder = 0
                half_activated_number_of_1b_adder = 0
                left_input = mapped_inputs
                baseline = nb_inputs_of_adder
                while left_input != 0:
                    baseline = baseline / 2
                    activated_depth = int(math.log2(baseline))
                    if left_input <= 1 and baseline == 1:  # special case
                        fully_activated_number_of_1b_adder += 0
                        half_activated_number_of_1b_adder += adder_input_pres
                        left_input = 0
                    elif left_input > baseline:
                        fully_activated_number_of_1b_adder += baseline * (adder_input_pres + 1) - (adder_input_pres + activated_depth + 1) + (adder_input_pres + activated_depth)
                        half_activated_number_of_1b_adder += 0
                        left_input = left_input - baseline
                    elif left_input < baseline:
                        half_activated_number_of_1b_adder += adder_input_pres + activated_depth
                    else:  # left_input == baseline
                        fully_activated_number_of_1b_adder += baseline * (adder_input_pres + 1) - (adder_input_pres + activated_depth + 1)
                        half_activated_number_of_1b_adder += adder_input_pres + activated_depth
                        left_input = left_input - baseline

            single_adder_tree_energy = fully_activated_number_of_1b_adder * logic_unit.get_1b_adder_energy() + \
                                       half_activated_number_of_1b_adder * logic_unit.get_1b_adder_energy_half_activated()
            nb_of_activation_times = mapped_cols * layer_act_operand_pres * macro_activation_times
            energy_adders = single_adder_tree_energy * nb_of_activation_times
        else:
            energy_adders = 0
        return energy_adders, adder_output_pres

    def get_adder_pv_energy(self, nb_inputs_of_adder_pv, input_precision, logic_unit, layer_act_operand_pres, input_bit_per_cycle, mapped_cols, macro_activation_times):
        """
        get the energy for adder tree with input having place value (pv)
        """
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = input_precision
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (math.log2(nb_inputs_of_adder_pv) - 0.5)
            nb_of_activation_times = mapped_cols * layer_act_operand_pres / input_bit_per_cycle * macro_activation_times
            energy_adders_pv = logic_unit.get_1b_adder_energy() * nb_of_1b_adder_pv * nb_of_activation_times
        return energy_adders_pv

    def get_energy_for_a_layer(self, layer, mapping):
        """
        get the imc array energy for specific layer with specific mapping
        """
        """check if operand precision defined in the layer is the same with in hardware template"""

        layer_const_operand = layer.constant_operands[0] # weight representation
        layer_const_operand_pres = layer.operand_precision[layer_const_operand]
        layer_act_operand = [operand for operand in layer.input_operands if operand != layer_const_operand][0] # activation representation
        layer_act_operand_pres = layer.operand_precision[layer_act_operand]
        weight_pres_in_hd_param = self.hd_param["weight_precision"]
        act_pres_in_hd_param = self.hd_param["input_precision"]

        # currently in the energy model, the input and weight precision defined in the workload file should be the same with in the hd input file.
        # this check can be removed if variable precision is supported in the future.
        assert layer_const_operand_pres == weight_pres_in_hd_param, \
                f"Weight precision defined in the workload [{layer_const_operand_pres}] not equal to the one defined in the hardware hd_param [{weight_pres_in_hd_param}]."
        assert layer_act_operand_pres == act_pres_in_hd_param, \
                f"Activation precision defined in the workload [{layer_act_operand_pres}] not equal to the one defined in the hardware hd_param [{act_pres_in_hd_param}]."

        """parameter extraction"""
        mapped_rows_total, mapped_rows_for_adder, mapped_cols, macro_activation_times = DimcArrayUnit.get_mapped_oa_dim(layer, self.wl_dim, self.bl_dim)
        self.mapped_rows_total = mapped_rows_total

        """energy calculation"""
        """energy of precharging"""
        energy_precharging, mapped_group_depth = DimcArrayUnit.get_precharge_energy(self.hd_param, self.logic_unit.tech_param, layer, mapping)
        self.mapped_group_depth = mapped_group_depth

        """energy of multiplier array"""
        energy_mults = self.get_mults_energy(self.hd_param, self.logic_unit, layer, mapped_rows_total, self.wl_dim_size, macro_activation_times)

        """energy of adder trees (type: RCA)"""
        energy_adders, adder_output_pres = self.get_adder_trees_energy(layer, self.logic_unit, mapped_rows_for_adder,
                                           self.bl_dim_size, mapped_cols, layer_act_operand_pres, macro_activation_times)

        """energy of adders_pv (type: RCA)"""
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        input_bit_per_cycle = self.hd_param["input_bit_per_cycle"]
        energy_adders_pv = self.get_adder_pv_energy(nb_inputs_of_adder_pv, adder_output_pres, self.logic_unit, layer_act_operand_pres,
                                                    input_bit_per_cycle, mapped_cols, macro_activation_times)

        """energy of accumulators (adder type: RCA)"""
        if input_bit_per_cycle == layer_act_operand_pres:
            energy_accumulators = 0
        else:
            accumulator_output_pres = self.hd_param["input_precision"]+self.hd_param["weight_precision"]+math.log2(self.bl_dim_size)
            nb_of_activation_times = mapped_cols * layer_act_operand_pres / input_bit_per_cycle * macro_activation_times
            energy_accumulators = (self.logic_unit.get_1b_adder_energy() + self.logic_unit.get_1b_reg_energy()) * \
                                  accumulator_output_pres * nb_of_activation_times

        self.energy_breakdown = { # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "mults": energy_mults,
            "adders": energy_adders,
            "adders_pv": energy_adders_pv,
            "accumulators": energy_accumulators
        }
        self.energy = sum([v for v in self.energy_breakdown.values()])
        return self.energy_breakdown

if __name__ == "__main__":
#
##### IMC hardware dimension illustration (keypoint: adders' accumulation happens on D2)
#
#       |<------------------------ D1 ----------------------------->| (nb_of_columns/macro = D1 * weight_precision)
#    -  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    \
#    ^  +                                                           +  +  D3 (nb_of_macros)
#    |  +         ^     +++++++                                     +   +  \
#    |  +         |     +  W  +                                     +   +
#    |  +   group_depth +++++++                                     +   +
#    |  +         |     +  W  +                                     +   +
#    |  +         v     +++++++                                     +   +
#    |  +                  |                                        +   +
#    |  +                  v                                        +   +
#    |  +               multipliers -\                              +   +
#    |  +        .                    \                             +   +
#       +        .                     - adders (DIMC)              +   +
#   D2  +        .                    / OR adcs (AIMC)              +   +
#       +               multipliers -/       |                      +   +
#    |  +                  ^                 |                      +   +
#    |  +                  |                 |                      +   +
#    |  +         ^     +++++++              v                      +   +
#    |  +         |     +  W  +          adders_pv (place value)    +   +
#    |  +   group_depth +++++++              |                      +   +
#    |  +         |     +  W  +              v                      +   +
#    |  +         v     +++++++         accumulators                +   +
#    |  +                                    |                      +   +
#    v  +                                    |                      +   +
#    -  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   +
#          +                                 |                        + +
#           +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   (nb_of_rows/macro = D2 * group_depth)    |
#                                            v
#                                        outputs
#

    tech_param_28nm = {
        "tech_node":0.028,              # unit: um
        "vdd":      0.9,                # unit: V
        "nd2_cap":  0.7/1e3,            # unit: pF
        "xor2_cap": 0.7*1.5/1e3,        # unit: pF
        "dff_cap":  0.7*3/1e3,          # unit: pF
        "nd2_area": 0.614/1e6,          # unit: mm^2
        "xor2_area":0.614*2.4/1e6,      # unit: mm^2
        "dff_area": 0.614*6/1e6,        # unit: mm^2
        "nd2_dly":  0.0478,             # unit: ns
        "xor2_dly": 0.0478*2.4,         # unit: ns
        # "dff_dly":  0.0478*3.4,         # unit: ns
    }
    dimensions = {
        "D1": 32/8, # wordline dimension
        "D2": 32,   # bitline dimension
        "D3": 1,    # nb_macros
    }  # {"D1": ("K", 4), "D2": ("C", 32),}

    """hd_param example for DIMC"""
    hd_param = {
        "pe_type":              "in_sram_computing",    # required for CostModelStage
        "imc_type":             "digital",              # "digital" or "analog". Or else: pure digital
        "input_precision":      8,      # activation precison
        "weight_precision":     8,      # weight precision
        "input_bit_per_cycle":  1,      # nb_bits of input/cycle
        "group_depth":          1,      # m factor
        "wordline_dimension":   "D1",   # wordline dimension
        # hardware dimension where input reuse happens (corresponds to the served dimension of input regs)
        "bitline_dimension":    "D2",   # bitline dimension
        # hardware dimension where accumulation happens (corresponds to the served dimension of output regs)
        "enable_cacti":         True,   # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
    }
    dimc = DimcArrayUnit(tech_param_28nm, hd_param, dimensions)
    dimc.get_area()
    dimc.get_delay()
    logger = _logging.getLogger(__name__)
    logger.info(f"Total IMC area (mm^2): {dimc.area}")
    logger.info(f"area breakdown: {dimc.area_breakdown}")
    logger.info(f"delay (ns): {dimc.delay}")
    logger.info(f"delay breakdown (ns): {dimc.delay_breakdown}")
    dimc.get_macro_level_peak_performance()
    exit()