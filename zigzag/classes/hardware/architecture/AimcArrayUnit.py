import numpy as np
import math
import copy
if __name__ == "__main__":
    from imc_unit import ImcUnit
    from DimcArrayUnit import DimcArrayUnit
    import logging as _logging
    _logging_level = _logging.INFO
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level,
                         format=_logging_format)
else:
    import logging as _logging
    from zigzag.classes.hardware.architecture.imc_unit import ImcUnit
    from zigzag.classes.hardware.architecture.DimcArrayUnit import DimcArrayUnit

###############################################################################################################
# README
#   . class AimcArrayUnit (defines the energy/area/delay cost of an ADC, a DAC and an AIMC array)
# How to use this file?
#   . This file is internally called in ZigZag-IMC framework.
#   . It can also be run independently, for mainly debugging. An example is given at the end of the file.
###############################################################################################################

class AimcArrayUnit(ImcUnit):
    def __init__(self,tech_param:dict, hd_param:dict, dimensions:dict):
        super().__init__(tech_param, hd_param, dimensions)

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        # not implemented
        # return {"operational_unit": self.unit, "dimensions": self.dimensions}
        pass

    def get_adc_cost(self):
        """single ADC cost calculation"""
        """area (mm^2)"""
        if self.hd_param["adc_resolution"] == 1:
            adc_area = 0
        else: # formula extracted and validated against 3 AIMC papers on 28nm
            k1 = -0.0369
            k2 = 1.206
            adc_area = 10**(k1*self.hd_param["adc_resolution"]+k2) * 2**self.hd_param["adc_resolution"] * (10**-6) # unit: mm^2
        """delay (ns)"""
        k3 = 0.00653 # ns
        k4 = 0.640 # ns
        adc_delay = self.hd_param["adc_resolution"] * (k3*self.dimensions["D2"] + k4) # unit: ns
        """energy (fJ)"""
        k5 = 100 # fF
        k6 = 0.001 # fF
        adc_energy = (k5 * self.hd_param["adc_resolution"] + k6 * 4**self.hd_param["adc_resolution"]) * self.logic_unit.tech_param["vdd"]**2 # unit: fJ
        adc_energy = adc_energy/1000 # unit: pJ
        return adc_area, adc_delay, adc_energy

    def get_dac_cost(self):
        """single DAC cost calculation"""
        """area (mm^2)"""
        dac_area = 0 # neglected
        """delay (ns)"""
        dac_delay = 0 # neglected
        """energy (fJ)"""
        if self.hd_param["input_bit_per_cycle"] == 1:
            dac_energy = 0
        else:
            k0 = 50e-3 # pF
            dac_energy = k0 * self.hd_param["input_bit_per_cycle"] * self.logic_unit.tech_param["vdd"]**2 # unit: pJ
        return dac_area, dac_delay, dac_energy

    ## get area of AIMC macros (cells, mults, adders, adders_pv, accumulators. Not include input/output regs)
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
            area_cells = single_cell_array_area * self.nb_of_banks  # total cell array area in the core
        else:
            # TODO: [jiacong] [TO BE SUPPORTED OR YOU CAN MODIFY YOURSELF]
            area_cells = None  # user-provided cell array area (from somewhere?)
            raise Exception(f"User-provided cell area is not supported yet.")

        # area of multiplier array
        area_mults = self.logic_unit.get_1b_multiplier_area() * w_pres * \
                     self.wl_dim_size * self.bl_dim_size * self.nb_of_banks

        # area of ADCs
        area_adcs = self.get_adc_cost()[0] * w_pres * self.wl_dim_size * self.nb_of_banks

        # area of DACs
        area_dacs = self.get_dac_cost()[0] * self.bl_dim_size * self.nb_of_banks

        # area of adders with place values after ADC conversion (type: RCA)
        nb_inputs_of_adder_pv = w_pres
        if nb_inputs_of_adder_pv == 1:
            nb_of_1b_adder_pv = 0
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            assert adder_depth_pv % 1 == 0, \
                f"[AimcArray] The value [{nb_inputs_of_adder_pv}] of [weight_precision] is not in the power of 2."
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            adder_input_precision = self.hd_param["adc_resolution"]
            nb_of_1b_adder_pv = adder_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (adder_depth_pv - 0.5)  # nb of 1b adders in a single place-value adder tree
            nb_of_1b_adder_pv *= self.wl_dim_size * self.nb_of_banks  # multiply with nb_of_adder_trees
        area_adders_pv = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_pv

        # area of accumulators (adder type: RCA)
        if self.hd_param["input_bit_per_cycle"] == self.hd_param["input_precision"]:
            area_accumulators = 0
        else:
            accumulator_output_pres = w_pres + self.hd_param["adc_resolution"] + self.hd_param["input_precision"] # output precision from adders_pv + required shifted bits
            nb_of_1b_adder_accumulator = accumulator_output_pres * self.wl_dim_size * self.nb_of_banks
            nb_of_1b_reg_accumulator = nb_of_1b_adder_accumulator # number of regs in an accumulator
            area_accumulators = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_accumulator + \
                                self.logic_unit.get_1b_reg_area() * nb_of_1b_reg_accumulator

        # total area of imc
        self.area_breakdown = { # unit: same with in input hd file
            "cells":    area_cells,
            "mults":    area_mults,
            "adcs":     area_adcs,
            "dacs":     area_dacs,
            "adders_pv":area_adders_pv,
            "accumulators": area_accumulators
        }
        self.area = sum([v for v in self.area_breakdown.values()])
        # return self.area_breakdown

    ## get delay of imc macros (worst path: dacs -> mults -> adcs -> adders -> accumulators)
    def get_delay(self):
        # delay of dacs
        dly_dacs = self.get_dac_cost()[1]

        # delay of multipliers
        dly_mults = self.logic_unit.get_1b_multiplier_dly()

        # delay of adcs
        dly_adcs = self.get_adc_cost()[1]

        # delay of adders_pv (adder type: RCA, worst path: in-to-sum -> in-to-sum -> ... -> in-to-cout -> cin-to-cout -> ... -> cin-to-cout)
        w_pres = self.hd_param["weight_precision"] # weight precision
        nb_inputs_of_adder_pv = w_pres
        if nb_inputs_of_adder_pv == 1:
            dly_adders_pv = 0
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            adder_pv_output_precision = nb_inputs_of_adder_pv + self.hd_param["adc_resolution"] # output precision from adders_pv
            dly_adders_pv = (adder_depth_pv-1) * self.logic_unit.get_1b_adder_dly_in2sum() + \
                            self.logic_unit.get_1b_adder_dly_in2cout() + \
                            (adder_pv_output_precision-1) * self.logic_unit.get_1b_adder_dly_cin2cout()

        # delay of accumulators (adder type: RCA)
        if self.hd_param["input_bit_per_cycle"] == self.hd_param["input_precision"]:
            dly_accumulators = 0
        else:
            accumulator_input_pres = adder_pv_output_precision
            accumulator_output_pres = self.hd_param["weight_precision"] + self.hd_param["adc_resolution"] + self.hd_param["input_precision"]  # output precision from adders_pv + required shifted bits
            dly_accumulators = self.logic_unit.get_1b_adder_dly_in2cout() + \
                               (accumulator_output_pres-accumulator_input_pres) * self.logic_unit.get_1b_adder_dly_cin2cout()

        # total delay of imc
        self.delay_breakdown = {
            "dacs":     dly_dacs,
            "mults":    dly_mults,
            "adcs":     dly_adcs,
            "adders_pv":dly_adders_pv,
            "accumulators": dly_accumulators
        }
        self.delay = sum([v for v in self.delay_breakdown.values()])
        # return self.delay_breakdown

    ## macro-level one-cycle energy of imc arrays (fully utilization, no weight updating)
    # (components: cells, mults, adders, adders_pv, accumulators. Not include input/output regs)
    def get_peak_energy_single_cycle(self):
        layer_const_operand_pres = self.hd_param["weight_precision"]
        layer_act_operand_pres = self.hd_param["input_precision"]
        """energy of precharging"""
        energy_precharging = 0

        """energy of DACs"""
        energy_dacs = self.get_dac_cost()[2] * self.bl_dim_size * self.nb_of_banks

        """energy of cell array (bitline accumulation, type: voltage-based)"""
        energy_cells = (self.logic_unit.tech_param["bl_cap"] * (self.logic_unit.tech_param["vdd"] ** 2) * layer_const_operand_pres) * \
                       self.wl_dim_size * self.bl_dim_size * self.nb_of_banks

        """energy of ADCs"""
        energy_adcs = self.get_adc_cost()[2] * layer_const_operand_pres * self.wl_dim_size * self.nb_of_banks

        """energy of multiplier array"""
        energy_mults = (self.logic_unit.get_1b_multiplier_energy() * layer_const_operand_pres) * \
                       self.bl_dim_size * self.wl_dim_size * self.nb_of_banks

        """energy of adders_pv (type: RCA)"""
        nb_inputs_of_adder_pv = layer_const_operand_pres
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = self.hd_param["adc_resolution"]
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (math.log2(nb_inputs_of_adder_pv) - 0.5)
            energy_adders_pv = nb_of_1b_adder_pv * self.logic_unit.get_1b_adder_energy() * self.wl_dim_size * self.nb_of_banks

        """energy of accumulators (adder type: RCA)"""
        if self.hd_param["input_bit_per_cycle"] == layer_act_operand_pres:
            energy_accumulators = 0
        else:
            accumulator_output_pres = layer_act_operand_pres + layer_const_operand_pres + math.log2(self.bl_dim_size)
            energy_accumulators = (self.logic_unit.get_1b_adder_energy() + self.logic_unit.get_1b_reg_energy()) * accumulator_output_pres * \
                                  self.wl_dim_size * self.nb_of_banks

        peak_energy_breakdown = {  # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "dacs": energy_dacs,
            "adcs": energy_adcs,
            "analog_bitlines": energy_cells,
            "mults": energy_mults,
            "adders_pv": energy_adders_pv,
            "accumulators": energy_accumulators
        }
        # peak_energy = sum([v for v in peak_energy_breakdown.values()])
        return peak_energy_breakdown

    ## macro-level peak performance of imc arrays (fully utilization, no weight updating)
    def get_macro_level_peak_performance(self):
        nb_of_macs_per_cycle = self.wl_dim_size * self.bl_dim_size / \
                               (self.hd_param["input_precision"] / self.hd_param["input_bit_per_cycle"]) * \
                               self.nb_of_banks

        self.get_area()
        self.get_delay()

        clock_cycle_period = self.delay  # unit: ns
        peak_energy_per_cycle = sum([v for v in self.get_peak_energy_single_cycle().values()])  # unit: pJ
        imc_area = self.area  # unit: mm^2

        tops_peak = nb_of_macs_per_cycle * 2 / clock_cycle_period / 1000
        topsw_peak = nb_of_macs_per_cycle * 2 / peak_energy_per_cycle
        topsmm2_peak = tops_peak / imc_area

        logger = _logging.getLogger(__name__)
        logger.info(f"Current macro-level peak performance:")
        logger.info(f"TOP/s: {tops_peak}, TOP/s/W: {topsw_peak}, TOP/s/mm^2: {topsmm2_peak}")

        return tops_peak, topsw_peak, topsmm2_peak

    def get_energy_for_a_layer(self, layer, mapping):
        """check if operand precision defined in the layer is supported"""
        # currently in the energy model, the input and weight precision defined in the workload file should be the same with in the hd input file.
        # this check can be removed if variable precision is supported in the future.

        layer_const_operand = layer.constant_operands[0]  # weight representation
        layer_const_operand_pres = layer.operand_precision[layer_const_operand]
        layer_act_operand = [operand for operand in layer.input_operands if operand != layer_const_operand][0]  # activation representation
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
        mapped_rows_total, mapped_rows_for_adder, mapped_cols, macro_activation_times = DimcArrayUnit.get_mapped_oa_dim(
            layer, self.wl_dim, self.bl_dim)
        self.mapped_rows_total = mapped_rows_total

        """energy calculation"""
        """energy of precharging"""
        energy_precharging, mapped_group_depth = DimcArrayUnit.get_precharge_energy(self.hd_param, self.logic_unit.tech_param, layer, mapping)
        self.mapped_group_depth = mapped_group_depth

        """energy of DACs"""
        energy_dacs = self.get_dac_cost()[2] * mapped_rows_total * \
                      layer_act_operand_pres / self.hd_param["input_bit_per_cycle"] * macro_activation_times

        """energy of cell array (bitline accumulation, type: voltage-based)"""
        energy_cells = (self.logic_unit.tech_param["bl_cap"] * (self.logic_unit.tech_param["vdd"]**2) * layer_const_operand_pres) * \
                       mapped_cols * \
                       self.bl_dim_size * \
                       layer_act_operand_pres / self.hd_param["input_bit_per_cycle"] * \
                       macro_activation_times

        """energy of ADCs"""
        energy_adcs = self.get_adc_cost()[2] * layer_const_operand_pres * mapped_cols * \
                      layer_act_operand_pres / self.hd_param["input_bit_per_cycle"] * macro_activation_times

        """energy of multiplier array"""
        energy_mults = (self.logic_unit.get_1b_multiplier_energy() * layer_const_operand_pres) *\
                       (mapped_rows_total * self.wl_dim_size) * \
                       (layer_act_operand_pres / self.hd_param["input_bit_per_cycle"]) * \
                       macro_activation_times

        """energy of adders_pv (type: RCA)"""
        nb_inputs_of_adder_pv = layer_const_operand_pres
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = self.hd_param["adc_resolution"]
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv-1) + nb_inputs_of_adder_pv*(math.log2(nb_inputs_of_adder_pv)-0.5)
            energy_adders_pv = nb_of_1b_adder_pv * self.logic_unit.get_1b_adder_energy() * mapped_cols * \
                               layer_act_operand_pres / self.hd_param["input_bit_per_cycle"] * macro_activation_times

        """energy of accumulators (adder type: RCA)"""
        if self.hd_param["input_bit_per_cycle"] == layer_act_operand_pres:
            energy_accumulators = 0
        else:
            accumulator_output_pres = layer_act_operand_pres + layer_const_operand_pres + math.log2(self.bl_dim_size)
            energy_accumulators = (self.logic_unit.get_1b_adder_energy() + self.logic_unit.get_1b_reg_energy()) * accumulator_output_pres * \
                                  mapped_cols * \
                                  layer_act_operand_pres / self.hd_param["input_bit_per_cycle"] * macro_activation_times

        self.energy_breakdown = { # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "dacs": energy_dacs,
            "adcs": energy_adcs,
            "analog_bitlines": energy_cells,
            "mults": energy_mults,
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

    """hd_param example for AIMC"""
    hd_param_aimc = {
        "pe_type":              "in_sram_computing",    # required for CostModelStage
        "imc_type":             "analog",               # "digital" or "analog". Or else: pure digital
        "input_precision":      8,      # activation precision
        "weight_precision":     8,      # weight precision
        "input_bit_per_cycle":  2,      # nb_bits of input/cycle
        "group_depth":          1,      # m factor
        "adc_resolution":       8,      # adc resolution
        "wordline_dimension":   "D1",   # wordline dimension
        "bitline_dimension":    "D2",   # bitline dimension
        "enable_cacti":         True,   # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
    }
    hd_param_aimc["adc_resolution"] = hd_param_aimc["input_bit_per_cycle"] + 0.5*math.log2(dimensions["D2"])
    aimc = AimcArrayUnit(tech_param_28nm, hd_param_aimc, dimensions)
    aimc.get_area()
    aimc.get_delay()
    logger = _logging.getLogger(__name__)
    logger.info(f"Total IMC area (mm^2): {aimc.area}")
    logger.info(f"area breakdown: {aimc.area_breakdown}")
    logger.info(f"delay (ns): {aimc.delay}")
    logger.info(f"delay breakdown (ns): {aimc.delay_breakdown}")
    aimc.get_macro_level_peak_performance()
    exit()