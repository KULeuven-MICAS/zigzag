"""
README
  . class DimcArray (defines the energy/area/delay cost of a DIMC array)
How to use this file?
  . This file is internally called in ZigZag-IMC framework.
#   . It can also be run independently, for mainly debugging. An example is given at the end of the file.
"""

import numpy as np
import math
import copy

from zigzag.workload.layer_node import LayerNode

if __name__ == "__main__" or __name__ == "DimcArray":
    # branch when the script is run locally or called by AimcArray.py
    from imc_unit import ImcUnit
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)
else:
    import logging as _logging
    from zigzag.hardware.architecture.imc_unit import ImcUnit


class UserSpatialMappingGenerator:
    """Dummy class to get rid of ruff lint check warnings.
    This should be removed and the code should be updated accordingly.
    """


class DimcArray(ImcUnit):
    # definition of a Digtal In-SRAM-Computing (DIMC) core
    # constraint:
    #     -- activation precision must be in the power of 2.
    #     -- input_bit_per_cycle must be in the power of 2.
    def __init__(self, tech_param: dict, hd_param: dict, dimensions: dict):
        # @param tech_param: technology related parameters
        # @param hd_param: IMC cores' parameters
        # @param dimensions: IMC cores' dimensions
        super().__init__(tech_param, hd_param, dimensions)

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        # not implemented
        # return {"operational_unit": self.unit, "dimensions": self.dimensions}
        pass

    def get_area(self):
        """! area of imc macros (cells, mults, adders, adders_pv, accumulators. Not include input/output regs)"""
        # area of cell array
        tech_node = self.logic_unit.tech_param["tech_node"]
        group_depth = self.hd_param["group_depth"]
        w_pres = self.hd_param["weight_precision"]
        if self.hd_param["enable_cacti"] == True:
            single_cell_array_area = self.get_single_cell_array_cost_from_cacti(
                tech_node, self.wl_dim_size, self.bl_dim_size, group_depth, w_pres
            )[1]
            # at this point, we have the area of single cell array. Then multiply it with the number of banks.
            area_cells = single_cell_array_area * self.nb_of_banks  # total cell array area in the core
        else:
            # TODO: [TO BE SUPPORTED OR YOU CAN MODIFY YOURSELF]
            area_cells = None  # user-provided cell array area (from somewhere?)
            raise Exception(f"User-provided cell area is not supported yet.")

        # area of multiplier array
        area_mults = (
            self.logic_unit.get_1b_multiplier_area()
            * self.hd_param["input_bit_per_cycle"]
            * w_pres
            * self.wl_dim_size
            * self.bl_dim_size
            * self.nb_of_banks
        )

        # area of adder trees (type: RCA)
        adder_input_pres = w_pres  # input precision of the adder tree
        nb_inputs_of_adder = self.bl_dim_size  # the number of inputs of the adder tree
        adder_depth = math.log2(nb_inputs_of_adder)
        assert (
            adder_depth % 1 == 0
        ), f"[DimcArray] The number of inputs [{nb_inputs_of_adder}] for the adder tree is not in the power of 2."
        adder_depth = int(adder_depth)  # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth  # output precision of the adder tree
        nb_of_1b_adder_in_single_adder_tree = nb_inputs_of_adder * (adder_input_pres + 1) - (
            adder_input_pres + adder_depth + 1
        )  # nb of 1b adders in a single adder tree
        nb_of_adder_trees = self.hd_param["input_bit_per_cycle"] * self.wl_dim_size * self.nb_of_banks
        area_adders = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_in_single_adder_tree * nb_of_adder_trees

        # area of extra adders with place values (pv) when input_bit_per_cycle>1 (type: RCA)
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        if nb_inputs_of_adder_pv == 1:
            nb_of_1b_adder_pv = 0  # number of 1b adder in an pv_adder tree
            nb_of_adder_trees_pv = 0  # number of pv_adder trees
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            input_precision_pv = adder_output_pres
            assert (
                adder_depth_pv % 1 == 0
            ), f"[DimcArray] The value [{nb_inputs_of_adder_pv}] of [input_bit_per_cycle] is not in the power of 2."
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            nb_of_1b_adder_pv = input_precision_pv * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (
                adder_depth_pv - 0.5
            )  # nb of 1b adders in a single place-value adder tree
            nb_of_adder_trees_pv = self.wl_dim_size * self.nb_of_banks
        area_adders_pv = self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_pv * nb_of_adder_trees_pv

        # area of accumulators (adder type: RCA)
        if self.hd_param["input_bit_per_cycle"] == self.hd_param["input_precision"]:
            area_accumulators = 0
        else:
            accumulator_output_pres = (
                self.hd_param["input_precision"] + self.hd_param["weight_precision"] + math.log2(self.bl_dim_size)
            )
            nb_of_1b_adder_accumulator = (
                accumulator_output_pres * self.wl_dim_size * self.nb_of_banks
            )  # number of 1b adder in all accumulators
            nb_of_1b_reg_accumulator = nb_of_1b_adder_accumulator  # number of regs in all accumulators
            area_accumulators = (
                self.logic_unit.get_1b_adder_area() * nb_of_1b_adder_accumulator
                + self.logic_unit.get_1b_reg_area() * nb_of_1b_reg_accumulator
            )
        # total area of imc
        self.area_breakdown = {  # unit: same with in input hd file
            "cells": area_cells,
            "mults": area_mults,
            "adders": area_adders,
            "adders_pv": area_adders_pv,
            "accumulators": area_accumulators,
        }
        self.area = sum([v for v in self.area_breakdown.values()])
        # return self.area_breakdown

    def get_delay(self):
        """delay of imc arrays (worst path: mults -> adders -> adders_pv -> accumulators)
        unit: ns (if CACTI is used). whatever it can be otherwise."""
        dly_mults = self.logic_unit.get_1b_multiplier_dly()

        # delay of adders (tree) (type: RCA)
        adder_input_pres = self.hd_param["weight_precision"]
        nb_inputs_of_adder = self.bl_dim_size
        adder_depth = math.log2(nb_inputs_of_adder)
        assert (
            adder_depth % 1 == 0
        ), f"[DimcArray] The number of inputs [{nb_inputs_of_adder}] for the adder tree is not in the power of 2."
        adder_depth = int(adder_depth)  # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth
        dly_adders = (
            (adder_depth - 1) * self.logic_unit.get_1b_adder_dly_in2sum()
            + self.logic_unit.get_1b_adder_dly_in2cout()
            + (adder_output_pres - 1 - 1) * self.logic_unit.get_1b_adder_dly_cin2cout()
        )

        # delay of adders_pv (type: RCA)
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        if nb_inputs_of_adder_pv == 1:
            dly_adders_pv = 0
            accumulator_input_pres = adder_output_pres
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            assert (
                adder_depth_pv % 1 == 0
            ), f"[DimcArray] The value [{nb_inputs_of_adder_pv}] of [input_bit_per_cycle] is not in the power of 2."
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            adder_pv_input_precision = adder_output_pres
            adder_pv_output_precision = (
                nb_inputs_of_adder_pv + adder_output_pres
            )  # output precision from adders_pv (depth + input_precision)
            accumulator_input_pres = adder_pv_output_precision
            dly_adders_pv = (
                (adder_depth_pv - 1) * self.logic_unit.get_1b_adder_dly_in2sum()
                + self.logic_unit.get_1b_adder_dly_in2cout()
                + (adder_pv_output_precision - adder_pv_input_precision - 1)
                * self.logic_unit.get_1b_adder_dly_cin2cout()
            )

        # delay of accumulators (adder type: RCA)
        accumulator_output_pres = (
            self.hd_param["input_precision"] + self.hd_param["weight_precision"] + math.log2(self.bl_dim_size)
        )
        accumulator_output_pres = int(accumulator_output_pres)  # float -> int for simplicity
        if accumulator_output_pres == accumulator_input_pres:  # no accumulator
            dly_accumulators = 0
        else:
            dly_accumulators = (
                self.logic_unit.get_1b_adder_dly_in2cout()
                + (accumulator_output_pres - accumulator_input_pres) * self.logic_unit.get_1b_adder_dly_cin2cout()
            )

        # total delay of imc
        self.delay_breakdown = {
            "mults": dly_mults,
            "adders": dly_adders,
            "adders_pv": dly_adders_pv,
            "accumulators": dly_accumulators,
        }
        self.delay = sum([v for v in self.delay_breakdown.values()])
        # return self.delay_breakdown

    def get_peak_energy_single_cycle(self):
        """
        macro-level one-cycle energy of imc arrays (fully utilization, no weight updating)
        (components: cells, mults, adders, adders_pv, accumulators. Not include input/output regs)
        """
        w_pres = self.hd_param["weight_precision"]
        # energy of precharging
        energy_precharging = 0

        # energy of multiplier array
        nb_of_mults = (
            self.hd_param["input_bit_per_cycle"] * w_pres * self.wl_dim_size * self.bl_dim_size * self.nb_of_banks
        )
        energy_mults = self.logic_unit.get_1b_multiplier_energy() * nb_of_mults

        # energy of adder trees (type: RCA)
        adder_input_pres = w_pres  # input precision of the adder tree
        nb_inputs_of_adder = self.bl_dim_size  # the number of inputs of the adder tree
        adder_depth = math.log2(nb_inputs_of_adder)
        assert (
            adder_depth % 1 == 0
        ), f"[DimcArray] The number of inputs [{nb_inputs_of_adder}] for the adder tree is not in the power of 2."
        adder_depth = int(adder_depth)  # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth  # output precision of the adder tree
        nb_of_1b_adder_in_single_adder_tree = nb_inputs_of_adder * (adder_input_pres + 1) - (
            adder_input_pres + adder_depth + 1
        )  # nb of 1b adders in a single adder tree
        nb_of_adder_trees = self.hd_param["input_bit_per_cycle"] * self.wl_dim_size * self.nb_of_banks
        energy_adders = self.logic_unit.get_1b_adder_energy() * nb_of_1b_adder_in_single_adder_tree * nb_of_adder_trees

        # energy of adders_pv (type: RCA)
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = adder_output_pres
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (
                math.log2(nb_inputs_of_adder_pv) - 0.5
            )
            nb_of_adder_trees_pv = self.wl_dim_size * self.nb_of_banks
            energy_adders_pv = self.logic_unit.get_1b_adder_energy() * nb_of_1b_adder_pv * nb_of_adder_trees_pv

        # energy of accumulators (adder type: RCA)
        if self.hd_param["input_bit_per_cycle"] == self.hd_param["input_precision"]:
            energy_accumulators = 0
        else:
            accumulator_output_pres = (
                self.hd_param["input_precision"] + self.hd_param["weight_precision"] + math.log2(self.bl_dim_size)
            )
            nb_of_1b_adder_accumulator = (
                accumulator_output_pres * self.wl_dim_size * self.nb_of_banks
            )  # number of 1b adder in all accumulators
            nb_of_1b_reg_accumulator = nb_of_1b_adder_accumulator  # number of regs in all accumulators
            energy_accumulators = (
                self.logic_unit.get_1b_adder_energy() * nb_of_1b_adder_accumulator
                + self.logic_unit.get_1b_reg_energy() * nb_of_1b_reg_accumulator
            )

        peak_energy_breakdown = {  # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "mults": energy_mults,
            "adders": energy_adders,
            "adders_pv": energy_adders_pv,
            "accumulators": energy_accumulators,
        }
        # peak_energy = sum([v for v in peak_energy_breakdown.values()])
        return peak_energy_breakdown

    def get_macro_level_peak_performance(self):
        """
        macro-level peak performance of imc arrays (fully utilization, no weight updating)
        """
        nb_of_macs_per_cycle = (
            self.wl_dim_size
            * self.bl_dim_size
            / (self.hd_param["input_precision"] / self.hd_param["input_bit_per_cycle"])
            * self.nb_of_banks
        )

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
        """
        get the imc array energy for specific layer with specific mapping
        check if operand precision defined in the layer is the same with in hardware template"""
        # activation/weight representation
        layer_act_operand, layer_const_operand = UserSpatialMappingGenerator.identify_layer_operand_representation(
            layer
        )

        layer_const_operand_pres = layer.operand_precision[layer_const_operand]
        layer_act_operand_pres = layer.operand_precision[layer_act_operand]
        weight_pres_in_hd_param = self.hd_param["weight_precision"]
        act_pres_in_hd_param = self.hd_param["input_precision"]

        # currently in the energy model, the input and weight precision defined in the workload file should be the same with in the hd input file.
        # this check can be removed if variable precision is supported in the future.
        assert (
            layer_const_operand_pres == weight_pres_in_hd_param
        ), f"Weight precision defined in the workload [{layer_const_operand_pres}] not equal to the one defined in the hardware hd_param [{weight_pres_in_hd_param}]."
        assert (
            layer_act_operand_pres == act_pres_in_hd_param
        ), f"Activation precision defined in the workload [{layer_act_operand_pres}] not equal to the one defined in the hardware hd_param [{act_pres_in_hd_param}]."

        # parameter extraction
        (
            mapped_rows_total,
            mapped_rows_for_adder,
            mapped_cols,
            macro_activation_times,
        ) = DimcArray.get_mapped_oa_dim(layer, self.wl_dim, self.bl_dim)
        self.mapped_rows_total = mapped_rows_total

        # energy calculation
        # energy of precharging
        energy_precharging, mapped_group_depth = DimcArray.get_precharge_energy(
            self.hd_param, self.logic_unit.tech_param, layer, mapping
        )
        self.mapped_group_depth = mapped_group_depth

        # energy of multiplier array
        energy_mults = self.get_mults_energy(
            self.hd_param,
            self.logic_unit,
            layer,
            mapped_rows_total,
            self.wl_dim_size,
            macro_activation_times,
        )

        # energy of adder trees (type: RCA)
        energy_adders, adder_output_pres = self.get_adder_trees_energy(
            layer,
            self.logic_unit,
            mapped_rows_for_adder,
            self.bl_dim_size,
            mapped_cols,
            layer_act_operand_pres,
            macro_activation_times,
        )

        # energy of adders_pv (type: RCA)
        nb_inputs_of_adder_pv = self.hd_param["input_bit_per_cycle"]
        input_bit_per_cycle = self.hd_param["input_bit_per_cycle"]
        energy_adders_pv = self.get_adder_pv_energy(
            nb_inputs_of_adder_pv,
            adder_output_pres,
            self.logic_unit,
            layer_act_operand_pres,
            input_bit_per_cycle,
            mapped_cols,
            macro_activation_times,
        )

        # energy of accumulators (adder type: RCA)
        if input_bit_per_cycle == layer_act_operand_pres:
            energy_accumulators = 0
        else:
            accumulator_output_pres = (
                self.hd_param["input_precision"] + self.hd_param["weight_precision"] + math.log2(self.bl_dim_size)
            )
            nb_of_activation_times = mapped_cols * layer_act_operand_pres / input_bit_per_cycle * macro_activation_times
            energy_accumulators = (
                (self.logic_unit.get_1b_adder_energy() + self.logic_unit.get_1b_reg_energy())
                * accumulator_output_pres
                * nb_of_activation_times
            )

        self.energy_breakdown = {  # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "mults": energy_mults,
            "adders": energy_adders,
            "adders_pv": energy_adders_pv,
            "accumulators": energy_accumulators,
        }
        self.energy = sum([v for v in self.energy_breakdown.values()])
        return self.energy_breakdown


if __name__ == "__main__":
    #
    # #### IMC hardware dimension illustration (keypoint: adders' accumulation happens on D2)
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
        "tech_node": 0.028,  # unit: um
        "vdd": 0.9,  # unit: V
        "nd2_cap": 0.7 / 1e3,  # unit: pF
        "xor2_cap": 0.7 * 1.5 / 1e3,  # unit: pF
        "dff_cap": 0.7 * 3 / 1e3,  # unit: pF
        "nd2_area": 0.614 / 1e6,  # unit: mm^2
        "xor2_area": 0.614 * 2.4 / 1e6,  # unit: mm^2
        "dff_area": 0.614 * 6 / 1e6,  # unit: mm^2
        "nd2_dly": 0.0478,  # unit: ns
        "xor2_dly": 0.0478 * 2.4,  # unit: ns
        # "dff_dly":  0.0478*3.4,         # unit: ns
    }
    dimensions = {
        "D1": 32 / 8,  # wordline dimension
        "D2": 32,  # bitline dimension
        "D3": 1,  # nb_macros
    }  # {"D1": ("K", 4), "D2": ("C", 32),}

    """hd_param example for DIMC"""
    hd_param = {
        "pe_type": "in_sram_computing",  # required for CostModelStage
        "imc_type": "digital",  # "digital" or "analog". Or else: pure digital
        "input_precision": 8,  # activation precison
        "weight_precision": 8,  # weight precision
        "input_bit_per_cycle": 1,  # nb_bits of input/cycle
        "group_depth": 1,  # m factor
        "wordline_dimension": "D1",  # wordline dimension
        # hardware dimension where input reuse happens (corresponds to the served dimension of input regs)
        "bitline_dimension": "D2",  # bitline dimension
        # hardware dimension where accumulation happens (corresponds to the served dimension of output regs)
        "enable_cacti": True,  # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
    }
    dimc = DimcArray(tech_param_28nm, hd_param, dimensions)
    dimc.get_area()
    dimc.get_delay()
    logger = _logging.getLogger(__name__)
    logger.info(f"Total IMC area (mm^2): {dimc.area}")
    logger.info(f"area breakdown: {dimc.area_breakdown}")
    logger.info(f"delay (ns): {dimc.delay}")
    logger.info(f"delay breakdown (ns): {dimc.delay_breakdown}")
    dimc.get_macro_level_peak_performance()
    exit()
