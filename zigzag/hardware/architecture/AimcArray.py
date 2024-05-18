"""
README
  . class AimcArray (defines the energy/area/delay cost of an ADC, a DAC and an AIMC array)
How to use this file?
  . This file is internally called in ZigZag-IMC framework.
  . It can also be run independently, for mainly debugging. An example is given at the end of the file.
"""
import math
from zigzag.hardware.architecture.DimcArray import UserSpatialMappingGenerator

if __name__ == "__main__":
    from imc_unit import ImcUnit
    from DimcArray import DimcArray
    import logging as _logging

    _logging_level = _logging.INFO
    _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    _logging.basicConfig(level=_logging_level, format=_logging_format)
else:
    import logging as _logging
    from zigzag.hardware.architecture.imc_unit import ImcUnit
    from zigzag.hardware.architecture.DimcArray import DimcArray


class AimcArray(ImcUnit):
    # definition of an Analog In-SRAM-Computing (DIMC) core
    # constraint:
    #     -- activation precision must be in the power of 2.
    #     -- input_bit_per_cycle must be in the power of 2.
    def __init__(self, cells_size: int, imc_data: dict):
        # @param cells_size: cell group size (unit: bit)
        # @param imc_data: IMC cores' parameters
        # @param dimension_sizes: IMC cores' dimension_sizes
        super().__init__(cells_size, imc_data)

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        pass

    def get_adc_cost(self) -> tuple:
        """single ADC cost calculation"""
        # area (mm^2)
        if self.imc_data["adc_resolution"] == 1:
            adc_area = 0
        else:  # formula extracted and validated against 3 AIMC papers on 28nm
            k1 = -0.0369
            k2 = 1.206
            adc_area = (
                    10 ** (k1 * self.imc_data["adc_resolution"] + k2) * 2 ** self.imc_data["adc_resolution"] * (
                    10 ** -6)
            )  # unit: mm^2
        # delay (ns)
        k3 = 0.00653  # ns
        k4 = 0.640  # ns
        adc_delay = self.imc_data["adc_resolution"] * (k3 * self.bl_dim_size + k4)  # unit: ns
        # energy (fJ)
        k5 = 100  # fF
        k6 = 0.001  # fF
        adc_energy = (k5 * self.imc_data["adc_resolution"] + k6 * 4 ** self.imc_data["adc_resolution"]
                      ) * self.tech_param["vdd"] ** 2  # unit: fJ
        adc_energy = adc_energy / 1000  # unit: pJ
        return adc_area, adc_delay, adc_energy

    def get_dac_cost(self) -> tuple:
        """single DAC cost calculation"""
        # area (mm^2)
        dac_area = 0  # neglected
        # delay (ns)
        dac_delay = 0  # neglected
        # energy (fJ)
        if self.imc_data["bit_serial_precision"] == 1:
            dac_energy = 0
        else:
            k0 = 50e-3  # pF
            dac_energy = k0 * self.imc_data["bit_serial_precision"] * self.tech_param["vdd"] ** 2  # unit: pJ
        return dac_area, dac_delay, dac_energy

    def get_area(self):
        """! get area of AIMC logics (mults, adders, adders_pv, accumulators. Exclude sram cells, input/output regs)"""
        # area of cells
        area_cells = 0  # TODO

        # area of multiplier array
        area_mults = (
            self.get_1b_multiplier_area() * self.weight_precision * self.wl_dim_size * self.bl_dim_size * self.nb_of_banks
        )

        # area of ADCs
        area_adcs = self.get_adc_cost()[0] * self.weight_precision * self.wl_dim_size * self.nb_of_banks

        # area of DACs
        area_dacs = self.get_dac_cost()[0] * self.bl_dim_size * self.nb_of_banks

        # area of adders with place values after ADC conversion (type: RCA)
        nb_inputs_of_adder_pv = self.weight_precision
        if nb_inputs_of_adder_pv == 1:
            nb_of_1b_adder_pv = 0
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            assert (
                    adder_depth_pv % 1 == 0
            ), f"[AimcArray] The value [{nb_inputs_of_adder_pv}] of [weight_precision] is not in the power of 2."
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            adder_input_precision = self.imc_data["adc_resolution"]
            nb_of_1b_adder_pv = adder_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (
                    adder_depth_pv - 0.5
            )  # nb of 1b adders in a single place-value adder tree
            nb_of_1b_adder_pv *= self.wl_dim_size * self.nb_of_banks  # multiply with nb_of_adder_trees
        area_adders_pv = self.get_1b_adder_area() * nb_of_1b_adder_pv

        # area of accumulators (adder type: RCA)
        if self.imc_data["bit_serial_precision"] == self.activation_precision:
            area_accumulators = 0
        else:
            accumulator_output_pres = (
                    self.activation_precision + self.imc_data["adc_resolution"] + self.weight_precision
            )  # output precision from adders_pv + required shifted bits
            nb_of_1b_adder_accumulator = accumulator_output_pres * self.wl_dim_size * self.nb_of_banks
            nb_of_1b_reg_accumulator = nb_of_1b_adder_accumulator  # number of regs in an accumulator
            area_accumulators = (
                    self.get_1b_adder_area() * nb_of_1b_adder_accumulator
                    + self.get_1b_reg_area() * nb_of_1b_reg_accumulator
            )

        # total logic area of imc macros (exclude cells)
        self.area_breakdown = {
            "cells": area_cells,
            "mults": area_mults,
            "adcs": area_adcs,
            "dacs": area_dacs,
            "adders_pv": area_adders_pv,
            "accumulators": area_accumulators,
        }
        self.area = sum([v for v in self.area_breakdown.values()])

    def get_delay(self):
        """! get delay of imc macros (worst path: dacs -> mults -> adcs -> adders -> accumulators)"""
        # delay of dacs
        dly_dacs = self.get_dac_cost()[1]

        # delay of multipliers
        dly_mults = self.get_1b_multiplier_dly()

        # delay of adcs
        dly_adcs = self.get_adc_cost()[1]

        # delay of adders_pv
        # adder type: RCA, worst path: in-to-sum -> in-to-sum -> ... -> in-to-cout -> cin-to-cout -> ... -> cin-to-cout
        nb_inputs_of_adder_pv = self.weight_precision
        if nb_inputs_of_adder_pv == 1:
            adder_pv_output_precision = 1
            dly_adders_pv = 0
        else:
            adder_depth_pv = math.log2(nb_inputs_of_adder_pv)
            adder_depth_pv = int(adder_depth_pv)  # float -> int for simplicity
            adder_pv_output_precision = (
                    nb_inputs_of_adder_pv + self.imc_data["adc_resolution"]
            )  # output precision from adders_pv
            dly_adders_pv = (
                    (adder_depth_pv - 1) * self.get_1b_adder_dly_in2sum()
                    + self.get_1b_adder_dly_in2cout()
                    + (adder_pv_output_precision - 1) * self.get_1b_adder_dly_cin2cout()
            )

        # delay of accumulators (adder type: RCA)
        if self.imc_data["bit_serial_precision"] == self.activation_precision:
            dly_accumulators = 0
        else:
            accumulator_input_pres = adder_pv_output_precision
            accumulator_output_pres = (
                    self.activation_precision + self.imc_data["adc_resolution"] + self.weight_precision
            )  # output precision from adders_pv + required shifted bits
            dly_accumulators = (
                    self.get_1b_adder_dly_in2cout()
                    + (accumulator_output_pres - accumulator_input_pres) * self.get_1b_adder_dly_cin2cout()
            )

        # total delay of imc
        self.delay_breakdown = {
            "dacs": dly_dacs,
            "mults": dly_mults,
            "adcs": dly_adcs,
            "adders_pv": dly_adders_pv,
            "accumulators": dly_accumulators,
        }
        self.delay = sum([v for v in self.delay_breakdown.values()])

    def get_peak_energy_single_cycle(self) -> dict:
        """! macro-level one-cycle energy of imc arrays (fully utilization, no weight updating)
        (components: cells, mults, adders, adders_pv, accumulators. Not include input/output regs)
        """
        # energy of precharging
        energy_precharging = 0

        # energy of DACs
        energy_dacs = self.get_dac_cost()[2] * self.bl_dim_size * self.nb_of_banks

        # energy of cell array (bitline accumulation, type: voltage-based)
        energy_cells = (
                (self.tech_param["bl_cap"] * (
                        self.tech_param["vdd"] ** 2) * self.weight_precision)
                * self.wl_dim_size
                * self.bl_dim_size
                * self.nb_of_banks
        )

        # energy of ADCs
        energy_adcs = self.get_adc_cost()[2] * self.weight_precision * self.wl_dim_size * self.nb_of_banks

        # energy of multiplier array
        energy_mults = (
                (self.get_1b_multiplier_energy() * self.weight_precision)
                * self.bl_dim_size
                * self.wl_dim_size
                * self.nb_of_banks
        )

        # energy of adders_pv (type: RCA)
        nb_inputs_of_adder_pv = self.weight_precision
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = self.imc_data["adc_resolution"]
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (
                    math.log2(nb_inputs_of_adder_pv) - 0.5
            )
            energy_adders_pv = (
                    nb_of_1b_adder_pv * self.get_1b_adder_energy() * self.wl_dim_size * self.nb_of_banks
            )

        # energy of accumulators (adder type: RCA)
        if self.imc_data["bit_serial_precision"] == self.activation_precision:
            energy_accumulators = 0
        else:
            accumulator_output_pres = (
                    self.activation_precision + self.imc_data["adc_resolution"] + self.weight_precision
            )  # output precision from adders_pv + required shifted bits
            energy_accumulators = (
                    (self.get_1b_adder_energy() + self.get_1b_reg_energy())
                    * accumulator_output_pres
                    * self.wl_dim_size
                    * self.nb_of_banks
            )

        peak_energy_breakdown = {  # unit: pJ (the unit is borrowed from CACTI)
            "precharging": energy_precharging,
            "dacs": energy_dacs,
            "adcs": energy_adcs,
            "analog_bitlines": energy_cells,
            "mults": energy_mults,
            "adders_pv": energy_adders_pv,
            "accumulators": energy_accumulators,
        }
        return peak_energy_breakdown

    def get_macro_level_peak_performance(self) -> tuple:
        """! macro-level peak performance of imc arrays (fully utilization, no weight updating)"""
        nb_of_macs_per_cycle = (
                self.wl_dim_size
                * self.bl_dim_size
                / (self.activation_precision / self.imc_data["bit_serial_precision"])
                * self.nb_of_banks
        )

        self.get_area()  # configure self.area and self.area_breakdown
        self.get_delay()  # configure self.delay and self.delay_breakdown

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
        # parameter extraction
        (
            mapped_rows_total,
            mapped_rows_for_adder,
            mapped_cols,
            macro_activation_times,
        ) = self.get_mapped_oa_dim(layer, self.wl_dim, self.bl_dim)
        self.mapped_rows_total = mapped_rows_total

        # # energy calculation
        # # energy of precharging
        energy_precharging, mapped_group_depth = self.get_precharge_energy(
            self.tech_param, layer, mapping
        )
        self.mapped_group_depth = mapped_group_depth

        # # energy of DACs
        energy_dacs = (
                self.get_dac_cost()[2]
                * mapped_rows_total
                * self.activation_precision
                / self.imc_data["bit_serial_precision"]
                * macro_activation_times
        )

        # # energy of cell array (bitline accumulation, type: voltage-based)
        energy_cells = (
                (self.tech_param["bl_cap"] * (
                        self.tech_param["vdd"] ** 2) * self.weight_precision)
                * mapped_cols
                * self.bl_dim_size
                * self.activation_precision
                / self.imc_data["bit_serial_precision"]
                * macro_activation_times
        )

        # # energy of ADCs
        energy_adcs = (
                self.get_adc_cost()[2]
                * self.weight_precision
                * mapped_cols
                * self.activation_precision
                / self.imc_data["bit_serial_precision"]
                * macro_activation_times
        )

        # # energy of multiplier array
        energy_mults = (
                (self.get_1b_multiplier_energy() * self.weight_precision)
                * (mapped_rows_total * self.wl_dim_size)
                * (self.activation_precision / self.imc_data["bit_serial_precision"])
                * macro_activation_times
        )

        # # energy of adders_pv (type: RCA)
        nb_inputs_of_adder_pv = self.weight_precision
        if nb_inputs_of_adder_pv == 1:
            energy_adders_pv = 0
        else:
            adder_pv_input_precision = self.imc_data["adc_resolution"]
            nb_of_1b_adder_pv = adder_pv_input_precision * (nb_inputs_of_adder_pv - 1) + nb_inputs_of_adder_pv * (
                    math.log2(nb_inputs_of_adder_pv) - 0.5
            )
            energy_adders_pv = (
                    nb_of_1b_adder_pv
                    * self.get_1b_adder_energy()
                    * mapped_cols
                    * self.activation_precision
                    / self.imc_data["bit_serial_precision"]
                    * macro_activation_times
            )

        # # energy of accumulators (adder type: RCA)
        if self.imc_data["bit_serial_precision"] == self.activation_precision:
            energy_accumulators = 0
        else:
            accumulator_output_pres = self.activation_precision + self.weight_precision + self.imc_data["adc_resolution"]
            energy_accumulators = (
                    (self.get_1b_adder_energy() + self.get_1b_reg_energy())
                    * accumulator_output_pres
                    * mapped_cols
                    * self.activation_precision
                    / self.imc_data["bit_serial_precision"]
                    * macro_activation_times
            )

        self.energy_breakdown = {  # unit: pJ (the unit borrowed from CACTI)
            "precharging": energy_precharging,
            "dacs": energy_dacs,
            "adcs": energy_adcs,
            "analog_bitlines": energy_cells,
            "mults": energy_mults,
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
    # imc_data example for AIMC
    imc_data = {
        "is_imc": True,
        "imc_type": "analog",
        "input_precision": [8, 8],
        "bit_serial_precision": 2,
        "adc_resolution": 8,
        "dimensions": ["D1", "D2"],
        "sizes": [32, 32],
    }
    cells_size = 8  # unit: bit

    aimc = AimcArray(cells_size, imc_data)
    aimc.get_area()
    aimc.get_delay()
    logger = _logging.getLogger(__name__)
    logger.info(f"Total IMC area (mm^2): {aimc.area}")
    logger.info(f"area breakdown: {aimc.area_breakdown}")
    logger.info(f"delay (ns): {aimc.delay}")
    logger.info(f"delay breakdown (ns): {aimc.delay_breakdown}")
    aimc.get_macro_level_peak_performance()
    exit()
