"""
This file includes:
  . class LogicUnit (defines the energy/area/delay cost of multipliers, adders, regs)
  . class ImcArray (provides initialization function, used for class DimcArray and AimcArray)
"""

import math

from zigzag.hardware.architecture.operational_unit import OperationalUnit

if __name__ == "__main__" or __name__ == "imc_unit":
    # branch when the script is run locally or called by A/DimcArray.py
    from get_cacti_cost import get_cacti_cost
else:
    from zigzag.hardware.architecture.get_cacti_cost import get_cacti_cost


class LogicUnit:
    """cost (energy, area, delay) of 1b adder, 1b multiplier, 1b register is defined in this class"""

    def __init__(self, tech_param: dict):
        """
            Input example:
            tech_param_28nm = {
            "vdd":      0.9,            # unit: V
            "nd2_cap":  0.7/1e3,        # unit: pF
            "nd2_area": 0.614/1e6,      # unit: mm^2
            "nd2_dly":  0.0478,         # unit: ns
            "xor2_cap": 0.7*1.5/1e3,    # unit: pF
            "xor2_area":0.614*2.4/1e6,  # unit: mm^2
            "xor2_dly": 0.0478*1.5,     # unit: ns
            "dff_cap":  0.7*3/1e3,      # unit: pF
            "dff_area": 0.0614*6/1e6,   # unit: mm^2
            "dff_dly":  0.0478*3.4,     # unit: ns
        }
        """
        # check input firstly
        self.check_tech_param(tech_param)
        # initialization
        self.tech_param = tech_param
        self.tech_param["wl_cap"] = (
            tech_param["nd2_cap"] / 2
        )  # wordline cap of each SRAM cell is treated as NAND2_cap/2
        self.tech_param["bl_cap"] = tech_param["nd2_cap"] / 2  # bitline cap of each SRAM cell is treated as NAND2_cap/2

    def check_tech_param(self, tech_param):
        required_param = [
            "tech_node",
            "vdd",
            "nd2_cap",
            "nd2_area",
            "nd2_dly",
            "xor2_cap",
            "xor2_area",
            "xor2_dly",
            "dff_cap",
            "dff_area",
        ]
        for ii_a, a in enumerate(required_param):
            if a not in tech_param.keys():
                raise Exception(f"[LogicUnit] Incorrect input, required param [{a}] not found.")
            if not (isinstance(tech_param[a], int) or isinstance(tech_param[a], float)):
                raise Exception(f"[LogicUnit] Incorrect input, value [{tech_param[a]}] of param [{a}] is not a num.")
            if tech_param[a] <= 0:
                raise Exception(f"[LogicUnit] Incorrect input, value [{tech_param[a]}] of param [{a}] is not positive.")

    def get_1b_adder_energy(self):
        """energy of 1b full adder
        Assume a 1b adder has 3 ND2 gate and 2 XOR2 gate"""
        adder_cap = 3 * self.tech_param["nd2_cap"] + 2 * self.tech_param["xor2_cap"]
        return adder_cap * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_adder_energy_half_activated(self):
        """energy of 1b full adder when 1 input is 0"""
        adder_cap = 2 * self.tech_param["xor2_cap"]
        return adder_cap * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_multiplier_energy(self):
        """energy of 1b multiplier
        1b mult includes 1 NOR gate, which is assumed as the same cost of ND2 gate
        why 0.5: considering weight stays constant during multiplication"""
        return 0.5 * self.tech_param["nd2_cap"] * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_reg_energy(self):
        """energy of 1b DFF"""
        return self.tech_param["dff_cap"] * (self.tech_param["vdd"] ** 2)  # unit: pJ

    def get_1b_adder_area(self):
        """area of 1b full adder
        Assume a 1b adder has 3 ND2 gate and 2 XOR2 gate"""
        adder_area = 3 * self.tech_param["nd2_area"] + 2 * self.tech_param["xor2_area"]
        return adder_area

    def get_1b_multiplier_area(self):
        """area of 1b multiplier
        1b mult includes 1 NOR gate, which is assumed as the same cost of ND2 gate"""
        return self.tech_param["nd2_area"]

    def get_1b_reg_area(self):
        """area of 1b DFF"""
        return self.tech_param["dff_area"]

    def get_1b_adder_dly_in2sum(self):
        """delay of 1b adder: input to sum-out"""
        adder_dly = 2 * self.tech_param["xor2_dly"]
        return adder_dly

    def get_1b_adder_dly_in2cout(self):
        """delay of 1b adder: input to carry-out"""
        adder_dly = self.tech_param["xor2_dly"] + 2 * self.tech_param["nd2_dly"]
        return adder_dly

    def get_1b_adder_dly_cin2cout(self):
        """delay of 1b adder: carry-in to carry-out"""
        adder_dly = 2 * self.tech_param["nd2_dly"]
        return adder_dly

    def get_1b_multiplier_dly(self):
        """delay of 1b multiplier
        1b mult includes 1 NOR gate, which is assumed as the same cost of ND2 gate"""
        return self.tech_param["nd2_dly"]

    def get_1b_reg_dly(self):
        """delay of 1b DFF
        why 0? Compared to others, it's negligible"""
        return 0


class ImcUnit:
    """definition of general initilization function for D/AIMC
    # TODO ImcUnit should inherit from OperationalUnit since ImcArray inherits from OperationalArray
    """

    def __init__(self, tech_param: dict, hd_param: dict, dimensions: dict):
        """check input firstly"""
        self.check_input(hd_param, dimensions)
        # initialization
        self.hd_param = hd_param
        self.dimensions = dimensions
        self.wl_dim = hd_param[
            "wordline_dimension"
        ]  # wl_dim should be the same with the dimension served by input_reg.
        self.bl_dim = hd_param[
            "bitline_dimension"
        ]  # bl_dim should be the same with the dimension served by output_reg.
        self.wl_dim_size = dimensions[self.wl_dim]  # dimension where wordline is
        self.bl_dim_size = dimensions[self.bl_dim]  # dimension where bitline (adder tree) is
        self.nb_of_banks = math.prod(
            [dimensions[oa_dim] for oa_dim in dimensions if oa_dim not in [self.wl_dim, self.bl_dim]]
        )
        # tech_param will be checked and initialized in LogicUnit class
        self.logic_unit = LogicUnit(tech_param)
        # parameters to be updated in function
        self.energy = None
        self.energy_breakdown = None
        self.area = None
        self.area_breakdown = None
        self.delay = None
        self.delay_breakdown = None
        self.mapped_rows_total = None
        self.mapped_group_depth = None

    def check_input(self, hd_param, dimensions):
        # check if required_hd_param is provided
        # check if there is any negative dimension value
        required_hd_param = [
            "imc_type",
            "input_precision",
            "weight_precision",
            "input_bit_per_cycle",
            "group_depth",
            "wordline_dimension",
            "bitline_dimension",
            "enable_cacti",
        ]
        for ii_a, a in enumerate(required_hd_param):
            if a not in hd_param.keys():
                raise Exception(f"[ImcArray] Incorrect hd_param, required param [{a}] not found.")
            if a == "imc_type":
                if hd_param[a] not in ["digital", "analog"]:
                    raise Exception(
                        f"[ImcArray] Incorrect imc_type in hd_param, either [analog] or [digital] is expected."
                    )
            elif a == "wordline_dimension" or a == "bitline_dimension":
                if not isinstance(hd_param[a], str) or hd_param[a] not in dimensions.keys():
                    raise Exception(f"[ImcArray] param [{a}] is not a str or is not a key in dimensions.")
            elif a == "enable_cacti":
                if not isinstance(hd_param[a], bool):
                    raise Exception(f"[ImcArray] param [{a}] is not bool (Ture, False).")
            else:
                if not (isinstance(hd_param[a], int) or isinstance(hd_param[a], float)):
                    raise Exception(
                        f"[ImcArray] Incorrect hd_param, value [{hd_param[a]}] of param [{a}] is not a num."
                    )
                if hd_param[a] <= 0:
                    raise Exception(
                        f"[ImcArray] Incorrect hd_param, value [{hd_param[a]}] of param [{a}] is not positive."
                    )
                if a == "input_bit_per_cycle" and hd_param[a] > hd_param["input_precision"]:
                    input_precision = hd_param["input_precision"]
                    raise Exception(
                        f"[ImcArray] Incorrect hd_param, value [{hd_param[a]}] of param [{a}] is bigger than [input_precision] ({input_precision})."
                    )
        for oa_dim in dimensions.keys():
            if dimensions[oa_dim] <= 0:
                raise Exception(
                    f"[ImcArray] Incorrect dimensions, value [{dimensions[a]}] of param [{a}] is not a positive number."
                )
        if hd_param["imc_type"] == "analog":
            a = "adc_resolution"
            if a not in hd_param.keys():
                raise Exception(f"[ImcArray] Incorrect hd_param, required param [{a}] not found.")
            # if adc_resolution is not a number or adc_resolution <= 0
            if (not (isinstance(hd_param[a], int) or isinstance(hd_param[a], float))) or (hd_param[a] <= 0):
                raise Exception(
                    f"[ImcArray] Incorrect hd_param, value [{hd_param[a]}] of param [{a}] is not a positive number."
                )

    def get_single_cell_array_cost_from_cacti(self, tech_node, wl_dim_size, bl_dim_size, group_depth, w_pres):
        """get the area, energy cost of a single macro (cell array) using CACTI
        this function is called when cacti is required for cost estimation
        @param tech_node:   the technology node (e.g. 0.028, 0.032, 0.022 ... unit: um)
        @param wl_dim_size: the size of dimension where wordline is.
        @param bl_dim_size: the size of dimension where bitline (adder tree) is.
        @param group_depth: the size of each cell group (number of SRAM cells on local bitline)
        @param w_pres:      weight precision (number of SRAM cells required to store a operand)
        """
        cell_array_size = wl_dim_size * bl_dim_size * group_depth * w_pres / 8  # array size. unit: byte
        array_bw = wl_dim_size * w_pres  # imc array bandwidth. unit: bit

        # we will call cacti to get the area (mm^2), access_time (ns), r_cost (nJ/access), w_cost (nJ/access)
        if __name__ == "imc_unit":
            cacti_path = "../../cacti/cacti_master"
        else:
            cacti_path = "zigzag/classes/cacti/cacti_master"
        access_time, area, r_cost, w_cost = get_cacti_cost(
            cacti_path=cacti_path,
            tech_node=tech_node,
            mem_type="sram",
            mem_size_in_byte=cell_array_size,
            bw=array_bw,
        )
        return access_time, area, r_cost, w_cost
