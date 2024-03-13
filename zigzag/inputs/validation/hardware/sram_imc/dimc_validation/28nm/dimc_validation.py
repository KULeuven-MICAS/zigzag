import pdb
from dimc_validation_subfunc import *

"""
ISSCC2022, 15.5 (50% input toggle rate, 50% weight sparsity)
"""
dimc_ISSCC2022_15_5 = { # https://ieeexplore.ieee.org/document/9731762 (28nm)
    'paper_idx':            'ISSCC2022, 15.5',
    'input_toggle_rate':    0.5,
    'weight_sparsity':      0.5,
    'activation_precision': 8,
    'weight_precision':     8,
    'output_precision':     8, # output precision (unit: bit)
    'input_precision':      2,
    'input_channel':        32, # how many input in parallel (per bank)
    'output_channel':       6, # how many output in parallel (per bank)
    'booth_encoding':       True,
    'multiplier_precision': 8,
    'adder_input_precision':9,
    'accumulator_input_precision': 14,
    'accumulator_precision':32,
    'reg_accumulator_precision': 32,
    'reg_input_bitwidth':   32*2,
    'pipeline':             False,
    'vdd':                  0.9, # V
    'rows':                 32, # equal to the number of input channels
    'cols':                 48,
    'banks':                64, # number of cores
    'area':                 0.9408, # mm2
    'tclk':                 1/195*1000, # ns
    'TOP/s':                6144*195*(10**-6),
    'TOP/s/W':              36.63,
    'unit_area':            0, # um2
    'unit_delay':           0, #ns
    'unit_cap':             0 #fF
    }
cacti_ISSCC2022_15_5 = { # 256B, bw: 48
    'delay':                0.0669052, #ns
    'r_energy':             0.000221196*10**6/64*81, #fJ
    'w_energy':             0.000328423*10**6/64*81, #fJ
    'area':                 0.00065545 #mm2
    }

"""
ISSCC2022, 11.7 (50% input sparsity, unknown weight sparsity, average performance reported)
"""
dimc_ISSCC2022_11_7 = { # https://ieeexplore.ieee.org/document/9731545 (28nm)
    'paper_idx':            'ISSCC2022, 11.7',
    'input_toggle_rate':    0.5, # assumption (this paper will not be used for energy validation)
    'weight_sparsity':      0.9, # assumption (this paper will not be used for energy validation)
    'activation_precision': 8,
    'weight_precision':     8,
    'output_precision':     21,
    'input_precision':      1,
    'input_channel':        32, # how many input in parallel (per bank)
    'output_channel':       1, # how many output in parallel (per bank)
    'booth_encoding':       False,
    'multiplier_precision': 8,
    'adder_input_precision':16,
    'accumulator_input_precision': 8,
    'accumulator_precision':16,
    'reg_accumulator_precision': 16,
    'reg_input_bitwidth':   32,
    'pipeline':             True,
    'reg_pipeline_precision':8,
    'vdd':                  0.9, # V
    'rows':                 32*16, # equal to the number of input channels
    'cols':                 8*4,
    'banks':                2, # number of cores
    'area':                 0.03, # mm2
    'tclk':                 3, # ns
    'TOP/s':                0.0054,
    'TOP/s/W':              22,
    'unit_area':            0, # um2
    'unit_delay':           0, #ns
    'unit_cap':             0 #fF
    }

cacti_ISSCC2022_11_7 = { # 2048B, bw: 64
    'delay':                0.0944664, #ns
    'r_energy':             0.5643*1000/64*81, #fJ
    'w_energy':             0.607*1000/64*81, #fJ
    'area':                 0.00396 #mm2
    }

"""
ISSCC2023, 7.2 (50% input sparsity, 50% weight sparsity)
"""
dimc_ISSCC2023_7_2 = { # https://ieeexplore.ieee.org/document/10067260/
    'paper_idx':            'ISSCC2023, 7.2',
    'input_toggle_rate':    0.5,
    'weight_sparsity':      0.5,
    'activation_precision': 8,
    'weight_precision':     8,
    'output_precision':     23,
    'input_precision':      2,
    'input_channel':        128, # how many input in parallel (per bank)
    'output_channel':       8, # how many output in parallel (per bank)
    'booth_encoding':       False,
    'multiplier_precision': 1,
    'adder_input_precision':2,
    'accumulator_input_precision': 17,
    'accumulator_precision':23,
    'reg_accumulator_precision': 23,
    'reg_input_bitwidth':   2,
    'pipeline':             False,
    'reg_pipeline_precision':None,
    'vdd':                  0.9, # V
    'rows':                 64, 
    'cols':                 128, # equal to the number of input channels
    'banks':                8, # number of cores
    'area':                 0.1462, # mm2
    'tclk':                 1000/182, # ns
    'TOP/s':                None,
    'TOP/s/W':              19.5,
    'unit_area':            0, # um2
    'unit_delay':           0, #ns
    'unit_cap':             0 #fF
    }
cacti_value_ISSCC2023_7_2 = { # here I temporarily use: 1024 B, bw: 64 (no 128 in raw data)
    'delay':                0.0914947, #ns
    'r_energy':             0.401656*1000/64*81, #fJ
    'w_energy':             0.855128*1000/64*81, #fJ
    'area':                 0.00193147 #mm2
    }


if __name__ == '__main__':
    unit_area = 0.614 #um2
    unit_delay = 0.0478 #ns
    unit_cap = 0.7 #fF
    dimc_ISSCC2022_15_5['unit_area'] = unit_area #um2
    dimc_ISSCC2022_11_7['unit_area'] = unit_area #um2
    dimc_ISSCC2023_7_2['unit_area'] = unit_area #um2
    dimc_ISSCC2022_15_5['unit_delay'] = unit_delay #ns
    dimc_ISSCC2022_11_7['unit_delay'] = unit_delay #ns
    dimc_ISSCC2023_7_2['unit_delay'] = unit_delay #ns
    dimc_ISSCC2022_15_5['unit_cap'] = unit_cap #fF
    dimc_ISSCC2022_11_7['unit_cap'] = unit_cap #fF
    dimc_ISSCC2023_7_2['unit_cap'] = unit_cap #fF
    print(dimc_cost_estimation(dimc_ISSCC2022_15_5, cacti_ISSCC2022_15_5) )
    print(dimc_cost_estimation(dimc_ISSCC2022_11_7, cacti_ISSCC2022_11_7), 'Energy value does not make sense for this work (3rd value)') # no energy validation for this
    print(dimc_cost_estimation(dimc_ISSCC2023_7_2, cacti_value_ISSCC2023_7_2))
