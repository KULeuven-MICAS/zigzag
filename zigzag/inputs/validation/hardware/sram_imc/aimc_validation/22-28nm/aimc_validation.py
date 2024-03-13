import pdb
from aimc1_validation_subfunc import *
from aimc2_validation_subfunc import *
from aimc3_validation_subfunc import *

"""
CICC2021 (Assume 100% input toggle rate, 0% weight sparsity)
"""
aimc1 = { # https://ieeexplore.ieee.org/document/9431575 (22nm)
    'paper_idx':            'CICC2021',
    'input_toggle_rate':    1, # assumption
    'weight_sparsity':      0, # assumption
    'activation_precision': 7,
    'weight_precision':     2,
    'output_precision':     6, # output precision (unit: bit)
    'input_precision':      7,
    'input_channel':        1024, # how many input in parallel (per bank)
    'output_channel':       512, # how many output in parallel (per bank)
    'adc_resolution':       6,
    'dac_resolution':       7,
    'booth_encoding':       False,
    'multiplier_precision': 2,
    'adder_input_precision':None,
    'accumulator_precision':None,
    'reg_accumulator_precision': None,
    'reg_input_bitwidth':   None,
    'pipeline':             False,
    'vdd':                  0.8, # V
    'rows':                 1024, # equal to the number of input channels
    'cols':                 1024,
    'banks':                1, # number of cores
    'compact_rule':         False, # not used
    'area':                 1.9425, # mm2 (in code, the area will scale from 28nm -> 22nm)
    'tclk':                 1000/22.5, # ns (assume tclk doesn't scale with technology)
    'TOP/s':                None,
    'TOP/s/W':              1050,  # (in code, the energy will scale from 28nm -> 22nm)
    'unit_area':            0.614, # um2
    'unit_delay':           0.0478, #ns
    'unit_cap':             0.7, #fF
    'dac_energy_k0':        50 #fF (energy validation fitting parameter, which is taken directly from the value in TinyML paper)
    }
cacti1 = { # 131072B, bw: 1024
    'delay':                0.106473, #ns 
    'r_energy':             None, # not used
    'w_energy':             None, # not used
    'area':                 0.24496704 #mm2
    }
    
"""
JSSC2023 (Assume 100% input toggle rate, 0% weight sparsity)
"""
aimc2 = { # https://ieeexplore.ieee.org/document/9896828/ (28nm)
    'paper_idx':            'JSSC2023',
    'input_toggle_rate':    1, # assumption
    'weight_sparsity':      0, # assumption
    'activation_precision': 8,
    'weight_precision':     8,
    'output_precision':     20, # output precision (unit: bit)
    'input_precision':      8,
    'input_channel':        16, # how many input in parallel (per bank)
    'output_channel':       12, # how many output in parallel (per bank)
    'adc_resolution':       5,
    'dac_resolution':       2,
    'booth_encoding':       False,
    'multiplier_precision': 2,
    'adder_input_precision':12,
    'accumulator_input_precision':16,
    'accumulator_precision':20,
    'reg_accumulator_precision': 20,
    'reg_input_bitwidth':   None,
    'pipeline':             True,
    'vdd':                  0.9, # V
    'rows':                 32*16, # equal to the number of input channels
    'cols':                 8*2*12, # *2 for column MUX
    'banks':                4, # number of cores
    'compact_rule':         True,
    'area':                 0.468, # mm2
    'tclk':                 7.2, # ns
    'TOP/s':                None,
    'TOP/s/W':              15.02,
    'unit_area':            0.614, # um2
    'unit_delay':           0.0478, #ns
    'unit_cap':             0.7, #fF
    'dac_energy_k0':        50 #fF
    }
cacti2 = { #98304b , bw: 96
    'delay':                0.16111872, #ns
    'r_energy':             None, #fJ @ 0.9V # not used
    'w_energy':             None, #fJ @ 0.9V # not used
    'area':                 0.0360450648 #mm2
    }
    
"""
ISSCC2023, 7.8 (Assume 37.5% input toggle rate, 50% weight sparsity)
"""
aimc3 = { # https://ieeexplore.ieee.org/document/10067289 (22nm)
    'paper_idx':            'ISSCC2023, 7.8',
    'input_toggle_rate':    0.375, # assumption
    'weight_sparsity':      0.5, # assumption
    'activation_precision': 8,
    'weight_precision':     8,
    'output_precision':     24, # output precision (unit: bit)
    'input_precision':      1,
    'input_channel':        8, # how many input in parallel (per bank)
    'output_channel':       256, # how many output in parallel (per bank)
    'adc_resolution':       3,
    'dac_resolution':       0,
    'booth_encoding':       False,
    'multiplier_precision': 1,
    'adder_input_precision':None,
    'accumulator_precision':None,
    'reg_accumulator_precision': None,
    'reg_input_bitwidth':   None,
    'pipeline':             False,
    'vdd':                  0.8, # V @ 22nm
    'rows':                 64,
    'cols':                 256,
    'banks':                8, # number of cores
    'compact_rule':         True,
    'area':                 1.88, # mm2 (in code, the area will scale from 28nm -> 22nm)
    'tclk':                 1000/364, # ns
    'TOP/s':                None,
    'TOP/s/W':              18.7,  # (in code, the area will scale from 28nm -> 22nm)
    'unit_area':            0.614, # um2
    'unit_delay':           0.0478, #ns
    'unit_cap':             0.7, #fF
    'dac_energy_k0':        50 #fF
    }
cacti3 = { # 64*256, bw: 256
    'delay':                0.0722227, #ns not used # delay of array will be merged into ADC delay
    'r_energy':             None, #fJ @ 0.9V # not used
    'w_energy':             None, #fJ @ 0.9V # not used
    'area':                 0.004505472 #mm2
    }



if __name__ == '__main__':
    """
    For energy fitting, fit: dac_energy_k0
    For area fitting, fit: cell scaling factor (2 for now), constant in ADC formula
    For delay fitting, fit: constant in ADC formula
    """
#    print(aimc1_cost_estimation(aimc1, cacti1) ) # aimc1
#    print(aimc2_cost_estimation(aimc2, cacti2) ) # aimc2
    print(aimc3_cost_estimation(aimc3, cacti3) ) # aimc3

