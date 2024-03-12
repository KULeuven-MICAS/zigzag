import pdb

"""
ISSCC2023, 16.3 (50% input sparsity, 50% weight sparsity)
"""
dimc_ISSCC2023_16_3 = {
    'paper_idx':            'ISSCC2023, 16.3',
    'input_toggle_rate':    0.5,
    'weight_sparsity':      0.5,
    'activation_precision': 8,
    'weight_precision':     8,
    'output_precision':     8, #not used
    'input_precision':      1,
    'input_channel':        128, # how many input in parallel (per bank)
    'output_channel':       8, # how many output in parallel (per bank)
    'booth_encoding':       False,
    'multiplier_precision': 1,
    'adder_input_precision':4,
    'accumulator_input_precision': 9,
    'accumulator_precision':17,
    'reg_accumulator_precision': 17,
    'reg_input_bitwidth':   1,
    'pipeline':             False,
    'reg_pipeline_precision':6,
    'vdd':                  0.9, # V
    'rows':                 128, 
    'cols':                 128, # equal to the number of input channels
    'banks':                4, # number of cores
    'area':                 0.269, # mm2
    'tclk':                 1000/400, # ns
    'TOP/s':                None,
    'TOP/s/W':              275,
    'unit_area':            0.614, # um2
    'unit_delay':           0.0478, #ns
    'unit_cap':             0.7 #fF
    }
cacti_value_ISSCC2023_16_3 = { # rows: 256, bw: 64
    'delay':                0.0944664, #ns
    'r_energy':             0.000691128*1000/64*81, #fJ
    'w_energy':             0.00102207*1000/64*81, #fJ
    'area':                 0.00416728 #mm2
    }


if __name__ == '__main__':
    from dimc_validation_subfunc4 import *
    print(dimc_cost_estimation4(dimc_ISSCC2023_16_3, cacti_value_ISSCC2023_16_3))
