workload = {
    0: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 56, 'C': 1, 'OY': 540, 'OX': 960, 'FY': 5, 'FX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1: {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ox][oy]',
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 56, 'OY': 540, 'OX': 960},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'operand_source_dimension_mapping': {'I':{'OX':'OX', 'OY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    2: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3,},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    3: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=2*ox+fx-2', 'iy=2*oy+fy-2'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3,},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    4: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3,},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [3]},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    5: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3,},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [4]},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    6: {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ox][oy]',
        'loop_dim_size': {'B': 1, 'K': 56, 'C': 12, 'OY': 540, 'OX': 960},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'OX':'OX', 'OY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    12: {'equation': 'O[b][k][oy][ox]+=W[k][c][fx][fy]*I[b][c][ix][iy]',
        'loop_dim_size': {'B': 1, 'K': 56, 'C': 56, 'OY': 540, 'OX': 960, 'FX': 3, 'FY': 3},
        'equation_relations': ['ix=2*ox+fx-2', 'iy=2*oy+fy-2'],
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    'add1': {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 56, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [6], 'B': [12]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}},

    7: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 56, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3,},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': ['add1']},
        'constant_operands' : ['W'],
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
}
