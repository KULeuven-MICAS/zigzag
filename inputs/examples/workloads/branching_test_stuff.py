workload = {
    0: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    2: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    3: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    4: {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 16, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [1], 'B': [3]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}
        }
    ,
    5: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [4]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    6: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    7: {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 16, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [6], 'B': [4]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}
        }
    ,
    8: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [7]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    9: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [8]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    10: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [9]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    11: {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 16, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [10], 'B': [8]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}
        }
    ,
    12: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [11]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1210: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [12]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1211: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 13, 'FX': 13},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1210]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1212: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1211]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1220: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [12]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    12210: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1220]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    12211: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 7, 'FX': 7},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [12210]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    12212: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [12211]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    12220: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1220]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    12221: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [12220]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    123: {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 16, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [12212], 'B': [1212]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}
        }
    ,
    13: {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 16, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [123], 'B': [12221]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}
        }
    ,
    14: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [13]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1410: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [14]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1411: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1410]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1412: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1411]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1413: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1412]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1420: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [14]},
        'operand_source_dimension_mapping': {'I':{'IX':'OX', 'IY':'OY', 'C':'K', 'B':'B'}},
        'constant_operands' : ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    15: {'equation': 'O[b][k][oy][ox]=A[b][k][oy][ox]+B[b][k][oy][ox]',
         'loop_dim_size': {'B': 1, 'K': 16, 'OY': 56, 'OX': 56},
         'operand_precision': {'O': 8, 'O_final': 8, 'A': 8, 'B': 8},
         'operand_source': {'A': [1420], 'B': [1413]},
         'operand_source_dimension_mapping': {'A':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}, 'B':{'OX':'OX', 'OY':'OY', 'K':'K', 'B':'B'}},
         'core_allocation': 1,
         'spatial_mapping': {'D2': ('K', 4)},  # Must match with the dimensions of core 1
         'memory_operand_links':{'O': 'O', 'A': 'I1', 'B': 'I1'}
        }
}
