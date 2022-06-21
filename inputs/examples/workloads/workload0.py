workload = {
    0: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',  # for cost model testing purpose
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 4, 'C': 8, 'OY': 2002, 'OX': 2002, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 8), 'D2': ('K', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 8, 'C': 4, 'OY': 2002, 'OX': 2002, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 8), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    2: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 4, 'C': 4, 'OY': 2002, 'OX': 2002, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('C', 4), 'D2': ('K', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'source_storage_level': {'I': 1}
        }
    ,
    3: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 1, 'K': 48, 'C': 16, 'OY': 2016, 'OX': 2016, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 16), 'D2': ('C', 16)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'source_storage_level': {'I': 1}
        }
    ,
}
