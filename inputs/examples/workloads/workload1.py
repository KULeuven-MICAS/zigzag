workload = {
    1: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 16, 'C': 3, 'OY': 112, 'OX': 112, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'output_destination': ...,
        'core_allocation': 2,
        'spatial_mapping': {'D1': ('OX', 14), 'D2': ('FX', 3)},  # Must match with the dimensions of core 2
        'temporal_mapping': [('B', 2), ('K', 16), ('C', 3), ('OY', 112), ('OX', ), ('FY', 3)],  # For now just order of loops and internally we will allocate them to the memories
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    2: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },


    3: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    4: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    'add1': {'equation': 'O[g][b][k][oy][ox]=X[g][b][k][oy][ox]+Y[g][b][k][oy][ox]',
             'loop_dim_size': {'B': 2, 'K': 64, 'OY': 56, 'OX': 56, 'G': 1},
             'operand_precision': {'O': 24, 'O_final': 8, 'X': 24, 'Y': 24},
             'operand_source': {'X': [3], 'Y': [4]},
             'memory_operand_links': {'O': 'O', 'X': 'I2', 'Y': 'I1'}
             },

    5: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': ['add1']},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    6: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 128, 'C': 64, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    7: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 128, 'C': 64, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [6]},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        },

    'add2': {'equation': 'O[g][b][k][oy][ox]=A[g][b][k][oy][ox]+B[g][b][k][oy][ox]',
             'loop_dim_size': {'B': 2, 'K': 64, 'OY': 56, 'OX': 56, 'G': 1},
             'operand_precision': {'O': 24, 'O_final': 8, 'A': 24, 'B': 24},
             'operand_source': {'A': [5], 'B': [7]},
             'memory_operand_links': {'O': 'O', 'A': 'I2', 'B': 'I1'}
             },

    8: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 256, 'C': 128, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': ['add2']},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
}
