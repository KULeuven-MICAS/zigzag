workload = {
    1: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 16, 'C': 3, 'OY': 112, 'OX': 112, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'next_node': [2, 'add1']},

    2: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},    # current layer's Input ('I') comes from layer 1's Output.
        'next_node': [3, 4]},

    3: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'next_node': ['concat']},

    4: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 56, 'OX': 56, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'next_node': ['concat']},

    'concat': {'equation': 'O[g][b][k][oy][ox]=O1[g1][b1][k1][oy1][ox1]++O2[g2][b2][k2][oy2][ox2]',
               'loop_dim_size': {'B1': 2, 'K1': 64, 'OY1': 56, 'OX1': 56, 'G1': 1,
                                 'B2': 2, 'K2': 64, 'OY2': 56, 'OX2': 56, 'G2': 1},
               'operand_precision': {'O': 24, 'O1': 24, 'O2': 24},
               'operand_source': {'O1': [3], 'O2': [4]},
               'next_node': ['add1']},

    'add1': {'equation': 'O[g][b][k][oy][ox]=O1[g][b][k][oy][ox]+O2[g][b][k][oy][ox]',
             'loop_dim_size': {'B': 2, 'K': 64, 'OY': 56, 'OX': 56, 'G': 1},
             'operand_precision': {'O': 24, 'O_final': 8, 'O1': 24, 'O2': 24},
             'operand_source': {'O1': [1], 'O2': ['concat']},
             'next_node': [5]},

    5: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 64, 'C': 16, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': ['add1']},
        'next_node': [6, 'add2']},

    6: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 128, 'C': 64, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'next_node': [7]},

    7: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 128, 'C': 64, 'OY': 14, 'OX': 14, 'FY': 3, 'FX': 3, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [6]},
        'next_node': ['add2']},

    'add2': {'equation': 'O[g][b][k][oy][ox]=O1[g][b][k][oy][ox]+O2[g][b][k][oy][ox]',
             'loop_dim_size': {'B': 2, 'K': 64, 'OY': 56, 'OX': 56, 'G': 1},
             'operand_precision': {'O': 24, 'O_final': 8, 'O1': 24, 'O2': 24},
             'operand_source': {'O1': [5], 'O2': [7]},
             'next_node': [8]},

    8: {'equation': 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'B': 2, 'K': 256, 'C': 128, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'G': 1},
        'operand_precision': {'O': 24, 'O_final': 24, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': ['add2']},
        'next_node': []}

}
