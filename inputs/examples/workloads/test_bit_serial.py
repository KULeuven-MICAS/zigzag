workload = {
    0: {  # conv1, stride 2
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx][wb]*I[b][c][ix][iy]',
        'dimension_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 112, 'OX': 112, 'FY': 7, 'FX': 7, 'WB': 8},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 1, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },

}
