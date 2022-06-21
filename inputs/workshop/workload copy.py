workload = {
    0: {'equation': 'O[k][b]+=A[k][c]*B[c][b]',
        'loop_dim_size': {'K': 4, 'B': 4, 'C': 4},
        'operand_precision': {'O': 16, 'O_final': 8, 'A': 8, 'B': 8},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 4)},  # Must match with the dimensions of core 1
        'temporal_ordering': [('B', 4), ('C', 4)],
        'memory_operand_links': {'O': 'O', 'A': 'I1', 'B': 'I2'}
        }
}
