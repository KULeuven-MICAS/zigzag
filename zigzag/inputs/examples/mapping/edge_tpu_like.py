mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {'D1': ('K', 8), 'D2': ('C', 8), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        "memory_operand_links": {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },
    'Add': {
        "core_allocation": 1,
        "spatial_mapping": {'D1': ('G', 8), 'D2': ('C', 1), 'D3': ('OX', 1), 'D4': ('OY', 1)},
        "memory_operand_links": {'O': 'O', 'X': 'I2', 'Y': 'I1'}
    },
    'Pooling': {
        "core_allocation": 1,
        "spatial_mapping": {'D1': ('G', 8), 'D2': ('C', 1), 'D3': ('OX', 1), 'D4': ('OY', 1)},
        "memory_operand_links": {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },
}