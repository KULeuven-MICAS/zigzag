mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {'D1': ('K', 16), 'D2': ('C', 16), 'D3': ('OX', 2), 'D4': ('OY', 2)},
        "memory_operand_links": {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },
    'Add': {
        "core_allocation": 1,
        "spatial_mapping": {'D1': ('G', 16), 'D2': ('C', 1), 'D3': ('OX', 1), 'D4': ('OY', 1)},
        "memory_operand_links": {'O': 'O', 'X': 'I2', 'Y': 'I1'}
    },
}