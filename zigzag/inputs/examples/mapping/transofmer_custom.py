mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("D", 8),
            "D2": ("L", 8),
            "D3": ("C", 4),
            "D4": ("K", 1),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "add": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("D", 8),
            "D2": ("L", 1),
            "D3": ("C", 1),
            "D4": ("K", 1),
        },
        "memory_operand_links": {
            "O": "O",
            "W": "I2",
            "I": "I1",
        },
    },
    "matmul": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("D", 8),
            "D2": ("C", 8),
            "D3": ("L", 1),
            "D4": ("K", 1),
        },  # !NOTE spatial dimension can be used for more than one loop dimension
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "matmul_project": {  # (LxC)*(CxD)->(LxD)
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("D", 8),
            "D2": ("C", 8),
            "D3": ("L", 4),
            "D4": ("L", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "matmul_attn": {  # (LxD)*(DxK)->(LxK)
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("K", 8),
            "D2": ("D", 8),
            "D3": ("L", 4),
            "D4": ("L", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "matmul_fc1": {  # (LxD)(DxDFC)->(LxDFC)
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("DFC", 8),
            "D2": ("D", 8),
            "D3": ("L", 4),
            "D4": ("L", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "add_fc": {  # (LxDFC)+(LxDFC)
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("L", 8),
            "D2": ("DFC", 1),
            # "D3": ("DFC", 1),
            # "D4": ("L", 1),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "matmul_fc2": {  # (LxDFC)(DFCxD)->(LxD)
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("D", 8),
            "D2": ("DFC", 8),
            "D3": ("L", 4),
            "D4": ("L", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    # "Conv": {
    #     "core_allocation": 1,
    #     "spatial_mapping": {
    #         "D1": ("D", 32),
    #         "D2": ("C", 32),
    #     },
    #     "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    # },
}
