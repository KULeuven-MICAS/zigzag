mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {
            # "D1": ("P", 8),
            # "D2": ("Q", 8),
            # "D3": ("P", 4),
            # "D4": ("P", 4),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    # "add": {
    #     "core_allocation": 1,
    #     "spatial_mapping": {
    #         # "D1": ("P", 8),
    #         # "D2": ("R", 1),
    #         # "D3": ("C", 1),
    #         # "D4": ("K", 1),
    #     },
    #     "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    # },
    # "matmul": {  # (PxQ)*(QxR)->(PxR)
    #     "core_allocation": 1,
    #     "spatial_mapping": {
    #         # "D1": ("R", 8),
    #         # "D2": ("Q", 8),
    #         # "D3": ("P", 4),
    #         # "D4": ("P", 4),
    #     },
    #     "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    # },
    # "softmax": {  # (PxQ)*(QxR)->(PxR)
    #     "core_allocation": 1,
    #     "spatial_mapping": {
    #         "D1": ("R", 1),
    #         # "D2": ("Q", 8),
    #         # "D3": ("P", 4),
    #         # "D4": ("P", 4),
    #     },
    #     "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    # },
}
