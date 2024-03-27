mapping = {
    "default": {
        "core_allocation": 1,
        # "spatial_mapping": {"D1": ("OX", 25), "D2": (("FX", 3), ("FY", 3))},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        "spatial_mapping_hint": {"D1": ["K", "OX"], "D2": ["C", "FX", "FY"]},
    },
    "Add": {  # to avoid errors when the workload is manually defined and contains Add layers.
        "core_allocation": 1,
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
        "spatial_mapping_hint": {"D1": ["G"], "D2": ["C"]},
    },
}
