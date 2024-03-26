mapping = {
    "Gemm": {  # Gemm
        "spatial_mapping": {"D1": ("M", 8), "D2": ("N", 8), "D3": ("K", 8)},
        "temporal_ordering": [
            # Innermost loop
            ("K", 8),
            ("N", 8),
            ("M", 8),
            ("K", 8),
            ("N", 8),
            ("M", 8),
            # Outermost loop
        ],
        "core_allocation": 1,
        "memory_operand_links": {
            "O": "O",
            "B": "I2",
            "A": "I1",
        },
    },
}