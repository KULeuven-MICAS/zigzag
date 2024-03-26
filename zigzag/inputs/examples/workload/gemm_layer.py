workload = {
    0: {  # example Gemm layer
        "operator_type": "Gemm",
        "equation": "O[m][n]+=A[m][k]*B[k][n]",
        "dimension_relations": [],
        "loop_dim_size": {
            "M": 8*8*8,
            "K": 8*8*8,
            "N": 8*8*8,
        },
        "operand_precision": {"O": 32, "O_final": 8, "B": 8, "A": 8},
        "operand_source": {"B": [], "A": []},
        "constant_operands": ["B"],
        "operand_source_dimension_mapping": {},
    },
}