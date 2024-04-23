# Llama 2 13B params
batch_size = 1
hidden_dim = 5120  # hidden dimension of embedding
seq_len = 16384  # actually 32000  # sequence length (note: this is dynamic during inference!)
num_heads = 1  # number of attention heads
fc_dim = 13824  # number of nodes in fully connected layer

workload = {
    # # MatMuls have sizes (PxQ)*(QxR)->(PxR)
    # 0: {  # Embedding
    #     "operator_type": "add",
    #     "equation": "O[p]=I[p]+W[p]",
    #     "loop_dim_size": {
    #         "P": hidden_dim,
    #     },
    #     "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
    #     "operand_source": {"W": [], "I": []},
    #     "constant_operands": ["I", "W"],
    # },
    0: {  # K projection
        "operator_type": "matmul",
        "equation": "O[p][r]+=I[p][q]*W[q][r]",
        "loop_dim_size": {
            "P": seq_len,
            "R": hidden_dim,
            "Q": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["W"],
    },
    # 2: {  # Q projection
    #     "operator_type": "matmul",
    #     "equation": "O[p][r]+=I[p][q]*W[q][r]",
    #     "loop_dim_size": {
    #         "P": seq_len,
    #         "R": hidden_dim,
    #         "Q": hidden_dim,  # inner loop variable
    #     },
    #     "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
    #     "operand_source": {"W": [], "I": [0]},
    #     "constant_operands": ["W"],
    #     # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    # },
    # 3: {  # V projection
    #     "operator_type": "matmul",
    #     "equation": "O[p][r]+=I[p][q]*W[q][r]",
    #     "loop_dim_size": {
    #         "P": seq_len,
    #         "R": hidden_dim,
    #         "Q": hidden_dim,  # inner loop variable
    #     },
    #     "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
    #     "operand_source": {"W": [], "I": [0]},
    #     "constant_operands": ["W"],
    #     # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    # },
    1: {  # KQ MM (Q should also be transposed...)
        "operator_type": "matmul",
        "equation": "O[p][r]+=I[p][q]*W[q][r]",
        "loop_dim_size": {
            "P": seq_len,
            "R": seq_len,
            "Q": hidden_dim,
        },
        "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
        "operand_source": {"I": [0], "W": [0]},
        "constant_operands": [],  # Note that none are constant
        # "operand_source_dimension_mapping": {"I": {"P": "Q"}},
    },
    # 5: {  # softmax(KQ^T)
    #     "operator_type": "softmax",  # ! not yet supported
    #     "equation": "O[p][r]+=I[p][r]*W[p][r]",
    #     "loop_dim_size": {
    #         "P": 1,
    #         "R": 1,
    #     },
    #     "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
    #     "operand_source": {"I": [1], "W": [2]},
    #     "constant_operands": [],  # Note that none are constant
    # },
    2: {  # KQ^T * V matmul
        "operator_type": "matmul",
        "equation": "O[p][r]+=I[p][q]*W[q][r]",
        "loop_dim_size": {
            "P": seq_len,
            "R": hidden_dim,
            "Q": seq_len,
        },
        "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
        "operand_source": {"I": [1], "W": [0]},
        "constant_operands": [],  # Note that none are constant
    },
    3: {  # fully connected
        "operator_type": "matmul",
        "equation": "O[p][r]+=I[p][q]*W[q][r]",
        "loop_dim_size": {
            "P": seq_len,
            "R": fc_dim,
            "Q": hidden_dim,
        },
        "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
        "operand_source": {"I": [2], "W": []},
        "constant_operands": ["W"],
    },
    4: {  # fully connected - add bias
        "operator_type": "add",
        "equation": "O[p][r]=I[p][r]+W[p][r]",
        "loop_dim_size": {
            "P": seq_len,
            "R": fc_dim,
        },
        "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
        "operand_source": {"I": [3], "W": []},
        "constant_operands": ["W"],
    },
    5: {  # fully connected
        "operator_type": "matmul",
        "equation": "O[p][r]+=I[p][q]*W[q][r]",
        "loop_dim_size": {
            "P": seq_len,
            "Q": fc_dim,
            "R": hidden_dim,
        },
        "operand_precision": {"O": 16, "O_final": 16, "W": 16, "I": 16},
        "operand_source": {"I": [4], "W": []},
        "constant_operands": ["W"],
    },
}
