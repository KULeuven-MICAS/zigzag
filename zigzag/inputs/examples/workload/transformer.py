batch_size = 1
hidden_dim = 5120  # hidden dimension of embedding
seq_len = 16384  # sequence length (note: this is dynamic during inference!)
num_heads = 1  # number of attention heads
fc_dim = 13824  # number of nodes in fully connected layer

workload = {
    # B: batch_size
    # MatMuls have sizes (PxQ)*(QxR)->(PxR)
    0: {  # Embedding
        "operator_type": "add",
        "equation": "O[b][p]=I[b][p]+W[p]",  # !NOTE No difference between += and =
        "loop_dim_size": {
            "B": 1,
            "P": hidden_dim,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
    },
    1: {  # K projection
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": hidden_dim,
            "Q": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}}, # TODO what does this do?
    },
    2: {  # Q projection
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": hidden_dim,
            "Q": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    },
    3: {  # V projection
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": hidden_dim,
            "Q": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    },
    4: {  # KQ MM (Q should also be transposed...)
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": seq_len,
            "Q": hidden_dim,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [1], "W": [2]},
        "constant_operands": [],  # Note that none are constant
        # "operand_source_dimension_mapping": {"I": {"P": "Q"}},
    },
    5: {  # softmax(KQ^T)
        "operator_type": "softmax",  # ! not yet supported
        "equation": "O[b][p][r]+=I[b][p][r]*W[p][r]",
        "loop_dim_size": {
            "B": 1,
            "P": 1,
            "R": 1,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [1], "W": [2]},
        "constant_operands": [],  # Note that none are constant
    },
    6: {  # KQ^T * V matmul
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": hidden_dim,
            "Q": seq_len,  # inner loop varaible
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [3], "W": [5]},  # TODO doet source veranderen iets?
        "constant_operands": [],  # Note that none are constant
    },
    7: {  # fully connected
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": fc_dim,
            "Q": hidden_dim,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [6], "W": []},
        "constant_operands": ["W"],
    },
    8: {  # fully connected - add bias
        "operator_type": "add",
        "equation": "O[b][p][r]=I[b][p][r]+W[p][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "R": fc_dim,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [7], "W": []},
        "constant_operands": ["W"],
    },
    9: {  # fully connected
        "operator_type": "matmul",
        "equation": "O[b][p][r]+=I[b][p][q]*W[q][r]",
        "loop_dim_size": {
            "B": 1,
            "P": seq_len,
            "Q": fc_dim,
            "R": hidden_dim,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"I": [8], "W": []},
        "constant_operands": ["W"],
    },
}
