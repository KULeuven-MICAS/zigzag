batch_size = 1
hidden_dim = 128  # hidden dimension of embedding
seq_len = 64  # sequence length (note: this is dynamic during inference!)
num_heads = 1  # number of attention heads
fc_dim = 512  # number of nodes in fully connected layer

workload = {
    # B: batch_size
    # D: hidden_dim
    # L: seq_len
    # H: num_heads
    # C: hidden_dim or seq_len (inner loop variable)
    # K: seq_len (inner loop variable)
    0: {  # Embedding
        "operator_type": "add",
        "equation": "O[b][d]=I[b][d]+W[d]",  # !NOTE No difference between += and =
        "loop_dim_size": {
            "B": 1,
            "D": hidden_dim,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
    },
    1: {  # K projection
        "operator_type": "matmul_project",
        "equation": "O[b][l][d]+=I[b][l][c]*W[b][c][d]",
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "D": hidden_dim,
            "C": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}}, # TODO what does this do?
    },
    2: {  # Q projection
        "operator_type": "matmul_project",
        "equation": "O[b][l][d]+=I[b][l][c]*W[b][c][d]",
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "D": hidden_dim,
            "C": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    },
    3: {  # V projection
        "operator_type": "matmul_project",
        "equation": "O[b][l][d]+=I[b][l][c]*W[b][c][d]",
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "D": hidden_dim,
            "C": hidden_dim,  # inner loop variable
        },
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": [0]},
        "constant_operands": ["W"],
        # "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    },
    4: {  # KQ MM (Q should also be transposed...)
        "operator_type": "matmul_attn",
        "equation": "O[b][l][k]+=I[b][l][d]*W[b][d][k]",  # I/W are K/Q matrices
        "loop_dim_size": {
            "B": 1,
            "D": hidden_dim,
            "L": seq_len,
            "K": seq_len,
        },
        "operand_precision": {"O": 16, "O_final": 8, "I": 8, "W": 8},
        "operand_source": {"I": [1], "W": [2]},
        "constant_operands": [],  # Note that none are constant
    },
    5: {  # softmax(KQ^T)
        "operator_type": "softmax",  # ! not yet supported
        "equation": "O[b][l][k]+=I[b][l][d]*W[b][d][k]",  # I/W are K/Q matrices
        "loop_dim_size": {
            "B": 1,
            "D": 1,
            "L": 1,
            "K": 1,
        },
        "operand_precision": {"O": 16, "O_final": 8, "I": 8, "W": 8},
        "operand_source": {"I": [1], "W": [2]},
        "constant_operands": [],  # Note that none are constant
    },
    6: {  # KQ^T * V matmul
        "operator_type": "matmul_project",
        "equation": "O[b][l][d]+=I[b][l][c]*W[c][d]",  # W is V matrix, I is LxL
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "D": hidden_dim,
            "C": seq_len,  # inner loop varaible
        },
        "operand_precision": {"O": 16, "O_final": 8, "I": 8, "W": 8},
        "operand_source": {"I": [3], "W": [5]},  # TODO doet source veranderen iets?
        "constant_operands": [],  # Note that none are constant
    },
    7: {  # fully connected
        "operator_type": "matmul_fc1",
        "equation": "O[b][l][dfc]+=I[b][l][d]*W[d][dfc]",
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "DFC": fc_dim,
            "D": hidden_dim,
        },
        "operand_precision": {"O": 16, "O_final": 8, "I": 8, "W": 8},
        "operand_source": {"I": [6], "W": []},
        "constant_operands": ["W"],
    },
    8: {  # fully connected - add bias
        "operator_type": "add_fc",
        "equation": "O[b][l][dfc]+=I[b][l][dfc]+W[l][dfc]",
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "DFC": fc_dim,
        },
        "operand_precision": {"O": 16, "O_final": 8, "I": 8, "W": 8},
        "operand_source": {"I": [7], "W": []},
        "constant_operands": ["W"],
    },
    9: {  # fully connected
        "operator_type": "matmul_fc2",
        "equation": "O[b][l][d]+=I[b][l][dfc]*W[dfc][d]",
        "loop_dim_size": {
            "B": 1,
            "L": seq_len,
            "DFC": fc_dim,
            "D": hidden_dim,
        },
        "operand_precision": {"O": 16, "O_final": 8, "I": 8, "W": 8},
        "operand_source": {"I": [8], "W": []},
        "constant_operands": ["W"],
    },
    # 1: {  # fc, from resnet18
    #     "operator_type": "Conv",
    #     "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
    #     "dimension_relations": ["ix=1*ox+0*fx", "iy=1*oy+0*fy"],
    #     "loop_dim_size": {
    #         "B": 1,
    #         "K": 50,
    #         "C": 32,
    #         "OY": 1,
    #         "OX": 1,
    #         "FY": 1,
    #         "FX": 1,
    #     },
    #     "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    #     "operand_source": {"W": [], "I": [0]},
    #     "constant_operands": ["W", "I"],
    #     "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY", "C": "G"}},
    # },
    # 2: {  # Add bias
    #     "operator_type": "Add",
    #     # "equation": "O[b][k][oy][ox]+=I[b][c][iy][ix]+W[c]",
    #     "equation": "O[b][k]=I[b][c]+W[c]",  # ? Dit kan precies ook
    #     "dimension_relations": ["ix=1*ox+0*fx", "iy=1*oy+0*fx"],
    #     "loop_dim_size": {
    #         "B": 1,
    #         "OY": 1,
    #         "OX": 1,
    #         "FY": 1,
    #         "FX": 1,
    #         "K": 32,
    #         "C": 32,
    #     },
    #     "operand_precision": {"O": 8, "O_final": 8, "I": 8, "W": 8},
    #     "operand_source": {"W": [], "I": [1]},
    #     "constant_operands": ["W"],
    # },
}
