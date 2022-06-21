workload = {
    1: {  # start convolutional layer with 3x3 filter size and a stride of 2, output size is 28x28
        'equation': 'O[k][oy][ox]+=W[k][c][fy][fx]*I[c][ix][iy]',
        'equation_relations': ['ix=2*ox+fx-2', 'iy=2*oy+fy-2'],  # stride = 2 for ix and iy
        'loop_dim_size': {'K': 8, 'C': 8, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('OX', 14), 'D2': ('FX', 3)},  # Dimensions must be present in core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
    }
    ,
    2: {  # first depthwise convolution layer, 3x3 filter and 8 input/output channels
        'equation': 'O[g][oy][ox]+=W[g][fy][fx]*I[g][ix][iy]',
        # depthwise convolution using g to denote the in-out channels
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 8},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},  # This layers input comes from layer 1
        'source_storage_level': {'I': 1},  # This layers input is stored in level 1 of the 'I1' memory hierarchy
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('OX', 14), 'D2': ('FX', 3)},  # Dimensions must be present in core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    3: {  # first pointwise convolution layer, 1x1 filter and 8 input/output channels
        'equation': 'O[k][oy][ox]+=W[k][c]*I[c][ox][oy]',  # pointwise convolution (fx and fy omitted because 1x1)
        'equation_relations': [],  # no relationship between equation dimensions because ix=ox and iy=oy for pointwise
        'loop_dim_size': {'K': 8, 'C': 8, 'OY': 28, 'OX': 28},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'source_storage_level': {'I': 1},  # This layers input is stored in level 1 of the 'I1' memory hierarchy
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('OX', 14), 'D2': ('OY', 4)},  # Dimensions must be present in core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
    }
    ,
    4: {  # second depthwise convolution layer, 3x3 filter and 8 input/output channels
        'equation': 'O[g][oy][ox]+=W[g][fy][fx]*I[g][ix][iy]',
        # depthwise convolution using g to denote the in-out channels
        'equation_relations': ['ix=ox+fx-1', 'iy=oy+fy-1'],
        'loop_dim_size': {'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 8},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [3]},  # This layers input comes from layer 1
        'source_storage_level': {'I': 1},  # This layers input is stored in level 1 of the 'I1' memory hierarchy
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('OX', 14), 'D2': ('FX', 3)},  # Dimensions must be present in core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    5: {  # second pointwise convolution layer, 1x1 filter and 8 input/output channels
        'equation': 'O[k][oy][ox]+=W[k][c]*I[c][ox][oy]',  # pointwise convolution (fx and fy omitted because 1x1)
        'equation_relations': [],  # no relationship between equation dimensions because ix=ox and iy=oy for pointwise
        'loop_dim_size': {'K': 8, 'C': 8, 'OY': 28, 'OX': 28},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [4]},
        'source_storage_level': {'I': 1},  # This layers input is stored in level 1 of the 'I1' memory hierarchy
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('OX', 14), 'D2': ('OY', 4)},  # Dimensions must be present in core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
    }
    ,
    'add1': {  # element-wise add the outputs of layer 1 and layer 5
        'equation': 'O[k][oy][ox]=X[k][oy][ox]+Y[k][oy][ox]',
        'loop_dim_size': {'K': 8, 'OY': 28, 'OX': 28},
        'operand_precision': {'O': 24, 'O_final': 8, 'X': 24, 'Y': 24},
        'operand_source': {'X': [1], 'Y': [5]},
        'memory_operand_links': {'O': 'O', 'X': 'I2', 'Y': 'I1'}
    }
    ,
}
