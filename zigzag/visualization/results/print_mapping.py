from copy import deepcopy


def create_printing_block(row, col):
    return [[' '] * col for _ in range(row)]


def modify_printing_block(old_block, start_row, start_col, new_str):
    new_block = deepcopy(old_block)
    new_block[start_row][start_col:start_col + len(new_str)] = new_str
    return new_block


def print_printing_block(printing_block):
    print()
    for i in range(len(printing_block)):
        print(''.join(printing_block[i]))


def print_good_tm_format(tm, mem_name, cme_name, mem_op_to_layer_op):
    op_list = list(tm.keys())
    tm_list = [tp for li in tm[op_list[0]] for tp in li]

    # get required interval between operands (e.g., 'W', 'I', 'O'), based on actual mem name length
    max_mem_name_len = 0
    for operand in op_list:
        for lv in range(len(mem_name[operand])):
            if len(mem_name[operand][lv]) > max_mem_name_len:
                max_mem_name_len = len(mem_name[operand][lv])
    interval = max_mem_name_len + 10

    tot_row = 2 * (len(tm_list) + 1) + 8
    tot_col = int(2 * (len(tm_list) + 1) + 3.75 * interval)
    tot_col_cut = 2 * (len(tm_list) + 1) + interval
    tm_block = create_printing_block(tot_row, tot_col)
    title = f' Temporal Mapping - {cme_name} '
    dash = '*' * int((tot_col - len(title)) / 2)
    tm_block = modify_printing_block(tm_block, 1, 0, dash + title + dash)
    i = 2
    for op in mem_op_to_layer_op.keys():
        tm_block = modify_printing_block(tm_block, i, 1, f'{op} ({mem_op_to_layer_op[op]}): ' + str(tm[op]))
        i += 1
    tm_block = modify_printing_block(tm_block, 6, 0, '-' * tot_col)
    tm_block = modify_printing_block(tm_block, 7, 1, 'Temporal Loops')
    tm_block = modify_printing_block(tm_block, 8, 0, '-' * tot_col)
    finish_row = 2 * len(tm_list) + 7
    for i, li in enumerate(tm_list):
        tm_block = modify_printing_block(tm_block, finish_row - 2 * i, len(tm_list) - i,
                                         'for ' + str(li[0]) + ' in ' + '[0:' + str(li[1]) + ')')
        tm_block = modify_printing_block(tm_block, 2 * (i + 1) + 1 + 7, 0, '-' * tot_col)

    # print mem name to each level
    for idx, operand in enumerate(op_list):
        column_position = tot_col_cut + idx * interval
        tm_block = modify_printing_block(tm_block, 7, column_position, operand)
        i = 0
        for level, lv_li in enumerate(tm[operand]):
            for _ in lv_li:
                tm_block = modify_printing_block(tm_block, finish_row - 2 * i, column_position,
                                                 str(mem_name[operand][level]))
                i += 1
    # tm_block = modify_printing_block(tm_block, finish_row + 2, 1,
    #                                  "(Notes: Temporal Mapping starts from the innermost memory level. MAC level is out of Temporal Mapping's scope.)")
    print_printing_block(tm_block)


def print_mapping(cme):
    tm = cme.temporal_mapping.mapping_dic_stationary
    mem_op_to_layer_op = cme.mem_op_to_layer_op
    layer_op_to_mem_op = cme.layer_op_to_mem_op
    mem_name = {}
    for (mem_op, mems_all_levels) in cme.accelerator.cores[0].mem_hierarchy_dict.items():
        layer_op = mem_op_to_layer_op[mem_op]
        mem_name[layer_op] = []
        for mem_a_level in mems_all_levels:
            mem_name[layer_op].append(mem_a_level.name)
        assert len(tm[layer_op]) == len(mem_name[layer_op]), \
            f"Temporal mapping level {len(tm[layer_op])} and memory hierarchy level {len(mem_name[layer_op])} of operand {layer_op} do not match."
    cme_name = str(cme)
    print_good_tm_format(tm, mem_name, cme_name, layer_op_to_mem_op)

if __name__ == '__main__':
    import pickle
    with open('../list_of_cmes.pickle', 'rb') as handle:
        list_of_cme = pickle.load(handle)
    for cme in list_of_cme:
        print_mapping(cme)
