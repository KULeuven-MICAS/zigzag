from zigzag.utils import pickle_deepcopy


def get_temporal_spatial_loops(cme):
    core = cme.accelerator.get_core(cme.layer.core_allocation)
    mol = cme.layer.memory_operand_links
    operands = list(mol.keys())
    tm = pickle_deepcopy(cme.temporal_mapping.mapping_dic_stationary)
    tls = [loop for level in tm["O"] for loop in level]
    temporal_loops = []
    all_mem_names = set()
    for tl in tls:
        mem_names = []
        for layer_op, mem_op in mol.items():
            # Find in which memory level this temporal loop is stored
            contains = [tl in level for level in tm[layer_op]]
            level = contains.index(True)
            # Remove it from the tm dict
            idx = tm[layer_op][level].index(tl)
            tm[layer_op][level].pop(idx)
            # Get the name of this memory level
            mem_name = core.get_memory_level(mem_op, level).memory_instance.name
            mem_names.append(mem_name)
            all_mem_names.add(mem_name)
        mem_names = tuple(mem_names)
        temporal_loops.append((tl[0], (0, tl[1]), mem_names))
    temporal_loops.reverse()
    sls = [l for level in cme.spatial_mapping_dict_int["O"] for l in level]
    spatial_loops = [(sl[0], (0, sl[1]), ('', '', '')) for sl in sls]
    spatial_loops.reverse()
    memories = list(all_mem_names)
    return temporal_loops, spatial_loops, memories

def print_mapping(cme, OFFSETS=2):
    # Extract the temporal loops, spatial loops, and memories from the cme
    temporal_loops, spatial_loops, memories = get_temporal_spatial_loops(cme)
    LOOP_COLUMN_WIDTH = max([len(l[0]) for l in temporal_loops]) + 14 + OFFSETS*(len(temporal_loops) + len(spatial_loops)) + 5
    MEMORY_COLUMN_WIDTH = max([len(i) for i in memories]) + 5
    """
    Prints a structured representation of a CostModelEvaluation mapping.
    
    :param cme: The CostModelEvaluation to print the mapping of.
    :param offsets: The number of spaces to offset nested loops.
    """
    def print_single_loop(loop_var, loop_range, memory, loop_str, indent):
        """
        Prints a single loop with its memory assignment at the given indentation level.
        """
        print(f"{' ' * indent}{loop_str} {loop_var} in [{loop_range[0]}, {loop_range[1]}):".ljust(LOOP_COLUMN_WIDTH), end="")
        print(f"{memory[0]:<{MEMORY_COLUMN_WIDTH}}{memory[1]:<{MEMORY_COLUMN_WIDTH}}{memory[2]:<{MEMORY_COLUMN_WIDTH}}")
        print(''.ljust(LOOP_COLUMN_WIDTH + 3*MEMORY_COLUMN_WIDTH, '-'))

    def recursive_print(loops, loop_str, offset=0, indent=True):
        """
        Recursively prints loops and their nested structure.
        """
        if not loops:
            return offset
        loop_var, loop_range, memory = loops[0]
        print_single_loop(loop_var, loop_range, memory, loop_str, offset)
        if indent:
            new_offset = recursive_print(loops[1:], loop_str, offset + OFFSETS, indent=indent)
        else:
            new_offset = recursive_print(loops[1:], loop_str, offset, indent=indent)
        return new_offset
    
    def print_header(text: str, operands: list):
        assert len(operands) == 3, "Three operands are expected"
        print(''.ljust(LOOP_COLUMN_WIDTH + 3*MEMORY_COLUMN_WIDTH, '='))
        print(f"{text.ljust(LOOP_COLUMN_WIDTH)}", end="")
        print(f"{operands[0]:<{MEMORY_COLUMN_WIDTH}}{operands[1]:<{MEMORY_COLUMN_WIDTH}}{operands[2]:<{MEMORY_COLUMN_WIDTH}}")
        print(''.ljust(LOOP_COLUMN_WIDTH + 3*MEMORY_COLUMN_WIDTH, '='))
    

    # Print Temporal loops header
    operands = list(cme.layer.memory_operand_links.keys())
    print_header("Temporal Loops", operands)
    # Start recursive temporal loops printing
    indent = recursive_print(temporal_loops, loop_str='for', offset=0, indent=True)
    # Print Spatial loops header
    operands = ["", "", ""]
    print_header("Spatial Loops", operands)
    # Start recursive spatial loops printing
    indent = recursive_print(spatial_loops, loop_str='parfor', offset=indent, indent=False)

if __name__ == "__main__":
    # Example usage
    import pickle
    with open("zigzag/visualization/list_of_cmes.pickle", "rb") as fp:
        cmes = pickle.load(fp)
    cme = cmes[0]
    print_mapping(cme)
