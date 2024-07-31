from zigzag.cost_model.cost_model import (
    CostModelEvaluation,
    CostModelEvaluationABC,
    CumulativeCME,
)
from zigzag.datatypes import Constants, LayerDim, LayerOperand, UnrollFactor
from zigzag.utils import pickle_deepcopy


def get_temporal_spatial_loops(
    cme: CostModelEvaluation,
) -> tuple[
    list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]],
    list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]],
    list[str],
]:
    """
    # TODO documentation, split this up into multiple, sensible functions
    """
    core = cme.accelerator.get_core(cme.layer.core_allocation[0])
    operand_links = cme.layer.memory_operand_links

    tm: dict[LayerOperand, list[list[tuple[LayerDim, UnrollFactor]]]] = pickle_deepcopy(
        cme.temporal_mapping.mapping_dic_stationary
    )
    tls = [loop for level in tm[Constants.OUTPUT_LAYER_OP] for loop in level]
    temporal_loops: list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]] = []
    all_mem_names: set[str] = set()
    for tl in tls:
        mem_names: list[str] = []
        for layer_op in operand_links.layer_operands:
            mem_op = operand_links.layer_to_mem_op(layer_op)
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
        mem_names_tuple = tuple(mem_names)
        temporal_loops.append((tl[0], (0, tl[1]), mem_names_tuple))
    temporal_loops.reverse()
    sls = [x for level in cme.spatial_mapping_dict_int[Constants.OUTPUT_LAYER_OP] for x in level]
    spatial_loops: list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]] = [
        (sl[0], (0, sl[1]), ("", "", "")) for sl in sls
    ]
    spatial_loops.reverse()
    memories = list(all_mem_names)
    return temporal_loops, spatial_loops, memories


def print_mapping(cme: CostModelEvaluationABC, offsets: int = 2):
    """
    Prints a structured representation of a CostModelEvaluation mapping.

    :param cme: The CostModelEvaluation to print the mapping of.
    :param offsets: The number of spaces to offset nested loops.
    """
    # Skip CumulativeCMEs
    if isinstance(cme, CumulativeCME):
        return
    assert isinstance(cme, CostModelEvaluation)

    # Extract the temporal loops, spatial loops, and memories from the cme
    temporal_loops, spatial_loops, memories = get_temporal_spatial_loops(cme)
    loop_column_width = (
        max([len(x[0].name) for x in temporal_loops]) + 14 + offsets * (len(temporal_loops) + len(spatial_loops)) + 5
    )
    memory_column_width = max([len(i) for i in memories]) + 5

    def print_single_loop(
        loop_var: LayerDim,
        loop_range: tuple[int, UnrollFactor],
        memory: tuple[str, ...],
        loop_str: str,
        indent: int,
    ):
        """
        Prints a single loop with its memory assignment at the given indentation level.
        """
        print(
            f"{' ' * indent}{loop_str} {loop_var} in [{loop_range[0]}, {loop_range[1]}):".ljust(loop_column_width),
            end="",
        )
        print(f"{memory[0]:<{memory_column_width}}{memory[1]:<{memory_column_width}}{memory[2]:<{memory_column_width}}")
        print("".ljust(loop_column_width + 3 * memory_column_width, "-"))

    def recursive_print(
        loops: list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]],
        loop_str: str,
        offset: int = 0,
        indent: bool = True,
    ) -> int:
        """
        Recursively prints loops and their nested structure.
        """
        if not loops:
            return offset
        loop_var, loop_range, memory = loops[0]
        print_single_loop(loop_var, loop_range, memory, loop_str, offset)
        if indent:
            new_offset = recursive_print(loops[1:], loop_str, offset + offsets, indent=indent)
        else:
            new_offset = recursive_print(loops[1:], loop_str, offset, indent=indent)
        return new_offset

    def print_header(text: str, operands: list[str]):
        assert len(operands) == 3, "Three operands are expected"
        print("".ljust(loop_column_width + 3 * memory_column_width, "="))
        print(f"{text.ljust(loop_column_width)}", end="")
        print(
            f"{operands[0]:<{memory_column_width}}{operands[1]:<{memory_column_width}}"
            f"{operands[2]:<{memory_column_width}}"
        )
        print("".ljust(loop_column_width + 3 * memory_column_width, "="))

    print(f"Loop ordering for {cme.layer.name}")
    # Print Temporal loops header
    operands = list(map(lambda x: x.name, cme.layer.memory_operand_links.layer_operands))
    print_header("Temporal Loops", operands)
    # Start recursive temporal loops printing
    indent = recursive_print(temporal_loops, loop_str="for", offset=0, indent=True)
    # Print Spatial loops header
    operands = ["", "", ""]
    print_header("Spatial Loops", operands)
    # Start recursive spatial loops printing
    indent = recursive_print(spatial_loops, loop_str="parfor", offset=indent, indent=False)
    print()
