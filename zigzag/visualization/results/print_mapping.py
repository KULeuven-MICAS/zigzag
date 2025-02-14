from zigzag.cost_model.cost_model import (
    CostModelEvaluation,
    CostModelEvaluationABC,
    CumulativeCME,
)
from zigzag.datatypes import LayerDim, UnrollFactor
from zigzag.mapping.utils import get_memory_names, get_spatial_loops, get_temporal_loops


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
    temporal_loops = get_temporal_loops(cme)
    temporal_loops.reverse()
    spatial_loops = get_spatial_loops(cme)
    memories = get_memory_names(cme)
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
