from math import prod
from typing import TypeAlias

from zigzag.datatypes import LayerDim, LayerOperand, PrLoop, UnrollFactor
from zigzag.utils import pickle_deepcopy
from zigzag.workload.layer_attributes import LayerDimSizes
from zigzag.workload.layer_node import LayerNode

SpatialMappingPerMemLvl: TypeAlias = dict[LayerOperand, list[list[tuple[LayerDim, UnrollFactor | float]]]]


def decouple_pr_loop(mapping_dict: SpatialMappingPerMemLvl, layer_node: "LayerNode") -> SpatialMappingPerMemLvl:
    """! This function decouples the pr loops into data size (r loops) and data reuse (ir loops).
    It also provides a transferred mapping dictionary in which the pr loops are replaced by r and ir loops.
    # TODO cleanup
    """

    relevancy_info = layer_node.loop_relevancy_info
    r_ir_operand_loop_lut: dict[LayerOperand, list[LayerDim]] = {
        layer_op: relevancy_info.get_r_layer_dims(layer_op) + relevancy_info.get_ir_layer_dims(layer_op)
        for layer_op in layer_node.layer_operands
    }
    pr_operand_loop_lut: dict[LayerOperand, PrLoop] = {
        layer_op: relevancy_info.get_pr_layer_dims(layer_op)
        for layer_op in layer_node.layer_operands
        if len(relevancy_info.get_pr_layer_dims(layer_op)) > 0
    }

    pr_operand_list = list(pr_operand_loop_lut.keys())
    mapping_dict_reform: SpatialMappingPerMemLvl = pickle_deepcopy(mapping_dict)

    # current and below level pr data size
    cabl_pr_data_size: dict[LayerOperand, dict[LayerDim, list[list[float]]]] = {}
    # current and below level pr data reuse
    cabl_pr_data_reuse: dict[LayerOperand, dict[LayerDim, list[list[float]]]] = dict()

    # each single pr loop data size
    per_pr_data_size: dict[LayerOperand, dict[LayerDim, list[list[float]]]] = dict()
    # each single pr loop data reuse
    per_pr_data_reuse: dict[LayerOperand, dict[LayerDim, list[list[float]]]] = dict()

    for operand in pr_operand_list:
        # initialize current and below level pr loop size
        cabl_pr_lp_size: dict[LayerDim, dict[LayerDim, UnrollFactor]] = {
            pr_data_dim: {pr_loop_dim: 1 for pr_loop_dim in pr_operand_loop_lut[operand][pr_data_dim]}
            for pr_data_dim in pr_operand_loop_lut[operand]
        }

        # initialize current and below level pr data size
        cabl_pr_data_size[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] for pr_data_dim in pr_operand_loop_lut[operand]
        }

        # initialize current and below level pr data reuse
        cabl_pr_data_reuse[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] for pr_data_dim in pr_operand_loop_lut[operand]
        }

        # initialize per pr loop data size
        per_pr_data_size[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] for pr_data_dim in pr_operand_loop_lut[operand]
        }

        # initialize per pr loop data reuse
        per_pr_data_reuse[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))]
            for pr_data_dim in pr_operand_loop_lut[operand].keys()
        }

        # update the cabl_pr_lp_size by multiply pr loop size across architectural level
        for level, loop_list in enumerate(mapping_dict[operand]):
            for loop_type, loop_size in loop_list:
                if loop_type in r_ir_operand_loop_lut[operand]:
                    continue
                for pr_data_dim in pr_operand_loop_lut[operand].keys():
                    if any(lp_type == loop_type for lp_type in pr_operand_loop_lut[operand][pr_data_dim]):
                        cabl_pr_lp_size[pr_data_dim][loop_type] *= loop_size

                        # compute pr related data dimension size and data dimension reuse at current and below joint
                        # levels based on pr_funcs (dynamic functions extracted in LayerNode). Each pr loop is decoupled
                        # into r and ir loops.
                        pr_loop_combined_to_r = layer_node.calc_tensor_dim(
                            pr_data_dim, LayerDimSizes(cabl_pr_lp_size[pr_data_dim])
                        )
                        pr_loop_combined_to_ir = prod(cabl_pr_lp_size[pr_data_dim].values()) / pr_loop_combined_to_r
                        cabl_pr_data_size[operand][pr_data_dim][level].append(pr_loop_combined_to_r)
                        cabl_pr_data_reuse[operand][pr_data_dim][level].append(pr_loop_combined_to_ir)

        # compute pr related data dimension size and data dimension reuse at each level for each pr loop
        # based on cabl_pr_data_size/cabl_pr_data_reuse """
        for pr_data_dim in cabl_pr_data_size[operand].keys():
            data_size_list = cabl_pr_data_size[operand][pr_data_dim]
            data_reuse_list = cabl_pr_data_reuse[operand][pr_data_dim]
            previous_data_size = 1
            previous_data_data_reuse = 1
            for level, va_list in enumerate(data_size_list):
                for idx in range(len(va_list)):
                    per_pr_data_size[operand][pr_data_dim][level].append(
                        data_size_list[level][idx] / previous_data_size
                    )
                    per_pr_data_reuse[operand][pr_data_dim][level].append(
                        data_reuse_list[level][idx] / previous_data_data_reuse
                    )
                    previous_data_size = data_size_list[level][idx]
                    previous_data_data_reuse = data_reuse_list[level][idx]

        mapping_dict_reform[operand] = replace_pr_loop_in_mapping(
            mapping_dict[operand],
            per_pr_data_size[operand],
            per_pr_data_reuse[operand],
            pr_operand_loop_lut[operand],
            r_ir_operand_loop_lut[operand],
        )

    return mapping_dict_reform


def replace_pr_loop_in_mapping(
    single_operand_mapping: list[list[tuple[LayerDim, UnrollFactor]]],
    per_pr_data_size: dict[LayerDim, list[list[float]]],
    per_pr_data_reuse: dict[LayerDim, list[list[float]]],
    pr_operand_loop_lut: PrLoop,
    r_ir_operand_loop_lut: list[LayerDim],
) -> list[list[tuple[LayerDim, UnrollFactor]]]:
    """! This function replaces all pr loops in a mapping of a single operand with r and ir loops."""
    mapping_new: list[list[tuple[LayerDim, UnrollFactor]]] = pickle_deepcopy(single_operand_mapping)

    for level, loop_list in enumerate(single_operand_mapping):
        # Introduce the current level pr loop index to distinguish different pr loops at the same architectural level
        cl_pr_lp_idx_local = {pr_data_dim: 0 for pr_data_dim in pr_operand_loop_lut.keys()}
        cl_pr_lp_idx_global = 0
        for idx, (loop_type, _) in enumerate(loop_list):
            if loop_type in r_ir_operand_loop_lut:
                continue
            for pr_data_dim in pr_operand_loop_lut.keys():
                if any(lp_type == loop_type for lp_type in pr_operand_loop_lut[pr_data_dim]):
                    # replace the pr loop in the mapping by r loop
                    pr_idx_local = cl_pr_lp_idx_local[pr_data_dim]
                    pr_idx_global = cl_pr_lp_idx_global
                    mapping_new[level][idx + pr_idx_global] = (
                        pr_data_dim.create_r_version(),
                        per_pr_data_size[pr_data_dim][level][pr_idx_local],
                    )
                    # insert ir loop after the r loop
                    # NOTE: Here we insert the ir loop after/above the r loop, which indicates that we ignore the input
                    # FIFO effect during current level feeds data to below level. We could also insert the ir loop
                    # before/below the r loop, which leads to more energy-efficient mapping if the innermost ir loop
                    # merging down is enabled.
                    mapping_new[level].insert(
                        idx + pr_idx_global + 1,
                        (
                            pr_data_dim.create_ir_version(),
                            per_pr_data_reuse[pr_data_dim][level][pr_idx_local],
                        ),
                    )
                    # update the pr loop index
                    cl_pr_lp_idx_local[pr_data_dim] += 1
                    cl_pr_lp_idx_global += 1

    return mapping_new
