from typing import TypeAlias
from typing import Dict, List
from typing import TYPE_CHECKING
from math import prod
from copy import deepcopy

from zigzag.classes.mapping.spatial.SpatialMapping import LayerDimStr, UnrollFactor
from zigzag.classes.workload.layer_node import Relevancy
from zigzag.utils import pickle_deepcopy

if TYPE_CHECKING:
    from zigzag.classes.workload.layer_node import LayerNode

SpatialMappingPerMEMLvl: TypeAlias = dict[str, list[list[tuple[LayerDimStr, UnrollFactor]]]]


class Loop:
    """!  Collect information of each single loop tuple in mapping.
    Applied range: from the lowest architectural level to the current level.
    """

    def __init__(self, loop: tuple, MAC_op: int, data_elem: int):
        """!  The class constructor
        @param loop
        @param MAC_op
        @param data_elem
        """
        self.loop = loop
        self.MAC_op = MAC_op
        self.data_elem = data_elem
        self.reuse = MAC_op / data_elem

    def __str__(self):
        return str(self.loop)

    def __repr__(self):
        return str(self.loop)


def decouple_pr_loop(mapping_dict: SpatialMappingPerMEMLvl, layer_node: "LayerNode") -> SpatialMappingPerMEMLvl:
    """!  This function decouples the pr loops into data size (r loops) and data reuse (ir loops).
    It also provides a transferred mapping dictionary in which the pr loops are replaced by r and ir loops.
    """

    operand_loop_dim = {op: layer_node.operand_loop_dim[op] for op in mapping_dict.keys()}
    r_ir_operand_loop_LUT = {
        op: relevance[Relevancy.R] + relevance[Relevancy.IR] for (op, relevance) in operand_loop_dim.items()
    }
    pr_operand_loop_LUT = {
        op: relevance[Relevancy.PR] for (op, relevance) in operand_loop_dim.items() if relevance[Relevancy.PR] != {}
    }
    pr_operand_list = list(pr_operand_loop_LUT.keys())
    mapping_dict_reform: SpatialMappingPerMEMLvl = pickle_deepcopy(mapping_dict)  # type: ignore

    # current and below level pr data size
    cabl_pr_data_size: dict = {}
    # current and below level pr data reuse
    cabl_pr_data_reuse: dict = {}

    # each single pr loop data size
    per_pr_data_size: dict = {}
    # each single pr loop data reuse
    per_pr_data_reuse: dict = {}

    for operand in pr_operand_list:

        # initialize current and below level pr loop size
        cabl_pr_lp_size = {
            pr_data_dim: {pr_loop_dim: 1 for pr_loop_dim in pr_operand_loop_LUT[operand][pr_data_dim]}
            for pr_data_dim in pr_operand_loop_LUT[operand]
        }

        # initialize current and below level pr data size
        cabl_pr_data_size[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] for pr_data_dim in pr_operand_loop_LUT[operand]
        }

        # initialize current and below level pr data reuse
        cabl_pr_data_reuse[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] for pr_data_dim in pr_operand_loop_LUT[operand]
        }

        # initialize per pr loop data size
        per_pr_data_size[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))] for pr_data_dim in pr_operand_loop_LUT[operand]
        }

        # initialize per pr loop data reuse
        per_pr_data_reuse[operand] = {
            pr_data_dim: [[] for _ in range(len(mapping_dict[operand]))]
            for pr_data_dim in pr_operand_loop_LUT[operand].keys()
        }

        # update the cabl_pr_lp_size by multiply pr loop size across architectural level
        for level, loop_list in enumerate(mapping_dict[operand]):
            for loop_type, loop_size in loop_list:
                if loop_type in r_ir_operand_loop_LUT[operand]:
                    continue
                for pr_data_dim in pr_operand_loop_LUT[operand].keys():
                    if any(lp_type == loop_type for lp_type in pr_operand_loop_LUT[operand][pr_data_dim]):
                        cabl_pr_lp_size[pr_data_dim][loop_type] *= loop_size

                        # compute pr related data dimension size and data dimension reuse at current and below joint levels
                        # based on pr_funcs (dynamic functions extracted in LayerNode). Each pr loop is decoupled into r and ir loops.
                        pr_loop_combined_to_r = layer_node.calc_tensor_dim(cabl_pr_lp_size[pr_data_dim], pr_data_dim)
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
            pr_operand_loop_LUT[operand],
            r_ir_operand_loop_LUT[operand],
        )

    # return mapping_dict_reform, cabl_pr_data_size, cabl_pr_data_reuse, per_pr_data_size, per_pr_data_reuse
    return mapping_dict_reform


def replace_pr_loop_in_mapping(
    single_operand_mapping: dict,
    per_pr_data_size: dict,
    per_pr_data_reuse: dict,
    pr_operand_loop_LUT: dict,
    r_ir_operand_loop_LUT: list,
):
    """!  This function replaces all pr loops in a mapping of a single operand with r and ir loops.
    @param single_operand_mapping
    @param per_pr_data_size
    @param per_pr_data_reuse
    @param pr_operand_loop_LUT
    @param r_ir_operand_loop_LUT
    """
    mapping_new: Dict = pickle_deepcopy(single_operand_mapping)  # type: ignore

    for level, loop_list in enumerate(single_operand_mapping):
        # Introduce the current level pr loop index to distinguish different pr loops at the same architectural level
        cl_pr_lp_idx_local = {pr_data_dim: 0 for pr_data_dim in pr_operand_loop_LUT.keys()}
        cl_pr_lp_idx_global = 0
        for idx, (loop_type, loop_size) in enumerate(loop_list):
            if loop_type in r_ir_operand_loop_LUT:
                continue
            for pr_data_dim in pr_operand_loop_LUT.keys():
                if any(lp_type == loop_type for lp_type in pr_operand_loop_LUT[pr_data_dim]):
                    # replace the pr loop in the mapping by r loop
                    pr_idx_local = cl_pr_lp_idx_local[pr_data_dim]
                    pr_idx_global = cl_pr_lp_idx_global
                    mapping_new[level][idx + pr_idx_global] = (
                        pr_data_dim + "_r",
                        per_pr_data_size[pr_data_dim][level][pr_idx_local],
                    )
                    # insert ir loop after the r loop
                    # NOTE: Here we insert the ir loop after/above the r loop, which indicates that we ignore the input FIFO effect
                    # during current level feeds data to below level. We could also insert the ir loop before/below the r loop,
                    # which leads to more energy-efficient mapping if the innermost ir loop merging down is enabled.
                    mapping_new[level].insert(
                        idx + pr_idx_global + 1,
                        (
                            pr_data_dim + "_ir",
                            per_pr_data_reuse[pr_data_dim][level][pr_idx_local],
                        ),
                    )
                    # update the pr loop index
                    cl_pr_lp_idx_local[pr_data_dim] += 1
                    cl_pr_lp_idx_global += 1

    return mapping_new


def calc_data_size_MAC_count_per_loop(mapping_dict_reform: Dict, operand_loop_dim_reform: Dict):
    """! This function generates detailed information for each single loop item for each operand."""
    detailed_mapping_dict = deepcopy(mapping_dict_reform)
    for operand, mapping_list in mapping_dict_reform.items():
        MAC_count = 1
        data_elem = 1
        for level, loop_list in enumerate(mapping_dict_reform[operand]):
            for idx, (loop_type, loop_size) in enumerate(loop_list):
                MAC_count *= loop_size
                if loop_type in operand_loop_dim_reform[operand][Relevancy.R]:
                    data_elem *= loop_size
                detailed_mapping_dict[operand][level][idx] = Loop(
                    (loop_type, loop_size), round(MAC_count), round(data_elem)
                )
    return detailed_mapping_dict
