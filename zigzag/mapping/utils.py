from typing import TypeAlias

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import Constants, LayerDim, UnrollFactor
from zigzag.utils import pickle_deepcopy

TemporalLoopsType: TypeAlias = list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]]


def get_spatial_loops(cme: CostModelEvaluation):

    sls = [x for level in cme.spatial_mapping_dict_int[Constants.OUTPUT_LAYER_OP] for x in level]
    spatial_loops: list[tuple[LayerDim, tuple[int, UnrollFactor], tuple[str, ...]]] = [
        (sl[0], (0, sl[1]), ("", "", "")) for sl in sls
    ]
    spatial_loops.reverse()
    return spatial_loops


def get_temporal_loops(cme: CostModelEvaluation):
    operand_links = cme.layer.memory_operand_links
    tm = pickle_deepcopy(cme.temporal_mapping.mapping_dic_stationary)
    tls = [loop for level in tm[Constants.OUTPUT_LAYER_OP] for loop in level]
    temporal_loops: TemporalLoopsType = []
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
            mem_name = cme.accelerator.get_memory_level(mem_op, level).memory_instance.name
            mem_names.append(mem_name)
            all_mem_names.add(mem_name)
        mem_names_tuple = tuple(mem_names)
        temporal_loops.append((tl[0], (0, tl[1]), mem_names_tuple))
    return temporal_loops


def get_memory_names(cme: CostModelEvaluation):
    temporal_loops = get_temporal_loops(cme)
    all_mem_names: set[str] = set()
    for tl in temporal_loops:
        all_mem_names.update(tl[2])
    return list(all_mem_names)
