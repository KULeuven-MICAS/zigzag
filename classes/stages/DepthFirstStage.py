import math
import logging
from collections import defaultdict
from typing import Tuple, Dict, List, DefaultDict, Callable, Any
import pickle
import networkx as nx

from classes.depthfirst.data_copy_layer import DataCopyAction, DataCopyLayer
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.hardware.architecture.memory_level import MemoryLevel
from classes.opt.temporal.loma.memory_allocator import MemHierarchyTooSmallException
from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
from classes.workload.layer_node import LayerNode
from utils import pickle_deepcopy
from classes.stages.WorkloadStage import WorkloadStage
import classes.io.input_config as inputs
from  math import prod

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: Feature: multicore





# To avoid unpickable lambda's
def return_0():
    return 0
def return_empty_list():
    return []
def return_0_0_list():
    return [0,0]
def return_emtpystr_0_list():
    return ['', 0]
def return_emtpystr_0_tuple():
    return ('', 0)
def return_empty_dict():
    return {}


def get_largest_alive_size(corrected_workload: DNNWorkload, permanent_data: Dict[LayerNode, int] = {}):
    # find maximum size of feature map stuff in memory at same time,
    # to then reserve this space before figuring out where cache should go
    current_size = [0]
    largest_input_plus_output_size = [0]
    def add_to_size(s):
        current_size[0] += s
        largest_input_plus_output_size[0] = max(current_size[0], largest_input_plus_output_size[0])
    first = True
    to_remove = {}
    for n in nx.topological_sort(corrected_workload):

        si = sum(n.operand_size_elem[I] * n.operand_precision[I] for I in n.input_operands if I not in n.constant_operands)
        # subtract existing, add this. Note that this is necessary as input may be greater than it was output of
        # a previous layer due to caching refetch
        for in_node, _ in corrected_workload.in_edges(n):
            add_to_size(-to_remove[in_node][1])
        add_to_size(si)

        s = n.operand_size_elem['O'] * n.operand_precision['O_final']
        add_to_size(s)


        add_to_size(-si)
        # re add input part that was removed above because of overlap
        for in_node, _ in corrected_workload.in_edges(n):
            add_to_size(to_remove[in_node][1])

        else:
            for in_node, _ in corrected_workload.in_edges(n):
                to_remove[in_node][0].remove(n)
                if len(to_remove[in_node][0]) == 0:
                    add_to_size(-to_remove[in_node][1])
                    if n in permanent_data:
                        add_to_size(permanent_data[n])
                    del to_remove[in_node]

        to_remove[n] = [[], n.operand_size_elem['O'] * n.operand_precision['O_final']]
        for _, out_node in corrected_workload.out_edges(n):
            to_remove[n][0].append(out_node)
        to_remove[n][1] = s
    largest_input_plus_output_size = largest_input_plus_output_size[0]
    return largest_input_plus_output_size


def get_effective_output_to_cache(to_cache, node):
    """
    From multiple required to_caches with lag, get the total to_cache with lag
    :param to_cache:
    :param node:
    :return:
    """
    assert(node not in to_cache or not to_cache[node] or all(to_cache[node][i][0] == to_cache[node][next(iter(to_cache[node]))][0] for i in to_cache[node]))
    if node not in to_cache:
        return 0, 0
    if not to_cache[node]:
        return 0, 0
    lag = max(to_cache[node][i][2] for i in to_cache[node])
    leftmost = min(to_cache[node][i][2] - to_cache[node][i][1] for i in to_cache[node])
    dim = to_cache[node][next(iter(to_cache[node]))][0]
    to_cache = lag-leftmost
    return dim, to_cache, lag

def backpropagate_tilesize(workload: DNNWorkload, on_i, tile_x, tile_y,
                                       use_horizontal_caching=False,
                                       make_horizontal_caching=False,
                                       use_vertical_caching=False,
                                       make_vertical_caching=False)\
                    -> Tuple[List[LayerNode],
                             Dict[LayerNode, Tuple[str, int]],
                             Dict[Tuple[LayerNode, str], Tuple[str, int]],
                             Dict[LayerNode, Tuple[str, int]],
                             Dict[Tuple[LayerNode, str], Tuple[str, int]]]:
                """
                Backpropagates a tilesize of (OX, OY) = (tile_x, tile_y) at the output nodes to (OX, OY) of earlier
                layers (nodes in workload).
                Note: all of this is for a fully-in-regime tile
                :param workload: the workload to apply this on
                :param tile_x:
                :param tile_y:
                :param use_horizontal_caching: assume horizontal caching is used
                :param use_vertical_caching: assume vertical caching is used
                :return: (usefull_nodes, overlap_columns_out, overlap_columns_in, overlap_rows_out, overlap_rows_in).
                First a list of nodes contributing to the output node currently handled (with index on_i).
                Then two both dicts containing for every layer how many columns if it's output/input should be cached
                for reuse.
                Then the same for rows
                """

                if on_i is None:
                    on_i_list = range(len(list(node for node, out_degree in workload.out_degree() if out_degree == 0)))
                else:
                    on_i_list = [on_i]
                usefull_nodes_output_sizes = defaultdict(return_0_0_list)
                node_output_sizes_bits = {}
                columns_to_cache_output = defaultdict(return_empty_dict)  # of output!
                columns_to_cache_input = defaultdict(return_emtpystr_0_tuple)
                rows_to_cache_output = defaultdict(return_empty_dict)  # of output!
                rows_to_cache_input = defaultdict(return_emtpystr_0_tuple)




                for on_i in on_i_list:
                    on: LayerNode = list(node for node, out_degree in workload.out_degree() if out_degree == 0)[on_i]
                    if tile_x is not None:
                        on.loop_dim_size['OX'] = tile_x
                    if tile_y is not None:
                        on.loop_dim_size['OY'] = tile_y
                    usefull_nodes_output_sizes[on] = [on.loop_dim_size['OX'], on.loop_dim_size['OY']]
                    nodes_to_trace_over = [(on, 'OX', 'OY', 0, 0)]


                    # construct ROI of output patch (with usefull nodes)
                    largest_input_plus_output_size = 0
                    while len(nodes_to_trace_over) > 0:
                        n: LayerNode = nodes_to_trace_over.pop(0)
                        n, df_output_loop_X, df_output_loop_Y, lag_h, lag_v = n
                        n.extract_layer_info()
                        output_size = n.operand_size_elem['O'] * n.operand_precision['O_final']
                        node_output_sizes_bits[n] = output_size



                        # for input operands
                        # get input operand source
                        for I, input_node in n.input_operand_source.items():
                            input_loops = n.calc_tensor_dims(I, n.loop_dim_size)

                            df_input_loop_X : str
                            if df_output_loop_X in input_loops:
                                df_input_loop_X = df_output_loop_X
                            else:
                                for l in n.operand_loop_dim[I]['pr']:
                                    if df_output_loop_X in n.operand_loop_dim[I]['pr'][l]:
                                        df_input_loop_X = l
                                        break
                            df_input_loop_Y : str
                            if df_output_loop_Y in input_loops:
                                df_input_loop_Y = df_output_loop_Y
                            else:
                                for l in n.operand_loop_dim[I]['pr']:
                                    if df_output_loop_Y in n.operand_loop_dim[I]['pr'][l]:
                                        df_input_loop_Y = l
                                        break

                            # find strides
                            if df_input_loop_X == df_output_loop_X:
                                SX = 1
                            else:
                                SX = n.pr_scaling_factors[df_input_loop_X][df_output_loop_X.lower()]
                            if df_input_loop_Y == df_output_loop_Y:
                                SY = 1
                            else:
                                SY = n.pr_scaling_factors[df_input_loop_Y][df_output_loop_Y.lower()]

                            input_overlap_X = max(0,
                                                  input_loops[df_input_loop_X]
                                                  - n.loop_dim_size[df_output_loop_X] * SX)
                            input_overlap_Y = max(0,
                                                  input_loops[df_input_loop_Y]
                                                  - n.loop_dim_size[df_output_loop_Y] * SY)
                            # find lag
                            newlag_h = SX * lag_h \
                                          - (input_loops[df_input_loop_X] - n.loop_dim_size[df_output_loop_X] * SX + SX -1)//2 \
                                          + input_overlap_X

                            newlag_v = SY * lag_v \
                                          - (input_loops[df_input_loop_Y] - n.loop_dim_size[df_output_loop_Y] * SY + SY -1)//2 \
                                          + input_overlap_Y


                            columns_to_cache_input[(n, I)] = df_input_loop_X, input_overlap_X
                            rows_to_cache_input[(n, I)] = df_input_loop_Y, input_overlap_Y
                            if use_horizontal_caching:
                                input_loops[df_input_loop_X] -= input_overlap_X
                            if use_vertical_caching:
                                input_loops[df_input_loop_Y] -= input_overlap_Y

                            IX = input_loops[df_input_loop_X]
                            IY = input_loops[df_input_loop_Y]


                            usefull_nodes_output_sizes[input_node][0] = max(
                                                                          usefull_nodes_output_sizes[input_node][0],
                                                                          IX)
                            usefull_nodes_output_sizes[input_node][1] = max(
                                                                          usefull_nodes_output_sizes[input_node][1],
                                                                          IY)

                            input_node.loop_dim_size[n.operand_source_dimension_mapping[I][df_input_loop_X]] = \
                                usefull_nodes_output_sizes[input_node][0]
                            input_node.loop_dim_size[n.operand_source_dimension_mapping[I][df_input_loop_Y]] = \
                                usefull_nodes_output_sizes[input_node][1]

                            columns_to_cache_output[input_node][n] = [
                                    n.operand_source_dimension_mapping[I][df_input_loop_X],
                                    input_overlap_X if make_horizontal_caching else 0,
                                    newlag_h]
                            newlag_h = get_effective_output_to_cache(columns_to_cache_output, input_node)[2]

                            rows_to_cache_output[input_node][n] = [
                                    n.operand_source_dimension_mapping[I][df_input_loop_Y],
                                    input_overlap_Y if make_vertical_caching else 0,
                                    newlag_v]
                            newlag_v = get_effective_output_to_cache(rows_to_cache_output, input_node)[2]


                            nodes_to_trace_over.insert(0, (input_node,
                                                             n.operand_source_dimension_mapping[I][df_input_loop_X],
                                                             n.operand_source_dimension_mapping[I][df_input_loop_Y],
                                                             newlag_h,
                                                             newlag_v))


                return list(usefull_nodes_output_sizes.keys()), \
                       {k: get_effective_output_to_cache(columns_to_cache_output, k)[:-1] for k in columns_to_cache_output}, \
                       columns_to_cache_input, \
                       {k: get_effective_output_to_cache(rows_to_cache_output, k)[:-1] for k in rows_to_cache_output}, \
                       rows_to_cache_input


class DepthFirstStage(Stage):
    """
    Stage that processes a workload in a depth first style manner.
    Yielded results are per-layer accumulated results across tiles.
    Subcallable may only yield a single value!
    """
    def __init__(self, list_of_callables, *,
                 accelerator,
                 workload,
                 df_tilesize_x:int, df_tilesize_y:int,
                 df_horizontal_caching:bool, df_vertical_caching:bool,
                 **params):
        """
        Initialize the pipeline by initializing the workload and spatial mapping converison loma pipelines.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **params)
        self.tilesize_x = df_tilesize_x
        self.tilesize_y = df_tilesize_y
        self.horizontal_caching = df_horizontal_caching
        self.vertical_caching = df_vertical_caching
        self.workload = workload
        self.accelerator = accelerator

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        return str(self)

    def run(self):

        nb_workload_output_nodes = \
            len(list(node for node, out_degree in self.workload.out_degree() if out_degree == 0))

        cost_model_evaluations_per_layer = defaultdict(return_empty_list)

        # compute full feature map sizes first
        backpropagate_tilesize(self.workload, None, None, None, False, False, False, False)


        for on_i in range(nb_workload_output_nodes):
            usefull_workload = pickle_deepcopy(self.workload)
            usefull_nodes = list(node for node, out_degree in usefull_workload.out_degree() if out_degree == 0)[on_i:on_i+1]
            nodes_to_track = usefull_nodes[:]
            while nodes_to_track:
                next_nodes_to_track = []
                for input_node, _ in list(usefull_workload.in_edges(nodes_to_track)):
                    next_nodes_to_track.append(input_node)
                usefull_nodes += next_nodes_to_track
                nodes_to_track = next_nodes_to_track

            for n in usefull_workload.nodes():
                if n not in usefull_nodes:
                    usefull_workload.remove_node(n)

            on = usefull_nodes[0]

            total_OX, total_OY = on.loop_dim_size['OX'], on.loop_dim_size['OY']

            run_for_tilesize_cache = {}



            def run_for_tilesize(tile_x, tile_y, weights_from_top,
                                 make_horizontal_cache_ml=None,
                                 use_horizontal_cache_ml=None,
                                 horizontal_write_columns=None,
                                 horizontal_read_columns=None,
                                 make_vertical_cache_ml=None,
                                 use_vertical_cache_ml=None,
                                 vertical_write_rows=None,
                                 vertical_read_rows=None,
                                 w_ml=None,
                                 tile_x0=None, tile_y0=None): #only for info printing
                # caching results for runtime optimization
                args = (tile_x, tile_y, weights_from_top, make_horizontal_cache_ml, use_horizontal_cache_ml,
                        tuple(horizontal_write_columns.items()), tuple(horizontal_read_columns.items()),
                        make_vertical_cache_ml, use_vertical_cache_ml,
                        tuple(vertical_write_rows.items()), tuple(vertical_read_rows.items()))
                if args in run_for_tilesize_cache:
                    return run_for_tilesize_cache[args]

                # not in cache => actually computing this tile
                logger.debug("Running for " + str(tuple(a if not isinstance(a, MemoryLevel) else a.memory_instance.name for a in args)) + f"at position {tile_x0}, {tile_y0}")


                best_cme_per_layer_tile = defaultdict(return_0)
                usefull_workload_tile = pickle_deepcopy(usefull_workload)
                usefull_nodes, _, _, _, _= backpropagate_tilesize(usefull_workload_tile, on_i, tile_x, tile_y,
                                                             use_horizontal_caching=use_horizontal_cache_ml is not None,
                                                             make_horizontal_caching=make_horizontal_cache_ml is not None,
                                                             use_vertical_caching=use_vertical_cache_ml is not None)

                largest_input_plus_output_size = get_largest_alive_size(usefull_workload_tile)

                mh = self.accelerator.get_core(next(usefull_workload_tile.topological_sort()).core_allocation).memory_hierarchy

                memories_to_take_for_W = []
                for i, ml in enumerate(mh.get_memory_levels(next(usefull_workload_tile.topological_sort()).memory_operand_links['W'])):
                    memories_to_take_for_W.append(ml)
                    if ml == w_ml:
                        break

                # lookup dictionary to know where a previous layer stored its output
                # to know where to find the input at a current layer
                memories_to_take_for_I_lookup_as_from_O = {}
                # run the newly adapted workload
                to_copy = []  # holds still to proces DataMoveActions
                layer : LayerNode
                for layer in usefull_workload_tile.topological_sort():

                    # throwing stuff at the wall to see what sticks
                    extra_output_memlevels_because_it_did_not_work = 0
                    extra_input_memlevels_because_it_did_not_work = 0
                    to_copy_orig = to_copy[:]
                    while True:  # serves more as a goto target here
                        logger.debug(f"Current layer: {layer}")
                        is_networks_input = usefull_workload_tile.in_degree(layer) == 0
                        is_networks_output = usefull_workload_tile.out_degree(layer) == 0
                        to_copy = to_copy_orig[:]

                        # TODO: better way to get relevant input operands?
                        #  If so, do on lot's of places in this file
                        input_size = sum(layer.operand_size_elem[I] * layer.operand_precision[I] for I in layer.input_operands if I not in layer.constant_operands)

                        logger.debug("Input size: " + str(input_size))
                        if is_networks_input:  # just all of them
                            memories_to_take_for_I = list(mh.get_memory_levels(layer.memory_operand_links[next(I for I in layer.input_operands if I not in layer.constant_operands)]))
                        else:
                            # loweset ml that fits all the inputs of this layer and is not unrolled
                            memories_to_take_for_I = []
                            I = next(I for I in layer.input_operands if I not in layer.constant_operands)
                            for i, ml in enumerate(mh.get_memory_levels(layer.memory_operand_links[I])):
                                memories_to_take_for_I.append(ml)
                                if ml.memory_instance.size >= input_size and  ml.unroll_count == 1:
                                    break

                        layer_output_size = layer.operand_size_elem['O'] * layer.operand_precision['O_final']
                        logger.debug("Output size: " + str(layer_output_size))

                        if is_networks_output:  # just take all of them
                            memories_to_take_for_O = list(mh.get_memory_levels(layer.memory_operand_links['O']))
                        else:
                            # lowest ml that fits
                            # - the output
                            # - the input, if this ml is also the highest one to be included for inputs
                            #  ... and is not unrolled
                            memories_to_take_for_O = []
                            for i, ml in enumerate(mh.get_memory_levels(layer.memory_operand_links['O'])):
                                memories_to_take_for_O.append(ml)
                                if ml.memory_instance.size >= layer_output_size + \
                                        (input_size if memories_to_take_for_I[-1] else 0) and ml.unroll_count == 1:
                                    break
                        # throwing this at wall stuff: take more levels if already proven that more are needed
                        memories_to_take_for_I.extend(mh.get_memory_levels(layer.memory_operand_links[next(I for I in layer.input_operands if I not in layer.constant_operands)])[len(memories_to_take_for_I):len(memories_to_take_for_I)+extra_input_memlevels_because_it_did_not_work])
                        memories_to_take_for_O.extend(mh.get_memory_levels(layer.memory_operand_links['O'])[len(memories_to_take_for_O):len(memories_to_take_for_O)+extra_output_memlevels_because_it_did_not_work])

                        logger.debug('top to take for I  {}'.format(memories_to_take_for_I[-1]))

                        # register to_copy stuff (DataCopyActions to be)
                        if not is_networks_input:
                            # output -> input
                            for I in layer.input_operand_source:
                                stored_as_output = memories_to_take_for_I_lookup_as_from_O[layer.input_operand_source[I]]
                                if memories_to_take_for_I[-1] != stored_as_output[0]:
                                    # the inputs where not stored where they need to be stored now -> copy them
                                    to_copy.append((('O', I), stored_as_output[0], memories_to_take_for_I[-1], stored_as_output[1]))
                                    logger.debug("Added to copy output -> input: " + str(tuple(a if not isinstance(a, MemoryLevel) else a.memory_instance.name for a in to_copy[-1])))


                            # caches for horizontal resue -> input
                            if use_horizontal_cache_ml is not None and use_horizontal_cache_ml != memories_to_take_for_I[-1]:
                                for I in layer.input_operands:
                                    if I not in layer.constant_operands:
                                        # copy cached inputs
                                        amount_bits_factors = {k: layer.loop_dim_size[k] for k in layer.operand_loop_dim[I]['r']}
                                        for k in layer.operand_loop_dim[I]['pr']:
                                            amount_bits_factors[k] = layer.calc_tensor_dims(I, layer.loop_dim_size)[k]
                                        amount_bits_factors[horizontal_read_columns[(layer.id, I)][0]] = horizontal_read_columns[(layer.id, I)][1]
                                        amount_bits_factors['precision'] = layer.operand_precision['O_final']

                                        # practically doing this but more flexible in terms of loops and operand names
                                        # amount_bits_factors = horizontal_read_columns[layer.id] \
                                        #                       , LayerNode.return_lambda(layer.pr_funcs_LUT['IY'])(layer.loop_dim_size) \
                                        #                       , layer.loop_dim_size.get('C', 1) \
                                        #                       , layer.operand_precision['I']
                                        amount_bits = prod(amount_bits_factors.values())
                                        if amount_bits>0:
                                            to_copy.append((('CI',I), use_horizontal_cache_ml, memories_to_take_for_I[-1], amount_bits))
                                            logger.debug("Added to copy cached input (H) -> input: " + str(tuple(a if not isinstance(a, MemoryLevel) else a.memory_instance.name for a in to_copy[-1])) + " = " + '*'.join(str(f) for f in amount_bits_factors.values()))

                            # caches for verticale reuse -> input
                            if use_vertical_cache_ml is not None and vertical_cache_ml != memories_to_take_for_I[-1]:
                                for I in layer.input_operands:
                                    if I not in layer.constant_operands:
                                        column_dim = vertical_read_rows[(layer.id, I)][2]
                                        nb_columns = layer.loop_dim_size[column_dim] if column_dim in layer.loop_dim_size \
                                                        else layer.calc_tensor_dims(I, layer.loop_dim_size)[column_dim]

                                        if use_horizontal_cache_ml is not None:
                                            # this part overlaps and is already loaded because of horizontal reuse caching
                                            nb_columns -= horizontal_read_columns[(layer.id, I)][1]

                                        amount_bits_factors = {k: layer.loop_dim_size[k] for k in layer.operand_loop_dim[I]['r']}
                                        for k in layer.operand_loop_dim[I]['pr']:
                                            amount_bits_factors[k] = layer.calc_tensor_dims(I, layer.loop_dim_size)[k]

                                        amount_bits_factors[vertical_read_rows[(layer.id, I)][2]] = nb_columns
                                        amount_bits_factors[vertical_read_rows[(layer.id, I)][0]] = vertical_read_rows[(layer.id, I)][1]
                                        amount_bits_factors['precision'] = layer.operand_precision['O_final']
                                        amount_bits = prod(amount_bits_factors.values())
                                        if amount_bits>0:
                                            to_copy.append((('CI',I), use_vertical_cache_ml, memories_to_take_for_I[-1], amount_bits))
                                            logger.debug("Added to copy cached input (V) -> input: " + str(tuple(a if not isinstance(a, MemoryLevel) else a.memory_instance.name for a in to_copy[-1])) + " = " + '*'.join(str(f) for f in amount_bits_factors.values()))



                        # give subsequent layers info on where to find there inputs = outputs of this layer
                        memories_to_take_for_I_lookup_as_from_O[layer] = memories_to_take_for_O[-1], layer_output_size

                        logger.debug('Top to take for O  {}'.format(memories_to_take_for_O[-1]))

                        # copy of the accelerator, which will have adjusted memory hierarchy
                        accelerator_mem_level_removed = pickle_deepcopy(self.accelerator)
                        accelerator_mem_level_removed.name += "_sub"
                        # the memory hierarchy that will be adjusted
                        memhier = accelerator_mem_level_removed.get_core(layer.core_allocation).memory_hierarchy



                        # We now know which memory levels to include, remove everything above it
                        if not is_networks_input:
                            while memories_to_take_for_I[-1] not in memhier.get_operator_top_level(layer.memory_operand_links[I])[0]:

                                removed, _ = memhier.remove_operator_top_level(layer.memory_operand_links[I])
                                accelerator_mem_level_removed.get_core(layer.core_allocation).\
                                    recalculate_memory_hierarchy_information()

                        if not is_networks_output:
                            while memories_to_take_for_O[-1] not in memhier.get_operator_top_level(layer.memory_operand_links['O'])[0]:

                                removed, _ = memhier.remove_operator_top_level(layer.memory_operand_links['O'])
                                accelerator_mem_level_removed.get_core(layer.core_allocation).\
                                    recalculate_memory_hierarchy_information()


                        # Adapt the memory stack for 'layers' like 'add' where two operands come from the same part of the
                        # Memory hierarchy. In this case, I2 is removed completely, and then everywhere I1 is present, I2
                        # is also made present and one of the two layer operands is mapped to I2
                        if [layer.memory_operand_links[I] for I in layer.input_operands] == ['I1', 'I1']:
                            while memhier.remove_operator_top_level('I2')[0]:
                                pass
                            for ml in memhier.nodes:
                                if 'I1' in ml.operands:
                                    ml.operands.append('I2')
                                    ml.mem_level_of_operands['I2'] = ml.mem_level_of_operands['I1']
                                    l = list(ml.port_alloc_raw)
                                    l.append(ml.port_alloc_raw[ml.operands.index('I1')].copy())
                                    ml.port_alloc_raw = tuple(l)
                                    for p in (ml.port_list):
                                        for sold in p.served_op_lv_dir[:]:
                                            if sold[0] == 'I1':
                                                p.add_port_function(tuple(['I2'] + list(sold[1:])))
                            memhier.nb_levels['I2'] = memhier.nb_levels['I1']
                            layer.memory_operand_links[layer.input_operands[1]] = 'I2'
                            accelerator_mem_level_removed.get_core(layer.core_allocation).recalculate_memory_hierarchy_information()


                        # remove too high weight memory levels
                        if not weights_from_top:
                            if layer.constant_operands:
                                while memories_to_take_for_W[-1] not in memhier.get_operator_top_level(layer.memory_operand_links[layer.constant_operands[0]])[0]:
                                    removed, _ = memhier.remove_operator_top_level(layer.memory_operand_links['W'])
                                    accelerator_mem_level_removed.get_core(layer.core_allocation).\
                                        recalculate_memory_hierarchy_information()

                        # process still to process DataMoveActions
                        to_copy_actions = []
                        for tc in to_copy:
                            if tc[0][1] == 'CO':
                                dest_op = layer.memory_operand_links['O']
                            else:
                                dest_op = layer.memory_operand_links[tc[0][1]]

                            source_op = 'O'
                            if tc[0][0] == 'CI':
                                source_op = layer.memory_operand_links[tc[0][1]]  # assume same operand as input (memorylevel is shared anyways)

                            source_memlevel = self.accelerator.get_core(layer.core_allocation).memory_hierarchy.get_memorylevel_with_id(tc[1].get_id())

                            if tc[0][1] == 'CO':
                                dest_memlevel = self.accelerator.get_core(layer.core_allocation)\
                                                    .memory_hierarchy.get_memorylevel_with_id(tc[2].get_id())
                            else:
                                dest_memlevel = memhier.get_memorylevel_with_id(tc[2].get_id())
                            to_copy_actions.append(
                                DataCopyAction(tc[3],
                                               (source_op, source_memlevel.mem_level_of_operands[source_op]),
                                               (dest_op, dest_memlevel.mem_level_of_operands[dest_op]),
                                               self.accelerator.get_core(layer.core_allocation)))
                        dcl = DataCopyLayer(f'datamovement_b4_{layer}', to_copy_actions, self.accelerator, layer.core_allocation)  # will save to return this after the substage ran correctly

                        # done processing these, clear the array
                        to_copy.clear()



                        # create and call substage
                        try: # try except to catch edge cases where reserved space turns out not to be enough, in which case we throw stuff at the wall
                            kwargs_to_pass = self.kwargs.copy()
                            kwargs_to_pass['accelerator'] = accelerator_mem_level_removed
                            kwargs_to_pass['layer'] = layer
                            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs_to_pass)
                            sub_stage_gen = sub_stage.run()
                            cme, extra_info = next(sub_stage_gen)

                            _exhausted = object()
                            if next(sub_stage_gen, _exhausted) is not _exhausted:
                                raise ValueError("Subcallable of df_pipeline yields more than one result, which df_pipeline does not know what to do with. Consider using a reduce.MinimalEnergyStage.")

                        except MemHierarchyTooSmallException as e:  # start throwing
                            if extra_output_memlevels_because_it_did_not_work == extra_input_memlevels_because_it_did_not_work:  # equal state, increase output
                                extra_output_memlevels_because_it_did_not_work += 1
                            elif extra_output_memlevels_because_it_did_not_work > extra_input_memlevels_because_it_did_not_work:  # increasing output did not work, try input
                                extra_input_memlevels_because_it_did_not_work += 1
                                extra_output_memlevels_because_it_did_not_work -= 1
                            else:  # tried input => raise to both
                                extra_output_memlevels_because_it_did_not_work += 1  # now both are increased
                            logger.info(f"MemoryLevels were too optimistic => retrying for this layer with extra levels {extra_output_memlevels_because_it_did_not_work} for O and {extra_input_memlevels_because_it_did_not_work} for I")
                            continue  # do again with extra memorylevel now

                        best_cme_per_layer_tile[f'datamovement_b4_{layer}'] = dcl, [f'datamovement_b4_{layer}']
                        best_cme_per_layer_tile[layer.id] = cme, [layer, extra_info]

                        # Register DataCopyActions for outputs
                        # To deal with the overlap, we first do caches for vertical reuse, as by doing this the overlap
                        # will have passed (or ended up) in the ml for horizontal reuse caching as well,
                        # as the ml for caching for vertical reuse is at least as high as the one for horizontal reuse
                        subtract_vertical_from_horizontal = None # amount to not copy for horizontal, as its already copied for vertical cache
                        if make_vertical_cache_ml is not None and make_vertical_cache_ml != memories_to_take_for_O[-1] \
                            and layer.id in vertical_write_rows:
                            amount_bits_factors = {k: layer.loop_dim_size[k] for k in layer.operand_loop_dim['O']['r']}
                            amount_bits_factors[vertical_write_rows[layer.id][0]] = vertical_write_rows[layer.id][1]
                            amount_bits_factors['precision'] = layer.operand_precision['O_final']
                            # functionally doing this but more generic
                            # amount_bits_factors = (vertical_write_rows[layer.id][1] \
                            #                       , layer.loop_dim_size.get('K', 1) \
                            #                       , layer.loop_dim_size['OX'] \
                            #                       , layer.operand_precision['O_final'])
                            amount_bits = prod(amount_bits_factors.values())
                            subtract_vertical_from_horizontal = vertical_write_rows[layer.id]
                            if amount_bits>0:
                                to_copy.append((('O', 'CO'), memories_to_take_for_O[-1], make_vertical_cache_ml, amount_bits))
                                logger.debug("Added to copy output (V) -> cached_output: " + str(tuple(a if not isinstance(a, MemoryLevel) else a.memory_instance.name for a in to_copy[-1])) + " = " + '*'.join(str(f) for f in amount_bits_factors.values()))

                        # and now for horizontal reuse
                        if make_horizontal_cache_ml is not None and make_horizontal_cache_ml != memories_to_take_for_O[-1] \
                                and layer.id in horizontal_write_columns:
                            amount_bits_factors = {k: layer.loop_dim_size[k] for k in layer.operand_loop_dim['O']['r']}
                            amount_bits_factors[horizontal_write_columns[layer.id][0]] = horizontal_write_columns[layer.id][1]
                            amount_bits_factors['precision'] = layer.operand_precision['O_final']
                            if subtract_vertical_from_horizontal is not None:
                                amount_bits_factors[subtract_vertical_from_horizontal[0]] -= subtract_vertical_from_horizontal[1]

                            amount_bits = prod(amount_bits_factors.values())
                            if amount_bits>0:
                                to_copy.append((('O', 'CO'), memories_to_take_for_O[-1], make_horizontal_cache_ml, amount_bits))
                                logger.debug("Added to copy output (H) -> cached_output: " + str(tuple(a if not isinstance(a, MemoryLevel) else a.memory_instance.name for a in to_copy[-1])) + " = " + '*'.join(str(f) for f in amount_bits_factors.values()))
                        break
                run_for_tilesize_cache[args] = best_cme_per_layer_tile
                return best_cme_per_layer_tile


            # NOTODO: basically in this whole code, make sure that memorylevels used to pass feature map tiles between \
            # layers are memories for both I and O, otherwise none of this makes any sense
            # NOPE => now code checks to whether input memorylevel is output memorylevel of predecessor anyways,
            # so this can just deal with that

            determine_horizontal_caching_ml_for_tilesize_cache = {}
            def determine_caching_for_tilesize(tile_x:int,
                                               tile_y:int,
                                               first_x:bool,
                                               first_y:bool,
                                               horizontal_caching:bool,  # unused, oops
                                               vertical_caching:bool  # unused oops
                                               )\
                                                    -> Tuple[MemoryLevel,
                                                             MemoryLevel,
                                                             Dict[LayerNode, int],
                                                             Dict[LayerNode, int],
                                                             Dict[LayerNode, int],
                                                             Dict[LayerNode, int],
                                                             MemoryLevel]:
                # cache results for runtime optimization
                tilesize_to_determine_cachesize = (tile_x, tile_y, first_x, first_y, horizontal_caching, vertical_caching)
                if tilesize_to_determine_cachesize in determine_horizontal_caching_ml_for_tilesize_cache:
                    return determine_horizontal_caching_ml_for_tilesize_cache[tilesize_to_determine_cachesize]


                workload_copy = pickle_deepcopy(usefull_workload)

                usefull_nodes, \
                to_cache_horizontally_out, \
                to_cache_horizontally_in, \
                to_cache_vertically_out, \
                to_cache_vertically_in = backpropagate_tilesize(
                                               workload_copy,
                                               on_i, # still the output node in question
                                               tile_x, tile_y, #tilesize
                                               not first_x,  # use horizontal caching
                                               True,    # make horizontal caching
                                               not first_y,    # use vertical caching
                                               True)    # make vertical caching

                largest_input_plus_output_size = get_largest_alive_size(workload_copy)


                to_cache_horizontally_bits_in = {}
                for n, I in to_cache_horizontally_in:
                    dim, size = to_cache_horizontally_in[(n,I)]
                    if dim in n.operand_loop_dim[I]['r']:
                        assert size == 0 # should be the case for r loops
                        elems = 0
                        # elems = n.operand_size_elem[I] // n.operand_loop_dim[I]['r'][dim] * size
                    else:
                        elems = n.operand_size_elem[I] // n.calc_tensor_dims(I, n.loop_dim_size)[dim] * size
                    to_cache_horizontally_bits_in[(n, I)] = elems * n.operand_precision[I]

                to_cache_horizontally_bits_out = {}
                for n in to_cache_horizontally_out:
                    dim, size = to_cache_horizontally_out[n]
                    elems = n.operand_size_elem['O'] // n.loop_dim_size[dim] * size
                    to_cache_horizontally_bits_out[n] = elems * n.operand_precision['O_final']

                to_cache_vertically_bits_in = {}
                for n, I in to_cache_vertically_in:
                    dim, size = to_cache_vertically_in[(n, I)]
                    if dim in n.operand_loop_dim[I]['r']:
                        assert size == 0
                        elems = 0
                    else:
                        h_dim = to_cache_horizontally_in[n, I][0]
                        h_dim_total_size = n.calc_tensor_dims(I, self.workload.get_node_with_id(n.id).loop_dim_size)[h_dim]
                        h_dim_tile_size = n.calc_tensor_dims(I, n.loop_dim_size)[h_dim]
                        v_dim_tile_size = n.calc_tensor_dims(I, n.loop_dim_size)[dim]
                        elems = n.operand_size_elem[I] // h_dim_tile_size * h_dim_total_size // v_dim_tile_size * size
                    to_cache_vertically_bits_in[(n, I)] = elems * n.operand_precision[I]

                to_cache_overlapped_bits_in = {}
                for n,I in to_cache_vertically_in:
                    v_dim, v_size = to_cache_vertically_in[(n,I)]
                    h_dim, h_size = to_cache_horizontally_in[(n,I)]
                    if v_dim in n.loop_dim_size or h_dim in n.loop_dim_size:
                        assert to_cache_vertically_bits_in[n,I] * to_cache_horizontally_bits_in[n,I] == 0
                        to_cache_overlapped_bits_in[(n, I)] = 0
                    else:
                        h_tilesize = n.calc_tensor_dims(I, n.loop_dim_size)[h_dim]
                        v_tilesize = n.calc_tensor_dims(I, n.loop_dim_size)[v_dim]
                        elems = n.operand_size_elem[I] // h_tilesize * h_size // v_tilesize * v_size
                        to_cache_overlapped_bits_in[(n, I)] = elems * n.operand_precision[I]

                # if first tile in a row, we don't have all horizontal caches at once (except at the end)
                # So we figger out at every point in the network what is the number of features alive + all caches
                # created up until that point
                peak_f_mem_size = get_largest_alive_size(workload_copy)
                i_mem_op = next(n.memory_operand_links['I'] for n in usefull_nodes if 'I' in n.memory_operand_links)

                o_mem_op = next(n.memory_operand_links['O'] for n in usefull_nodes if 'O' in n.memory_operand_links)
                # get lowest memory that fits largest fm, to reserve space that can not be used for weights
                # and is also not unrolled
                memories_to_reserve_for_IO = []
                for i, ml in enumerate(mh.get_memory_levels(o_mem_op)):
                    memories_to_reserve_for_IO.append(ml)
                    if i_mem_op in ml.operands and ml.memory_instance.size >= largest_input_plus_output_size and ml.unroll_count == 1:
                        break
                io_mem_level = memories_to_reserve_for_IO[-1]


                if horizontal_caching:
                    if first_x:
                        peak_f_mem_size = get_largest_alive_size(workload_copy, to_cache_horizontally_bits_out)
                    else:
                        # Otherwise all caches are present
                        peak_f_mem_size = largest_input_plus_output_size + sum(to_cache_horizontally_bits_in.values())
                    peak_f_mem_size_h = peak_f_mem_size
                    horizontal_caching_ml : MemoryLevel
                    first_layer = next(usefull_workload.topological_sort()) # need some layer to get some info
                    memory_hierarchy = self.accelerator.get_core(first_layer.core_allocation).memory_hierarchy

                    # Find lowest memory level that
                    #  - fits horizontal at the same time as largest amount of alive features (see if statement above=> peak_mem size)
                    #  - is shared between inputs and outputs
                    #  - is not unrolled
                    for i, ml in enumerate(memory_hierarchy.get_memory_levels(first_layer.memory_operand_links['O'])):
                        if ml.memory_instance.size >= peak_f_mem_size and i_mem_op in ml.operands and ml.unroll_count == 1:
                            horizontal_caching_ml = ml
                            logger.info(f"Caching for horizontal reuse in memorylevel {ml} for tile {tile_x}x{tile_y}" +
                                        (" (firstx)" if first_x else "") + (" (firsty)" if first_y else ""))
                            break

                    if vertical_caching:
                        # Find lowest memory level that
                        #  - fits horizontal at the same time as largest amount of alive features (see if statement above=> peak_mem size)
                        #    and now also caches for vertical reuse
                        #  - is shared between inputs and outputs
                        #  - is not unrolled
                        vertical_caching_ml : MemoryLevel
                        peak_f_mem_size = sum(to_cache_vertically_bits_in.values()) + peak_f_mem_size
                        for i, ml in enumerate(memory_hierarchy.get_memory_levels(o_mem_op)):
                            size_to_fit = peak_f_mem_size
                            if ml == horizontal_caching_ml:
                                size_to_fit -= sum(to_cache_overlapped_bits_in.values())  # only subtracting this if they are the same memorylevel, Otherwise we assume stored double
                            if ml.memory_instance.size >= size_to_fit and i_mem_op in ml.operands and ml.unroll_count ==  1:
                                vertical_caching_ml = ml
                                logger.info(f"Caching for vertical reuse in memorylevel {ml}")
                                break
                    else:
                        vertical_caching_ml = None
                else:
                    horizontal_caching_ml = None
                    vertical_caching_ml = None

                # weights have lowest priority, so lowest memorylevel that fits
                # - all weights
                # - if shared with I/O also all features and all caches, if included for
                weights_total = sum(sum(layer.operand_size_bit[W] for W in layer.constant_operands) for layer in usefull_nodes)
                w_mem_op = next(n.memory_operand_links['W'] for n in usefull_nodes if 'W' in n.memory_operand_links)
                # lowest memory that also fits the weights
                memories_to_take_for_W = []
                for i, ml in enumerate(mh.get_memory_levels(w_mem_op)):
                    memories_to_take_for_W.append(ml)
                    size_to_fit = weights_total
                    if ml in memories_to_reserve_for_IO:
                        size_to_fit = weights_total + largest_input_plus_output_size
                    if ml == horizontal_caching_ml:
                        size_to_fit = weights_total + peak_f_mem_size_h
                    if ml == vertical_caching_ml:
                        size_to_fit = weights_total + peak_f_mem_size
                    if ml == horizontal_caching_ml and ml == vertical_caching_ml:
                        size_to_fit = weights_total + peak_f_mem_size - sum(to_cache_overlapped_bits_in.values())  # only subtracting this if they are the same memorylevel, Otherwise we assume stored double
                    if ml.memory_instance.size >= size_to_fit:
                        break
                weight_ml = memories_to_take_for_W[-1]

                # how many new columns to put in cache for this cache
                # This is just to_cache_horizontally, except for very small tilesizes where part of this is already
                # in cache from previous tile (So some inputs serve more than two tiles)
                # (same for rows) => need not to be written every time again
                if first_x:
                    columns_to_write = to_cache_horizontally_out
                else:
                    columns_to_write = {l: (to_cache_horizontally_out[l][0],
                                             min(
                                                 to_cache_horizontally_out[l][1],
                                                  l.loop_dim_size[to_cache_horizontally_out[l][0]]
                                             )) for l in to_cache_horizontally_out}
                if first_y:
                    rows_to_write = to_cache_vertically_out
                else:
                    rows_to_write = {l : (to_cache_vertically_out[l][0],
                                          min(to_cache_vertically_out[l][1],
                                              l.loop_dim_size[to_cache_vertically_out[l][0]]))
                                         for l in to_cache_vertically_out}

                columns_to_read = defaultdict(return_0) if first_x else to_cache_horizontally_in
                rows_to_read = defaultdict(return_0) if first_y else to_cache_vertically_in

                #dataformat change, no new intelligence here
                columns_to_write = {l.id: columns_to_write[l] for l in columns_to_write}
                columns_to_read = {(l[0].id, l[1]): columns_to_read[l] for l in columns_to_read}
                rows_to_write = {l.id: rows_to_write[l] for l in rows_to_write}
                rows_to_read = {(l[0].id, l[1]): tuple(list(rows_to_read[l]) + [to_cache_horizontally_in[l][0]]) for l in rows_to_read}

                # caching of this functions output for runtime optimization
                determine_horizontal_caching_ml_for_tilesize_cache[tilesize_to_determine_cachesize] = \
                    (horizontal_caching_ml, vertical_caching_ml, columns_to_write, columns_to_read, rows_to_write, rows_to_read, weight_ml)

                return determine_horizontal_caching_ml_for_tilesize_cache[tilesize_to_determine_cachesize]


            leftover_tilesize_x = total_OX % self.tilesize_x
            if leftover_tilesize_x == 0:
                leftover_tilesize_x = self.tilesize_x
            leftover_tilesize_y = total_OY % self.tilesize_y
            if leftover_tilesize_y == 0:
                leftover_tilesize_y = self.tilesize_y



            ############################################
            # A first iteration through all tiles to
            # deal with all stuff that must be consistent
            # across tiles, such as
            # - vertical caching ml across rows  (level = highest overall)
            # - horizontal caching ml across neighbouring columns (level = max of 2 horizontal neighbours)
            # - weight ml (highest overall)
            ############################################


            mh : MemoryHierarchy = self.accelerator.get_core(next(nx.topological_sort(self.workload)).core_allocation).memory_hierarchy
            mh_sorted = list(nx.topological_sort(mh))


            highest_w_ml = None
            highest_vertical_cache_ml = None
            horizontal_caching_info = {}  # key = x0,y0; values = tuples of make_ml, write_columns, read_columns
            vertical_caching_info = {}  # similar
            y0 = 0
            while y0 < total_OY:
                this_tile_y = self.tilesize_y if y0 != 0 else leftover_tilesize_y
                assert y0 + this_tile_y <= total_OY
                x0 = 0
                while x0 < total_OX:
                    this_tile_x = self.tilesize_x if x0 != 0 else leftover_tilesize_x
                    assert x0 + this_tile_x <= total_OX

                    horizontal_caching_ml,\
                    vertical_cache_ml,\
                    horizontal_cache_write_columns,\
                    horizontal_cache_read_columns,\
                    vertical_write_rows,\
                    vertical_read_rows,\
                    w_ml                          = determine_caching_for_tilesize(this_tile_x, this_tile_y,
                                                                                   x0 == 0, y0 == 0,
                                                                                   self.horizontal_caching,
                                                                                   self.vertical_caching)
                    if self.horizontal_caching:
                        if x0 + self.tilesize_x >= total_OX:
                            horizontal_caching_info[(x0, y0)] = (None, defaultdict(return_emtpystr_0_list), horizontal_cache_read_columns)
                        elif x0 == 0:
                            horizontal_caching_info[(x0, y0)] = horizontal_caching_ml, \
                                                              horizontal_cache_write_columns,\
                                                              horizontal_cache_read_columns
                        else:
                            horizontal_caching_info[(x0, y0)] = horizontal_caching_ml, \
                                                              horizontal_cache_write_columns,\
                                                              horizontal_cache_read_columns
                            # If caching ml of this tile is higher than for previous tile, make sure previous tile
                            # makes to this memory level
                            previous = horizontal_caching_info[(x0 - self.tilesize_x, y0)][0]
                            if previous != horizontal_caching_ml:  # (line not required algorithmically, but makes it faster)
                                previous = previous.get_id()
                                this = horizontal_caching_ml.get_id()
                                for ml in mh_sorted:
                                    if ml == previous:  # previous hit first so new one is higher
                                        highest_vertical_cache_ml = this
                                        horizontal_caching_ml[(x0 - self.tilesize_x, y0)][0] = horizontal_caching_ml
                                        break
                    else:
                        horizontal_caching_info[(x0, y0)] = (None, defaultdict(return_emtpystr_0_list), defaultdict(return_0))

                    if self.vertical_caching and self.horizontal_caching:
                        if highest_vertical_cache_ml is None:
                            highest_vertical_cache_ml = vertical_cache_ml
                        else:
                            # get the highest memory of the two
                            if highest_vertical_cache_ml != vertical_cache_ml:  # (line not required algorithmically, but makes it faster)
                                h = mh.get_memorylevel_with_id(highest_vertical_cache_ml.get_id())
                                new = mh.get_memorylevel_with_id(vertical_cache_ml.get_id())
                                for ml in mh_sorted:
                                    if ml == h:  # h hit first, so new one is higher
                                        highest_vertical_cache_ml = new
                                        for k in vertical_caching_info:
                                            if vertical_caching_info[k][0] is not None:
                                                vertical_caching_info[k] = (highest_vertical_cache_ml, ) + vertical_caching_info[k][1:]
                                        break

                        if y0 + self.tilesize_y >= total_OY:
                            vertical_caching_info[(x0,y0)] = (None, defaultdict(return_emtpystr_0_list), vertical_read_rows)
                        else:
                            vertical_caching_info[(x0, y0)] = (highest_vertical_cache_ml, vertical_write_rows, vertical_read_rows)
                    else:
                        vertical_caching_info[(x0, y0)] = (None, defaultdict(return_emtpystr_0_list), defaultdict(return_0))

                    if highest_w_ml is None:
                        highest_w_ml = w_ml
                    else:
                        if highest_w_ml != w_ml:  # (line not required algorithmically, but makes it faster)
                            h = mh.get_memorylevel_with_id(highest_w_ml.get_id())
                            new = mh.get_memorylevel_with_id(w_ml.get_id())
                            for ml in mh_sorted:
                                if ml == h:  # h hit first, so new one is higher
                                    highest_w_ml = new
                                    break
                                if ml == new:
                                    highest_w_ml = h
                                    break

                    x0 += this_tile_x
                y0 += this_tile_y


            ##################################
            # Actually do all the tiles
            ##################################
            weights_from_top = True
            y0 = 0

            previous_y0 = None
            while y0 < total_OY:
                this_tile_y = self.tilesize_y if y0 != 0 else leftover_tilesize_y
                assert y0 + this_tile_y <= total_OY
                previous_x0 = None
                x0 = 0
                while x0 < total_OX:
                    this_tile_x = self.tilesize_x if x0 != 0 else leftover_tilesize_x
                    assert x0 + this_tile_x <= total_OX

                    make_horizontal_cache_ml, horizontal_cache_write_columns, horizontal_cache_read_columns = \
                        horizontal_caching_info[(x0, y0)]
                    make_vertical_cache_ml, vertical_write_rows, vertical_read_rows = \
                        vertical_caching_info[(x0, y0)]
                    if x0:
                        use_horizontal_cache_ml = horizontal_caching_info[(previous_x0, y0)][0]
                    else:
                        use_horizontal_cache_ml = None
                    if y0:
                        use_vertical_cache_ml = vertical_caching_info[(x0, previous_y0)][0]
                    else:
                        use_vertical_cache_ml = None

                    assert( self.horizontal_caching or use_horizontal_cache_ml is None)
                    assert( (self.vertical_caching and self.horizontal_caching) or use_vertical_cache_ml is None)
                    assert( x0 + self.tilesize_x < total_OX or make_horizontal_cache_ml is None)
                    assert( y0 + self.tilesize_y < total_OY or make_vertical_cache_ml is None)


                    #Actually run a tile here:
                    energy_of_this_tile = \
                                    run_for_tilesize(this_tile_x,
                                                     this_tile_y,
                                                     weights_from_top,
                                                     make_horizontal_cache_ml,
                                                     use_horizontal_cache_ml,
                                                     horizontal_cache_write_columns,
                                                     horizontal_cache_read_columns,
                                                     make_vertical_cache_ml,  # make vertical cache ml
                                                     use_vertical_cache_ml,  # make vertical cache ml
                                                     vertical_write_rows,  # write rows of cache if not last line of tiles
                                                     vertical_read_rows,
                                                     highest_w_ml,
                                                     x0, y0)
                    # see if result was already seen
                    # If so, just increase its multiplier (runtime optimization)
                    for k, v in energy_of_this_tile.items():
                        try:
                            # v is in there, adjust multiplier. Otherwise exception is raised
                            index = [i[0] for i in cost_model_evaluations_per_layer[k]].index(v)
                            #Note: CostModelEvaluation has no __eq__ implementation and therefore resorts to comparing
                            #based on id, which is already due to run_for_tilesize actually the exact same instance
                            #due to that it caches its outputs
                            cost_model_evaluations_per_layer[k][index][1] += 1
                        except ValueError:
                            # v was not in there, add it with multiplier 1
                            cost_model_evaluations_per_layer[k].append([v, 1])

                    # after first tile, weights no longer need to come from top memory level
                    weights_from_top = False

                    # next horizontal cache ml to read from (use) (in the next tile), is the one we wrote to (make) now
                    if self.horizontal_caching:
                        use_horizontal_cache_ml = make_horizontal_cache_ml
                    previous_x0 = x0
                    x0 += this_tile_x
                previous_y0=y0
                y0 += this_tile_y
            assert y0 == total_OY
            assert x0 == total_OX


        # Done, accumulate and yield results
        for k in cost_model_evaluations_per_layer:
            total = None
            for cme, mul in cost_model_evaluations_per_layer[k]:
                cme, extra_info = cme
                if total is None:
                    total = cme * mul
                else:
                    total += cme * mul
            total.accelerator = self.accelerator
            yield total, cost_model_evaluations_per_layer[k]

        # Print some stuff to logger
        longest_layer_name = max(len(str(l)) for l in cost_model_evaluations_per_layer)
        for layer, cm_list in cost_model_evaluations_per_layer.items():
            logger.info(("Layer {layer:>"+str(longest_layer_name)+"s} best energy found = {best_energy:20,d}  with latency {latency:30,d}")
                        .format(layer=str(layer),
                                best_energy=int(sum(cm[0].energy_total*mul for cm, mul in cm_list)),
                                latency=int(sum(cm[0].latency_total1 * mul for cm, mul in cm_list))))





