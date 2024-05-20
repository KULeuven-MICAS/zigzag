from typing import Any, Generator
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import LayerDim, LayerOperand, MemoryOperand

from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.stages.Stage import Stage, StageCallable

import networkx as nx
from zigzag.workload.Workload import Workload
from zigzag.workload.DummyNode import DummyNode

import logging

from zigzag.workload.layer_node import LayerNode
from zigzag.workload.LayerNodeABC import LayerNodeABC

logger = logging.getLogger(__name__)


class SearchUnusedMemoryStage(Stage):
    def __init__(
        self, list_of_callables: list[StageCallable], *, accelerator: Accelerator, workload: Workload, **kwargs: Any
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        # TODO fix this!
        core_id = accelerator.cores[0].id  # correct only for single-core hardware
        core = accelerator.get_core(core_id)
        # record of all memory levels
        self.core_mem_level_list = core.memory_hierarchy.mem_level_list
        # record of top mem level per layer: dict[layer_idx, mem_level]
        self.mem_update_list: dict[str, list[dict[MemoryOperand, int]]] = {}
        # record of input, output size per layer (unit: bit): dict[layer_idx: [operand: size]]
        self.each_layer_IO_data_size: dict[str, list[dict[MemoryOperand, int]]] = {}
        # record of the index of the highest weight memory level
        self.mem_update_weight: int = len(self.core_mem_level_list) - 1
        # record of the weight size of entire workload (unit: bit)
        self.weight_size_entire_workload: int = 0
        # record of executable layer name and index: dict[LayerNode, layer_idx]
        self.layer_list: dict[LayerNodeABC, int] = {}

        # derive input, output size per layer and weight size of entire workload
        for idx, layer in enumerate(workload.topological_sort()):
            if isinstance(layer, DummyNode):
                continue

            # input, weight operand name in hardware
            (
                act_operand_in_layer,
                weight_operand_in_layer,
                act_operand_in_hardware,
                weight_operand_in_hardware
            ) = SearchUnusedMemoryStage.get_act_weight_operand_names(layer=layer)
            output_operand_in_layer = layer.output_operand
            output_operand_in_hardware = layer.memory_operand_links[output_operand_in_layer]

            # update info for special layers: "Adder", which does not have weight
            if len(layer.constant_operands) == 0:
                # special case when workload input is from .py rather than .onnx:
                # the constant operands list is empty for "Adder" layers
                input_data_size = 0
                for operand in layer.input_operands:
                    input_data_size += layer.operand_size_bit[operand]
                self.mem_update_list[str(idx)] = [
                    {operand: -1}
                    for operand in core.mem_hierarchy_dict.keys()
                    if operand in [output_operand_in_hardware, act_operand_in_hardware]
                ]
                self.each_layer_IO_data_size[str(idx)] = [
                    {
                        output_operand_in_hardware: layer.operand_size_bit[output_operand_in_layer],
                        act_operand_in_hardware: input_data_size,
                    }
                ]
                self.layer_list[layer] = idx
                continue

            # update info for regular layers
            self.mem_update_list[str(idx)] = [
                {act_operand_in_hardware: -1},
                {output_operand_in_hardware: -1}
            ]
            self.each_layer_IO_data_size[str(idx)] = [
                {
                    act_operand_in_hardware: layer.operand_size_bit[act_operand_in_layer],
                    output_operand_in_hardware: layer.operand_size_bit[output_operand_in_layer]
                }
            ]
            self.weight_size_entire_workload += layer.operand_size_bit[weight_operand_in_layer]
            self.layer_list[layer] = idx

    def run(self, workload_data_always_from_top_mem: bool = False):
        # @param workload_data_always_from_top_mem: input of 1st layer, output of last layer must in the top mem level

        # update allowed the lowest mem level per operand per layer
        self.update_top_mem_level()

        if workload_data_always_from_top_mem:
            self.update_mem_level_for_loading_data()

        sub_stage: Stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            accelerator=self.accelerator,
            workload=self.workload,
            mem_update_list=self.mem_update_list,
            mem_update_weight=self.mem_update_weight,
            layer_list=self.layer_list,
            **self.kwargs,
        )
        for cme, (layer, extra_info) in sub_stage.run():
            yield cme, (layer, extra_info)

    def update_top_mem_level(self):
        """
        Update mem_update_list and mem_update_weight according to the algorithm description at the file beginning.
        # TODO cleanup, split into functions
        """
        # remove dummy (non-conv) layers (ReLU, Pooling..) in the layer graph
        self.remove_dummy_nodes_in_workload()

        # calculate the allowed lowest mem level per operand per layer
        for id, layer in enumerate(self.workload.topological_sort()):
            if isinstance(layer, DummyNode):
                continue

            # original layer index before removing dummy nodes
            curr_id = self.layer_list[layer]
            # input, weight operand name in hardware
            act_operand, const_operand = self.get_act_weight_operand_names(layer=layer)
            # output operand name in hardware
            output_operand: MemoryOperand = layer.memory_operand_links[layer.output_operand]

            is_branch_starting_node = True if self.workload.out_degree(layer) > 1 else False
            is_branch_final_node = (
                True
                if self.workload.out_degree(layer) == 1
                and self.workload.in_degree(list(self.workload.successors(layer))[0]) > 1
                else False
            )



            if id != 0:  # not the first layer
                # Assign mem_udpate_list[layer]["I"] = mem_udpate_list[previous_layer]["O"]
                prev_layer: LayerNode = list(self.workload.predecessors(layer))[0]  # previous layer node (object)
                prev_layer_id = self.layer_list[prev_layer]  # previous layer id
                # output representation in memory of previous layer
                prev_layer_output_layer_op = prev_layer.output_operand
                prev_layer_output_mem_op = prev_layer.memory_operand_links.layer_to_mem_op(prev_layer_output_layer_op)
                # starting node of branches
                for ele in self.mem_update_list[f"{prev_layer_id}"]:
                    if prev_layer_output_mem_op in ele:
                        prev_layer_output_level = ele[prev_layer_output_mem_op]
                        self.update_IO_mem_level(curr_id, act_operand, prev_layer_output_level)

            # # branch starting node or branch final node or permited dummy nodes (e.g. Adder layer)
            # # Update input, weight, output mem level for branch starting node and branch final node
            # # Find the top mem level for input if it is the first layer, update mem_udpate_list of current layer
            if branch_starting_node or branch_final_node:
                if id == 0:  # the first layer
                    for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                        served_operands = list(
                            mem.mem_level_of_operands.keys()
                        )  # Check the served operand of current mem
                        if act_operand in served_operands:
                            # update the input mem level of current layer if it is the first layer
                            self.update_IO_mem_level(curr_id, act_operand, curr_mem_level)
                            break
                # Find the top mem level for output, update mem_update_list of current layer
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    if output_operand in served_operands:
                        # update the output mem level of current layer
                        self.update_IO_mem_level(curr_id, output_operand, curr_mem_level)
                        break
                # Find the top mem level for weight, update mem_update_weight of current layer to the top weight mem level if mem_update_weight is bigger
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    if const_operand in served_operands:  # identify the top weight mem level
                        # We need to check if the current mem serve all oa dims, otherwise we will not decrease
                        # the mem_update_weight.
                        # The reason is if the current mem not serve all oa dims, the mapping will impact the memory
                        # utilization, so solely comparing with total memory size will be incorrect.
                        mem_serve_all_oa_dims = self.check_if_mem_serve_all_oa_dims(mem, self.accelerator)
                        # mem_update_weight is bigger than the top weight mem level
                        if (curr_mem_level < self.mem_update_weight) and mem_serve_all_oa_dims:
                            self.mem_update_weight = curr_mem_level
                        break
            # # node (layer) that is not a branch starting node or a branch final node
            else:
                # # Iterate the memory level and update input, weight, output mem level
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    # Update input, weight, output mem level
                    avail_mem_size = mem.memory_instance.size * mem.unroll_count  # available hardware mem size

                    try:
                        # we need to grab the next layer name, which is a non-Adder layer for sure
                        # if next layer is an Adder layer, then branch_final_node=True for the current layer,
                        # so, the simulation will not reach to this "else" branch.
                        next_layer: LayerNode = list(self.workload.successors(layer))[0]
                        # next, we find out the layer representation for the act operand of the next layer
                        const_layer_operand_of_next_layer = next_layer.constant_operands[0]
                        act_layer_operand_of_next_layer = [
                            operand
                            for operand in next_layer.input_operands
                            if operand != const_layer_operand_of_next_layer
                        ][0]
                        # then, we will fetch the mem representation for the act operand of the next layer
                        act_mem_operand_of_next_layer = next_layer.memory_operand_links[act_layer_operand_of_next_layer]
                        # check if the current mem level serve the act operand in the next layer
                        mem_serve_act_in_next_layer = (
                            True if (act_mem_operand_of_next_layer in served_operands) else False
                        )
                    except IndexError:  # there is no next layer, which means the current layer is the last layer
                        # As for the last layer, we will instead check
                        # if the mem serves act operand of the current layer.
                        mem_serve_act_in_next_layer = True if (act_operand in served_operands) else False

                    # ["I", "O"] both in mem.served_operands
                    mem_serve_io_both = (
                        True if mem_serve_act_in_next_layer and (output_operand in served_operands) else False
                    )
                    # mem.served_operands = ["W"]
                    mem_serve_weight = True if (const_operand in served_operands) else False
                    # we need to change served_operands if the current layer is an Adder layer,
                    # for the ease of calculation of required input data size.
                    # Since an Adder layer has two inputs,
                    # but in each_layer_IO_data_size, data size of two inputs are put under one key,
                    # so we have to update served_operands to ensure the key used in each_layer_IO_data_size is in it.

                    # the layer type is an Adder layer, which has multiple input operands
                    if len(layer.constant_operands) == 0 and mem_serve_io_both:
                        served_operands = [
                            output_operand,
                            layer.memory_operand_links[layer.input_operands[0]],
                        ]

                    if mem_serve_io_both or mem_serve_weight:
                        required_IO_data_size = sum(
                            [
                                self.each_layer_IO_data_size[f"{curr_id}"][0][operand]
                                for operand in served_operands
                                if operand != const_operand
                            ]
                        )
                        # required size to put data in current mem level
                        required_weight_size = (
                            self.weight_size_entire_workload if const_operand in served_operands else 0
                        )
                        required_total_size = required_IO_data_size + required_weight_size

                        # sum(layer[operand_size] for operand in mem.operands) <= mem.size
                        if required_total_size <= avail_mem_size:
                            if mem_serve_io_both:
                                if id == 0:
                                    # update input mem level
                                    self.update_IO_mem_level(curr_id, act_operand, curr_mem_level)
                                # update output mem level
                                self.update_IO_mem_level(curr_id, output_operand, curr_mem_level)
                            # For weight, we need to check if the current mem serve all oa dims, otherwise we will not
                            # decrease the mem_update_weight.
                            # The reason is if the current mem not serve all oa dims, the mapping will impact the memory
                            # utilization, so solely comparing with total memory size will be incorrect.
                            mem_serve_all_oa_dims = self.check_if_mem_serve_all_oa_dims(mem, self.accelerator)
                            # update weight mem level
                            if (curr_mem_level < self.mem_update_weight) and mem_serve_all_oa_dims and mem_serve_weight:
                                self.mem_update_weight = curr_mem_level
        # [OPTIONAL CHECK] assert check if there is -1 value in mem_update_list
        # [NOTE] Until here, if there is still -1 value in mem_update_list, it means the size of top mem level for IO is not big enough.
        for layer_ele in self.mem_update_list.values():
            for operand_dict in layer_ele:
                assert (
                    list(operand_dict.values())[0] >= 0
                ), "SearchUnusedMemoryStage finishes abnormally, there are still layers with top mem levels not figured out."

    @staticmethod
    def get_act_weight_operand_names(layer: LayerNode
    ) -> tuple[LayerOperand, LayerOperand or None, MemoryOperand, MemoryOperand or None]:
        # the function is also called within imc cost model
        if len(layer.constant_operands) == 1:
            # regular layers
            weight_operand_in_layer: LayerOperand = layer.constant_operands[0]
            weight_operand_in_hardware: MemoryOperand = layer.memory_operand_links[weight_operand_in_layer]
            act_operand_in_layer: LayerOperand = [
                operand for operand in layer.input_operands if operand != weight_operand_in_layer][0]
            act_operand_in_hardware: MemoryOperand = layer.memory_operand_links[act_operand_in_layer]
        elif len(layer.constant_operands) == 0:
            # Adder layers
            weight_operand_in_layer: None = None
            weight_operand_in_hardware: None = None
            act_operand_in_layer: LayerOperand = layer.input_operands[0]
            act_operand_in_hardware: MemoryOperand = layer.memory_operand_links[act_operand_in_layer]
        else:
            # len(layer.constant_operands) == 2, both input, activation exist in layer.constant_operands
            pr_loop_keys = tuple(layer.pr_loop.keys())
            related_loop_dict: dict[LayerOperand, list[LayerDim]] = {
                layer_op: layer.equation.get_r_layer_dims(layer_op)
                for layer_op in layer.equation.get_contained_operands()
            }
            for operand_in_layer, related_loop in related_loop_dict.items():
                if pr_loop_keys[0] in related_loop:
                    act_operand_in_layer: LayerOperand = operand_in_layer
                    break
            act_operand_in_hardware: MemoryOperand = layer.memory_operand_links[act_operand_in_layer]
            weight_operand_in_layer: LayerOperand = [x for x in layer.constant_operands if x != act_operand_in_layer][0]
            weight_operand_in_hardware: MemoryOperand = layer.memory_operand_links.layer_to_mem_op(weight_operand_in_layer)
        return act_operand_in_layer, weight_operand_in_layer, act_operand_in_hardware, weight_operand_in_hardware

    def check_if_mem_serve_all_oa_dims(self, mem: MemoryLevel, accelerator: Accelerator):
        # check if mem serve all hardare dimensions
        core = accelerator.cores[0]
        operational_array = core.operational_array
        oa_dim_nb = len(operational_array.oa_dim_sizes)
        mem_served_oa_dim_nb = mem.served_dimensions.nb_dims
        return mem_served_oa_dim_nb == oa_dim_nb

    def update_mem_level_for_loading_data(self):
        """!
        [OPTIONAL FUNCTION] This is an optional function.
        Depending on your requirement, sometimes data loading from the top mem level and offloading to the top mem level is a must.
        If that is the your case, add this function to self.run().
        Otherwise, if the input is generated on-chip at the lowest possible input mem level and the output is stored on-chip at the lowest possible output mem level, remove this function from self.run().
        [FUNCTION OBJECT]
        Update mem_update_list of first and last layer, so that the input data of first layer still is loaded from top input mem level and the output of last layer still is offloaded to top output mem level
        """
        self.remove_dummy_nodes_in_workload()  # remove dummy nodes for the ease of telling the branch starting or final nodes

        # Update mem_update_list and mem_update_weight
        for id, layer in enumerate(self.workload.topological_sort()):
            if isinstance(layer, DummyNode):
                continue

            # act representation
            act_operand = layer.memory_operand_links[
                [operand for operand in layer.input_operands if operand not in layer.constant_operands][0]
            ]
            output_layer_op = layer.output_operand  # output representation
            output_mem_op = layer.memory_operand_links.layer_to_mem_op(output_layer_op)
            curr_id = self.layer_list[layer]  # current layer id (key) in mem_udpate_list
            if id == 0:  # the first layer: update activation mem level to the top possible mem level
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    if act_operand in served_operands:
                        # update the input mem level of current layer if it is the first layer
                        self.update_IO_mem_level(curr_id, act_operand, curr_mem_level)
                        break
            if id == len(self.layer_list) - 1:  # the last layer: update output mem level to the top possible mem level
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    if output_mem_op in served_operands:
                        # update the output mem level of current layer if it is the last layer
                        self.update_IO_mem_level(curr_id, output_mem_op, curr_mem_level)
                        break

    def remove_dummy_nodes_in_workload(self):
        """! Remove dummy nodes (layers) in the graph (assume there is no branch from a non-dummy
        node to dummy node) Redirect the outgoing edges of dummy nodes to non-dummy nodes
        Method:
        for each dummy node, add edges between its predecessor nodes and successor nodes;
        then remove the dummy node.
        """
        dummy_nodes = [node for node in self.workload.nodes() if isinstance(node, DummyNode)]
        for dummy_node in dummy_nodes:
            for successor_node in list(self.workload.successors(dummy_node)):
                for predecessor_node in list(self.workload.predecessors(dummy_node)):
                    self.workload.add_edge(predecessor_node, successor_node)
        # visualize the resulted graph
        # import matplotlib.pyplot as plt
        # pos = nx.spring_layout(self.workload)
        # nx.draw(self.workload, pos, with_labels=True, node_color="lightblue", font_weight="bold")
        # plt.show()
        self.workload.remove_nodes_from(dummy_nodes)

    def update_IO_mem_level(self, layer_id: int, operand: MemoryOperand, target_level: int):
        """! Update self.mem_update_list as:
        self.mem_update_list[layer_id][operand_index][operand] = target_level
        """
        for pos, ele in enumerate(self.mem_update_list[f"{layer_id}"]):
            if list(ele.keys())[0] == f"{operand}":
                self.mem_update_list[f"{layer_id}"][pos][operand] = target_level
