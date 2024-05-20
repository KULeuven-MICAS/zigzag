from typing import Any
from zigzag.datatypes import LayerDim, LayerOperand, MemoryOperand

from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.stages.Stage import Stage, StageCallable

import networkx as nx  # kept for debugging
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
        self.mem_update_list: dict[str, dict[MemoryOperand, int]] = {}
        # record of input, output size per layer (unit: bit): dict[layer_idx: [operand: size]]
        self.each_layer_IO_data_size: dict[str, list[dict[MemoryOperand, int]]] = {}
        # record of the weight size of entire workload (unit: bit)
        self.weight_size_entire_workload: int = 0
        # record of executable layer name and index: dict[LayerNode, layer_idx]
        self.layer_list: dict[LayerNodeABC, int] = {}

        # derive top mem level of input, weight, output
        layer_0 = [layer for layer in self.workload.topological_sort()][0]
        (
            __,
            __,
            act_operand_in_hardware,
            weight_operand_in_hardware
        ) = SearchUnusedMemoryStage.get_act_weight_operand_names(layer=layer_0)
        output_operand_in_layer = layer_0.output_operand
        output_operand_in_hardware = layer_0.memory_operand_links[output_operand_in_layer]
        self.top_mem_level_act: int = -1
        self.top_mem_level_weight: int = -1
        self.top_mem_level_output: int = -1
        for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
            served_operands = list(mem.mem_level_of_operands.keys())
            if act_operand_in_hardware in served_operands and curr_mem_level > self.top_mem_level_act:
                self.top_mem_level_act = curr_mem_level
            if weight_operand_in_hardware in served_operands and curr_mem_level > self.top_mem_level_weight:
                self.top_mem_level_weight = curr_mem_level
            if output_operand_in_hardware in served_operands and curr_mem_level > self.top_mem_level_output:
                self.top_mem_level_output = curr_mem_level

        # record of the index of the weight memory level
        self.mem_update_weight: int = self.top_mem_level_weight

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
                self.mem_update_list[str(idx)] = {
                    act_operand_in_hardware: -1,
                    output_operand_in_hardware: -1
                }
                self.each_layer_IO_data_size[str(idx)] = [
                    {
                        output_operand_in_hardware: layer.operand_size_bit[output_operand_in_layer],
                        act_operand_in_hardware: input_data_size,
                    }
                ]
                self.layer_list[layer] = idx
                continue

            # update info for regular layers
            self.mem_update_list[str(idx)] = {
                    act_operand_in_hardware: -1,
                    output_operand_in_hardware: -1
            }
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
        """
        # remove dummy (non-conv) layers (ReLU, Pooling..) in the layer graph
        self.remove_dummy_nodes_in_workload()

        # calculate the allowed lowest mem level per operand per layer
        layer_list_without_dummy = [layer for layer in self.workload.topological_sort()]
        for id, layer in enumerate(layer_list_without_dummy):
            # handler for the first layer
            if layer == layer_list_without_dummy[0]:
                is_first_layer = True
            else:
                is_first_layer = False

            # handler for the final layer
            if layer == layer_list_without_dummy[-1]:
                is_final_layer = True
            else:
                is_final_layer = False

            # original layer index before removing dummy nodes
            curr_id = self.layer_list[layer]
            # activation, weight operand name
            (
                act_operand_in_layer,
                weight_operand_in_layer,
                act_operand_in_hardware,
                weight_operand_in_hardware
            ) = SearchUnusedMemoryStage.get_act_weight_operand_names(layer=layer)
            # output operand name
            output_operand_in_layer: LayerOperand = layer.output_operand
            output_operand_in_hardware: MemoryOperand = layer.memory_operand_links[output_operand_in_layer]

            is_branch_starting_node = True if self.workload.out_degree(layer) > 1 else False
            is_branch_final_node = (
                True
                if self.workload.out_degree(layer) == 1
                   and self.workload.in_degree(list(self.workload.successors(layer))[0]) > 1
                else False
            )

            if not is_first_layer:
                # propagate output mem level of the previous layer to input mem level of current layer
                prev_layer: LayerNode = list(self.workload.predecessors(layer))[0]
                prev_layer_id = self.layer_list[prev_layer]
                prev_layer_output_operand_in_layer = prev_layer.output_operand
                prev_layer_output_operand_in_hardware = prev_layer.memory_operand_links.layer_to_mem_op(
                    prev_layer_output_operand_in_layer)
                # starting node of branches
                mem_level_record_prev_layer = self.mem_update_list[f"{prev_layer_id}"]
                assert prev_layer_output_operand_in_hardware in mem_level_record_prev_layer.keys()
                prev_layer_output_level = mem_level_record_prev_layer[prev_layer_output_operand_in_hardware]
                self.update_IO_mem_level(curr_id, act_operand_in_hardware, prev_layer_output_level)

            # update input, weight, output mem level for branch starting node and branch final node
            if is_branch_starting_node or is_branch_final_node:
                if is_first_layer:
                    self.update_IO_mem_level(curr_id, act_operand_in_hardware, self.top_mem_level_act)
                self.update_IO_mem_level(curr_id, output_operand_in_hardware, self.top_mem_level_output)
            else:
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())
                    avail_mem_size = mem.memory_instance.size * mem.unroll_count
                    # check if curr_mem_level serve the next layer input
                    if not is_final_layer:
                        # grab the next layer name, which is a non-Adder layer for sure
                        next_layer: LayerNode = list(self.workload.successors(layer))[0]
                        (
                            next_layer_act_operand_in_layer,
                            next_layer_weight_operand_in_layer,
                            next_layer_act_operand_in_hardware,
                            next_layer_weight_operand_in_hardware
                        ) = SearchUnusedMemoryStage.get_act_weight_operand_names(layer=next_layer)
                        mem_serve_act_in_next_layer = (
                            True if (next_layer_act_operand_in_hardware in served_operands) else False
                        )
                    else:
                        # instead check if the mem serves act operand of the current layer
                        mem_serve_act_in_next_layer = True if (act_operand_in_hardware in served_operands) else False

                    # both next layer input and current layer output in mem.served_operands
                    mem_serve_io_both = (
                        True if mem_serve_act_in_next_layer and (
                                    output_operand_in_hardware in served_operands) else False
                    )
                    mem_serve_weight = True if (weight_operand_in_hardware in served_operands) else False

                    # we need to change served_operands if the current layer is an Adder layer,
                    # since the act operand name of Adder layers may be different from the next layer
                    is_adder_layer = True if len(layer.constant_operands) == 0 else False
                    if is_adder_layer and mem_serve_io_both:
                        served_operands = [output_operand_in_hardware, act_operand_in_hardware]

                    if mem_serve_io_both or mem_serve_weight:
                        required_IO_data_size = sum(
                            [
                                self.each_layer_IO_data_size[f"{curr_id}"][0][operand]
                                for operand in served_operands
                                if operand != weight_operand_in_hardware
                            ]
                        )
                        required_weight_size = (
                            self.weight_size_entire_workload if weight_operand_in_hardware in served_operands else 0
                        )
                        required_total_size = required_IO_data_size + required_weight_size

                        if required_total_size <= avail_mem_size:
                            if mem_serve_io_both:
                                if is_first_layer:
                                    # update input mem level
                                    self.update_IO_mem_level(curr_id, act_operand_in_hardware, curr_mem_level)
                                # update output mem level
                                self.update_IO_mem_level(curr_id, output_operand_in_hardware, curr_mem_level)
                            # weight mem level must serve all oa dims
                            mem_serve_all_oa_dims = self.check_if_mem_serve_all_oa_dims(mem, self.accelerator)
                            # update weight mem level
                            if (curr_mem_level < self.mem_update_weight) and mem_serve_all_oa_dims and mem_serve_weight:
                                self.mem_update_weight = curr_mem_level
        # assert check if there is -1 value in mem_update_list
        for layer_info in self.mem_update_list.values():
            for mem_level_in_info in layer_info.values():
                assert (mem_level_in_info >= 0), \
                    f"There are still layers with top mem levels not figured out."

    @staticmethod
    def get_act_weight_operand_names(layer: LayerNode
                                     ) -> tuple[
        LayerOperand, LayerOperand or None, MemoryOperand, MemoryOperand or None]:
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
            weight_operand_in_hardware: MemoryOperand = layer.memory_operand_links.layer_to_mem_op(
                weight_operand_in_layer)
        return act_operand_in_layer, weight_operand_in_layer, act_operand_in_hardware, weight_operand_in_hardware

    def check_if_mem_serve_all_oa_dims(self, mem: MemoryLevel, accelerator: Accelerator):
        # check if mem serve all hardare dimensions
        core = accelerator.cores[0]
        operational_array = core.operational_array
        oa_dim_nb = len(operational_array.oa_dim_sizes)
        mem_served_oa_dim_nb = mem.served_dimensions.nb_dims
        return mem_served_oa_dim_nb == oa_dim_nb

    def update_mem_level_for_loading_data(self):
        """! [OPTIONAL FUNCTION] This is an optional function. Depending on your requirement, sometimes data loading
        from the top mem level and offloading to the top mem level is a must. If that is the case, add this function
        to self.run(). Otherwise, if the input is generated on-chip at the lowest possible input mem level and the
        output is stored on-chip at the lowest possible output mem level, remove this function from self.run(). [
        FUNCTION OBJECT] Update mem_update_list of first and last layer, so that the input data of first layer still
        is loaded from top input mem level and the output of last layer still is offloaded to top output mem level
        """
        self.remove_dummy_nodes_in_workload()

        # Update mem_update_list and mem_update_weight
        layer_list_without_dummy = [layer for layer in self.workload.topological_sort()]
        for id, layer in enumerate(layer_list_without_dummy):
            # handler for the first layer
            if layer == layer_list_without_dummy[0]:
                is_first_layer = True
            else:
                is_first_layer = False

            # handler for the final layer
            if layer == layer_list_without_dummy[-1]:
                is_final_layer = True
            else:
                is_final_layer = False

            (
                act_operand_in_layer,
                weight_operand_in_layer,
                act_operand_in_hardware,
                weight_operand_in_hardware
            ) = SearchUnusedMemoryStage.get_act_weight_operand_names(layer=layer)
            output_operand_in_layer = layer.output_operand
            output_operand_in_hardware = layer.memory_operand_links.layer_to_mem_op(output_operand_in_layer)
            curr_id = self.layer_list[layer]
            if is_first_layer:
                self.update_IO_mem_level(curr_id, act_operand_in_hardware, self.top_mem_level_act)
            if is_final_layer:
                self.update_IO_mem_level(curr_id, output_operand_in_hardware, self.top_mem_level_output)

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
        self.mem_update_list[layer_id][operand] = target_level
        """
        for layer_op, mem_lv in self.mem_update_list[f"{layer_id}"].items():
            if layer_op == operand and ((mem_lv == -1) or (mem_lv > target_level)):
                self.mem_update_list[f"{layer_id}"][layer_op] = target_level
