from typing import Any
import logging
from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.workload.Workload import WorkloadABC
from zigzag.workload.layer_node import LayerNode
from zigzag.workload.LayerNodeABC import LayerNodeABC

logger = logging.getLogger(__name__)


class SearchUnusedMemoryStage(Stage):
    """! Class for searching lowest allowed memory level per operand per layer"""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        workload: WorkloadABC[LayerNodeABC],
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        core_id = accelerator.cores[0].id  # correct only for single-core hardware
        core = accelerator.get_core(core_id)
        # Remove dummy (non-conv) layers (ReLU, Pooling..) in the layer graph
        self.workload_no_dummy = self.workload.get_copy_no_dummy()

        # record of all memory levels
        self.core_mem_level_list = core.memory_hierarchy.mem_level_list
        # record of top mem level per layer: dict[layer_idx, mem_level]
        self.mem_update_list: dict[int, dict[MemoryOperand, int]] = {}
        # record of input, output size per layer (unit: bit): dict[layer_idx: [operand: size]]
        self.each_layer_io_data_size: dict[int, list[dict[MemoryOperand, int]]] = {}
        # record of the weight size of entire workload (unit: bit)
        self.weight_size_entire_workload: int = 0

        # Derive top mem level of input, weight, output
        layer_0 = self.workload_no_dummy.node_list[0]
        act_layer_op = layer_0.get_act_layer_op()
        weight_layer_op = layer_0.get_weight_layer_op()
        output_layer_op = layer_0.output_operand
        self.top_mem_level_act = self.__get_top_mem_level_for_op(
            layer_0.memory_operand_links.layer_to_mem_op(act_layer_op)
        )
        self.top_mem_level_weight = self.__get_top_mem_level_for_op(
            layer_0.memory_operand_links.layer_to_mem_op(weight_layer_op)
        )
        self.top_mem_level_output = self.__get_top_mem_level_for_op(
            layer_0.memory_operand_links.layer_to_mem_op(output_layer_op)
        )

        # record of the index of the weight memory level
        self.mem_update_weight = self.top_mem_level_weight

        # derive input, output size per layer and weight size of entire workload
        self.__calc_sizes_per_layer()

    def __get_top_mem_level_for_op(self, mem_op: MemoryOperand) -> int:
        """For the given MemoryOperand, return the level of the highest memory that serves this operand. Return -1
        if no memories serve this operand."""
        top_level = -1
        for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
            served_operands = mem.mem_level_of_operands.keys()
            if mem_op in served_operands:
                top_level = max(curr_mem_level, top_level)

        return top_level

    def __calc_sizes_per_layer(self) -> None:
        for layer in self.workload_no_dummy.topological_sort():
            act_layer_op = layer.get_act_layer_op()
            weight_layer_op = layer.get_weight_layer_op()
            output_layer_op = layer.output_operand
            act_mem_op = layer.memory_operand_links.layer_to_mem_op(act_layer_op)
            output_mem_op = layer.memory_operand_links.layer_to_mem_op(output_layer_op)

            # Initialize
            self.mem_update_list[layer.id] = {act_mem_op: -1, output_mem_op: -1}

            # All inputs are variable, e.g. Add or MatMul
            if len(layer.constant_operands) == 0:
                input_data_size = sum([layer.operand_size_bit[operand] for operand in layer.input_operands])
                self.each_layer_io_data_size[layer.id] = [
                    {
                        act_mem_op: input_data_size,
                        output_mem_op: layer.operand_size_bit[output_layer_op],
                    }
                ]
            else:
                # update info for regular layers
                self.each_layer_io_data_size[layer.id] = [
                    {
                        act_mem_op: layer.operand_size_bit[act_layer_op],
                        output_mem_op: layer.operand_size_bit[output_layer_op],
                    }
                ]
                self.weight_size_entire_workload += layer.operand_size_bit[weight_layer_op]

    def run(self, workload_data_always_from_top_mem: bool = False):
        """@param workload_data_always_from_top_mem: input of 1st layer, output of last layer must in the top mem
        level"""
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
            **self.kwargs,
        )
        for cme, (layer, extra_info) in sub_stage.run():
            yield cme, (layer, extra_info)

    def update_top_mem_level(self):
        """
        Update mem_update_list and mem_update_weight according to the algorithm description at the file beginning.
        # TODO should be split into multiple functions
        """
        # calculate the allowed lowest mem level per operand per layer
        layer_list_without_dummy: list[LayerNode] = list(self.workload_no_dummy.topological_sort())
        for layer in layer_list_without_dummy:
            is_first_layer = layer == layer_list_without_dummy[0]
            is_final_layer = layer == layer_list_without_dummy[-1]

            act_layer_op = layer.get_act_layer_op()
            weight_layer_op = layer.get_weight_layer_op()
            output_layer_op = layer.output_operand
            act_mem_op = layer.memory_operand_links.layer_to_mem_op(act_layer_op)
            weight_mem_op = layer.memory_operand_links.layer_to_mem_op(weight_layer_op)
            output_mem_op = layer.memory_operand_links.layer_to_mem_op(output_layer_op)

            is_branch_starting_node = self.workload_no_dummy.get_out_degree_for_layer(layer) > 1
            is_branch_final_node = (self.workload_no_dummy.get_out_degree_for_layer(layer) == 1) and (
                self.workload_no_dummy.get_out_degree_for_layer(
                    next(self.workload_no_dummy.get_successors_for_layer(layer))
                )
                > 1
            )

            if not is_first_layer:
                # propagate output mem level of the previous layer to input mem level of current layer
                prev_layer = next(self.workload_no_dummy.get_predecessors_for_layer(layer))
                prev_layer_output_layer_op = prev_layer.output_operand
                prev_layer_output_mem_op = prev_layer.memory_operand_links.layer_to_mem_op(prev_layer_output_layer_op)
                # starting node of branches
                mem_level_record_prev_layer = self.mem_update_list[prev_layer.id]
                # assert prev_layer_output_mem_op in mem_level_record_prev_layer.keys()
                prev_layer_output_level = mem_level_record_prev_layer[prev_layer_output_mem_op]
                self.update_io_mem_level(layer.id, act_mem_op, prev_layer_output_level)

            # update input, weight, output mem level for branch starting node and branch final node
            if is_branch_starting_node or is_branch_final_node:
                if is_first_layer:
                    self.update_io_mem_level(layer.id, act_mem_op, self.top_mem_level_act)
                self.update_io_mem_level(layer.id, output_mem_op, self.top_mem_level_output)
            else:
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())
                    avail_mem_size = mem.memory_instance.size * mem.unroll_count
                    # check if curr_mem_level serve the next layer input
                    if not is_final_layer:
                        # grab the next layer name, which is a non-Adder layer for sure
                        next_layer = next(self.workload_no_dummy.get_successors_for_layer(layer))
                        next_layer_act_layer_op = next_layer.get_act_layer_op()
                        next_layer_act_mem_op = next_layer.memory_operand_links.layer_to_mem_op(next_layer_act_layer_op)

                        mem_serve_act_in_next_layer = next_layer_act_mem_op in served_operands
                    else:
                        # instead check if the mem serves act operand of the current layer
                        mem_serve_act_in_next_layer = act_mem_op in served_operands

                    # both next layer input and current layer output in mem.served_operands
                    mem_serve_io_both = mem_serve_act_in_next_layer and (output_mem_op in served_operands)
                    mem_serve_weight = weight_mem_op in served_operands

                    # we need to change served_operands if the current layer is an Adder layer,
                    # since the act operand name of Adder layers may be different from the next layer
                    is_adder_layer = True if len(layer.constant_operands) == 0 else False
                    if is_adder_layer and mem_serve_io_both:
                        served_operands = [output_mem_op, act_mem_op]

                    if mem_serve_io_both or mem_serve_weight:
                        required_io_data_size = sum(
                            [
                                self.each_layer_io_data_size[layer.id][0][operand]
                                for operand in served_operands
                                if operand != weight_mem_op
                            ]
                        )
                        required_weight_size = (
                            self.weight_size_entire_workload if weight_mem_op in served_operands else 0
                        )
                        required_total_size = required_io_data_size + required_weight_size

                        if required_total_size <= avail_mem_size:
                            if mem_serve_io_both:
                                if is_first_layer:
                                    self.update_io_mem_level(layer.id, act_mem_op, curr_mem_level)
                                self.update_io_mem_level(layer.id, output_mem_op, curr_mem_level)
                            # weight mem level must serve all oa dims
                            mem_serve_all_oa_dims = self.check_if_mem_serve_all_oa_dims(mem, self.accelerator)
                            # update weight mem level
                            if (curr_mem_level < self.mem_update_weight) and mem_serve_all_oa_dims and mem_serve_weight:
                                self.mem_update_weight = curr_mem_level

        # assert check if there is -1 value in mem_update_list
        for layer_info in self.mem_update_list.values():
            for mem_level_in_info in layer_info.values():
                assert mem_level_in_info >= 0, "There are still layers with top mem levels not figured out."

    def check_if_mem_serve_all_oa_dims(self, mem: MemoryLevel, accelerator: Accelerator):
        """! Function to check if mem serve all hardware dimensions"""
        core = accelerator.cores[0]
        operational_array = core.operational_array
        oa_dim_nb = len(operational_array.dimension_sizes)
        mem_served_oa_dim_nb = len(mem.served_dimensions)
        return mem_served_oa_dim_nb == oa_dim_nb

    def update_mem_level_for_loading_data(self):
        """! [OPTIONAL FUNCTION] This is an optional function. Depending on your requirement, sometimes data loading
        from the top mem level and offloading to the top mem level is a must. If that is the case, add this function
        to self.run(). Otherwise, if the input is generated on-chip at the lowest possible input mem level and the
        output is stored on-chip at the lowest possible output mem level, remove this function from self.run(). [
        FUNCTION OBJECT] Update mem_update_list of first and last layer, so that the input data of first layer still
        is loaded from top input mem level and the output of last layer still is offloaded to top output mem level
        """

        # Update mem_update_list and mem_update_weight
        layers_without_dummy: list[LayerNode] = list(self.workload_no_dummy.topological_sort())
        for layer in layers_without_dummy:
            is_first_layer = layer == layers_without_dummy[0]
            is_final_layer = layer == layers_without_dummy[-1]

            act_layer_op = layer.get_act_layer_op()
            output_layer_op = layer.output_operand
            act_mem_op = layer.memory_operand_links.layer_to_mem_op(act_layer_op)
            output_mem_op = layer.memory_operand_links.layer_to_mem_op(output_layer_op)

            if is_first_layer:
                self.update_io_mem_level(layer.id, act_mem_op, self.top_mem_level_act)
            if is_final_layer:
                self.update_io_mem_level(layer.id, output_mem_op, self.top_mem_level_output)

    def update_io_mem_level(self, layer_id: int, operand: MemoryOperand, target_level: int):
        """! Update self.mem_update_list as:
        self.mem_update_list[layer_id][operand] = target_level
        """
        for layer_op, mem_lv in self.mem_update_list[layer_id].items():
            if layer_op == operand and ((mem_lv == -1) or (mem_lv > target_level)):
                self.mem_update_list[layer_id][layer_op] = target_level
