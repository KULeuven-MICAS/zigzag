from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.utils import pickle_deepcopy
from zigzag.classes.stages.Stage import Stage
from typing import Generator

import logging

logger = logging.getLogger(__name__)

#################### Description ####################
## This stage must be processed behind WorkloadStage.
## This stage removes unused memory level found by SearchUnusedMemoryStage.
################### Pseudo-code ####################
## Initialization:
##  target_act_mem_level, target_output_mem_level: get from mem_update_list
##  target_const_mem_level = mem_udpate_weight
## 1. Modify mem structure:
## for mem in mem_levels(sort_order: from bottom to top):
##   if ['I'] in mem.served_operand and mem.mem_level > target_act_mem_level:
##     remove ['I'] in mem.served_operand, mem_port_alloc
##   if ['O'] in mem.served_operand and mem.mem_level > target_output_mem_level:
##     remove ['O'] in mem.served_operand, mem_port_alloc
##   if ['W'] in mem.served_operand and mem.mem_level > target_const_mem_level:
##     remove ['W'] in mem.served_operand, mem_port_alloc
## 2. Remove unused memory
## for mem in mem_levels(sort_order: from top to bottom):
##   if mem.served_operand == empty:
##     do not add the current mem into the modified architecture
#####################################################


class RemoveUnusedMemoryStage(Stage):
    def __init__(
        self,
        list_of_callables,
        *,
        accelerator,
        layer,
        mem_update_list,
        mem_update_weight,
        layer_list,
        **kwargs,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.layer = layer
        self.layer_list = layer_list
        self.mem_update_list = mem_update_list
        self.mem_update_weight = mem_update_weight

    def run(self) -> Generator:
        modified_accelerator = self.generate_accelerator_with_removing_unused_memory()
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            accelerator=modified_accelerator,
            layer=self.layer,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def generate_accelerator_with_removing_unused_memory(self):
        ## Remove nouse memory level according to update_mem_list and mem_update_weight
        curr_id = self.layer_list[
            self.layer
        ]  # current layer id (key) in mem_udpate_list
        curr_id = str(curr_id)
        output_operand = self.layer.memory_operand_links[
            self.layer.output_operand
        ]  # output representation in memory
        core = next(iter(self.accelerator.cores))
        operational_array = core.operational_array
        memory_hierarchy = core.memory_hierarchy

        if len(self.layer.constant_operands) == 1:
            act_operand = self.layer.memory_operand_links[
                [
                    operand
                    for operand in self.layer.input_operands
                    if operand not in self.layer.constant_operands
                ][0]
            ]  # act representation in memory
            const_operand = self.layer.memory_operand_links[
                self.layer.constant_operands[0]
            ]  # weight representation in memory
        elif len(self.layer.constant_operands) == 0:
            # special case when defining workload manually:
            # the constant operands list is empty for such as "Adder" layers
            # for input operand, we will represent all inputs as one input, since only their data size is used for required mem size calculation.
            act_operand = self.layer.memory_operand_links[
                self.layer.input_operands[0]
            ]  # act representation in memory
            const_operand = self.layer.memory_operand_links[
                self.layer.input_operands[1]
            ]  # weight representation in memory
        else:
            # special case when defining workload manually:
            # both I and W are considered as constant operands for the first layer
            pr_loop_keys = tuple(self.layer.pr_loop.keys())
            for (
                operand,
                related_loop,
            ) in self.layer.operand_dimensionality_order.items():
                if pr_loop_keys[0] in related_loop:
                    act_operand = operand
            weight_operand: list = [
                x for x in self.layer.constant_operands if x != act_operand
            ]  # weight representation in layer
            assert len(weight_operand) == 1
            weight_operand: str = weight_operand[0]
            act_operand = self.layer.memory_operand_links[
                act_operand
            ]  # map from layer representation to hardware memory representation
            const_operand = self.layer.memory_operand_links[
                weight_operand
            ]  # weight representation in memory

        # Find target_act/const/output_mem_level
        for pos, ele in enumerate(self.mem_update_list[curr_id]):
            if list(ele.keys())[0] == act_operand:
                target_act_mem_level = self.mem_update_list[curr_id][pos][act_operand]
            if list(ele.keys())[0] == output_operand:
                target_output_mem_level = self.mem_update_list[curr_id][pos][
                    output_operand
                ]
        if len(self.layer.constant_operands) == 0:
            # special case when defining workload manually:
            # the constant operands list is empty for such as "Adder" layers
            # Here we make a trick: treating the other input as const_operand
            for pos, ele in enumerate(self.mem_update_list[curr_id]):
                if list(ele.keys())[0] == act_operand:
                    target_const_mem_level = self.mem_update_list[curr_id][pos][
                        act_operand
                    ]
        else:
            target_const_mem_level = self.mem_update_weight

        # Initialize the new memory hierarchy
        mh_name = memory_hierarchy.name
        new_mh_name = mh_name + "-without-unused-memory"
        new_memory_hierarchy = MemoryHierarchy(operational_array, new_mh_name)

        # Add memories to the new memory hierarchy with the correct attributes
        for curr_mem_level, memory_level in enumerate(memory_hierarchy.mem_level_list):
            memory_instance = memory_level.memory_instance
            operands = tuple(memory_level.operands)
            port_alloc = memory_level.port_alloc_raw
            served_dimensions_vec = memory_level.served_dimensions_vec
            assert len(served_dimensions_vec) >= 1
            served_dimensions = served_dimensions_vec[0]

            new_memory_instance = pickle_deepcopy(memory_instance)
            new_operands = []
            new_port_alloc = []
            if (act_operand in operands) and curr_mem_level <= target_act_mem_level:
                new_operands.append(act_operand)
                index_in_operands = operands.index(act_operand)
                new_port_alloc.append(port_alloc[index_in_operands])
            if (const_operand in operands) and curr_mem_level <= target_const_mem_level:
                new_operands.append(const_operand)
                index_in_operands = operands.index(const_operand)
                new_port_alloc.append(port_alloc[index_in_operands])
            if (
                output_operand in operands
            ) and curr_mem_level <= target_output_mem_level:
                new_operands.append(output_operand)
                index_in_operands = operands.index(output_operand)
                new_port_alloc.append(port_alloc[index_in_operands])
            new_operands = tuple(new_operands)
            new_port_alloc = tuple(new_port_alloc)
            new_served_dimensions = pickle_deepcopy(served_dimensions)
            if len(new_operands) > 0:
                new_memory_hierarchy.add_memory(
                    memory_instance=new_memory_instance,
                    operands=new_operands,
                    port_alloc=new_port_alloc,
                    served_dimensions=new_served_dimensions,
                )

        # Create the new core
        id = core.id
        dataflows = core.dataflows
        new_id = id
        new_dataflows = pickle_deepcopy(dataflows)
        new_core = Core(
            id=new_id,
            operational_array=operational_array,
            memory_hierarchy=new_memory_hierarchy,
            dataflows=new_dataflows,
        )

        # Create the new accelerator
        name = self.accelerator.name
        new_name = name + "-removing-nouse-mem"
        new_cores = {new_core}
        new_accelerator = Accelerator(
            name=new_name,
            core_set=new_cores,
        )

        logger.info(f"Update mem architecture for layer {self.layer}...")

        # RemoveUnusedMemoryStage.visulize_modified_memory_structure(new_memory_hierarchy)

        return new_accelerator

    @staticmethod
    def visulize_modified_memory_structure(new_memory_hierarchy):
        # Visualization for debugging
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(new_memory_hierarchy)
