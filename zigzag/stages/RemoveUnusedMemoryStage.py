from typing import Any
import logging


from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.memory_level import ServedMemDimensions
from zigzag.hardware.architecture.memory_port import DataDirection, PortAllocation
from zigzag.workload.layer_node import LayerNode
from zigzag.utils import pickle_deepcopy
from zigzag.stages.Stage import Stage, StageCallable

logger = logging.getLogger(__name__)


class RemoveUnusedMemoryStage(Stage):
    """! This stage must be processed behind WorkloadStage. It stage removes unused memory level found by
    SearchUnusedMemoryStage.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        mem_update_list: dict[int, dict[MemoryOperand, int]],
        mem_update_weight: int,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.layer = layer
        self.mem_update_list = mem_update_list
        self.mem_update_weight = mem_update_weight

    def run(self):
        modified_accelerator = self.generate_accelerator_with_removing_unused_memory()
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            accelerator=modified_accelerator,
            layer=self.layer,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def generate_accelerator_with_removing_unused_memory(self) -> Accelerator:
        # Remove no-use memory level according to update_mem_list and mem_update_weight
        curr_id = self.layer.id
        core = next(iter(self.accelerator.cores))
        operational_array = core.operational_array
        memory_hierarchy = core.memory_hierarchy

        act_layer_op = self.layer.get_act_layer_op()
        weight_layer_op = self.layer.get_weight_layer_op()
        output_layer_op = self.layer.output_operand
        act_mem_op = self.layer.memory_operand_links.layer_to_mem_op(act_layer_op)
        weight_mem_op = self.layer.memory_operand_links.layer_to_mem_op(weight_layer_op)
        output_mem_op = self.layer.memory_operand_links.layer_to_mem_op(output_layer_op)

        # Find target_act/const/output_mem_level
        target_act_mem_level = [
            targeted_mem_lv
            for operand_in_hardware, targeted_mem_lv in self.mem_update_list[curr_id].items()
            if operand_in_hardware == act_mem_op
        ].pop()
        target_output_mem_level = [
            targeted_mem_lv
            for operand_in_hardware, targeted_mem_lv in self.mem_update_list[curr_id].items()
            if operand_in_hardware == output_mem_op
        ].pop()

        if len(self.layer.constant_operands) == 0:
            # two inputs for Adder layers are both act
            target_const_mem_level = [
                targeted_mem_lv
                for operand_in_hardware, targeted_mem_lv in self.mem_update_list[curr_id].items()
                if operand_in_hardware == act_mem_op
            ].pop()
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
            port_alloc: PortAllocation = memory_level.port_alloc_raw
            served_dimensions = memory_level.served_dimensions

            new_memory_instance: MemoryInstance = pickle_deepcopy(memory_instance)
            new_operands: list[MemoryOperand] = []
            new_port_alloc_data: dict[MemoryOperand, dict[DataDirection, str]] = {}

            if (act_mem_op in operands) and curr_mem_level <= target_act_mem_level:
                new_operands.append(act_mem_op)
                new_port_alloc_data[act_mem_op] = port_alloc.get_alloc_for_mem_op(act_mem_op)
                # TODO I think you need
            if (weight_mem_op in operands) and curr_mem_level <= target_const_mem_level:
                new_operands.append(weight_mem_op)
                new_port_alloc_data[weight_mem_op] = port_alloc.get_alloc_for_mem_op(weight_mem_op)
            if (output_mem_op in operands) and curr_mem_level <= target_output_mem_level:
                new_operands.append(output_mem_op)
                new_port_alloc_data[output_mem_op] = port_alloc.get_alloc_for_mem_op(output_mem_op)

            new_port_alloc = PortAllocation(new_port_alloc_data)
            new_served_dimensions: ServedMemDimensions = pickle_deepcopy(served_dimensions)
            if len(new_operands) > 0:
                new_memory_hierarchy.add_memory(
                    memory_instance=new_memory_instance,
                    operands=new_operands,
                    port_alloc=new_port_alloc,
                    served_dimensions=new_served_dimensions,
                )

        # Create the new core
        new_core = Core(
            core_id=core.id,
            operational_array=operational_array,
            memory_hierarchy=new_memory_hierarchy,
        )

        # Create the new accelerator
        name = self.accelerator.name
        new_name = name + "-removing-nouse-mem"
        new_cores = {new_core}
        new_accelerator = Accelerator(
            name=new_name,
            core_set=new_cores,
        )

        logger.info("Update mem architecture for layer %s...", self.layer)

        return new_accelerator
