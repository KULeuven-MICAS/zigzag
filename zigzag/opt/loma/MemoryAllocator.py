from collections import defaultdict
from math import prod

import numpy as np

from zigzag.datatypes import (
    Constants,
    LayerDim,
    LayerOperand,
    MemoryOperand,
    UnrollFactor,
    UnrollFactorInt,
)
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping, TemporalMappingDict
from zigzag.opt.loma.Loop import Loop
from zigzag.workload.layer_attributes import LayerDimSizes
from zigzag.workload.layer_node import LayerNode


class MemoryHierarchyTooSmallException(Exception):
    """Indicates that the layer does not fit entirely within the memory hierarchy for this temporal ordering"""


class MemoryTooSmallException(Exception):
    """Indicates that some memory instance is too small to support this temporal ordering"""


class MemoryAllocator:
    """! Class that handles allocation of a loop ordering to the memories in the hierarchy."""

    def __init__(
        self,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        ordering: list[tuple[LayerDim, UnrollFactorInt]],
    ):
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.ordering = ordering

        # Initialize operands (having local copies speeds up the code)
        self.layer_and_mem_ops = self.layer.memory_operand_links.layer_and_mem_ops()
        self.layer_ops = [layer_op for layer_op, _ in self.layer_and_mem_ops]
        self.mem_ops = [mem_ops for _, mem_ops in self.layer_and_mem_ops]
        self.mem_to_layer_op = {mem_op: layer_op for layer_op, mem_op in self.layer_and_mem_ops}

        # Bit precision for the different mem ops
        self.precision: dict[MemoryOperand, int] = {
            mem_op: self.layer.operand_precision[layer_op] for layer_op, mem_op in self.layer_and_mem_ops
        }
        self.precision[Constants.FINAL_OUTPUT_MEM_OP] = self.layer.operand_precision.final_output_precision

        # Initialize the unallocated loops with the ordering for each operand
        self.unallocated = {mem_op: [Loop(dim, size) for (dim, size) in self.ordering] for mem_op in self.mem_ops}

        # Initialize the allocated loops with the spatial mapping at the operand level for each operand
        self.allocated: dict[MemoryOperand, list[Loop]] = {}
        for layer_op, mem_op in self.layer_and_mem_ops:
            self.allocated[mem_op] = [
                Loop(dim, size, "spatial") for (dim, size) in self.spatial_mapping.get_unrolling(op=layer_op, level=0)
            ]

        # Initialize the level of memory hierarchy for each layer operand at 1 (first memory level).
        # This information is required to fetch the correct spatial loops after we have allocated temporal loops.
        self.mem_level = {layer_op: 1 for layer_op in self.layer_ops}

        # Initialize the temporal mapping dict, which is appended to throughout the allocation process.
        # It is a dictionary with keys the different layer operands and values a list of lists.
        # The sublists represent the memory levels for that operand and contain the loops allocated to that level.
        self.temporal_mapping_dict: TemporalMappingDict = {layer_op: [] for layer_op in self.layer_ops}

    def run(self):
        """! Run the memory allocation process.
        Start by the lowest memory hierarchy level and allocate as much loops as possible
        for the different operands. The spatial unrolling has to be taken into account at
        each memory level in the hierarchy.
        """

        # self.nodes contains the different memory nodes in bottom-up fashion
        core_id = self.layer.core_allocation[0]
        memory_hierarchy = self.accelerator.get_core(core_id).memory_hierarchy
        top_levels = {mem_op: memory_hierarchy.get_operand_top_level(mem_op) for mem_op in self.mem_ops}
        for node in memory_hierarchy.topological_sort():
            self.allocate_node(node, top_levels)

        # After all the nodes have been allocated, we can create the TemporalMapping
        # object from the dictionary we have built
        temporal_mapping = TemporalMapping(self.temporal_mapping_dict, self.layer)
        return temporal_mapping

    def allocate_node(self, node: MemoryLevel, top_levels: dict[MemoryOperand, MemoryLevel]):
        """! Allocate a single memory node with the best loops that remain in the unallocated loop ordering.
        @param node: The MemoryLevel to which we will allocate loops.
        @param top_levels: A list of MemoryLevels for each mem_op that is the highest MemoryLevel that stores that
          mem_op.
        #TODO cleanup
        """

        # Select the mem operands that are required for this layer (e.g. pooling has no weights so one mem
        # op less)
        filtered_mem_ops = [op for op in node.operands if op in self.mem_ops]
        # Get the capacity of this memory node (in bits)
        mem_capacity = node.memory_instance.size

        # For all the mem_ops, find the max amount of unallocated loops we could allocate
        all_sizes = {mem_op: self.calc_size_slices(mem_op, mem_capacity) for mem_op in filtered_mem_ops}

        # Now that we have this for all the mem_ops, call function that finds the best
        # combination of loops to minimize the number of accesses to the level above
        best_loop_idxs = self.find_best_loop_combination(filtered_mem_ops, all_sizes, node, top_levels)

        for best_loop_idx, mem_op in zip(best_loop_idxs, filtered_mem_ops):
            # Now that we have the combination of loop_idx for each mem_op, add them
            # to the allocated loops and remove them from the unallocated loops
            loops_to_allocate = self.unallocated[mem_op][:best_loop_idx].copy()
            self.allocated[mem_op] += loops_to_allocate
            del self.unallocated[mem_op][:best_loop_idx]

            # Add the loops to allocate to the level-by-level temporal_mapping_dict
            # The key of this dict is the layer_op and not the mem_op
            layer_op = self.mem_to_layer_op[mem_op]
            self.temporal_mapping_dict[layer_op].append([(loop.layer_dim, loop.size) for loop in loops_to_allocate])

            # This memory node that stores one or more mem_ops might be
            # spatially unrolled, add these spatially unrolled loops to
            # the list of allocated loops now, so that the next memory nodes
            # correctly see this spatial unrolling.
            # For this we require the level of memory we are evaluating for this op.
            mem_level_op = self.mem_level[layer_op]
            spatial_loops = self.spatial_mapping.get_unrolling(op=layer_op, level=mem_level_op)
            for loop_dim, loop_size in spatial_loops:
                spatial_loop = Loop(layer_dim=loop_dim, size=loop_size, loop_type="spatial")
                self.allocated[mem_op].append(spatial_loop)

            # Check if this node (i.e. MemoryLevel) is the highest level of memory hierarchy.
            # If this is the case and we haven't allocated all loops, raise an exception.
            if node == top_levels[mem_op] and self.unallocated[mem_op]:  # if top level and unallocated not empty
                raise MemoryHierarchyTooSmallException(
                    f"Highest MemoryLevel for {mem_op} = {node} too small to store all loops."
                )

            # Increment the mem_level we are currently at for this layer_op by 1
            self.mem_level[layer_op] += 1

    def get_precision(self, mem_op: MemoryOperand, layer_op: LayerOperand, unallocated_loops: list[Loop]):
        """Get the precision at which this tensor will have to be stored in the MemoryLevel node.
        For output it can be either the partial sum precision, or the final sum precision.
        This depends on if all the irrelevant loops were allocated in a previous MemoryLevel.
        Which in turn means all remaining unallocated loops for this MemoryLevel must not contain any ir loops.
        Moreover, there might be unallocated spatial loops."""
        if mem_op == Constants.OUTPUT_MEM_OP:
            ir_dims = self.layer.loop_relevancy_info.get_ir_layer_dims(layer_op)
            unallocated_spatial_dims = [
                dim for dim, _ in self.spatial_mapping.get_unrolling_all(layer_op, self.mem_level[layer_op])
            ]
            unallocated_temporal_dims = [unallocated_loop.layer_dim for unallocated_loop in unallocated_loops]
            unallocated_dims = unallocated_spatial_dims + unallocated_temporal_dims

            # If there is still an irrelevant unallocated loop dimension, pick the full precision
            precision = (
                self.precision[Constants.OUTPUT_MEM_OP]
                if any([dim in ir_dims for dim in unallocated_dims])
                else self.precision[Constants.FINAL_OUTPUT_MEM_OP]
            )
        else:
            precision = self.precision[mem_op]
        return precision

    def calc_size_slices(
        self, mem_op: MemoryOperand, mem_capacity: int, db_support: bool = False
    ) -> list[UnrollFactor]:
        """! Calculate the required memory size to store different slices of the unallocated loops, with 'mem_capacity'
        as an upper bound.
        @param mem_capacity Capacity of the memory node in bits.
        @param db_support Double buffering support of this node
        """
        layer_op = self.mem_to_layer_op[mem_op]
        allocated_loops = self.allocated[mem_op]
        unallocated_loops = self.unallocated[mem_op]
        sizes: list[UnrollFactor] = []
        precision = self.get_precision(mem_op, layer_op, unallocated_loops)

        # If this memory supports double buffering get the size it would take to allocate everything
        if db_support:
            all_loops = allocated_loops + unallocated_loops[: len(unallocated_loops) + 1]
            all_loops_size = self.calc_loops_size(all_loops, layer_op, precision)

        # Go through all slices (includes empty slice)
        for i in range(len(unallocated_loops) + 1):
            unallocated_slice = unallocated_loops[:i]
            loops = allocated_loops + unallocated_slice
            size = self.calc_loops_size(loops, layer_op, precision)
            # double size allocated if the node uses double buffering
            if db_support:
                if len(unallocated_loops[i:]) > 0 and size < all_loops_size:  # type: ignore
                    size *= 2
            if size <= mem_capacity:
                sizes.append(size)
            else:
                if i == 0:  # This means we can't even store the already allocated loops
                    raise MemoryTooSmallException(
                        f"Memory capacity overflow for mem_op {mem_op}. loops={loops} size={size} "
                        f"mem_capacity={mem_capacity}"
                    )
                break  # Stop as soon as we have added a loop that overflows the memory
        return sizes

    def calc_loops_size(
        self,
        loops: list[Loop],
        layer_op: LayerOperand,
        precision: int,
    ) -> UnrollFactor:
        """! Calculate the 'mem_op' tensor size required for all the loops in 'loops'.
        @param loops: The loops we want to calculate the size for.
        @para layer_op: The layer operand we are calculating the size for.
        @param precision: number of bits for this layer operand.
        """

        # First we compute the size of all loop dimensions present in this layer given the loops in 'loops'.
        all_dim_sizes: dict[LayerDim, UnrollFactor] = defaultdict(lambda: 1)
        for loop in loops:
            all_dim_sizes[loop.layer_dim] *= loop.size

        tensor_size = self.layer.calc_tensor_size(layer_op, LayerDimSizes(all_dim_sizes))
        tensor_size_bits = tensor_size * precision
        return tensor_size_bits

    def find_best_loop_combination(
        self,
        mem_ops: list[MemoryOperand],
        all_sizes: dict[MemoryOperand, list[UnrollFactor]],
        node: MemoryLevel,
        top_levels: dict[MemoryOperand, MemoryLevel],
    ) -> list[int]:
        """! Find the best combination of loops from different mem_ops. Best is defined as the combination that
        minimizes the number of accesses to the memory level above.
        # TODO cleanup
        """
        # TODO: Take into account the operand precision which can change based on the loops picked
        mem_capacity = node.memory_instance.size

        # nb_operations = self.__main_inputs.layer.total_mac_count
        # all_accesses = {mem_op: [nb_operations//size for size in all_sizes[mem_op]] for mem_op in mem_ops}

        # If for one of the mem_ops this is the top level memory, we have to enforce that all unallocated loops
        # will be allocated for this operand. We do this by changing the sizes for this mem_op to only include
        # the last number (which represents the size to store all the ops).
        # Because we modify the sizes, we add an offset to the loop_idx for this mem_op.
        loop_idx_offsets = {mem_op: 0 for mem_op in mem_ops}
        for mem_op, sizes in all_sizes.items():
            if node == top_levels[mem_op]:
                loop_idx_offsets[mem_op] = len(sizes) - 1  # offset is number of original sizes - 1
                all_sizes[mem_op] = [sizes[-1]]

        all_accesses: dict[MemoryOperand, list[float]] = {mem_op: [] for mem_op in mem_ops}
        for mem_op in mem_ops:
            # The number of accesses to the level above is determined through the reuse we have for the different
            # choices of temporal loops. accesses = # total temporal loop iterations / temporal reuse
            # The temporal reuse = # allocated loop iterations / operand size
            # Thus: accesses = # total iterations / (# allocated iterations / size)
            # Thus: accesses = (# total iterations / # allocated iterations) * size
            # Thus: accesses = # unallocated iterations * size
            for i, size in enumerate(all_sizes[mem_op]):
                # slice of unallocated loops for this operand size
                unallocated_loops = self.unallocated[mem_op][(i + loop_idx_offsets[mem_op]) :]
                unallocated_iterations = prod((unallocated_loop.size for unallocated_loop in unallocated_loops))
                if node == top_levels[mem_op]:
                    accesses = 0
                else:
                    accesses = unallocated_iterations * size
                all_accesses[mem_op].append(accesses)

        all_max_nb_loops = {mem_op: len(all_sizes[mem_op]) for mem_op in mem_ops}
        all_max_nb_loops_list = list(all_max_nb_loops.values())
        best_loop_idxs = [0 for _ in mem_ops]
        best_accesses = np.inf
        nb_combinations = prod(len(sizes) for sizes in all_sizes.values())
        for i in range(nb_combinations):
            size_comb = 0
            accesses_comb = 0
            current_loop_idxs: list[int] = []
            for mem_op_idx, mem_op in enumerate(mem_ops):
                this_max_nb_loops = all_max_nb_loops_list[mem_op_idx]
                current_loop_idx = (i // prod(all_max_nb_loops_list[mem_op_idx + 1 :])) % this_max_nb_loops
                current_loop_idxs.append(current_loop_idx + loop_idx_offsets[mem_op])
                size_comb += all_sizes[mem_op][current_loop_idx]
                accesses_comb += all_accesses[mem_op][current_loop_idx]
            if size_comb > mem_capacity:
                if i == 0:
                    raise MemoryTooSmallException(
                        """The memory can't store all loops assigned to lower level memories. Likely due to spatial 
                        unrolling."""
                    )
                continue
            if accesses_comb <= best_accesses:
                best_accesses = accesses_comb
                best_loop_idxs = current_loop_idxs
        return best_loop_idxs
