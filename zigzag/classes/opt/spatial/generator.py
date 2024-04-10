from copy import deepcopy
from dataclasses import dataclass
import itertools
import logging
import math
from matplotlib.hatch import SmallCircles
from pydantic import BaseModel

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.dimension import Dimension
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.classes.opt.spatial.SpatialMapping import (
    LayerDim,
    SpatialMapping,
    SpatialMappingHint,
    MappingSingleOADim,
)


class UserSpatialMappingGenerator:
    """! Class that generates valid user-format spatial mappings."""

    def __init__(
        self,
        layer: LayerNode,
        accelerator: Accelerator,
        provided_mapping: SpatialMapping = SpatialMapping({}),
        enable_mix_spatial_mapping_generation=True,
        maximize_hardware_utilization=True,
        enable_weight_diagonal_mapping=True,
    ) -> None:
        """!  The class constructor
        @param layer
        @param accelerator
        """
        self.layer = layer
        self.accelerator = accelerator
        # TODO Watch this type conversion
        self.provided_mapping = SpatialMapping(provided_mapping)  # type: ignore
        # Functional parameters
        self.enable_mix_spatial_mapping_generation = enable_mix_spatial_mapping_generation
        self.enable_weight_diagonal_mapping = enable_weight_diagonal_mapping

        core: Core = self.accelerator.get_core(core_id=layer.core_allocation)
        # operational array dimensions
        self.oa_dims = core.operational_array.dimensions
        # Lowest memory levels
        self.innermost_levels = core.memory_hierarchy.get_inner_memories()

        # TODO move this conversion further up the chain
        self.spatial_mapping_hint: SpatialMappingHint = SpatialMappingHint(
            self.layer.user_spatial_mapping_hint
        )
        # TODO this does the same as `complete_user_spatial_mapping_hint`
        self.spatial_mapping_hint.complete_with_defaults(
            self.oa_dims, {LayerDim(x) for x in self.layer.loop_dim_list}
        )
        # Limit the number of SpatialMappings generated
        self.max_nb_mappings = 3

    def run(self):
        return self.generate_user_spatial_mappings()

    def get_max_unrolling(self) -> SpatialMapping:
        """! Generate a SpatialMapping that contains the maximal unroll factor for every Operational
        Array Dimension and every Layer Dimension. Note that this is NOT a valid mapping as each
        OA Dimension contains ALL Layer Dimensions, maximally unrolled."""

        # Initialize and limit unrolling to the size of the layer or the size of the OA Dimension
        max_unrolling = SpatialMapping(
            {
                oa_dim: MappingSingleOADim(
                    {
                        LayerDim(layer_dim): int(min(layer_size, oa_dim.size))
                        for layer_dim, layer_size in self.layer.loop_dim_size.items()
                    }
                )
                for oa_dim in self.oa_dims
            }
        )

        # Scale max unrollings if it is limited by memory structure
        for mem_level in self.innermost_levels:
            for mem_op in mem_level.operands:
                layer_op = self.layer.get_layer_operand(mem_op)
                # Either write BW (to write outputs away) or read BW (to read inputs)
                mem_bandwidth = mem_level.write_bw if (layer_op == "O") else mem_level.read_bw
                # Bit precision of layer operand
                precision = self.layer.operand_precision[layer_op]
                irrelevant_dimensions = self.layer.get_operand_irrelevant_dimensions(layer_op)

                for oa_dim in mem_level.served_dimensions:
                    # Iterate over all possible LayerDims and rescale max unroll factor
                    for layer_dim, unrolling_size in max_unrolling[oa_dim].items():
                        # If not irrelevant, it is (partially) relevant. Limit based on BW and operand precision.
                        if layer_dim not in irrelevant_dimensions:
                            try:
                                max_multicast_elements = mem_bandwidth // precision
                            except ZeroDivisionError:
                                max_multicast_elements = unrolling_size
                            max_unrolling[oa_dim][layer_dim] = min(
                                max_multicast_elements, unrolling_size
                            )

        return max_unrolling

    def generate_user_spatial_mapping_single_dim(
        self, unroll_hints: set[LayerDim], max_unrolling: MappingSingleOADim, oa_dim_size: int
    ) -> list[MappingSingleOADim]:
        """! Generate a list of possible mappings for the given Operational Array Dimension.
        This list is sorted according to hardware utilization of in this Dimension"""

        # All suggested mappings
        possible_mappings: list[MappingSingleOADim] = []

        # Simple case: maximally unroll over one LayerDim only
        for layer_dim in unroll_hints:
            # Filter out unrollings equal to 1
            if max_unrolling[layer_dim] >= 1:
                possible_mappings.append(MappingSingleOADim({layer_dim: max_unrolling[layer_dim]}))

        if self.enable_mix_spatial_mapping_generation:
            # Add mappings that are combinations of multiple LayerDim
            combination_mappings = self.generate_layer_dim_combinations(max_unrolling, unroll_hints)
            # Make sure new combinations do not exceed OA Dimension size
            possible_mappings += list(
                filter(lambda x: x.get_utilization() <= oa_dim_size, combination_mappings)
            )

        return possible_mappings

    def generate_user_spatial_mappings(self):
        """!  Generator that yields user-defined spatial mappings.
        User-defined means across operational array dimensions.
        For example, this might yield {'D1': (C, 16), 'D2': (K,16)}
        In essence it works as follows:
        \code{.py}
        for each operational array dimension oa_dim (D1, D2, ...):
             for each layer operand layer_op (W, I, O, ...):
              if oa_dim not in served_dimensions(layer_op):
                  continue
              else:
                  for layer dimensions layer_dim (B, K, ...) in the layer:
                      if layer_dim is irrelevant for layer_op:
                          layer_dim can be unrolled maximally
                        if layer_dim is not irrelevant for layer_op:
                          layer_dim can be unrolled if the BW allows it (assumes flexible "bus" reads)
        \endcode
        """

        max_unrollings = self.get_max_unrolling()

        # Start from given mapping if provided, create empty one instead
        mapping_template = (
            self.provided_mapping if self.provided_mapping is not None else SpatialMapping({})
        )

        mapping_template.check_and_reduce(max_unrollings, self.oa_dims)

        oa_dims_to_fill = [x for x in self.oa_dims if x not in mapping_template]
        # For each OA Dimension to fill, generate a list of MappingSingleOADim candidates
        mappings_per_oa_dim: list[list[MappingSingleOADim]] = [
            self.generate_user_spatial_mapping_single_dim(
                self.spatial_mapping_hint[oa_dim], max_unrollings[oa_dim], oa_dim.size
            )
            for oa_dim in oa_dims_to_fill
        ]

        candidate_mappings: list[SpatialMapping] = []
        for combination in itertools.product(*mappings_per_oa_dim):
            candidate = mapping_template.copy()
            for idx, oa_dim in enumerate(oa_dims_to_fill):
                candidate[oa_dim] = combination[idx]
            # Candidate can be invalid if unrollings of LayerDim exceed LayerDim size from workload
            if candidate.is_valid(max_unrollings, self.oa_dims):
                candidate_mappings.append(candidate)

        # Sort according to hardware utilization
        candidate_mappings = sorted(
            candidate_mappings, key=lambda x: x.get_utilization(), reverse=True
        )

        utilizations = [x.get_utilization() for x in candidate_mappings]
        # Count how many times the best utilization occurs
        nb_top_utilizations = utilizations.count(utilizations[0])

        for i in range(max(nb_top_utilizations, self.max_nb_mappings)):
            candidate = candidate_mappings[i]
            if self.enable_weight_diagonal_mapping:
                candidate = self.add_input_pr_spatial_loop(candidate)
            yield candidate

        # # For every operational array dimension, we initialize it by maximally unrolling all layer dimensions.
        # # Later these will be restricted if the memory structure doesn't allow for this unrolling.
        # # Note that max_unrolling is not a valid mapping as each OA Dimension contains the maximal
        # # unroll factor for ALL Layer Dimensions

        # # Indexed by Operational Array Dimension index
        # max_unrolling : list[dict[LayerDim, int]] = [
        #     {
        #         LayerDim(layer_dim): int(min(layer_size, oa_dim.size))
        #         for layer_dim, layer_size in self.layer.loop_dim_size.items()
        #     }
        #     for oa_dim in oa_dims_copy
        # ]

        # If the operational array dimension is a served dimension of the lowest memory level,
        # we ought to limit the unrolling for the relevant and partially relevant loop dimensions
        # for mem_level in self.innermost_levels:
        #     # served_dimensions: set[Dimension] =
        #     for mem_op in mem_level.operands:
        #         layer_op = self.layer.get_layer_operand(mem_op)
        #         # Either write BW (to write outputs away) or read BW (to read inputs)
        #         mem_bandwidth = mem_level.write_bw if (layer_op == "O") else mem_level.read_bw
        #         # Bit precision of layer operand
        #         precision = self.layer.operand_precision[layer_op]
        #         irrelevant_dimensions = self.layer.get_operand_irrelevant_dimensions(layer_op)

        #         for served_dimension in mem_level.served_dimensions:
        #             # Iterate over all possible LayerDims and rescale max unroll factor
        #             for layer_dim, unrolling_size in max_unrolling[served_dimension].items():
        #                 # If not irrelevant, it is (partially) relevant. Limit based on BW and operand precision.
        #                 if layer_dim not in irrelevant_dimensions:
        #                     try:
        #                         max_multicast_elements = mem_bandwidth // precision
        #                     except ZeroDivisionError:
        #                         max_multicast_elements = unrolling_size
        #                     max_unrolling[served_dimension][layer_dim] = min(
        #                         max_multicast_elements, unrolling_size
        #                     )

        # for oa_dim, max_unrolling_single_dim in enumerate(max_unrolling):
        #     # for oa_dim in oa_dims_copy:
        #     if oa_dim not in served_dimensions:
        #         continue

        #     for layer_dim, unrolling_size in mapping_single_dim.items():
        #         if layer_dim in irrelevant_dimensions:
        #             continue
        #         # If not irrelevant, it is (partially) relevant. Limit based on BW and operand precision.
        #         try:
        #             max_multicast_elements = mem_bandwidth // precision
        #         except ZeroDivisionError:
        #             max_multicast_elements = unrolling_size
        #         mapping_single_dim[layer_dim] = min(max_multicast_elements, unrolling_size)

        # At this point the unrolled layer dimensions are maximal (wrt the served dimensions and
        # bandwidth of the lowest memory level). The unrolling size might not be a factor of the
        # layer dimension size, which is required (for non greedy mapping).
        # Convert the unrolling size to be a factor of the layer dimension size.

        # # All possible (valid) spatial mappings
        # mappings: list[SpatialMapping] = []

        # for oa_dim in self.oa_dims:
        #     mappings_this_dim: list[MappingSingleOADim] = []

        #     # If the mapping is user-provided, it will be directly stored in the pool, instead of being generated.
        #     if self.defined_mapping is not None and oa_dim in self.defined_mapping:

        #         for layer_dim, unrolling_size in self.defined_mapping[oa_dim].items():
        #             # Remove user defined LayerDim unrollings that are not part of this layer
        #             if layer_dim in self.layer.loop_dim_size.keys():
        #                 # scale down the defined_mapping size if it exceeds the layer dim size
        #                 unrolling_size = min(
        #                     unrolling_size, self.layer.loop_dim_size[layer_dim.name]
        #                 )
        #                 mappings_this_dim.append(MappingSingleOADim({layer_dim: unrolling_size}))

        #         # # loop_to_reform = tuple(loop_to_reform)
        #         # if len(loop_to_reform) == 0:
        #         #     loop_to_reform = None
        #         # oa_dim_unrollings = [loop_to_reform]

        #     else:
        #         # Incorporate unrolling hits
        #         # NOTE if no hints were provided by user, the mapping hints contain all layer operands
        #         unrolling_hints = self.spatial_mapping_hint[oa_dim]
        #         for layer_dim, unrolling_size in max_unrolling[oa_dim].items():
        #             # layer_dim_size = self.layer.loop_dim_size[layer_dim]
        #             # If e.g. the unrolling size is 10 (because operational array dimension size is 10)
        #             # but the layer dimension size is 14, this would result in a temporal remainder of 14/10.
        #             # In that case we change the unrolling size to 7 (to be a factor of 14).
        #             # We have to make sure the unrolling size is a divisor of the layer dimension size:
        #             # Jan 18 2023: Commented this out as LomaStage allows greedy mapping by adding one more temporal iteration
        #             # while layer_dim_size % unrolling_size != 0:
        #             #     unrolling_size -= 1  # decrement the unrolling by 1

        #             # If the unrolling_size is not 1 and the layer dim is in the user-provided hints,
        #             # add it to the unrollings for this oa_dim

        #             if unrolling_size != 1 and layer_dim in unrolling_hints:
        #                 mappings_single_dim.append(MappingSingleOADim({layer_dim: unrolling_size}))

        #         if self.enable_mix_spatial_mapping_generation:
        #             # Now all unrollings in mappings_single_dim are for single layer dimension.
        #             # If mix spatial mapping is enabled, we will add the mix unrollings to mappings_single_dim next.
        #             mappings_single_dim = self.append_mix_spatial_unrollings(
        #                 mappings_single_dim,
        #                 unrolling_hints,
        #                 oa_dim=oa_dim,
        #             )

        #         if self.maximize_hardware_utilization:
        #             mappings_single_dim = sorted(
        #                 mappings_single_dim, key=lambda x: x.get_utilization(), reverse=True  # type: ignore
        #             )

        #             # # Sort oa_dim_unrollings so values follow a descending order.
        #             # (
        #             #     oa_dim_unrollings,
        #             #     hardware_utilization,
        #             # ) = self.sort_oa_dim_unrollings_in_the_order_of_utilization(
        #             #     oa_dim_unrollings, descending=True
        #             # )

        #         # In case there are no unrollings (of size > 1) possible, add a single unrolling of size 1.
        #         # The loop dimension we pick is randomly chosen as the first loop dimension in the layer.
        #         # The loop dimension chosen shouldn't matter as the size of unrolling is 1 anyway.
        #         # if len(mappings_single_dim) == 0:
        #         #     mappings_single_dim.append(None)

        #     mappings.append(mappings_single_dim)

        # # Now we have for each operational array dimension the layer dimensions and size they can be unrolled without fractional remainder.
        # # Now we have to combine them into user-defined spatial mappings.
        # yield_count = 0
        # # used to control the yield count when maximize_hardware_utilization == True
        # yield_count_limit = 9
        # for combination in itertools.product(*unrollings):
        #     if self.maximize_hardware_utilization and yield_count >= yield_count_limit:
        #         # 2 means: only check the top 2 spatial mapping with the highest hardware utilization
        #         # Modify "2" to other numbers if you want to check on more spatial mappings.
        #         break

        #     legal_spatial_loop, left_layer_dim_size = self.check_spatial_loop_legality(combination)
        #     if not legal_spatial_loop:
        #         continue
        #     # Zip the combination (which is a (layer_dim, layer_size) for each oa_dim with the oa_dim names.
        #     oa_dim_names = [_.name for _ in self.oa_dims]

        #     user_spatial_mapping = {
        #         k: v for (k, v) in zip(oa_dim_names, combination) if v is not None
        #     }

        #     # Add act ir loop if it is weight stationary and the innermost memories serve for act.
        #     if self.enable_weight_diagonal_mapping:
        #         user_spatial_mapping = self.add_input_pr_spatial_loop(
        #             user_spatial_mapping=user_spatial_mapping,
        #             left_layer_dim_size=left_layer_dim_size,
        #         )
        #     yield user_spatial_mapping
        #     yield_count += 1
        # # If yield_count==0, it means there is no legal spatial mapping found.
        # # One reason is that the spatial mapping provided by the user has exceeded the layer dim size,
        # # therefore the loop cannot pass the check.
        # # The other reason could be: there is a layer dim mapped on multiple oa dims,
        # # so the product has exceeded the layer dim size.
        # # For a quick fix on the second cause, we will reform the sm loop only for single layer dim mapping.
        # if yield_count == 0:
        #     for combination in itertools.product(*unrollings):
        #         is_mix_comb = False
        #         for loop in combination:
        #             if self.is_nested_tuple(loop):
        #                 is_mix_comb = True
        #                 continue
        #         if is_mix_comb:
        #             # The fix is not applied for mix sm loop.
        #             continue
        #         if self.maximize_hardware_utilization and yield_count >= yield_count_limit:
        #             # 2 means: only check the top 2 spatial mapping with the highest hardware utilization
        #             # Modify "2" to other numbers if you want to check on more spatial mappings.
        #             break
        #         (
        #             new_combination,
        #             left_layer_dim_size,
        #         ) = self.shrink_combination(
        #             combination,
        #         )
        #         # Zip the combination (which is a (layer_dim, layer_size) for each oa_dim with the oa_dim names.
        #         oa_dim_names = [oa_dim.name for oa_dim in self.oa_dims]

        #         user_spatial_mapping = {
        #             oa_dim_name: unrolling
        #             for (oa_dim_name, unrolling) in zip(oa_dim_names, new_combination)
        #             if unrolling is not None
        #         }
        #         # Add act ir loop if it is weight stationary and the innermost memories serve for act.
        #         if self.enable_weight_diagonal_mapping:
        #             user_spatial_mapping = self.add_input_pr_spatial_loop(
        #                 user_spatial_mapping=user_spatial_mapping,
        #                 left_layer_dim_size=left_layer_dim_size,
        #             )
        #         yield user_spatial_mapping
        #         yield_count += 1

        # assert (
        #     yield_count > 0
        # ), "There is no legal spatial mapping found. Please make sure the provided spatial mappings do not exceed the layer dimension size."

    # def shrink_combination(self, combination):
    #     """! Shrink combination when a layer dimension is mapped on multiple operational array
    #     dimensions"""
    #     new_combination = combination
    #     legal_spatial_loop, left_layer_dim_size = self.check_spatial_loop_legality(
    #         combination=new_combination
    #     )
    #     while not legal_spatial_loop:
    #         new_combination_next = list(new_combination)
    #         for layer_dim, layer_dim_size in left_layer_dim_size.items():
    #             if layer_dim_size < 1:
    #                 scaled_success = False
    #                 for oa_index in range(
    #                     len(new_combination_next) - 1, -1, -1
    #                 ):  # reverse order on oa dims
    #                     (
    #                         mapped_layer_dim,
    #                         mapped_layer_dim_size,
    #                     ) = new_combination_next[oa_index]
    #                     if mapped_layer_dim_size > 1:
    #                         # shrink the mapped layer dim size
    #                         mapped_layer_dim_size -= 1
    #                         new_combination_next[oa_index] = (
    #                             mapped_layer_dim,
    #                             mapped_layer_dim_size,
    #                         )
    #                         scaled_success = True
    #                         break
    #                     else:
    #                         # because a layer can be mapped on multiple oa dims, we will move to the next oa dim.
    #                         pass
    #                 # assert: if not scaled_success,
    #                 # it means the sm loop cannot pass the check, even though all mapped size on this layer dim is 1
    #                 assert scaled_success, (
    #                     f"The spatial loop cannot meet the current hardware dimension after scaling, "
    #                     f"Current spatial loop: {new_combination}"
    #                 )
    #         new_combination_next = tuple(new_combination_next)
    #         # Next we will judge if new_combination_next is a legal loop
    #         # If it is, then we will keep the current combination, rather than new_combination_next,
    #         # the reason is: new_combination can cover the entire layer dim, but new_combination_next is smaller than
    #         # the layer dim, therefore the actual sm loop for the layer dim is a decimal number.
    #         # In that case, we will ceil it up to mimic the real case on hardware.
    #         (
    #             legal_spatial_loop,
    #             left_layer_dim_size_next,
    #         ) = self.check_spatial_loop_legality(combination=new_combination_next)
    #         if not legal_spatial_loop:
    #             new_combination = new_combination_next
    #             left_layer_dim_size = left_layer_dim_size_next
    #         else:
    #             for layer_dim, layer_dim_size in left_layer_dim_size.items():
    #                 # A special case when we will use new_combination_next when legal_spatial_loop == True
    #                 # This case is when new_combination_next exactly match the layer dim size (left size == 1)
    #                 if layer_dim_size < 1 and left_layer_dim_size_next[layer_dim] == 1:
    #                     new_combination = new_combination_next
    #                     left_layer_dim_size = left_layer_dim_size_next
    #                     break
    #     return new_combination, left_layer_dim_size

    def generate_layer_dim_combinations(
        self,
        max_unrolling: MappingSingleOADim,
        unrolling_hints: set[LayerDim],
    ) -> list[MappingSingleOADim]:
        """! Given a list of MappingSingleOADim where each item only contains a single Layer Dimension,
        generate new MappingSingleOADim instances that each contain multiple Layer Dimensions (`mixed`),
        constrained to the maximal Operational Array dimension that corresponds to the MappingSingleOADim
        instance.
        """
        # Not possible to create new combinations if less than 1 LayerDim is available to unroll
        if len(unrolling_hints) <= 1 or max_unrolling.get_nb_unrolled_dims() <= 1:
            return []

        # assert all(
        #     x.get_nb_unrolled_dims == 1 for x in mappings_single_layer_dim
        # ), "Given mapping contains more than one layer dimension: function assumes only single layer per mapping"

        prime_pool: dict[LayerDim, list[int]] = {
            layer_dim: self.prime_factors(max_unroll)
            for layer_dim, max_unroll in max_unrolling.items()
            if max_unroll > 1
        }

        # Create new combinations
        new_mappings: list[MappingSingleOADim] = []
        # Number of Layer Dimensions combined in new Mapping
        for combination_len in range(2, len(unrolling_hints) + 1):
            # e.g. ("C", "K") or ("C", "K", "G")
            for layer_dim_comb in itertools.combinations(unrolling_hints, combination_len):
                to_combine: list[list[int]] = [
                    prime_pool[layer_dim] for layer_dim in layer_dim_comb
                ]
                # e.g. (2, 4) which corresponds to ("C", "K")
                for combination in itertools.product(*to_combine):
                    # e.g. (("C", 2), ("K", 4))
                    mapping_zip = zip(layer_dim_comb, combination)
                    new_mappings.append(
                        MappingSingleOADim(
                            {layer_dim: unroll_value for layer_dim, unroll_value in mapping_zip}
                        )
                    )

        return new_mappings

    def add_input_pr_spatial_loop(self, spatial_mapping: SpatialMapping) -> SpatialMapping:
        """! This function is used to support diagonal spatial mapping
        when input/activation is served in the innermost memories and the weight is stationary.
        """

        layer_dim_size_remainder = {
            LayerDim(layer_dim): layer_size
            for layer_dim, layer_size in self.layer.loop_dim_size.items()
        }
        for _, mapping_single_oa_dim in spatial_mapping.items():
            for layer_dim, unroll_factor in mapping_single_oa_dim.items():
                layer_dim_size_remainder[layer_dim] //= unroll_factor

        # get the link from layer op to mem op
        layer_op_to_mem_op: dict = self.layer.memory_operand_links
        # check if it is weight stationary.
        # keep the spatial loop as it was if it is not weight stationary.
        if len(self.layer.constant_operands) > 1:
            return spatial_mapping
        # # get weight operand name
        # const_operand = layer.constant_operands[0]  # weight representation
        # # get activation operand name
        # act_operand = [
        #     operand for operand in layer.input_operands if operand != const_operand
        # ][0]
        act_operand, const_operand = self.identify_layer_operand_representation(self.layer)
        # get output operand name
        output_operand = self.layer.output_operand
        # get name of OX, OY (weight ir layer dims)
        weight_ir_layer_dims: list[LayerDim] = [
            LayerDim(x) for x in self.layer.operand_loop_dim[const_operand]["ir"]
        ]
        # get the oa_dim name served by input / output innermost memory level
        for mem_level in self.innermost_levels:
            mem_ops = mem_level.operands
            if layer_op_to_mem_op[act_operand] in mem_ops:
                act_served_oa_dim: set[Dimension] = mem_level.served_dimensions  # type: ignore
            if layer_op_to_mem_op[output_operand] in mem_ops:
                output_served_oa_dim: set[Dimension] = mem_level.served_dimensions  # type: ignore
        # check if act is not served in the innermost memories, or act/output is not multicasting on only one dimension.
        # keep the spatial loop as it was if act is not served.
        if "act_served_oa_dim" not in locals() or len(act_served_oa_dim) != 1:  # type: ignore
            return spatial_mapping
        if "output_served_oa_dim" not in locals() or len(output_served_oa_dim) != 1:  # type: ignore
            return spatial_mapping

        # TODO why only first element?
        act_served_oa_dim: Dimension = list(act_served_oa_dim)[0]  # type: ignore
        output_served_oa_dim: Dimension = list(output_served_oa_dim)[0]  # type: ignore
        act_served_oa_dim_size = act_served_oa_dim.size
        output_served_oa_dim_size = output_served_oa_dim.size

        # check if OX / OY in spatial_mapping_hint. Or else target_layer_dim will be empty.
        target_layer_dim: list[LayerDim] = []  # OX or OY or both
        for layer_dim in weight_ir_layer_dims:
            if layer_dim in self.spatial_mapping_hint[act_served_oa_dim]:
                target_layer_dim.append(layer_dim)

        # no further execution if OX / OY unrolling is not in spatial_mapping_hint
        if len(target_layer_dim) == 0:
            return spatial_mapping

        # ###########################################
        # Get existed mapping size on act_served_oa_dim, which will be added with OX, OY later.
        if act_served_oa_dim in spatial_mapping:  # there already is sm loop
            mapping_served_dim = spatial_mapping[act_served_oa_dim]
            exist_act_loop_size = mapping_served_dim.get_utilization()
        else:  # there is no sm loop mapped on act served dim
            exist_act_loop_size = 1

        # Check if the existed mapping size is more than half of current oa dim size.
        # If so, it means there is no space for extra mapping even with a size of 2.
        # In that case, we will do nothing but return the orignal spatial mapping
        if exist_act_loop_size * 2 > act_served_oa_dim_size:
            return spatial_mapping

        # fetch pr loop pairs for activation, e.g. {"IX": ["OX", "FX"]}
        # TODO is this a valid type cast?
        act_pr_layer_dims: dict[LayerDim, list[LayerDim]] = self.layer.operand_loop_dim[
            act_operand
        ]["pr"]

        # Next we get existed mapping size on output_served_oa_dim
        # there are two classes of mapping:
        # (1) ir mapping to weight, e.g. "C"
        # (2) r mapping to weight, e.g. "FX", "FY" (kernel size)

        # We firstly create a dict for later recording down existed r mapping to weight
        # it will be like:
        # weight_r_loop = {"OX": {"FX": 1}, "OY": {"FY": 1}}
        weight_r_loop: dict[LayerDim, dict[LayerDim, int]] = (
            {}
        )  # here we put a nested dict for recording
        loops_name_for_kernel_size: list[LayerDim] = []
        pr_sm_link: dict[LayerDim, LayerDim] = (
            {}
        )  # here we record down the link between pr loops, e.g. link["FX"]="OX"

        for weight_ir_layer_dim in weight_ir_layer_dims:
            for [layer_dim1, layer_dim2] in act_pr_layer_dims.values():
                if weight_ir_layer_dim in [layer_dim1, layer_dim2]:
                    break
            # as we are unsure in act_pr_layer_dims, it is [OX, FX] or [FX, OX], we consider two possibilities.
            if layer_dim1 == weight_ir_layer_dim:  # if the first one is OX / OY
                weight_r_loop[layer_dim1] = {layer_dim2: 1}  # 1 by default
                loops_name_for_kernel_size.append(layer_dim2)
                pr_sm_link[layer_dim2] = layer_dim1
            else:  # layer_dim2 == weight_ir_layer_dim, the second one is OX / OY
                weight_r_loop[layer_dim2] = {layer_dim1: 1}  # 1 by default
                loops_name_for_kernel_size.append(layer_dim1)
                pr_sm_link[layer_dim1] = layer_dim2

        # Next we will update the dict, and also find the mapping size (weight ir loop size) we do not care out.
        weight_ir_loop_size = 1  # default value
        sm_loop = spatial_mapping[output_served_oa_dim]
        for layer_dim, unroll_factor in sm_loop.items():
            if layer_dim in loops_name_for_kernel_size:  # layer_dim in ["FX", "FY"]
                paired_pr_layer_dim = pr_sm_link[layer_dim]  # "FX" -> "OX", "FY" -> "OY"
                weight_r_loop[paired_pr_layer_dim][layer_dim] *= unroll_factor
            else:  # not care
                weight_ir_loop_size *= unroll_factor

        # At this point, we already know what sm mapping existed.
        # ###########################################

        # Next we will try to add possible OX / OY mapping
        # find all possible OX / OY mapping breakdown and put them in the pool
        # It looks like:
        # sm_pools = {"OX": [("OX",2),("OX",5),("OX",5)], "OY": [("OY",2),("OY",5),("OY",5)]}
        sm_pools_to_add: dict[LayerDim, list[tuple[LayerDim, int]]] = {}
        for layer_dim in target_layer_dim:
            layer_size = self.layer.loop_dim_size[layer_dim.name]
            layer_size_breakdown: list[int] = self.prime_factors(layer_size)

            # try to find the maximum OX / OY and add it to the list
            # (1) check on act_served_oa_dim (ceil down to integer)
            max_allowed_dim_size_on_act_served_dim = math.floor(
                act_served_oa_dim_size / exist_act_loop_size
            )
            # (2) check on output_served_oa_dim
            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
            for key in weight_r_loop:
                if key != layer_dim:
                    ir_layer_dim_to_current_layer_dim = key
            existed_pr_mapping_but_ir_to_current_layer_dim = list(
                weight_r_loop[ir_layer_dim_to_current_layer_dim].values()
            )[0]
            max_allowed_dim_size_on_output_served_dim = (
                output_served_oa_dim_size
                / weight_ir_loop_size
                / existed_pr_mapping_but_ir_to_current_layer_dim
            ) - (existed_pr_mapping - 1)
            # ceil down to integer
            max_allowed_dim_size_on_output_served_dim = math.floor(
                max_allowed_dim_size_on_output_served_dim
            )
            max_allowed_target_dim_size = min(
                max_allowed_dim_size_on_act_served_dim,
                max_allowed_dim_size_on_output_served_dim,
            )
            # check whether the element in layer_size_breakdown is allowed to add
            legal_layer_size_breakdown = []
            for factor in layer_size_breakdown:
                if (
                    factor <= max_allowed_target_dim_size
                    and factor <= layer_dim_size_remainder[layer_dim]
                ):
                    legal_layer_size_breakdown.append(factor)
            if len(legal_layer_size_breakdown) > 0:
                sm_pools_to_add[layer_dim] = [
                    (layer_dim, size) for size in legal_layer_size_breakdown
                ]

        # check if there is anything in the pool
        if len(sm_pools_to_add) == 0:
            return spatial_mapping

        # Generate possible combination
        # In the for loop below, we will first try only with OX or OY. Then with their combination.
        # In the end, we will only keep the best one, which has the maximal value of OX*OY.
        # If there are multiple combs having the same OX*OY, we will keep the first one, as their cost are the same.
        best_comb = []  # list initialization
        best_comb_size = 0  # reference value to find the best comb
        target_layer_dim = list(filter(lambda x: x in sm_pools_to_add, target_layer_dim))
        allowed_dim_comb_length = (
            len(target_layer_dim) if self.enable_mix_spatial_mapping_generation else 1
        )

        for dim_comb_length in range(1, allowed_dim_comb_length + 1):
            for dim_comb in itertools.combinations(target_layer_dim, dim_comb_length):
                # we will create a temporal pools for each dim combination
                sm_pools_mix: list[tuple[LayerDim, int]] = []
                for layer_dim in dim_comb:
                    sm_pools_mix += sm_pools_to_add[layer_dim]
                max_comb_length = len(sm_pools_mix)  # the max possible length of combination
                for comb_length in range(1, max_comb_length + 1):
                    for comb in itertools.combinations(sm_pools_mix, comb_length):
                        # At this point, in comb, we have a possible OX / OY mapping
                        # First we get current comb size
                        # Example: comb_mapping = {"OX": 5, "OY", 10}
                        comb_mapping: dict[LayerDim, int] = {}
                        for layer_dim in dim_comb:
                            comb_mapping[layer_dim] = 1  # default value
                        for element in comb:
                            layer_dim = element[0]
                            mapping_size = element[1]
                            comb_mapping[layer_dim] *= mapping_size
                        # Skip if current unrolling on a layer_dim is 1, which means it has been checked already.
                        curr_comb_already_checked = False
                        for unroll_size in comb_mapping.values():
                            if unroll_size == 1:
                                curr_comb_already_checked = True
                                break
                        if curr_comb_already_checked:
                            continue
                        # We will check if this comb is possible
                        # (1) check on layer_dim_size_remainder
                        curr_comb_illegal = False
                        for unroll_dim, unroll_size in comb_mapping.items():
                            if unroll_size > layer_dim_size_remainder[unroll_dim]:
                                curr_comb_illegal = True
                                break
                        if curr_comb_illegal:
                            continue
                        # (2) check on act_served_oa_dim
                        comb_size = math.prod([v for v in comb_mapping.values()])
                        required_oa_dim_size = exist_act_loop_size * comb_size
                        if required_oa_dim_size > act_served_oa_dim_size:
                            continue  # the comb is not possible on act_served_oa_dim
                        # (3) check on output_served_oa_dim
                        required_oa_dim_size = weight_ir_loop_size
                        for layer_dim, pr_mapping_to_add in comb_mapping.items():
                            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
                            new_mapping_size = existed_pr_mapping + pr_mapping_to_add - 1
                            required_oa_dim_size *= new_mapping_size
                        if len(comb_mapping) == 1:  # only OX or OY
                            # add the other existed pr loop to required_oa_dim_size,
                            # because previously it is not counted in output_served_oa_dim_size.
                            sole_dim = list(comb_mapping.keys())[0]
                            the_other_pr_mapping_name = [
                                key for key in weight_r_loop.keys() if key != sole_dim
                            ][0]
                            the_other_pr_mapping_size = list(
                                weight_r_loop[the_other_pr_mapping_name].values()
                            )[0]
                            required_oa_dim_size *= the_other_pr_mapping_size
                        if required_oa_dim_size > output_served_oa_dim_size:
                            continue  # this comb is not possible on output_served_oa_dim
                        # (4) compare with best_comb
                        if comb_size > best_comb_size:
                            # reformat the comb and merge repetitive elements
                            # example: (("OX", 5), ("OY", 2))
                            new_comb: list[tuple[LayerDim, int]] = [
                                (layer_dim, mapping_size)
                                for (layer_dim, mapping_size) in comb_mapping.items()
                            ]
                            best_comb = new_comb

        # At this point, we get the best possible comb to add. Then we can add that to the current sm mapping
        # there already is sm loop previously
        if act_served_oa_dim in spatial_mapping:
            curr_mapping = spatial_mapping[act_served_oa_dim]
            for layer_dim, unroll in best_comb:
                curr_mapping[layer_dim] = unroll
        else:
            # Initialize new mapping
            spatial_mapping[act_served_oa_dim] = MappingSingleOADim(
                {layer_dim: unroll for layer_dim, unroll in best_comb}
            )

        return spatial_mapping

    # @staticmethod
    # def all_unique(items):
    #     return len(set(items)) == len(items)

    @staticmethod
    def prime_factors(n: int) -> list:
        # non-prime number decomposition
        assert n > 0, "Number for prime decomposition must be a positive integer"
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    # @staticmethod
    # def is_nested_tuple(obj):
    #     if isinstance(obj, tuple):
    #         for item in obj:
    #             if isinstance(item, tuple):
    #                 # If any item within the tuple is itself a tuple, it's a nested tuple
    #                 return True
    #     return False

    @staticmethod
    def identify_layer_operand_representation(layer):
        # activation representation: list (conv layers)
        act_operand = [
            operand
            for operand in layer.operand_loop_dim.keys()
            if len(layer.operand_loop_dim[operand]["pr"]) > 0
        ]
        if len(act_operand) == 0:  # true for fully-connected (fc) layers
            # weight representation (fc layers)
            const_operand = [
                operand
                for operand in layer.operand_loop_dim.keys()
                if len(layer.operand_loop_dim[operand]["ir"]) == 0
            ][0]
            # activation representation (fc layers)
            act_operand = [operand for operand in layer.input_operands if operand != const_operand][
                0
            ]
        else:
            act_operand = act_operand[0]
            # weight representation (conv layers)
            const_operand = [operand for operand in layer.input_operands if operand != act_operand][
                0
            ]
        return act_operand, const_operand
