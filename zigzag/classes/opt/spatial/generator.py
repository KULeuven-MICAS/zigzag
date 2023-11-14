from typing import Set
import itertools

from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.dimension import Dimension
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_array import OperationalArray

import math


## Class that generates valid user-format spatial mappings.
class UserSpatialMappingGenerator:
    ## The class constructor
    # @param layer
    # @param accelerator
    def __init__(
        self,
        layer,
        accelerator,
        defined_mapping=None,
        enable_mix_spatial_mapping_generation=False,
        maximize_hardware_utilization=True,
        enable_weight_diagonal_mapping=False,
    ) -> None:
        self.layer = layer
        self.accelerator = accelerator
        self.defined_mapping = defined_mapping
        self.enable_mix_spatial_mapping_generation = (
            enable_mix_spatial_mapping_generation
        )
        self.maximize_hardware_utilization = maximize_hardware_utilization
        self.enable_weight_diagonal_mapping = enable_weight_diagonal_mapping

    def run(self):
        return self.generate_user_spatial_mappings(
            enable_mix_spatial_mapping_generation=self.enable_mix_spatial_mapping_generation,
            maximize_hardware_utilization=self.maximize_hardware_utilization,
            enable_weight_diagonal_mapping=self.enable_weight_diagonal_mapping,
        )

    ## Generator that yields user-defined spatial mappings.
    # User-defined means across operational array dimensions.
    # For example, this might yield {'D1': (C, 16), 'D2': (K,16)}
    # In essence it works as follows:
    # \code{.py}
    # for each operational array dimension oa_dim (D1, D2, ...):
    #      for each layer operand layer_op (W, I, O, ...):
    #       if oa_dim not in served_dimensions(layer_op):
    #           continue
    #       else:
    #           for layer dimensions layer_dim (B, K, ...) in the layer:
    #               if layer_dim is irrelevant for layer_op:
    #                   layer_dim can be unrolled maximally
    #                 if layer_dim is not irrelevant for layer_op:
    #                   layer_dim can be unrolled if the BW allows it (assumes flexible "bus" reads)
    # \endcode
    def generate_user_spatial_mappings(
        self,
        enable_mix_spatial_mapping_generation,
        maximize_hardware_utilization,
        enable_weight_diagonal_mapping,
    ):
        core_id = self.layer.core_allocation
        core: Core = self.accelerator.get_core(core_id=core_id)
        operational_array: OperationalArray = core.operational_array
        oa_dims = operational_array.dimensions
        oa_dims_copy = operational_array.dimensions.copy()
        memory_hierarchy: MemoryHierarchy = core.memory_hierarchy
        innermost_levels = memory_hierarchy.get_inner_memories()
        defined_mapping = self.defined_mapping
        user_spatial_mapping_hint = self.layer.user_spatial_mapping_hint

        # For every operational array dimension, we initialize it by maximally unrolling all layer dimensions.
        # Later these will be restricted if the memory structure doesn't allow for this unrolling

        if defined_mapping is not None:
            for oa_dim in oa_dims:
                if defined_mapping.get(oa_dim.name) is not None:
                    oa_dims_copy.remove(oa_dim)
        oa_dim_unrolling = {
            oa_dim: {
                layer_dim: int(min(layer_size, oa_dim.size))
                for layer_dim, layer_size in self.layer.loop_dim_size.items()
            }
            for oa_dim in oa_dims_copy
        }

        for memory_level in innermost_levels:
            served_dimensions: Set[Dimension] = memory_level.served_dimensions
            mem_ops = memory_level.operands
            for mem_op in mem_ops:
                layer_op = self.layer.get_layer_operand(
                    mem_op=mem_op
                )  # get the layer operand
                if layer_op == "O":
                    mem_bandwidth = (
                        memory_level.write_bw
                    )  # partial outputs are written to the memory
                else:
                    mem_bandwidth = (
                        memory_level.read_bw
                    )  # inputs are read from the memory
                precision = self.layer.operand_precision[
                    layer_op
                ]  # bit precision of layer operand
                irrelevant_dimensions = self.layer.get_operand_irrelevant_dimensions(
                    layer_op
                )
                for oa_dim in oa_dims_copy:
                    if oa_dim not in served_dimensions:
                        continue
                    # If the operational array dimension is a served dimension of the lowest memory level,
                    # we ought to limit the unrolling for the relevant and partially relevant loop dimensions
                    for layer_dim, unrolling_size in oa_dim_unrolling[oa_dim].items():
                        if layer_dim in irrelevant_dimensions:
                            continue
                        # If not irrelevant, it is (partially) relevant. Limit based on BW and operand precision.
                        try:
                            max_multicast_elements = mem_bandwidth // precision
                        except ZeroDivisionError:
                            max_multicast_elements = unrolling_size
                        oa_dim_unrolling[oa_dim][layer_dim] = min(
                            max_multicast_elements, unrolling_size
                        )

        # At this point the unrolled layer dimensions are maximal (wrt the served dimensions and bandwidth of the lowest memory level).
        # The unrolling size might not be a factor of the layer dimension size, which is required (for non greedy mapping).
        # Convert the unrolling size to be a factor of the layer dimension size. At the same time convert them to a list.
        unrollings = []
        for oa_dim in oa_dims:
            # If the mapping is user-provided, it will be directly stored in the pool, instead of being generated.
            if (
                defined_mapping is not None
                and defined_mapping.get(oa_dim.name) is not None
            ):
                # scale down the defined_mapping size if it exceeds the layer dim size
                ori_loop = defined_mapping.get(oa_dim.name)
                loop_to_reform = []
                if self.is_nested_tuple(ori_loop):  # mix sm loop
                    for sub_loop in ori_loop:
                        sub_loop_dim = sub_loop[0]
                        sub_loop_size = sub_loop[1]
                        if sub_loop_dim in self.layer.loop_dim_size.keys():
                            if sub_loop_size > self.layer.loop_dim_size[sub_loop_dim]:
                                sub_loop_size = self.layer.loop_dim_size[sub_loop_dim]
                            loop_to_reform.append((sub_loop_dim, sub_loop_size))
                else:  # single layer sm loop
                    loop_dim = ori_loop[0]
                    loop_size = ori_loop[1]
                    if loop_dim in self.layer.loop_dim_size.keys():
                        if loop_size > self.layer.loop_dim_size[loop_dim]:
                            loop_size = self.layer.loop_dim_size[loop_dim]
                        loop_to_reform.append((loop_dim, loop_size))
                loop_to_reform = tuple(loop_to_reform)
                if len(loop_to_reform) == 0:
                    loop_to_reform = None
                oa_dim_unrollings = [loop_to_reform]
            else:
                oa_dim_unrollings = []
                oa_dim_unrolling_hints = user_spatial_mapping_hint[oa_dim.name]
                for layer_dim, unrolling_size in oa_dim_unrolling[oa_dim].items():
                    layer_dim_size = self.layer.loop_dim_size[layer_dim]
                    # If e.g. the unrolling size is 10 (because operational array dimension size is 10)
                    # but the layer dimension size is 14, this would result in a temporal remainder of 14/10.
                    # In that case we change the unrolling size to 7 (to be a factor of 14).
                    # We have to make sure the unrolling size is a divisor of the layer dimension size:
                    # Jan 18 2023: Commented this out as LomaStage allows greedy mapping by adding one more temporal iteration
                    # while layer_dim_size % unrolling_size != 0:
                    #     unrolling_size -= 1  # decrement the unrolling by 1

                    # If the unrolling_size is not 1 and the layer dim is in the user-provided hints,
                    # add it to the unrollings for this oa_dim
                    if unrolling_size != 1 and layer_dim in oa_dim_unrolling_hints:
                        oa_dim_unrollings.append((layer_dim, unrolling_size))

                if enable_mix_spatial_mapping_generation:
                    # Now all unrollings in oa_dim_unrollings are for single layer dimension.
                    # If mix spatial mapping is enabled, we will add the mix unrollings to oa_dim_unrollings next.
                    oa_dim_unrollings = self.append_mix_spatial_unrollings(
                        provided_oa_dim_unrollings=oa_dim_unrollings,
                        provided_oa_dim_unrolling_hints=oa_dim_unrolling_hints,
                        oa_dim=oa_dim,
                    )
                if maximize_hardware_utilization:
                    # Sort oa_dim_unrollings so values follow a descending order.
                    (
                        oa_dim_unrollings,
                        hardware_utilization,
                    ) = self.sort_oa_dim_unrollings_in_the_order_of_utilization(
                        oa_dim_unrollings, descending=True
                    )

                # In case there are no unrollings (of size > 1) possible, add a single unrolling of size 1.
                # The loop dimension we pick is randomly chosen as the first loop dimension in the layer.
                # The loop dimension chosen shouldn't matter as the size of unrolling is 1 anyway.
                if len(oa_dim_unrollings) == 0:
                    oa_dim_unrollings.append(None)

            unrollings.append(oa_dim_unrollings)

        # Now we have for each operational array dimension the layer dimensions and size they can be unrolled without fractional remainder.
        # Now we have to combine them into user-defined spatial mappings.
        # record down the number of yield
        yield_count = 0
        yield_count_limit = 2  # used to control the yield count when maximize_hardware_utilization == True
        for combination in itertools.product(*unrollings):
            if maximize_hardware_utilization and yield_count >= yield_count_limit:
                # 2 means: only check the top 2 spatial mapping with the highest hardware utilization
                # Modify "2" to other numbers if you want to check on more spatial mappings.
                break

            legal_spatial_loop, left_layer_dim_size = self.check_spatial_loop_legality(
                combination=combination, layer=self.layer
            )
            if not legal_spatial_loop:
                continue
            # Zip the combination (which is a (layer_dim, layer_size) for each oa_dim with the oa_dim names.
            oa_dim_names = [oa_dim.name for oa_dim in oa_dims]

            user_spatial_mapping = {
                oa_dim_name: unrolling
                for (oa_dim_name, unrolling) in zip(oa_dim_names, combination)
                if unrolling is not None
            }
            # Add act ir loop if it is weight stationary and the innermost memories serve for act.
            if enable_weight_diagonal_mapping:
                user_spatial_mapping = self.add_input_pr_spatial_loop_if_enabled(
                    layer=self.layer,
                    provided_user_spatial_mapping=user_spatial_mapping,
                    user_spatial_mapping_hint=user_spatial_mapping_hint,
                    innermost_levels=innermost_levels,
                    left_layer_dim_size=left_layer_dim_size,
                    enable_mix_spatial_mapping_generation=enable_mix_spatial_mapping_generation,
                )
            yield user_spatial_mapping
            yield_count += 1
        # If yield_count==0, it means there is no legal spatial mapping found.
        # One reason is that the spatial mapping provided by the user has exceeded the layer dim size,
        # therefore the loop cannot pass the check.
        # The other reason could be: there is a layer dim mapped on multiple oa dims,
        # so the product has exceeded the layer dim size.
        # For a quick fix on the second cause, we will reform the sm loop only for single layer dim mapping.
        if yield_count == 0:
            for combination in itertools.product(*unrollings):
                is_mix_comb = False
                for loop in combination:
                    if self.is_nested_tuple(loop):
                        is_mix_comb = True
                        continue
                if is_mix_comb:
                    # The fix is not applied for mix sm loop.
                    continue
                if maximize_hardware_utilization and yield_count >= yield_count_limit:
                    # 2 means: only check the top 2 spatial mapping with the highest hardware utilization
                    # Modify "2" to other numbers if you want to check on more spatial mappings.
                    break
                (
                    new_combination,
                    left_layer_dim_size,
                ) = self.shrink_combination_when_a_layer_dim_is_mapped_on_multiple_oa_dims(
                    combination=combination,
                    layer=self.layer,
                )
                # Zip the combination (which is a (layer_dim, layer_size) for each oa_dim with the oa_dim names.
                oa_dim_names = [oa_dim.name for oa_dim in oa_dims]

                user_spatial_mapping = {
                    oa_dim_name: unrolling
                    for (oa_dim_name, unrolling) in zip(oa_dim_names, new_combination)
                    if unrolling is not None
                }
                # Add act ir loop if it is weight stationary and the innermost memories serve for act.
                if enable_weight_diagonal_mapping:
                    user_spatial_mapping = self.add_input_pr_spatial_loop_if_enabled(
                        layer=self.layer,
                        provided_user_spatial_mapping=user_spatial_mapping,
                        user_spatial_mapping_hint=user_spatial_mapping_hint,
                        innermost_levels=innermost_levels,
                        left_layer_dim_size=left_layer_dim_size,
                        enable_mix_spatial_mapping_generation=enable_mix_spatial_mapping_generation,
                    )
                yield user_spatial_mapping
                yield_count += 1

        assert (
            yield_count > 0
        ), "There is no legal spatial mapping found. Please make sure the provided spatial mappings do not exceed the layer dimension size."

    def shrink_combination_when_a_layer_dim_is_mapped_on_multiple_oa_dims(
        self, combination, layer
    ):
        new_combination = combination
        legal_spatial_loop, left_layer_dim_size = self.check_spatial_loop_legality(
            combination=new_combination, layer=layer
        )
        while not legal_spatial_loop:
            new_combination_next = list(new_combination)
            for layer_dim, layer_dim_size in left_layer_dim_size.items():
                if layer_dim_size < 1:
                    scaled_success = False
                    for oa_index in range(
                        len(new_combination_next) - 1, -1, -1
                    ):  # reverse order on oa dims
                        (
                            mapped_layer_dim,
                            mapped_layer_dim_size,
                        ) = new_combination_next[oa_index]
                        if mapped_layer_dim_size > 1:
                            # shrink the mapped layer dim size
                            mapped_layer_dim_size -= 1
                            new_combination_next[oa_index] = (
                                mapped_layer_dim,
                                mapped_layer_dim_size,
                            )
                            scaled_success = True
                            break
                        else:
                            # because a layer can be mapped on multiple oa dims, we will move to the next oa dim.
                            pass
                    # assert: if not scaled_success,
                    # it means the sm loop cannot pass the check, even though all mapped size on this layer dim is 1
                    assert scaled_success, (
                        f"The spatial loop cannot meet the current hardware dimension after scaling, "
                        f"Current spatial loop: {new_combination}"
                    )
            new_combination_next = tuple(new_combination_next)
            # Next we will judge if new_combination_next is a legal loop
            # If it is, then we will keep the current combination, rather than new_combination_next,
            # the reason is: new_combination can cover the entire layer dim, but new_combination_next is smaller than
            # the layer dim, therefore the actual sm loop for the layer dim is a decimal number.
            # In that case, we will ceil it up to mimic the real case on hardware.
            (
                legal_spatial_loop,
                left_layer_dim_size_next,
            ) = self.check_spatial_loop_legality(
                combination=new_combination_next, layer=layer
            )
            if not legal_spatial_loop:
                new_combination = new_combination_next
                left_layer_dim_size = left_layer_dim_size_next
            else:
                for layer_dim, layer_dim_size in left_layer_dim_size.items():
                    # A special case when we will use new_combination_next when legal_spatial_loop == True
                    # This case is when new_combination_next exactly match the layer dim size (left size == 1)
                    if layer_dim_size < 1 and left_layer_dim_size_next[layer_dim] == 1:
                        new_combination = new_combination_next
                        left_layer_dim_size = left_layer_dim_size_next
                        break
        return new_combination, left_layer_dim_size

    def check_spatial_loop_legality(self, combination, layer):
        # Extra check on the total unrolling size of a layer dim, if it is mapped on >=2 dimensions.
        combination_check = {
            layer_dim: layer_size
            for layer_dim, layer_size in layer.loop_dim_size.items()
        }
        legal_spatial_loop = True  # initialization
        for unrolling_in_combination in combination:
            if unrolling_in_combination is None:
                continue
            if self.is_nested_tuple(unrolling_in_combination):
                for sub_unrolling_in_combination in unrolling_in_combination:
                    unrolling_layer_dim = sub_unrolling_in_combination[0]
                    unrolling_layer_size = sub_unrolling_in_combination[1]
                    if unrolling_layer_dim in combination_check.keys():
                        combination_check[unrolling_layer_dim] /= unrolling_layer_size
                    else:
                        # The unrolled layer dim does not exist in current layer.
                        # This only happens when the spatial mapping is user-defined, which
                        # contains non-existent layer dims in current layer.
                        pass
            else:
                unrolling_layer_dim = unrolling_in_combination[0]
                unrolling_layer_size = unrolling_in_combination[1]
                if unrolling_layer_dim in combination_check.keys():
                    combination_check[unrolling_layer_dim] /= unrolling_layer_size
                else:
                    # The unrolled layer dim does not exist in current layer.
                    # This only happens when the spatial mapping is user-defined, which
                    # contains non-existent layer dims in current layer.
                    pass
        for layer_dim, layer_size in combination_check.items():
            if layer_size < 1:  # the layer size/the unrolling size < 1
                # It means the unrolling size > the layer size, which is incorrect and impossible.
                legal_spatial_loop = False
                break
        return legal_spatial_loop, combination_check

    def append_mix_spatial_unrollings(
        self, provided_oa_dim_unrollings, provided_oa_dim_unrolling_hints, oa_dim
    ):
        # Create and append new mix spatial unrollings to original oa_dim_unrollings
        # An example of mix: (("K",2), ("OX", 2))
        oa_dim_unrollings = provided_oa_dim_unrollings
        oa_dim_unrolling_hints = provided_oa_dim_unrolling_hints
        if (
            len(oa_dim_unrollings) >= 2 and len(oa_dim_unrolling_hints) >= 2
        ):  # a mix of at least 2 layer dimension
            oa_dim_unrollings_further = []
            for (
                layer_dim,
                unrolling_size,
            ) in (
                oa_dim_unrollings
            ):  # to decompose the existed layer dimension into primes
                unrolling_size_breakdown = self.prime_factors(unrolling_size)
                # unrolling_size_breakdown = self.prime_factors(
                #     self.layer.loop_dim_size[layer_dim])  # NOTE: loop_dim_size
                oa_dim_unrollings_further += [
                    tuple([layer_dim, unrolling_size_further])
                    for unrolling_size_further in unrolling_size_breakdown
                ]

            for dim_comb_length in range(
                2, len(oa_dim_unrolling_hints) + 1
            ):  # different combination length of layer dimensions
                for dim_comb in itertools.combinations(
                    oa_dim_unrolling_hints, dim_comb_length
                ):  # different combination of layer dimensions
                    oa_dim_unrollings_comb_pool = [
                        element
                        for element in oa_dim_unrollings_further
                        if element[0] in dim_comb
                    ]
                    for comb_length in range(
                        2, len(oa_dim_unrollings_comb_pool) + 1
                    ):  # different combination length of unrolling elements
                        for comb in itertools.combinations(
                            oa_dim_unrollings_comb_pool, comb_length
                        ):  # different combination of unrolling elements
                            if (
                                len(set([element[0] for element in comb])) == 1
                            ):  # skip if layer_dim is all the same, as it already exists in oa_dim_unrollings
                                continue
                            if (
                                math.prod([element[1] for element in comb])
                                <= oa_dim.size
                            ):
                                # check if there are repetitive unrolling representation, e.g. (('K', 2), ('K', 2))
                                if not self.all_unique(
                                    [element[0] for element in comb]
                                ):
                                    # merge repetitive unrolling representation
                                    merged_comb = {}
                                    for key, value in comb:
                                        if key in merged_comb:
                                            merged_comb[key] *= value
                                        else:
                                            merged_comb[key] = value
                                    merged_comb = tuple(
                                        [
                                            (key, value)
                                            for key, value in merged_comb.items()
                                        ]
                                    )
                                    #############
                                    # Since the merger will cause the comb is entirely superior to one existing comb,
                                    # the block will kick out that inferior comb.
                                    # e.g. ((K,2), (K,2), (C,2)) becomes ((K,4), (C,2)) after the merger
                                    # since ((K,2), (C,2)) also exist in oa_dim_unrollings,
                                    # We want to kick out ((K,2), (C,2)) and only keep ((K,4), (C,2))
                                    len_merged_comb = len(merged_comb)
                                    for comb_index in range(0, len(oa_dim_unrollings)):
                                        curr_unrolling_to_check = oa_dim_unrollings[
                                            comb_index
                                        ]
                                        # first, find out all mix combs with the same length
                                        if (
                                            self.is_nested_tuple(
                                                curr_unrolling_to_check
                                            )
                                            and len(curr_unrolling_to_check)
                                            == len_merged_comb
                                        ):
                                            # second, find out combs with the same layer_dim
                                            combs_key_exactly_match = True
                                            curr_unroll_dims = [
                                                unroll[0]
                                                for unroll in curr_unrolling_to_check
                                            ]
                                            curr_unroll_sizes = [
                                                unroll[1]
                                                for unroll in curr_unrolling_to_check
                                            ]
                                            for (
                                                comb_layer_dim,
                                                comb_unroll_size,
                                            ) in merged_comb:
                                                if (
                                                    comb_layer_dim
                                                    not in curr_unroll_dims
                                                ):
                                                    combs_key_exactly_match = False
                                            # third, compare the mapped layer_dim size
                                            kick_out_curr_unrolling = True
                                            if combs_key_exactly_match:
                                                for (
                                                    comb_layer_dim,
                                                    comb_unroll_size,
                                                ) in merged_comb:
                                                    unroll_dim_index = (
                                                        curr_unroll_dims.index(
                                                            comb_layer_dim
                                                        )
                                                    )
                                                    curr_unroll_size = (
                                                        curr_unroll_sizes[
                                                            unroll_dim_index
                                                        ]
                                                    )
                                                    if (
                                                        comb_unroll_size
                                                        < curr_unroll_size
                                                    ):
                                                        kick_out_curr_unrolling = False
                                            # fourth, kick out the current unrolling if it is inferior
                                            if kick_out_curr_unrolling:
                                                del oa_dim_unrollings[comb_index]
                                                break
                                        else:
                                            pass
                                    #############
                                    result_comb = merged_comb
                                else:
                                    result_comb = comb
                                if (
                                    result_comb not in oa_dim_unrollings
                                ):  # avoid there are repetitive comb
                                    oa_dim_unrollings.append(result_comb)
        return oa_dim_unrollings

    def sort_oa_dim_unrollings_in_the_order_of_utilization(
        self, provided_oa_dim_unrollings, descending=True
    ):
        # Sort the found unrollings in the order of the hardware dimension utilization.
        # @param descending:
        #                 True -- the higher the mapping utilization is, the closer to the front it is.
        #                 False -- the lower the mapping utilization is, the closer to the front it is.
        oa_dim_unrollings = provided_oa_dim_unrollings
        if len(oa_dim_unrollings) > 1:
            # First we will record down the hardware utilization of each spatial unrolling in comb_value
            comb_value = []  # record down the value of each combination
            for comb in oa_dim_unrollings:
                if self.is_nested_tuple(comb):  # the comb is a mix sm loop
                    comb_value.append(math.prod([element[1] for element in comb]))
                else:  # the comb is a single-dim sm loop
                    comb_value.append(comb[1])
            # Next, we will append the index of each unrolling
            indexed_comb_value = list(
                enumerate(comb_value)
            )  # record down the index information of comb
            # Then, we will sort the values in order, depending on if @param "descending" is True or False
            sorted_comb_value = sorted(
                indexed_comb_value, key=lambda x: x[1], reverse=descending
            )  # sort in descending order if True
            # After, we will record down the original index of each unrolling, as in the sorted order
            descending_comb_index = [
                x[0] for x in sorted_comb_value
            ]  # index of combs with their values in descending order
            # Finally, we fetch the value corresponding to each index as in the sorted order
            oa_dim_unrollings = [
                oa_dim_unrollings[x] for x in descending_comb_index
            ]  # new list with a descending order
            # We will also return the hardware utilization info
            hardware_utilization = [x[1] for x in sorted_comb_value]
        elif len(oa_dim_unrollings) == 1:
            # If there is only one spatial mapping found, no need to reorder it.
            # We will return 1 as hardware_utilization.
            hardware_utilization = [1]
        else:
            # If there is an empty spatial mapping found, no need to reorder it.
            # We will return None as hardware_utilization.
            hardware_utilization = None
        return oa_dim_unrollings, hardware_utilization

    def add_input_pr_spatial_loop_if_enabled(
        self,
        layer,
        provided_user_spatial_mapping,
        user_spatial_mapping_hint,
        innermost_levels,
        left_layer_dim_size,
        enable_mix_spatial_mapping_generation,
    ):
        # This function is used to support diagonal spatial mapping
        # when input/activation is served in the innermost memories and the weight is stationary.
        user_spatial_mapping = provided_user_spatial_mapping
        # get the link from layer op to mem op
        layer_op_to_mem_op: dict = layer.memory_operand_links
        # check if it is weight stationary.
        # keep the spatial loop as it was if it is not weight stationary.
        if len(layer.constant_operands) > 1:
            return user_spatial_mapping
        # get weight operand name
        const_operand = layer.constant_operands[0]  # weight representation
        # get activation operand name
        act_operand = [
            operand for operand in layer.input_operands if operand != const_operand
        ][0]
        # get output operand name
        output_operand = layer.output_operand
        # get name of OX, OY (weight ir layer dims)
        weight_ir_layer_dims: list = layer.operand_loop_dim[const_operand]["ir"]
        # get the oa_dim name served by input / output innermost memory level
        for memory_level in innermost_levels:
            mem_ops = memory_level.operands
            if layer_op_to_mem_op[act_operand] in mem_ops:
                act_served_oa_dim: set = memory_level.served_dimensions
            if layer_op_to_mem_op[output_operand] in mem_ops:
                output_served_oa_dim: set = memory_level.served_dimensions
        # check if act is not served in the innermost memories, or act/output is not multicasting on only one dimension.
        # keep the spatial loop as it was if act is not served.
        if "act_served_oa_dim" not in locals() or len(act_served_oa_dim) != 1:
            return user_spatial_mapping
        if "output_served_oa_dim" not in locals() or len(output_served_oa_dim) != 1:
            return user_spatial_mapping

        act_served_oa_dim_name = list(act_served_oa_dim)[0].name
        output_served_oa_dim_name = list(output_served_oa_dim)[0].name
        act_served_oa_dim_size = list(act_served_oa_dim)[0].size
        output_served_oa_dim_size = list(output_served_oa_dim)[0].size

        # check if OX / OY in user_spatial_mapping_hint. Or else target_layer_dim will be empty.
        target_layer_dim = []  # OX or OY or both
        for layer_dim in weight_ir_layer_dims:
            if layer_dim in user_spatial_mapping_hint[act_served_oa_dim_name]:
                target_layer_dim.append(layer_dim)

        # no further execution if OX / OY unrolling is not in user_spatial_mapping_hint
        if len(target_layer_dim) == 0:
            return user_spatial_mapping

        ############################################
        # Get existed mapping size on act_served_oa_dim, which will be added with OX, OY later.
        if (
            act_served_oa_dim_name in user_spatial_mapping.keys()
        ):  # there already is sm loop
            sm_loop = user_spatial_mapping[act_served_oa_dim_name]
            if self.is_nested_tuple(sm_loop):  # a mix layer sm mapping
                exist_act_loop_size = 1
                for element in sm_loop:
                    exist_act_loop_size *= element[1]
            else:  # a single layer sm mapping
                exist_act_loop_size = sm_loop[1]
        else:  # there is no sm loop mapped on act served dim
            exist_act_loop_size = 1

        # Check if the existed mapping size is more than half of current oa dim size.
        # If so, it means there is no space for extra mapping even with a size of 2.
        # In that case, we will do nothing but return the orignal spatial mapping
        if exist_act_loop_size * 2 > act_served_oa_dim_size:
            return user_spatial_mapping

        # fetch pr loop pairs for activation, e.g. {"IX": ["OX", "FX"]}
        act_pr_layer_dims: dict = layer.operand_loop_dim[act_operand]["pr"]

        # Next we get existed mapping size on output_served_oa_dim
        # there are two classes of mapping:
        # (1) ir mapping to weight, e.g. "C"
        # (2) r mapping to weight, e.g. "FX", "FY" (kernel size)

        # We firstly create a dict for later recording down existed r mapping to weight
        # it will be like:
        # weight_r_loop = {"OX": {"FX": 1}, "OY": {"FY": 1}}
        weight_r_loop: dict = {}  # here we put a nested dict for recording
        loops_name_for_kernel_size: list = []
        pr_sm_link: dict = (
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
        sm_loop = user_spatial_mapping[output_served_oa_dim_name]
        if self.is_nested_tuple(sm_loop):  # a mix sm mapping
            for element in sm_loop:
                # save operation as above
                layer_dim = element[0]
                mapping_size = element[1]
                if layer_dim in loops_name_for_kernel_size:  # layer_dim in ["FX", "FY"]
                    paired_pr_layer_dim = pr_sm_link[
                        layer_dim
                    ]  # "FX" -> "OX", "FY" -> "OY"
                    weight_r_loop[paired_pr_layer_dim][layer_dim] *= mapping_size
                else:  # not care
                    weight_ir_loop_size *= mapping_size
        else:  # a single layer sm mapping
            layer_dim = sm_loop[0]
            mapping_size = sm_loop[1]
            if layer_dim in loops_name_for_kernel_size:  # layer_dim in ["FX", "FY"]
                paired_pr_layer_dim = pr_sm_link[
                    layer_dim
                ]  # "FX" -> "OX", "FY" -> "OY"
                weight_r_loop[paired_pr_layer_dim][layer_dim] *= mapping_size
            else:  # not care
                weight_ir_loop_size *= mapping_size

        # At this point, we already know what sm mapping existed.
        ############################################

        # Next we will try to add possible OX / OY mapping
        # find all possible OX / OY mapping breakdown and put them in the pool
        # It looks like:
        # sm_pools = {"OX": [("OX",2),("OX",5),("OX",5)], "OY": [("OY",2),("OY",5),("OY",5)]}
        sm_pools_to_add: dict = {}
        for layer_dim in target_layer_dim:
            layer_size = self.layer.loop_dim_size[layer_dim]
            layer_size_breakdown: list = self.prime_factors(layer_size)

            # try to find the maximum OX / OY and add it to the list
            # (1) check on act_served_oa_dim (ceil down to integer)
            max_allowed_dim_size_on_act_served_dim = math.floor(
                act_served_oa_dim_size / exist_act_loop_size
            )
            # (2) check on output_served_oa_dim
            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
            for key in weight_r_loop.keys():
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
                    and factor <= left_layer_dim_size[layer_dim]
                ):
                    legal_layer_size_breakdown.append(factor)
            if len(legal_layer_size_breakdown) > 0:
                sm_pools_to_add[layer_dim] = [
                    tuple([layer_dim, size]) for size in legal_layer_size_breakdown
                ]

        # check if there is anything in the pool
        if len(sm_pools_to_add) == 0:
            return user_spatial_mapping

        # Generate possible combination
        # In the for loop below, we will first try only with OX or OY. Then with their combination.
        # In the end, we will only keep the best one, which has the maximal value of OX*OY.
        # If there are multiple combs having the same OX*OY, we will keep the first one, as their cost are the same.
        best_comb = []  # list initialization
        best_comb_size = 0  # reference value to find the best comb
        target_layer_dim = [
            layer_dim
            for layer_dim in target_layer_dim
            if layer_dim in sm_pools_to_add.keys()
        ]
        if enable_mix_spatial_mapping_generation:
            allowed_dim_comb_length = len(target_layer_dim)
        else:
            allowed_dim_comb_length = 1
        for dim_comb_length in range(1, allowed_dim_comb_length + 1):
            for dim_comb in itertools.combinations(target_layer_dim, dim_comb_length):
                # we will create a temporal pools for each dim combination
                sm_pools_mix = []
                for layer_dim in dim_comb:
                    sm_pools_mix += sm_pools_to_add[layer_dim]
                max_comb_length = len(
                    sm_pools_mix
                )  # the max possible length of combination
                for comb_length in range(1, max_comb_length + 1):
                    for comb in itertools.combinations(sm_pools_mix, comb_length):
                        # At this point, in comb, we have a possible OX / OY mapping
                        # First we get current comb size
                        # Example: comb_mapping = {"OX": 5, "OY", 10}
                        comb_mapping: dict = {}
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
                        # (1) check on left_layer_dim_size
                        curr_comb_illegal = False
                        for unroll_dim, unroll_size in comb_mapping.items():
                            if unroll_size > left_layer_dim_size[unroll_dim]:
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
                        for layer_dim in comb_mapping.keys():
                            existed_pr_mapping = list(
                                weight_r_loop[layer_dim].values()
                            )[0]
                            pr_mapping_to_add = comb_mapping[layer_dim]
                            new_mapping_size = (
                                existed_pr_mapping + pr_mapping_to_add - 1
                            )
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
                            new_comb: list = [
                                (layer_dim, mapping_size)
                                for (layer_dim, mapping_size) in comb_mapping.items()
                            ]
                            best_comb = new_comb

        # At this point, we get the best possible comb to add. Then we can add that to the current sm mapping
        if len(best_comb) == 0:  # did not find any comb
            return user_spatial_mapping
        else:
            if (
                act_served_oa_dim_name in user_spatial_mapping.keys()
            ):  # there already is sm loop previously
                act_served_mapping_to_change = user_spatial_mapping[
                    act_served_oa_dim_name
                ]
                if self.is_nested_tuple(
                    act_served_mapping_to_change
                ):  # originally it is a mix mapping
                    reformed_sm = list(act_served_mapping_to_change) + best_comb
                else:  # originally it is a single layer mapping
                    reformed_sm = [act_served_mapping_to_change] + best_comb
            else:  # there is no sm loop on act served oa dim previously
                reformed_sm = best_comb
            reformed_sm = tuple(reformed_sm)
            user_spatial_mapping[act_served_oa_dim_name] = reformed_sm

        return user_spatial_mapping

    @staticmethod
    def all_unique(items):
        return len(set(items)) == len(items)

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

    @staticmethod
    def is_nested_tuple(obj):
        if isinstance(obj, tuple):
            for item in obj:
                if isinstance(item, tuple):
                    # If any item within the tuple is itself a tuple, it's a nested tuple
                    return True
        return False
