from typing import Set
import itertools

from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.dimension import Dimension
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_array import OperationalArray

## Class that generates valid user-format spatial mappings.
class UserSpatialMappingGenerator:
    ## The class constructor
    # @param layer
    # @param accelerator
    def __init__(self, layer, accelerator, defined_mapping=None, enable_mix_spatial_mapping=False, maximize_hardware_utilization=True) -> None:
        self.layer = layer
        self.accelerator = accelerator
        self.defined_mapping = defined_mapping
        self.enable_mix_spatial_mapping = enable_mix_spatial_mapping
        self.maximize_hardware_utilization = maximize_hardware_utilization

    def run(self):
        return self.generate_user_spatial_mappings(enable_mix_spatial_mapping=self.enable_mix_spatial_mapping,
                                                   maximize_hardware_utilization=self.maximize_hardware_utilization)

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
    def generate_user_spatial_mappings(self, enable_mix_spatial_mapping, maximize_hardware_utilization):
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
                    for (layer_dim, unrolling_size) in oa_dim_unrolling[oa_dim].items():
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
            if defined_mapping is not None and defined_mapping.get(oa_dim.name) is not None:
                oa_dim_unrollings = [defined_mapping.get(oa_dim.name)]
            else:
                oa_dim_unrollings = []
                oa_dim_unrolling_hints = user_spatial_mapping_hint[oa_dim.name]
                for (layer_dim, unrolling_size) in oa_dim_unrolling[oa_dim].items():
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

                if enable_mix_spatial_mapping:
                    # Now all unrollings in oa_dim_unrollings are for single layer dimension.
                    # If mix spatial mapping is enabled, we will add the mix unrollings to oa_dim_unrollings next.
                    oa_dim_unrollings = self.append_mix_spatial_unrollings(
                        provided_oa_dim_unrollings=oa_dim_unrollings,
                        provided_oa_dim_unrolling_hints=oa_dim_unrolling_hints,
                        oa_dim=oa_dim)
                if maximize_hardware_utilization:
                    # Sort oa_dim_unrollings so values follow a descending order.
                    oa_dim_unrollings = self.sort_oa_dim_unrollings_in_the_order_of_utilization(oa_dim_unrollings,
                                                                                                descending=True)
                    # Then only keep the combs in oa_dim_unrollings that have the highest oa_dim mapping utilization
                    # To reduce the simulation time, only keep the first two unrolling combs for each oa_dim.
                    # The closer to the front, the higher the oa_dim utilization rate.
                    oa_dim_unrollings = oa_dim_unrollings[0:2]

                # In case there are no unrollings (of size > 1) possible, add a single unrolling of size 1.
                # The loop dimension we pick is randomly chosen as the first loop dimension in the layer.
                # The loop dimension chosen shouldn't matter as the size of unrolling is 1 anyway.
                if len(oa_dim_unrollings) == 0:
                    oa_dim_unrollings.append(None)

            unrollings.append(oa_dim_unrollings)

        # Now we have for each operational array dimension the layer dimensions and size they can be unrolled without fractional remainder.
        # Now we have to combine them into user-defined spatial mappings.
        for combination in itertools.product(*unrollings):
            # Zip the combination (which is a (layer_dim, layer_size) for each oa_dim with the oa_dim names.
            oa_dim_names = [oa_dim.name for oa_dim in oa_dims]
            # Extra check on the total unrolling size of a layer dim, if it is mapped on >=2 dimensions.
            combination_check = {
                layer_dim: layer_size
                for layer_dim, layer_size in self.layer.loop_dim_size.items()
            }
            for unrolling_in_combination in combination:
                if unrolling_in_combination is None:
                    continue
                if self.is_nested_tuple(unrolling_in_combination):
                    for sub_unrolling_in_combination in unrolling_in_combination:
                        unrolling_layer_dim = sub_unrolling_in_combination[0]
                        unrolling_layer_size = sub_unrolling_in_combination[1]
                        combination_check[unrolling_layer_dim] /= unrolling_layer_size
                else:
                    unrolling_layer_dim = unrolling_in_combination[0]
                    unrolling_layer_size = unrolling_in_combination[1]
                    combination_check[unrolling_layer_dim] /= unrolling_layer_size
            for (layer_dim, layer_size) in combination_check.items():
                if layer_size < 1:  # the layer size/the unrolling size < 1
                    # It means the unrolling size > the layer size, which is incorrect and impossible.
                    continue

            user_spatial_mapping = {
                oa_dim_name: unrolling
                for (oa_dim_name, unrolling) in zip(oa_dim_names, combination)
                if unrolling is not None
            }
            yield user_spatial_mapping

    def append_mix_spatial_unrollings(self, provided_oa_dim_unrollings, provided_oa_dim_unrolling_hints, oa_dim):
        # Create and append new mix spatial unrollings to original oa_dim_unrollings
        # An example of mix: (("K",2), ("OX", 2))
        import math
        oa_dim_unrollings = provided_oa_dim_unrollings
        oa_dim_unrolling_hints = provided_oa_dim_unrolling_hints
        if len(oa_dim_unrollings) > 0 and len(oa_dim_unrolling_hints) >= 2:  # a mix of at least 2 layer dimension
            oa_dim_unrollings_further = []
            for (layer_dim, unrolling_size) in oa_dim_unrollings: # to decompose the existed layer dimension into primes
                unrolling_size_breakdown = self.prime_factors(
                    self.layer.loop_dim_size[layer_dim])  # NOTE: loop_dim_size
                oa_dim_unrollings_further += [tuple([layer_dim, unrolling_size_further]) for unrolling_size_further in
                                              unrolling_size_breakdown]

            for dim_comb_length in range(2, len(oa_dim_unrolling_hints) + 1):  # different combination length of layer dimensions
                for dim_comb in itertools.combinations(oa_dim_unrolling_hints, dim_comb_length):  # different combination of layer dimensions
                    oa_dim_unrollings_comb_pool = [element for element in oa_dim_unrollings_further if element[0] in dim_comb]
                    for comb_length in range(2, len(oa_dim_unrollings_comb_pool) + 1): # different combination length of unrolling elements
                        for comb in itertools.combinations(oa_dim_unrollings_comb_pool, comb_length): # different combination of unrolling elements
                            if len(set([element[0] for element in comb])) == 1:  # skip if layer_dim is all the same, as it already exists in oa_dim_unrollings
                                continue
                            if math.prod([element[1] for element in comb]) <= oa_dim.size:
                                # check if there are repetitive unrolling representation, e.g. (('K', 2), ('K', 2))
                                if not self.all_unique([element[0] for element in comb]):
                                    # merge repetitive unrolling representation
                                    merged_comb = {}
                                    for (key, value) in comb:
                                        if key in merged_comb:
                                            merged_comb[key] *= value
                                        else:
                                            merged_comb[key] = value
                                    merged_comb = tuple([(key, value) for key, value in merged_comb.items()])
                                    result_comb = merged_comb
                                else:
                                    result_comb = comb
                                if result_comb not in oa_dim_unrollings:  # avoid there are repetitive comb
                                    oa_dim_unrollings.append(result_comb)
        return oa_dim_unrollings

    def sort_oa_dim_unrollings_in_the_order_of_utilization(self, provided_oa_dim_unrollings, descending=True):
        # Sort the found unrollings in the order of the hardware dimension utilization.
        # @param descending:
        #                 True -- the higher the mapping utilization is, the closer to the front it is.
        #                 False -- the lower the mapping utilization is, the closer to the front it is.
        import math
        oa_dim_unrollings = provided_oa_dim_unrollings
        if len(oa_dim_unrollings) > 0:
            # First we will record down the hardware utilization of each spatial unrolling in comb_value
            comb_value = []  # record down the value of each combination
            for comb in oa_dim_unrollings:
                if self.is_nested_tuple(comb):  # the comb is a mix sm loop
                    comb_value.append(math.prod([element[1] for element in comb]))
                else:  # the comb is a single-dim sm loop
                    comb_value.append(comb[1])
            # Next, we will add the index of each unrolling
            indexed_comb_value = list(enumerate(comb_value))  # record down the index information of comb
            # Then, we will sort the values in order, depending on if @param descending is True or False
            sorted_comb_value = sorted(indexed_comb_value, key=lambda x: x[1],
                                       reverse=descending)  # sort in descending order if True
            # After, we will record down the original index of each unrolling, as in the sorted order
            descending_comb_index = [x[0] for x in
                                     sorted_comb_value]  # index of combs with their values in descending order
            # Finally, we fetch the value corresponding to each index as in the sorted order
            oa_dim_unrollings = [oa_dim_unrollings[x] for x in
                                 descending_comb_index]  # new list with a descending order
        return oa_dim_unrollings

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
