import itertools
import logging
import math
from typing import Any, Generator, Iterator
from urllib.parse import non_hierarchical
from typeguard import typechecked
from sympy import divisors, primefactors

from zigzag.datatypes import Dimension, LayerOperand
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.memory_level import ServedMemDimensions
from zigzag.workload.layer_attributes import MemoryOperandLinks
from zigzag.workload.layer_node import LayerNode
from zigzag.mapping.spatial_mapping import (
    LayerDim,
    SpatialMapping,
    MappingSingleOADim,
    SpatialMappingHint,
    UnrollFactor,
)

logger = logging.getLogger(__name__)


@typechecked
class UserSpatialMappingGenerator:
    """! Class that generates valid user-format spatial mappings."""

    def __init__(
        self,
        layer: LayerNode,
        accelerator: Accelerator,
        enable_mix_spatial_mapping_generation=False,
        enable_weight_diagonal_mapping=False,
        nb_mappings_generated=3,
    ) -> None:
        """!  The class constructor
        @param enable_mix_spatial_mapping_generation Indicate wether to generate `mixed` spatial mappings i.e. unroll
          multiple LayerDims over same OA Dim
        @param nb_mappings_generated Maximal number of mappings generated, to limit simulation time
        """
        assert nb_mappings_generated > 0

        self.layer = layer
        self.accelerator = accelerator
        # Mapping defined by the user: may or may not be complete
        self.provided_mapping = self.layer.user_spatial_mapping
        # Functional parameters
        self.enable_mix_spatial_mapping_generation = enable_mix_spatial_mapping_generation
        self.enable_weight_diagonal_mapping = enable_weight_diagonal_mapping
        self.nb_mappings_generated = nb_mappings_generated

        core_id = layer.core_allocation
        core = self.accelerator.get_core(core_id)
        self.oa_dims = core.operational_array.dimensions
        self.layer_dim_sizes = self.layer.layer_dim_sizes
        self.innermost_levels = core.memory_hierarchy.get_inner_memories()

        self.spatial_mapping_hint: SpatialMappingHint = self.layer.user_spatial_mapping_hint
        self.spatial_mapping_hint.complete_with_defaults(self.oa_dims, set(self.layer.layer_dims))

    def run(self) -> Generator[SpatialMapping, None, None]:
        return self.generate_user_spatial_mappings()

    def generate_user_spatial_mappings(self) -> Generator[SpatialMapping, None, None]:
        """!  Generator that yields SpatialMappings
        # TODO this function first does all the work before it yield the first element
        """
        max_unrollings = self.get_max_unrolling()

        # Start from given mapping if provided, create empty one instead
        mapping_template = self.provided_mapping if self.provided_mapping is not None else SpatialMapping.empty()
        mapping_template.initialize_oa_dims(self.oa_dims)
        mapping_template.check_and_reduce(max_unrollings, self.layer_dim_sizes.data)  # type: ignore

        oa_dims_to_fill = [x for x in self.oa_dims if x not in mapping_template]

        # Full Spatial Mapping is already defined by user
        if len(oa_dims_to_fill) == 0:
            assert mapping_template.is_valid(max_unrollings, self.layer_dim_sizes.data, self.oa_dims)  # type: ignore
            yield mapping_template
            return

        # For each OA Dimension to fill, generate a list of MappingSingleOADim candidates
        mappings_per_oa_dim: list[list[MappingSingleOADim]] = [
            list(
                self.generate_user_spatial_mapping_single_dim(
                    self.spatial_mapping_hint[oa_dim], max_unrollings[oa_dim], oa_dim.size
                )
            )
            for oa_dim in oa_dims_to_fill
        ]

        assert all([len(x) > 0 for x in mappings_per_oa_dim]), "No mapping found for some OADimension"

        candidate_mappings: list[SpatialMapping] = []
        for combination in itertools.product(*mappings_per_oa_dim):
            # Start from the user-defined mapping
            candidate = mapping_template.copy()
            for idx, oa_dim in enumerate(oa_dims_to_fill):
                candidate[oa_dim] = combination[idx]
            # Candidate can be invalid if unrollings of LayerDim exceed LayerDim size from workload
            if candidate.is_valid(max_unrollings, self.layer_dim_sizes.data, self.oa_dims):  # type: ignore
                candidate_mappings.append(candidate)

        assert len(candidate_mappings) > 0, "No valid SpatialMappings found"
        # assert len(candidate_mappings) == len(set(candidate_mappings)), "Generated mappings are not unique"

        # Sort according to expected performance
        candidate_mappings = sorted(candidate_mappings, key=lambda x: x.get_performance_indicator(), reverse=True)

        indicators = [x.get_performance_indicator() for x in candidate_mappings]
        nb_top_mappings = len([x for x in indicators if x == indicators[0]])

        # Limit the number of mappings generated
        for i in range(min(nb_top_mappings, self.nb_mappings_generated)):
            candidate = candidate_mappings[i]
            if self.enable_weight_diagonal_mapping:
                candidate = self.add_input_pr_spatial_loop(candidate)
            yield candidate

    def limit_unrolling_to_mem_bandwidth(self, mapping: SpatialMapping) -> SpatialMapping:
        """! Scale the given SpatialMapping in case any of the unrollings is limited by the memory structure"""
        for mem_level in self.innermost_levels:
            for mem_op in mem_level.operands:
                layer_op = self.layer.get_layer_operand(mem_op)
                # Either write BW (to write outputs away) or read BW (to read inputs)
                mem_bandwidth = mem_level.write_bw if (layer_op == "O") else mem_level.read_bw
                # Bit precision of layer operand
                precision = self.layer.operand_precision[layer_op]
                irrelevant_dimensions = self.layer.get_operand_irrelevant_layer_dims(layer_op)

                for oa_dim in mem_level.served_dimensions:
                    # Iterate over all possible LayerDims and rescale max unroll factor
                    for layer_dim, unrolling_size in mapping[oa_dim].items():
                        # If not irrelevant, it is (partially) relevant. Limit based on BW and operand precision.
                        if layer_dim not in irrelevant_dimensions:
                            max_multicast_elements = mem_bandwidth // precision if precision > 0 else unrolling_size
                            if max_multicast_elements < unrolling_size:
                                mapping[oa_dim][layer_dim] = max_multicast_elements
                                logger.warning(
                                    "Maximal spatial unrolling of %s at %s limited to %i due to bandwidth of %s",
                                    layer_dim,
                                    oa_dim,
                                    max_multicast_elements,
                                    mem_level.name,
                                )
        return mapping

    def get_max_unrolling(self) -> SpatialMapping:
        """! Generate a SpatialMapping that contains the maximal unroll factor for every Operational
        Array Dimension and every Layer Dimension. Note that this is NOT a valid mapping as each
        OA Dimension contains ALL Layer Dimensions, maximally unrolled."""

        # Initialize and limit unrolling to the size of the layer or the size of the OA Dimension
        max_unrolling = SpatialMapping(
            {
                oa_dim: MappingSingleOADim(
                    {
                        layer_dim: int(min(layer_size, oa_dim.size))
                        for layer_dim, layer_size in self.layer.layer_dim_sizes.items()
                    }
                )
                for oa_dim in self.oa_dims
            }
        )

        max_unrolling = self.limit_unrolling_to_mem_bandwidth(max_unrolling)
        return max_unrolling

    def generate_user_spatial_mapping_single_dim(
        self,
        unroll_hints: set[LayerDim],
        max_unrollings: MappingSingleOADim,
        oa_dim_size: int,
    ) -> Iterator[MappingSingleOADim]:
        """! Generate a list of possible mappings for the given Operational Array Dimension. Possible mappings include
        unrolling all LayerDims in `unroll_hints` for all unique divisors upto the maximal value defined in
        `max_unrolling`.
        """
        # Unroll a single LayerDim over this OA Dim
        for layer_dim in unroll_hints:
            max_factor: UnrollFactor = int(max_unrollings[layer_dim])
            # NOTE the unroll factor may equal one
            for factor in divisors(max_factor):
                yield MappingSingleOADim({layer_dim: factor})

        if self.enable_mix_spatial_mapping_generation:
            mixed_mappings = self.generate_mapping_single_dim_mixed(max_unrollings, unroll_hints)
            yield from filter(lambda x: x.get_utilization() <= oa_dim_size, mixed_mappings)

    def generate_mapping_single_dim_mixed(
        self,
        max_unrolling: MappingSingleOADim,
        unrolling_hints: set[LayerDim],
    ) -> Iterator[MappingSingleOADim]:
        """! Given an iterator of MappingSingleOADim where each item only contains a single Layer Dimension,
        generate new MappingSingleOADim instances that each contain multiple Layer Dimensions (`mixed`),
        constrained to the maximal Operational Array dimension that corresponds to the MappingSingleOADim
        instance.
        """

        unique_factor_pool: dict[LayerDim, list[int]] = {
            layer_dim: self.non_trivial_divisors(max_unroll)  # type: ignore
            for layer_dim, max_unroll in max_unrolling.items()
        }

        # Number of Layer Dimensions combined in new Mapping
        for combination_len in range(2, len(unrolling_hints) + 1):
            # e.g. ("C", "K") or ("C", "K", "G")
            for layer_dim_mix in itertools.combinations(unrolling_hints, combination_len):
                # e.g. ([3,2], [5,4]) which corresponds to ("C", "K")
                to_combine: list[list[int]] = [unique_factor_pool[layer_dim] for layer_dim in layer_dim_mix]
                # e.g. (2, 4) which corresponds to ("C", "K")
                for combination in itertools.product(*to_combine):
                    # e.g. (("C", 2), ("K", 4))
                    mapping_zip = zip(layer_dim_mix, combination)
                    yield MappingSingleOADim({layer_dim: unroll_value for layer_dim, unroll_value in mapping_zip})

    def add_input_pr_spatial_loop(self, spatial_mapping: SpatialMapping) -> SpatialMapping:
        """! This function is used to support diagonal spatial mapping
        when input/activation is served in the innermost memories and the weight is stationary.
        # TODO needs cleanup
        """
        # check if it is weight stationary.
        # TODO should it also skip when len == 0?
        if len(self.layer.constant_operands) > 1:
            return spatial_mapping

        # Convert LayerDimStr to LayerDim
        layer_dim_size_remainder: dict[LayerDim, UnrollFactor] = dict(self.layer.layer_dim_sizes.items())

        for x in spatial_mapping.get_all_contained_layer_dims():
            layer_dim_size_remainder[x] //= spatial_mapping.get_total_unrolling_of_layer_dim(x)

        act_operand, const_operand = self.identify_layer_operand_representation(self.layer)
        # No solution if there is no constant operand (e.g. for Matrix Multiply)
        if act_operand is None or const_operand is None:
            return spatial_mapping

        output_operand = self.layer.output_operand
        weight_ir_layer_dims: list[LayerDim] = self.layer.loop_relevancy_info.get_ir_layer_dims(const_operand)

        memory_operand_links: MemoryOperandLinks = self.layer.memory_operand_links

        # OA Dims that serve activations/output at innermost memory level. The memory level may only multicast on one dimension
        act_served_oa_dims_list: list[ServedMemDimensions] = [
            mem_level.served_dimensions
            for mem_level in self.innermost_levels
            if memory_operand_links[act_operand] in mem_level.operands and len(mem_level.served_dimensions) == 1
        ]
        output_served_oa_dims_list: list[ServedMemDimensions] = [
            mem_level.served_dimensions
            for mem_level in self.innermost_levels
            if memory_operand_links[output_operand] in mem_level.operands and len(mem_level.served_dimensions) == 1
        ]

        if len(act_served_oa_dims_list) == 0 or len(output_served_oa_dims_list) == 0:
            return spatial_mapping

        # Get served dims for arbitrary memory level
        act_served_oa_dims = act_served_oa_dims_list.pop()
        output_served_oa_dims = output_served_oa_dims_list.pop()
        # Get arbitrary served oa dim (there is only 1)
        act_served_oa_dim: Dimension = next(iter(act_served_oa_dims))
        output_served_oa_dim: Dimension = next(iter(output_served_oa_dims))

        # check if OX / OY in spatial_mapping_hint. Or else target_layer_dim will be empty.
        target_layer_dim: list[LayerDim] = [
            x for x in weight_ir_layer_dims if x in self.spatial_mapping_hint[act_served_oa_dim]
        ]

        # no further execution if OX / OY unrolling is not in spatial_mapping_hint
        if len(target_layer_dim) == 0:
            return spatial_mapping

        # Get existed mapping size on act_served_oa_dim, which will be added with OX, OY later.
        exist_act_loop_size = (
            spatial_mapping[act_served_oa_dim].get_utilization() if act_served_oa_dim in spatial_mapping else 1
        )

        # Check if the existed mapping size is more than half of current oa dim size.
        # If so, it means there is no space for extra mapping even with a size of 2.
        # In that case, we will do nothing but return the orignal spatial mapping
        if exist_act_loop_size * 2 > act_served_oa_dim.size:
            return spatial_mapping

        # fetch pr loop pairs for activation, e.g. {"IX": ["OX", "FX"]}
        act_pr_layer_dims = self.layer.loop_relevancy_info.get_pr_layer_dims(act_operand)

        # Next we get existed mapping size on output_served_oa_dim
        # there are two classes of mapping:
        # (1) ir mapping to weight, e.g. "C"
        # (2) r mapping to weight, e.g. "FX", "FY" (kernel size)

        # We firstly create a dict for later recording down existed r mapping to weight
        # it will be like:
        # weight_r_loop = {"OX": {"FX": 1}, "OY": {"FY": 1}}
        weight_r_loop: dict[LayerDim, dict[LayerDim, UnrollFactor]] = {}  # here we put a nested dict for recording
        loops_name_for_kernel_size: list[LayerDim] = []
        pr_sm_link: dict[LayerDim, LayerDim] = {}  # here we record down the link between pr loops, e.g. link["FX"]="OX"

        for weight_ir_layer_dim in weight_ir_layer_dims:
            # TODO this assumes act_pr_layer_dims has len 2
            [layer_dim1, layer_dim2] = act_pr_layer_dims
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
                weight_r_loop[paired_pr_layer_dim][layer_dim] *= unroll_factor  # type: ignore
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
            layer_size = self.layer.layer_dim_sizes[layer_dim]
            layer_size_breakdown: list[int] = primefactors(layer_size)

            # try to find the maximum OX / OY and add it to the list
            # (1) check on act_served_oa_dim (ceil down to integer)
            max_allowed_dim_size_on_act_served_dim = math.floor(act_served_oa_dim.size / exist_act_loop_size)
            # (2) check on output_served_oa_dim
            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
            for key in weight_r_loop:
                if key != layer_dim:
                    ir_layer_dim_to_current_layer_dim = key
            existed_pr_mapping_but_ir_to_current_layer_dim = list(
                weight_r_loop[ir_layer_dim_to_current_layer_dim].values()
            )[0]
            max_allowed_dim_size_on_output_served_dim = (
                output_served_oa_dim.size / weight_ir_loop_size / existed_pr_mapping_but_ir_to_current_layer_dim
            ) - (existed_pr_mapping - 1)
            # ceil down to integer
            max_allowed_dim_size_on_output_served_dim = math.floor(max_allowed_dim_size_on_output_served_dim)
            max_allowed_target_dim_size = min(
                max_allowed_dim_size_on_act_served_dim,
                max_allowed_dim_size_on_output_served_dim,
            )
            # check whether the element in layer_size_breakdown is allowed to add
            legal_layer_size_breakdown = []
            for factor in layer_size_breakdown:
                if factor <= max_allowed_target_dim_size and factor <= layer_dim_size_remainder[layer_dim]:
                    legal_layer_size_breakdown.append(factor)
            if len(legal_layer_size_breakdown) > 0:
                sm_pools_to_add[layer_dim] = [(layer_dim, size) for size in legal_layer_size_breakdown]

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
        allowed_dim_comb_length = len(target_layer_dim) if self.enable_mix_spatial_mapping_generation else 1

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
                        if required_oa_dim_size > act_served_oa_dim.size:
                            continue  # the comb is not possible on act_served_oa_dim
                        # (3) check on output_served_oa_dim
                        required_oa_dim_size = weight_ir_loop_size
                        for layer_dim, pr_mapping_to_add in comb_mapping.items():
                            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
                            new_mapping_size = existed_pr_mapping + pr_mapping_to_add - 1
                            required_oa_dim_size *= new_mapping_size
                        if len(comb_mapping) == 1:  # only OX or OY
                            # add the other existed pr loop to required_oa_dim_size,
                            # because previously it is not counted in output_served_oa_dim.size.
                            sole_dim = list(comb_mapping.keys())[0]
                            the_other_pr_mapping_name = [key for key in weight_r_loop.keys() if key != sole_dim][0]
                            the_other_pr_mapping_size = list(weight_r_loop[the_other_pr_mapping_name].values())[0]
                            required_oa_dim_size *= the_other_pr_mapping_size
                        if required_oa_dim_size > output_served_oa_dim.size:
                            continue  # this comb is not possible on output_served_oa_dim
                        # (4) compare with best_comb
                        if comb_size > best_comb_size:
                            # reformat the comb and merge repetitive elements
                            # example: (("OX", 5), ("OY", 2))
                            new_comb: list[tuple[LayerDim, int]] = [
                                (layer_dim, mapping_size) for (layer_dim, mapping_size) in comb_mapping.items()
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

    @staticmethod
    def non_trivial_divisors(n: int) -> list[int]:
        """! Return a list of divisors of `n`, excluding 1 and `n` itself"""
        return list(filter(lambda x: 1 < x < n, divisors(n)))

    @staticmethod
    def identify_layer_operand_representation(layer: LayerNode) -> tuple[LayerOperand | None, LayerOperand | None]:
        """
        # TODO requires documentation
        """
        # activation representation: list (conv layers)
        act_operands_pool: list[LayerOperand] = [
            op for op in layer.layer_operands if len(layer.loop_relevancy_info.get_pr_layer_dims(op)) > 0
        ]
        # true for fully-connected (fc) layers
        if len(act_operands_pool) == 0:
            # weight representation (fc layers)
            const_operands_pool = [
                op for op in layer.layer_operands if len(layer.loop_relevancy_info.get_ir_layer_dims(op)) == 0
            ]
            const_operand = None if len(const_operands_pool) == 0 else const_operands_pool.pop()
            # activation representation (fc layers)
            act_operands_pool = [operand for operand in layer.input_operands if operand != const_operand]
            act_operand = None if len(act_operands_pool) == 0 else act_operands_pool.pop()

        else:
            act_operand = act_operands_pool.pop()
            # weight representation (conv layers)
            const_operands_pool = [operand for operand in layer.input_operands if operand != act_operand]
            const_operand = None if len(const_operands_pool) == 0 else const_operands_pool.pop()

        return act_operand, const_operand
