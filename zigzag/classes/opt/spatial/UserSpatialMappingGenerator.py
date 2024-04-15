import itertools
import logging
import math
from typing import Iterator
from typeguard import typechecked
from sympy import divisors, primefactors

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.Dimension import Dimension
from zigzag.classes.hardware.architecture.memory_level import ServedMemDimensions
from zigzag.classes.workload.layer_node import LayerNode, MemOperandStr, OperandStr, Relevancy
from zigzag.classes.mapping.spatial.SpatialMapping import (
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
        provided_mapping: SpatialMapping = SpatialMapping.empty(),
        enable_mix_spatial_mapping_generation=False,
        enable_weight_diagonal_mapping=False,
        nb_mappings_generated=3,
    ) -> None:
        """!  The class constructor
        @param enable_mix_spatial_mapping_generation Indicate wether to generate `mixed` spatial mappings i.e. unroll
          multiple LayerDims over same OA Dim
        @param enable_weight_diagonal_mapping #TODO
        @param nb_mappings_generated Maximal number of mappings generated, to limit simulation time
        """
        self.layer = layer
        self.accelerator = accelerator
        self.provided_mapping = provided_mapping
        # Functional parameters
        self.enable_mix_spatial_mapping_generation = enable_mix_spatial_mapping_generation
        self.enable_weight_diagonal_mapping = enable_weight_diagonal_mapping
        self.nb_mappings_generated = nb_mappings_generated

        core: Core = self.accelerator.get_core(core_id=layer.core_allocation)
        self.oa_dims = core.operational_array.dimensions
        self.layer_dim_sizes = self.layer.layer_dim_sizes
        self.innermost_levels = core.memory_hierarchy.get_inner_memories()
        self.spatial_mapping_hint: SpatialMappingHint = self.layer.user_spatial_mapping_hint
        self.spatial_mapping_hint.complete_with_defaults(self.oa_dims, {LayerDim(x) for x in self.layer.loop_dim_list})

    def run(self):
        return self.generate_user_spatial_mappings()

    def generate_user_spatial_mappings(self) -> Iterator[SpatialMapping]:
        """!  Generator that yields SpatialMappings
        # TODO this function first does all the work before it yield the first element
        """
        max_unrollings = self.get_max_unrolling()

        # Start from given mapping if provided, create empty one instead
        mapping_template = self.provided_mapping if self.provided_mapping is not None else SpatialMapping.empty()
        mapping_template.check_and_reduce(max_unrollings, self.oa_dims, self.layer_dim_sizes)

        oa_dims_to_fill = [x for x in self.oa_dims if x not in mapping_template]
        # For each OA Dimension to fill, generate a list of MappingSingleOADim candidates
        mappings_per_oa_dim: list[list[MappingSingleOADim]] = [
            list(
                self.generate_user_spatial_mapping_single_dim(
                    self.spatial_mapping_hint[oa_dim], max_unrollings[oa_dim], oa_dim.size
                )
            )
            for oa_dim in oa_dims_to_fill
        ]

        candidate_mappings: list[SpatialMapping] = []
        for combination in itertools.product(*mappings_per_oa_dim):
            # Start from the user-defined mapping
            candidate = mapping_template.copy()
            for idx, oa_dim in enumerate(oa_dims_to_fill):
                candidate[oa_dim] = combination[idx]
            # Candidate can be invalid if unrollings of LayerDim exceed LayerDim size from workload
            if candidate.is_valid(max_unrollings, self.oa_dims, self.layer_dim_sizes):
                candidate_mappings.append(candidate)

        assert len(candidate_mappings) > 0, "No valid SpatialMappings found"
        assert len(candidate_mappings) == len(set(candidate_mappings)), "Generated mappings are not unique"

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
                irrelevant_dimensions = self.layer.get_operand_irrelevant_dimensions(layer_op)

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
                        LayerDim(layer_dim): int(min(layer_size, oa_dim.size))
                        for layer_dim, layer_size in self.layer.loop_dim_size.items()
                    }
                )
                for oa_dim in self.oa_dims
            }
        )

        return self.limit_unrolling_to_mem_bandwidth(max_unrolling)

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
            max_factor: UnrollFactor = max_unrollings[layer_dim]
            # NOTE the unroll factor may equal one
            for factor in divisors(max_factor):  # type: ignore # TODO fix UnrollFactor type
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
        """

        layer_dim_size_remainder: dict[LayerDim, UnrollFactor] = {
            LayerDim(layer_dim): layer_size for layer_dim, layer_size in self.layer.loop_dim_size.items()
        }
        for _, mapping_single_oa_dim in spatial_mapping.items():
            for layer_dim, unroll_factor in mapping_single_oa_dim.items():
                layer_dim_size_remainder[layer_dim] //= unroll_factor

        # get the link from layer op to mem op
        memory_operand_links: dict[OperandStr, MemOperandStr] = self.layer.memory_operand_links
        # check if it is weight stationary.
        # keep the spatial loop as it was if it is not weight stationary.
        if len(self.layer.constant_operands) > 1:
            return spatial_mapping

        act_operand, const_operand = self.identify_layer_operand_representation(self.layer)
        # No solution if there is no constant operand (e.g. for Matrix Multiply)
        if act_operand is None or const_operand is None:
            return spatial_mapping

        # get output operand name
        output_operand = self.layer.output_operand
        # get name of OX, OY (weight ir layer dims)
        weight_ir_layer_dims: list[LayerDim] = [
            LayerDim(x) for x in self.layer.operand_loop_dim[const_operand][Relevancy.IR]
        ]

        # TODO this code doesn't currently work
        # get the oa_dim name served by input / output innermost memory level
        for mem_level in self.innermost_levels:
            if memory_operand_links[act_operand] in mem_level.operands:
                act_served_oa_dims: ServedMemDimensions = mem_level.served_dimensions
            if memory_operand_links[output_operand] in mem_level.operands:
                output_served_oa_dims: ServedMemDimensions = mem_level.served_dimensions
        # check if act is not served in the innermost memories, or act/output is not multicasting on only one dimension.

        # keep the spatial loop as it was if act is not served.
        if "act_served_oa_dim" not in locals() or len(act_served_oa_dims) != 1:
            return spatial_mapping
        if "output_served_oa_dim" not in locals() or len(output_served_oa_dims) != 1:
            return spatial_mapping

        # TODO why only first element?
        act_served_oa_dim: Dimension = list(act_served_oa_dims)[0]
        output_served_oa_dim: Dimension = list(output_served_oa_dims)[0]
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
        act_pr_layer_dims = [LayerDim(x) for x in self.layer.operand_loop_dim[act_operand][Relevancy.PR]]

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
            layer_size_breakdown: list[int] = primefactors(layer_size)

            # try to find the maximum OX / OY and add it to the list
            # (1) check on act_served_oa_dim (ceil down to integer)
            max_allowed_dim_size_on_act_served_dim = math.floor(act_served_oa_dim_size / exist_act_loop_size)
            # (2) check on output_served_oa_dim
            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
            for key in weight_r_loop:
                if key != layer_dim:
                    ir_layer_dim_to_current_layer_dim = key
            existed_pr_mapping_but_ir_to_current_layer_dim = list(
                weight_r_loop[ir_layer_dim_to_current_layer_dim].values()
            )[0]
            max_allowed_dim_size_on_output_served_dim = (
                output_served_oa_dim_size / weight_ir_loop_size / existed_pr_mapping_but_ir_to_current_layer_dim
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
                            the_other_pr_mapping_name = [key for key in weight_r_loop.keys() if key != sole_dim][0]
                            the_other_pr_mapping_size = list(weight_r_loop[the_other_pr_mapping_name].values())[0]
                            required_oa_dim_size *= the_other_pr_mapping_size
                        if required_oa_dim_size > output_served_oa_dim_size:
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
    def identify_layer_operand_representation(layer: LayerNode) -> tuple[str | None, str | None]:
        # activation representation: list (conv layers)
        act_operands_pool: list[str] = [
            op for op in layer.operand_loop_dim if len(layer.operand_loop_dim[op][Relevancy.PR]) > 0
        ]
        # true for fully-connected (fc) layers
        if len(act_operands_pool) == 0:
            # weight representation (fc layers)
            const_operands_pool = [
                op for op in layer.operand_loop_dim if len(layer.operand_loop_dim[op][Relevancy.IR]) == 0
            ]
            const_operand = None if len(const_operands_pool) == 0 else const_operands_pool[0]
            # activation representation (fc layers)
            act_operands_pool = [operand for operand in layer.input_operands if operand != const_operand]
            act_operand = None if len(act_operands_pool) == 0 else act_operands_pool[0]

        else:
            act_operand = act_operands_pool[0]
            # weight representation (conv layers)
            const_operands_pool = [operand for operand in layer.input_operands if operand != act_operand]
            const_operand = None if len(const_operands_pool) == 0 else const_operands_pool[0]

        return act_operand, const_operand
