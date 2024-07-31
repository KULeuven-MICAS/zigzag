import copy
import itertools
import logging
import math
from typing import Any, Generator

from sympy import divisors, primefactors  # type: ignore

from zigzag.datatypes import (
    LayerDim,
    LayerOperand,
    OADimension,
    UnrollFactor,
    UnrollFactorInt,
)
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.memory_level import ServedMemDimensions
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.mapping.spatial_mapping import (
    MappingSingleOADim,
    SpatialMapping,
)
from zigzag.stages.SpatialMappingConversionStage import (
    SpatialMappingConversionStage,
)
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.utils import pickle_deepcopy
from zigzag.workload.layer_attributes import MemoryOperandLinks
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class SpatialMappingGeneratorStage(Stage):
    """! Pipeline stage that finds spatial mappings given a:
    - accelerator
    - core allocation
    - interconnection pattern on the allocated core
    - layer
    The spatial mappings are found using the interconnection pattern present on the core.
    The inner-most memory level served dimensions is used,
    as this is how the memories connect to the operational array."""

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        enable_mix_spatial_mapping_generation: bool = False,
        enable_weight_diagonal_mapping: bool = False,
        nb_mappings_generated: int = 3,
        **kwargs: Any,
    ):
        """
        @param enable_mix_spatial_mapping_generation Indicate wether to generate `mixed` spatial mappings i.e. unroll
        multiple LayerDims over same OA Dim
        @param nb_mappings_generated Maximal number of mappings generated, to limit simulation time
        """
        assert nb_mappings_generated > 0

        super().__init__(list_of_callables, **kwargs)

        self.accelerator = accelerator
        self.layer = layer
        self.provided_mapping = self.layer.spatial_mapping

        # Control parameters
        self.enable_mix_spatial_mapping_generation = enable_mix_spatial_mapping_generation
        self.enable_weight_diagonal_mapping = enable_weight_diagonal_mapping
        self.nb_mappings_generated = nb_mappings_generated

        self.layer_dim_sizes = self.layer.layer_dim_sizes
        core_id = layer.core_allocation[0]
        self.core = self.accelerator.get_core(core_id)
        self.oa_dim_sizes = self.core.operational_array.dimension_sizes
        self.memory_hierarchy = self.core.memory_hierarchy

        # Spatial mapping hint
        self.spatial_mapping_hint = self.layer.spatial_mapping_hint
        self.spatial_mapping_hint.clear_invalid_hits(self.layer.layer_dims)
        self.spatial_mapping_hint.complete_with_defaults(self.oa_dim_sizes, self.layer.layer_dims)

    def run(self):
        """! Generate SpatialMappings and convert to internal representation"""

        generated_mappings = list(self.generate_spatial_mappings())
        nb_generated_mappings = len(generated_mappings)
        assert nb_generated_mappings > 0, "No SpatialMappings found"

        for i, generated_mapping in enumerate(generated_mappings):
            self.layer.spatial_mapping = generated_mapping
            logger.info(
                "%s: Launching spatial mapping %i/%i :%s.",
                self.layer.name,
                (i + 1),
                nb_generated_mappings,
                generated_mapping,
            )

            # Modify the size of lower input mem to support weight diagonal spatial unrolling (for OX/OY)
            accelerator_under_test = (
                self.modify_innermost_input_mem_size(generated_mapping)
                if self.enable_weight_diagonal_mapping
                else self.accelerator
            )

            spatial_mapping_conversion_stage = SpatialMappingConversionStage(
                self.list_of_callables,
                accelerator=accelerator_under_test,
                layer=copy.copy(self.layer),
                **self.kwargs,
            )

            # Set the generated_mapping in the layer, as this is required by SpatialMappingConversionStage
            self.layer.spatial_mapping = generated_mapping

            for cme, extra_info in spatial_mapping_conversion_stage.run():
                # recover back the accelerator in case the memory size had been adjusted
                cme.accelerator = self.accelerator
                yield cme, (generated_mapping, extra_info)

    def generate_spatial_mappings(self) -> Generator[SpatialMapping, None, None]:
        """! Generator that yields SpatialMappings
        # TODO this function first does all the work before it yield the first element
        """
        max_unrollings = self.get_max_unrolling()

        # Start from the given mapping
        mapping_template = copy.deepcopy(self.provided_mapping)
        mapping_template.initialize_oa_dims(self.oa_dim_sizes)
        mapping_template.check_and_reduce(max_unrollings, self.layer_dim_sizes.data)

        oa_dims_to_fill = [x for x in self.oa_dim_sizes if x not in mapping_template]

        # Full Spatial Mapping is already defined by user
        if len(oa_dims_to_fill) == 0:
            assert mapping_template.is_valid(max_unrollings, self.layer_dim_sizes.data)
            mapping_template = self.limit_unrolling_to_mem_capacity(mapping_template)
            yield mapping_template
            return

        # For each OADimension to fill, create a generator of MappingSingleOADim candidates
        mappings_per_oa_dim: list[Generator[MappingSingleOADim, None, None]] = [
            self.generate_spatial_mapping_single_oa_dim(
                self.spatial_mapping_hint[oa_dim],
                max_unrollings[oa_dim],
                self.oa_dim_sizes[oa_dim],
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
            if candidate.is_valid(max_unrollings, self.layer_dim_sizes.data):
                candidate_mappings.append(candidate)

        assert len(candidate_mappings) > 0, "No valid SpatialMappings found"

        # Sort according to expected performance
        candidate_mappings = sorted(
            candidate_mappings,
            key=lambda x: x.get_performance_indicator(),
            reverse=True,
        )

        # Limit the number of mappings generated
        for i in range(min(self.nb_mappings_generated, len(candidate_mappings))):
            candidate = candidate_mappings[i]
            if self.enable_weight_diagonal_mapping:
                candidate = self.add_input_pr_spatial_loop(candidate)
            candidate = self.limit_unrolling_to_mem_capacity(candidate)
            yield candidate

    def limit_unrolling_to_mem_bandwidth(
        self, mapping: dict[OADimension, dict[LayerDim, UnrollFactorInt]]
    ) -> dict[OADimension, dict[LayerDim, UnrollFactorInt]]:
        """! Scale the given unroll factors such that they do not exceed the bandwidths of the memory structure"""

        def conditional_log(layer_dim: LayerDim, oa_dim: OADimension, value: int, mem_name: str):
            # Don't log if user has defined an unrolling for a different layer dim
            do_not_log = oa_dim in self.provided_mapping and layer_dim not in self.provided_mapping[oa_dim]
            if not do_not_log:
                logger.warning(
                    "Maximal spatial unrolling of %s at %s limited to %i due to bandwidth of %s",
                    layer_dim,
                    oa_dim,
                    value,
                    mem_name,
                )

        for mem_level in self.memory_hierarchy.get_inner_memories():
            for mem_op in mem_level.operands:
                layer_op = self.layer.memory_operand_links.mem_to_layer_op(mem_op)
                # Either write BW (to write outputs away) or read BW (to read inputs)
                mem_bandwidth = mem_level.write_bw if layer_op.is_output() else mem_level.read_bw
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
                                conditional_log(
                                    layer_dim,
                                    oa_dim,
                                    max_multicast_elements,
                                    mem_level.name,
                                )

                                mapping[oa_dim][layer_dim] = max_multicast_elements

        return mapping

    def limit_unrolling_to_mem_capacity(self, mapping: SpatialMapping) -> SpatialMapping:
        """! Scale the given unroll factors such that they do not exceed the capacity of the memory structure"""

        def limit_loop_unrolling(
            spatial_mapping: SpatialMapping, dims_to_limit: set[LayerDim], max_unrolling: float
        ) -> SpatialMapping:
            def adjust_unrolling_factors(factors: list[UnrollFactor], max_unrolling: float) -> list[UnrollFactor]:
                product = math.prod(factors)
                while product > max_unrolling:
                    max_factor = max(factors)
                    max_index = factors.index(max_factor)
                    factors[max_index] -= 1
                    product = math.prod(factors)
                return factors

            # Extract the unrolling factors for the limited dimensions
            unrolling_factors: list[UnrollFactor] = []
            for dim in dims_to_limit:
                for oa_dim in spatial_mapping:
                    loop_unrollings = spatial_mapping[oa_dim]
                    if dim in loop_unrollings:
                        unrolling_factors.append(loop_unrollings[dim])
            # Adjust the unrolling factors
            adjusted_factors = adjust_unrolling_factors(unrolling_factors.copy(), max_unrolling)
            # Create a mapping from the original to adjusted factors
            factor_map = {old: new for old, new in zip(unrolling_factors, adjusted_factors)}
            # Apply the adjusted factors back to the loop_dict
            limited_spatial_mapping = SpatialMapping({})
            for oa_dim in spatial_mapping:
                loop_unrollings = spatial_mapping[oa_dim]
                limited_mapping_single_oa_dim = MappingSingleOADim(
                    {
                        loop_dim: factor_map[factor] if loop_dim in dims_to_limit else factor
                        for loop_dim, factor in loop_unrollings.items()
                    }
                )
                limited_spatial_mapping[oa_dim] = limited_mapping_single_oa_dim
            return limited_spatial_mapping

        for mem_level in self.memory_hierarchy.get_inner_memories():
            for mem_op in mem_level.operands:
                layer_op = self.layer.memory_operand_links.mem_to_layer_op(mem_op)
                # Either write BW (to write outputs away) or read BW (to read inputs)
                mem_capacity = mem_level.memory_instance.size
                # Bit precision of layer operand
                precision = self.layer.operand_precision[layer_op]
                irrelevant_dimensions = self.layer.get_operand_irrelevant_layer_dims(layer_op)
                total_unrolling_size = 1
                relevant_oa_dims_spatial_mapping = SpatialMapping({})
                non_irrelevant_dimensions: set[LayerDim] = set()
                for oa_dim in mem_level.served_dimensions:
                    relevant_oa_dims_spatial_mapping[oa_dim] = mapping[oa_dim]
                    # Iterate over all possible LayerDims and rescale max unroll factor
                    for layer_dim, unrolling_size in mapping[oa_dim].items():
                        if layer_dim not in irrelevant_dimensions:
                            non_irrelevant_dimensions.add(layer_dim)
                            total_unrolling_size *= unrolling_size
                max_stored_elements = mem_capacity / precision if precision > 0 else float("inf")
                # Limit the total unrolling across the OA Dims to the capacity of the memory
                if max_stored_elements < total_unrolling_size:
                    logger.warning(
                        "Maximal spatial unrolling limited to %i due to capacity %i of %s.",
                        max_stored_elements,
                        mem_capacity,
                        mem_level.name,
                    )
                    limited_mapping = limit_loop_unrolling(
                        relevant_oa_dims_spatial_mapping, non_irrelevant_dimensions, max_stored_elements
                    )
                    for oa_dim, unrollings in limited_mapping.items():
                        mapping[oa_dim].update(unrollings)
        return mapping

    def get_max_unrolling(self) -> dict[OADimension, dict[LayerDim, UnrollFactorInt]]:
        """! Generate a SpatialMapping that contains the maximal unroll factor for every Operational
        Array OADimension and every Layer Dimensions. Note that this is NOT a valid mapping as each
        OADimension contains ALL Layer Dimensions, maximally unrolled."""

        # Initialize and limit unrolling to the size of the layer or the size of the OADimension
        max_unrolling = {
            oa_dim: {
                layer_dim: int(min(layer_size, self.oa_dim_sizes[oa_dim]))
                for layer_dim, layer_size in self.layer.layer_dim_sizes.items()
            }
            for oa_dim in self.oa_dim_sizes
        }

        max_unrolling = self.limit_unrolling_to_mem_bandwidth(max_unrolling)
        return max_unrolling

    def generate_spatial_mapping_single_oa_dim(
        self,
        unroll_hints: set[LayerDim],
        max_unrollings: dict[LayerDim, UnrollFactorInt],
        oa_dim_size: int,
    ) -> Generator[MappingSingleOADim, None, None]:
        """! Generate a list of possible mappings for the given OADimension. Possible mappings include
        unrolling all LayerDims in `unroll_hints` for all unique divisors upto the maximal value defined in
        `max_unrolling`.
        """
        # Unroll a single LayerDim over this OA Dim
        for layer_dim in unroll_hints:
            if layer_dim in max_unrollings.keys():
                max_factor: UnrollFactor = int(max_unrollings[layer_dim])
                # Start with largest unrollings
                for factor in sorted(divisors(max_factor), reverse=True):
                    yield MappingSingleOADim({layer_dim: factor})

        if self.enable_mix_spatial_mapping_generation:
            mixed_mappings = self.generate_mapping_single_oa_dim_mixed(max_unrollings, unroll_hints)
            yield from filter(lambda x: x.utilization <= oa_dim_size, mixed_mappings)

    def generate_mapping_single_oa_dim_mixed(
        self,
        max_unrolling: dict[LayerDim, UnrollFactorInt],
        unrolling_hints: set[LayerDim],
    ) -> Generator[MappingSingleOADim, None, None]:
        """! Given an iterator of MappingSingleOADim where each item only contains a single Layer Dimensions,
        generate new MappingSingleOADim instances that each contain multiple Layer Dimensions (`mixed`),
        constrained to the maximal Operational Array dimension that corresponds to the MappingSingleOADim
        instance.
        """

        unique_factor_pool: dict[LayerDim, list[int]] = {
            layer_dim: self.non_trivial_divisors(max_unroll) for layer_dim, max_unroll in max_unrolling.items()
        }

        # Make sure all layer dims are defined
        unique_factor_pool.update(
            {layer_dim: [0] for layer_dim in unrolling_hints if layer_dim not in unique_factor_pool}
        )

        # Number of Layer Dimensions combined in new Mapping
        for combination_len in range(2, len(unrolling_hints) + 1):
            # e.g. ("C", "K") or ("C", "K", "G")
            for layer_dim_mix in itertools.combinations(unrolling_hints, combination_len):
                # e.g. ([3,2], [5,4]) which corresponds to ("C", "K")
                to_combine: list[list[int]] = [unique_factor_pool[layer_dim] for layer_dim in layer_dim_mix]
                # e.g. (2, 4) which corresponds to ("C", "K")
                for combination in itertools.product(*to_combine):
                    # e.g. (("C", 2), ("K", 4))
                    if len(combination) > 1:
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

        for x in spatial_mapping.all_contained_layer_dims:
            layer_dim_size_remainder[x] //= spatial_mapping.get_total_unrolling_of_layer_dim(x)

        act_operand, const_operand = self.identify_layer_operand_representation(self.layer)
        # No solution if there is no constant operand (e.g. for Matrix Multiply)
        if act_operand is None or const_operand is None:
            return spatial_mapping

        output_operand = self.layer.output_operand
        weight_ir_layer_dims: list[LayerDim] = self.layer.loop_relevancy_info.get_ir_layer_dims(const_operand)

        memory_operand_links: MemoryOperandLinks = self.layer.memory_operand_links

        # OA Dims that serve activations/output at innermost memory level. The memory level may only multicast on one
        # dimension
        act_served_oa_dims_list: list[ServedMemDimensions] = [
            mem_level.served_dimensions
            for mem_level in self.memory_hierarchy.get_inner_memories()
            if memory_operand_links[act_operand] in mem_level.operands and len(mem_level.served_dimensions) == 1
        ]
        output_served_oa_dims_list: list[ServedMemDimensions] = [
            mem_level.served_dimensions
            for mem_level in self.memory_hierarchy.get_inner_memories()
            if memory_operand_links[output_operand] in mem_level.operands and len(mem_level.served_dimensions) == 1
        ]

        if len(act_served_oa_dims_list) == 0 or len(output_served_oa_dims_list) == 0:
            return spatial_mapping

        # Get served dims for arbitrary memory level
        act_served_oa_dims = act_served_oa_dims_list.pop()
        output_served_oa_dims = output_served_oa_dims_list.pop()
        # Get arbitrary served oa dim (there is only 1)
        act_served_oa_dim: OADimension = next(iter(act_served_oa_dims))
        output_served_oa_dim: OADimension = next(iter(output_served_oa_dims))

        # check if OX / OY in spatial_mapping_hint. Or else target_layer_dim will be empty.
        target_layer_dim: list[LayerDim] = [
            x for x in weight_ir_layer_dims if x in self.spatial_mapping_hint[act_served_oa_dim]
        ]

        # no further execution if OX / OY unrolling is not in spatial_mapping_hint
        if len(target_layer_dim) == 0:
            return spatial_mapping

        # Get existed mapping size on act_served_oa_dim, which will be added with OX, OY later.
        exist_act_loop_size = (
            spatial_mapping[act_served_oa_dim].utilization if act_served_oa_dim in spatial_mapping else 1
        )

        # Check if the existed mapping size is more than half of current oa dim size.
        # If so, it means there is no space for extra mapping even with a size of 2.
        # In that case, we will do nothing but return the orignal spatial mapping
        if exist_act_loop_size * 2 > self.oa_dim_sizes[act_served_oa_dim]:
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
            max_allowed_dim_size_on_act_served_dim = math.floor(
                self.oa_dim_sizes[act_served_oa_dim] / exist_act_loop_size
            )
            # (2) check on output_served_oa_dim
            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]

            ir_layer_dim_to_current_layer_dim = next(
                filter(lambda x: x != layer_dim, weight_r_loop.keys())  # pylint: disable=W0640
            )

            existed_pr_mapping_but_ir_to_current_layer_dim = list(
                weight_r_loop[ir_layer_dim_to_current_layer_dim].values()
            )[0]
            max_allowed_dim_size_on_output_served_dim = (
                self.oa_dim_sizes[output_served_oa_dim]
                / weight_ir_loop_size
                / existed_pr_mapping_but_ir_to_current_layer_dim
            ) - (existed_pr_mapping - 1)
            # ceil down to integer
            max_allowed_dim_size_on_output_served_dim = math.floor(max_allowed_dim_size_on_output_served_dim)
            max_allowed_target_dim_size = min(
                max_allowed_dim_size_on_act_served_dim,
                max_allowed_dim_size_on_output_served_dim,
            )
            # check whether the element in layer_size_breakdown is allowed to add
            legal_layer_size_breakdown: list[int] = []
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
                        if required_oa_dim_size > self.oa_dim_sizes[act_served_oa_dim]:
                            continue  # the comb is not possible on act_served_oa_dim
                        # (3) check on output_served_oa_dim
                        required_oa_dim_size = weight_ir_loop_size
                        for layer_dim, pr_mapping_to_add in comb_mapping.items():
                            existed_pr_mapping = list(weight_r_loop[layer_dim].values())[0]
                            new_mapping_size = existed_pr_mapping + pr_mapping_to_add - 1
                            required_oa_dim_size *= new_mapping_size
                        if len(comb_mapping) == 1:  # only OX or OY
                            # add the other existed pr loop to required_oa_dim_size,
                            # because previously it is not counted in output_served_self.oa_dim_sizes[oa_dim].
                            sole_dim = list(comb_mapping.keys())[0]
                            the_other_pr_mapping_name = [key for key in weight_r_loop if key != sole_dim][0]
                            the_other_pr_mapping_size = list(weight_r_loop[the_other_pr_mapping_name].values())[0]
                            required_oa_dim_size *= the_other_pr_mapping_size
                        if required_oa_dim_size > self.oa_dim_sizes[output_served_oa_dim]:
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
        """! Return a list of divisors of `n`, excluding 1"""
        return list(filter(lambda x: 1 < x <= n, divisors(n)))

    @staticmethod
    def identify_layer_operand_representation(
        layer: LayerNode,
    ) -> tuple[LayerOperand | None, LayerOperand | None]:
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

    def modify_innermost_input_mem_size(self, user_spatial_mapping: SpatialMapping) -> Accelerator:
        """!
        # TODO needs cleanup
        """
        # To support OX, OY unrolling, we will scale the lowest input mem size by OXu*OYu
        # to avoid the MemoryTooSmallException in loma stage.
        innermost_levels = self.memory_hierarchy.get_inner_memories()

        # check if it is weight stationary.
        # keep the spatial loop as it was if it is not weight stationary.
        if len(self.layer.constant_operands) != 1:
            return self.accelerator
        # get weight operand name
        const_operand = self.layer.constant_operands[0]  # weight representation
        # get activation operand name
        act_operand = [operand for operand in self.layer.input_operands if operand != const_operand][0]
        # get name of OX, OY (weight ir layer dims)
        weight_ir_layer_dims: list[LayerDim] = self.layer.loop_relevancy_info.get_ir_layer_dims(const_operand)
        # get the oa_dim name served by input innermost memory level
        act_innermost_mem_level = None
        act_served_oa_dims = ServedMemDimensions(set())
        for memory_level in innermost_levels:
            mem_ops = memory_level.operands
            if self.layer.memory_operand_links.layer_to_mem_op(act_operand) in mem_ops:
                act_innermost_mem_level = memory_level
                act_served_oa_dims: ServedMemDimensions = memory_level.served_dimensions

        # check if act is not served in the innermost memories, or it is uti-casting for act.
        # keep the spatial loop as it was if act is not served.
        if "act_served_oa_dim" not in locals() or len(act_served_oa_dims) != 1:
            return self.accelerator

        act_served_oa_dim: OADimension = next(iter(act_served_oa_dims))
        # get the mem scaling f#actor if OX, OY exist
        mem_scaling_factor: int = 1
        if act_served_oa_dim not in user_spatial_mapping:  # there is no sm loop
            pass
        else:  # there is sm loop on act served oa dim
            act_served_oa_mapping = user_spatial_mapping[act_served_oa_dim]
            for layer_dim, layer_size in act_served_oa_mapping.items():
                if layer_dim in weight_ir_layer_dims:
                    mem_scaling_factor *= int(layer_size)

        # scale the mem size
        if mem_scaling_factor == 1:
            # No need to change the input mem size
            return self.accelerator

        # Initialize the new memory hierarchy
        mh_name = self.memory_hierarchy.name
        new_mh_name = mh_name + "-supporting-diagonal-map"
        operational_array = self.core.operational_array
        new_memory_hierarchy = MemoryHierarchy(operational_array, new_mh_name)
        # Add memories to the new memory hierarchy with the correct attributes
        for memory_level in self.memory_hierarchy.mem_level_list:
            memory_instance = memory_level.memory_instance
            if memory_level == act_innermost_mem_level:
                # scale here. For others, keep them unchanged.
                prev_size = memory_instance.size
                new_size = memory_instance.size * mem_scaling_factor
                memory_instance.update_size(new_size)
                logger.info(
                    "Updated %s size from %i to %i",
                    memory_instance,
                    prev_size,
                    new_size,
                )

            new_memory_instance: MemoryInstance = pickle_deepcopy(memory_instance)
            new_operands = pickle_deepcopy(memory_level.operands)
            new_port_alloc = pickle_deepcopy(memory_level.port_alloc_raw)
            new_served_dimensions = pickle_deepcopy(memory_level.served_dimensions)
            new_memory_hierarchy.add_memory(
                memory_instance=new_memory_instance,
                operands=new_operands,
                port_alloc=new_port_alloc,
                served_dimensions=new_served_dimensions,
            )
        # Create the new core
        new_core = Core(
            core_id=self.core.id,
            operational_array=operational_array,
            memory_hierarchy=new_memory_hierarchy,
        )

        # Create the new accelerator
        name = self.accelerator.name
        new_name = name + "-supporting-diagonal-map"
        new_cores = {new_core}
        new_accelerator = Accelerator(
            name=new_name,
            core_set=new_cores,
        )
        return new_accelerator
