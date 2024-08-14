import logging
import operator
from math import factorial
from typing import Any, Generator

import numpy as np
from sympy.ntheory import factorint  # type: ignore
from tqdm import tqdm

from zigzag.datatypes import LayerDim, UnrollFactor
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping
from zigzag.opt.loma.MemoryAllocator import (
    MemoryAllocator,
    MemoryHierarchyTooSmallException,
    MemoryTooSmallException,
)
from zigzag.opt.loma.multipermute import (
    PermutationConstraint,
    StaticPositionsAndSizesConstraint,
    constrainded_permutations,
    permutations,
)
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class NoValidLoopOrderingFoundException(Exception):
    """Indicates that not a single valid temporal loop was found"""


class LomaEngine:
    """! Class that handles optimization of temporal mapping given a:
    - layer
    - spatial mapping
    - a memory hierarchy

    This optimization is carried out through loop order based memory allocation.
    For each ordering of the temporal loops, they are allocated bottom-up to the
    levels in the memory hierarchy.

    See https://ieeexplore.ieee.org/document/9458493 for more details.
    """

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        loma_lpf_limit: int | None = None,
        **kwargs: Any,
    ):
        """
        The memory hierarchy from the correct core is extracted from the accelerator.

        @param accelerator: accelerator to use the memory hierarchy of
        @param layer: layer to generate temporal mappings for
        @param spatial_mapping: SpatialMapping to use
        @param loma_lpf_limit:
        @param kwargs: further unused, for ease of calling only
        """
        self.lpf_limit = loma_lpf_limit

        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.constraints = []
        self.has_constraints = False

        # Extract the memory hierarchy from the accelerator
        # TODO: Take into account that data might be stored in lower level,
        # TODO: thus adapt the memory hierarchy.
        # TODO: The fact that there is a global buffer above the cores requires attention.
        core_id = layer.core_allocation[0]
        self.memory_hierarchy = accelerator.get_core(core_id).memory_hierarchy

        self.show_progress_bar = kwargs.get("loma_show_progress_bar", False)

    def set_constraints(self, constraints: list[PermutationConstraint]) -> None:
        self.constraints = constraints
        self.has_constraints = True

    def run(self) -> Generator[TemporalMapping, None, None]:
        """! Runs the LomaEngine
        @return Generator that yields all temporal mappings
        """
        # TODO: add the criterion(s) as inputs to this function.
        self.temporal_loop_dim_size = self.get_temporal_loops()  # get all the temporal loops to be scheduled
        self.update_min_lpf_factor(self.temporal_loop_dim_size)
        self.get_prime_factors()  # convert these to LPFs (loop prime factors)

        pbar = tqdm(total=self.nb_permutations) if self.show_progress_bar else None

        yielded = False
        for ordering in self.ordering_generator():
            allocator = MemoryAllocator(self.accelerator, self.layer, self.spatial_mapping, ordering)  # type: ignore
            # using try catch here because in the depth-first mode the highest level might not be big enough
            try:
                temporal_mapping = allocator.run()  # allocate this ordering to the memories
                yielded = True
                yield temporal_mapping
            except MemoryHierarchyTooSmallException:
                pass
            except MemoryTooSmallException:
                # Skip the ordering that crashed due to ordering (or spatial unrolling) not fitting in memory
                pass
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not yielded:
            raise NoValidLoopOrderingFoundException(
                f"No valid loop ordering was found for layer {self.layer}. Please make sure the data layout is "
                f"compatible with the architecture. Common causes of this error are: \n"
                f"- The spatial mapping is incompatible with the operational array dimensions\n"
                f"- The layer does not fit within the full memory hierarchy\n"
                f"- A single operand does not fit within the lowest memory level\n"
                f"- One of the layer dimensions cannot be split up in appropriate divisors\n"
            )

    def get_temporal_loops(self):
        """! Get all loops that have to be temporally scheduled given layer and spatial mapping.
        # TODO clean up (and make use of `LayerDimSizes` methods)
        """
        layer_dim_sizes = self.layer.layer_dim_sizes.copy()  # init with all loop sizes
        for (
            spatial_loop_dim,
            spatial_loop_size,
        ) in self.spatial_mapping.spatial_loop_dim_size:
            try:
                # Allow greedy mapping. If the spatial unrolling is not a multiple of the layer dimension size,
                # we take the ceil of the division, so there can be one extra temporal iteration.
                q = int(np.ceil(layer_dim_sizes[spatial_loop_dim] / spatial_loop_size))
                if q == 1:
                    del layer_dim_sizes[spatial_loop_dim]
                else:
                    layer_dim_sizes[spatial_loop_dim] = q
            except KeyError:
                # This might happen when the same LayerDim is defined multiple times in `spatial_loop_dim_size` and
                # deleted from `layer_dim_sizes`
                continue

        # Remove all dimensions with a temporal loop size of 1
        temporal_loop_dim_size_no_1s = {key: val for key, val in layer_dim_sizes.items() if val > 1}
        return temporal_loop_dim_size_no_1s

    def update_min_lpf_factor(self, loop_sizes: dict[LayerDim, UnrollFactor]):
        min_nb_temporal_loops = len(loop_sizes)
        if self.lpf_limit is not None and self.lpf_limit < min_nb_temporal_loops:
            logger.debug(
                "Updated layer %s's lpf limit from %i to %i lpfs.",
                self.layer,
                self.lpf_limit,
                min_nb_temporal_loops,
            )
            self.lpf_limit = min_nb_temporal_loops

    def get_prime_factors(self) -> None:
        """! Get the prime factors for all temporal loops.
        This is saved in three separate class attributes (temporal_loop_pfs, temporal_loop_pf_counts,
        temporal_loop_pf_count_sums)
        # TODO clean up (functions should not change class state variables...)
        """
        # temporal_loop_pfs: a dict that for each temporal loop dimension contains the prime factors
        # temporal_loop_pf_counts: a dict that for each temporal loop dimension contains the prime factor multiplicities
        # temporal_loop_pf_count_sums: a dict that for each temporal loop dimension contains the total amount of prime
        # factors

        temporal_loop_pfs: dict[LayerDim, tuple[int, ...]] = {}
        temporal_loop_pf_counts: dict[LayerDim, tuple[int, ...]] = {}
        temporal_loop_pf_count_sums: dict[LayerDim, int] = {}
        lpfs = []
        for tl_dim, tl_size in self.temporal_loop_dim_size.items():  # tl = temporal loop
            factors: dict[int, int] = factorint(tl_size)  # type: ignore
            pfs = []
            counts = []
            for pf, multiplicity in factors.items():  # type: ignore
                pfs.append(pf)  # type: ignore
                counts.append(multiplicity)  # type: ignore
                for _ in range(multiplicity):  # type: ignore
                    lpfs.append((tl_dim, pf))  # type: ignore
            temporal_loop_pfs[tl_dim] = tuple(pfs)  # type: ignore
            temporal_loop_pf_counts[tl_dim] = tuple(counts)  # type: ignore
            temporal_loop_pf_count_sums[tl_dim] = sum(counts)  # type: ignore

        # If there are no temporal LPFs generated, i.e. all loops are unrolled spatially,
        # we manually insert a loop of size 1
        if not lpfs:
            loop_dim = self.layer.layer_dims[0]
            temporal_loop_pfs = {loop_dim: (1,)}
            temporal_loop_pf_counts = {loop_dim: (1,)}
            temporal_loop_pf_count_sums = {loop_dim: 1}
            lpfs = [(loop_dim, 1)]

        logger.debug("Generated %i LPFs for layer %s.", len(lpfs), self.layer)

        self.temporal_loop_pfs: dict[LayerDim, tuple[int, ...]] = temporal_loop_pfs
        self.temporal_loop_pf_counts = temporal_loop_pf_counts
        self.temporal_loop_pf_count_sums = temporal_loop_pf_count_sums
        self.lpfs = lpfs

        # Limit the number of lpfs (if this is set in the settings)
        self.limit_lpfs()

        # Compute how many total permuatations we will have to consider
        self.compute_nb_permutations()

    def compute_nb_permutations(self):
        """! Compute the number of permutations that will have to be considered given the LPF distribution"""
        nb_permutations = factorial(sum(self.temporal_loop_pf_count_sums.values()))
        for nb_duplicated_pfs in self.temporal_loop_pf_counts.values():
            for nb_duplicated_pf in nb_duplicated_pfs:
                nb_permutations = int(nb_permutations / factorial(nb_duplicated_pf))
        self.nb_permutations = nb_permutations
        logger.debug(
            "Launching %s temporal loop order permutations.",
            f"{self.nb_permutations:,}",
        )

    def reduce_static_fps(self):
        try:
            static_sizes = next(
                (
                    v.static_positions_and_sizes.values()
                    for v in self.constraints
                    if isinstance(v, StaticPositionsAndSizesConstraint)
                )
            )

            for static_size in static_sizes:
                factor_list: dict[int, int] = factorint(static_size[1])  # type: ignore
                for factor, multiplicity in factor_list.items():  # type: ignore
                    index = self.temporal_loop_pfs[static_size[0]].index(factor)
                    if self.temporal_loop_pf_counts[static_size[0]][index] >= multiplicity:
                        self.temporal_loop_pf_counts[static_size[0]] = tuple(
                            count - multiplicity if i == index else count
                            for i, count in enumerate(self.temporal_loop_pf_counts[static_size[0]])  # type: ignore
                        )
                        # Add the static size to the count sums
                        if static_size[1] in self.temporal_loop_pfs[static_size[0]]:
                            self.temporal_loop_pf_counts[static_size[0]] = tuple(
                                count + 1 if i == index else count
                                for i, count in enumerate(self.temporal_loop_pf_counts[static_size[0]])  # type: ignore
                            )
                        else:
                            self.temporal_loop_pfs[static_size[0]] += (static_size[1],)
                            self.temporal_loop_pf_counts[static_size[0]] += (1,)
                        self.temporal_loop_pf_count_sums[static_size[0]] -= multiplicity - 1
        except StopIteration:
            # No static sizes
            pass

        lpfs: list[tuple[LayerDim, int]] = []
        for layer_dim, loop in self.temporal_loop_pfs.items():
            for pf, count in zip(loop, self.temporal_loop_pf_counts[layer_dim]):
                lpfs += list(((layer_dim, pf),) * count)
        self.lpfs = lpfs

    def find_smallest_non_static_pf(self, layer_dim: LayerDim) -> tuple[int, int]:
        static_sizes: list[tuple[LayerDim, int]] = []
        for constr in self.constraints:
            if isinstance(constr, StaticPositionsAndSizesConstraint):
                static_sizes = list(constr.static_positions_and_sizes.values())
        static_sizes_list = [static_sizes[i][1] for i in range(len(static_sizes)) if static_sizes[i][0] == layer_dim]
        smallvalue = 0
        already_first = False
        for i in range(len(self.temporal_loop_pfs[layer_dim])):
            if self.temporal_loop_pfs[layer_dim][i] not in static_sizes_list or self.temporal_loop_pf_counts[layer_dim][
                i
            ] > static_sizes_list.count(self.temporal_loop_pfs[layer_dim][i]):
                if already_first:
                    return (smallvalue, i)
                elif (
                    self.temporal_loop_pf_counts[layer_dim][i]
                    > static_sizes_list.count(self.temporal_loop_pfs[layer_dim][i]) + 1
                ):
                    return (i, i)
                else:
                    smallvalue = i
                    already_first = True
        return (smallvalue, len(self.temporal_loop_pfs[layer_dim]) - 1)

    def limit_lpfs(self) -> None:
        """! Function to limit the total number of loop prime factors present in this instance.
        This function scans the lpfs and while the number of lpfs is greater than self.lpf_limit it:
        - picks the loop dimension that has the most lpfs
        - merges the smallest two lpfs of that loop dimension (multiplying their values)
        """
        if self.has_constraints:
            self.reduce_static_fps()
        n_pf = sum(self.temporal_loop_pf_count_sums.values())
        if self.lpf_limit is None or n_pf <= self.lpf_limit:
            logger.debug("No lpf limiting performed for layer %s", self.layer)
            return
        while n_pf > self.lpf_limit:
            # Find the loop dimension with the most lpfs
            max_ld = max(self.temporal_loop_pf_count_sums.items(), key=operator.itemgetter(1))[0]
            # Get the prime factors of this loop dimension
            max_pfs: list[int] = list(self.temporal_loop_pfs[max_ld])
            # Get the multiplicity of these prime factors
            max_counts: list[int] = list(self.temporal_loop_pf_counts[max_ld])

            smallest_non_static = self.find_smallest_non_static_pf(max_ld)
            new_factor = max_pfs[smallest_non_static[0]] * max_pfs[smallest_non_static[1]]
            max_counts[smallest_non_static[0]] -= 1
            max_counts[smallest_non_static[1]] -= 1

            if new_factor in max_pfs:  # possible if not first iteration of while loop
                new_factor_idx = max_pfs.index(new_factor)
                max_counts[new_factor_idx] += 1
            else:  # the new factor is not yet present in the factors, insert so list remains sorted
                new_factor_idx = len([pf for pf in max_pfs if pf < new_factor])
                max_pfs.insert(new_factor_idx, new_factor)
                max_counts.insert(new_factor_idx, 1)  # first time this factor occurred, count = 1

            # Sanitize max_pfs and max_counts to remove all elements with multiplicity 0
            non_zero_idxs = [idx for idx, count in enumerate(max_counts) if count != 0]
            max_pfs = [max_pfs[non_zero_idx] for non_zero_idx in non_zero_idxs]
            max_counts = [max_counts[non_zero_idx] for non_zero_idx in non_zero_idxs]

            # Update the appropriate variables with these new factors and multiplicities
            self.temporal_loop_pfs[max_ld] = tuple(max_pfs)
            self.temporal_loop_pf_counts[max_ld] = tuple(max_counts)
            self.temporal_loop_pf_count_sums[max_ld] -= 1

            # Decrease the total number of factors by 1
            n_pf -= 1

        # Update self.lpfs for these new factors
        lpfs: list[tuple[LayerDim, int]] = []
        for layer_dim, loop in self.temporal_loop_pfs.items():
            for pf, count in zip(loop, self.temporal_loop_pf_counts[layer_dim]):
                lpfs += list(((layer_dim, pf),) * count)
        self.lpfs = lpfs

        logger.debug("Limited layer %s to %i lpfs.", self.layer, len(self.lpfs))
        return

    def ordering_generator(self) -> Generator[list[tuple[LayerDim, int]], None, None]:
        """! Generator that yields all orderings of the temporal loops."""
        if self.has_constraints:
            return constrainded_permutations(self.lpfs, self.constraints)  # type:ignore
        else:
            return permutations(self.lpfs)
