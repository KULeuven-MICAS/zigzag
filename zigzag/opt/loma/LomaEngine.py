from math import factorial
import operator
from typing import Any, Generator
from tqdm import tqdm
import numpy as np
from sympy.ntheory import factorint
import logging


from zigzag.datatypes import LayerDim
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.opt.loma.multipermute import permutations
from zigzag.opt.loma.MemoryAllocator import (
    MemoryHierarchyTooSmallException,
    MemoryTooSmallException,
    MemoryAllocator,
)
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class NoValidLoopOrderingFoundException(Exception):

    pass


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

        # Extract the memory hierarchy from the accelerator
        # TODO: Take into account that data might be stored in lower level,
        # TODO: thus adapt the memory hierarchy.
        # TODO: The fact that there is a global buffer above the cores requires attention.
        core_id = layer.core_allocation[0]
        self.memory_hierarchy: MemoryHierarchy = accelerator.get_core(core_id).memory_hierarchy

        self.show_progress_bar = kwargs.get("loma_show_progress_bar", False)

    def run(self):
        """! Runs the LomaEngine
        @return Generator that yields all temporal mappings
        """
        # TODO: add the criterion(s) as inputs to this function.
        self.get_temporal_loops()  # get all the temporal loops
        self.get_prime_factors()  # convert these to LPFs (loop prime factors)

        pbar = tqdm(total=self.nb_permutations) if self.show_progress_bar else None

        yielded = False
        for ordering in self.ordering_generator():
            allocator = MemoryAllocator(self.accelerator, self.layer, self.spatial_mapping, ordering)
            # using try catch here because in the depth-first mode the highest level might not be big enough
            try:
                temporal_mapping = allocator.run()  # allocate this ordering to the memories
                yielded = True
                yield temporal_mapping
            except MemoryHierarchyTooSmallException:
                pass
            except MemoryTooSmallException:
                pass  # Skip the ordering that crashed due to ordering (+su) not fitting in memory
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not yielded:
            # TODO this warning is unclear: an invalid spatial mapping is not necessarily its cause
            raise NoValidLoopOrderingFoundException(
                f"No valid loop ordering was found for layer {self.layer}. Please make sure the spatial mapping is "
                f"compatible with the architecture."
            )

    def get_temporal_loops(self) -> None:
        """! Get all loops that have to be temporally scheduled given layer and spatial mapping.
        # TODO clean up (and make use of `LayerDimSizes` methods)
        """
        layer_dim_sizes = self.layer.layer_dim_sizes.copy()  # init with all loop sizes
        for spatial_loop in self.spatial_mapping.spatial_loop_dim_size:
            (spatial_loop_dim, spatial_loop_size) = spatial_loop
            # Allow greedy mapping. If the spatial unrolling is not a multiple of the layer dimension size,
            # we take the ceil of the division, so there can be one extra temporal iteration.
            q = int(np.ceil(layer_dim_sizes[spatial_loop_dim] / spatial_loop_size))
            # q, rem = divmod(layer_dim_sizes[spatial_loop_dim], spatial_loop_size)
            # assert rem == 0, "Division of dimension size by spatial unrolling size is not an integer"
            if q == 1:
                del layer_dim_sizes[spatial_loop_dim]
            else:
                layer_dim_sizes[spatial_loop_dim] = q

        # Remove all dimensions with a temporal loop size of 1
        temporal_loop_dim_size_no_1s = {key: val for key, val in layer_dim_sizes.items() if val > 1}

        self.temporal_loop_dim_size = temporal_loop_dim_size_no_1s
        min_nb_temporal_loops = len(self.temporal_loop_dim_size)
        if self.lpf_limit is not None and self.lpf_limit < min_nb_temporal_loops:
            logger.debug(
                f"Updated layer {self.layer}'s lpf limit from {self.lpf_limit} to {min_nb_temporal_loops} lpfs."
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
        # temporal_loop_pf_count_sums: a dict that for each temporal loop dimension contains the total amount of prime factors

        temporal_loop_pfs: dict[LayerDim, tuple[int, ...]] = {}
        temporal_loop_pf_counts: dict[LayerDim, tuple[int, ...]] = {}
        temporal_loop_pf_count_sums: dict[LayerDim, int] = {}
        lpfs = []
        for tl_dim, tl_size in self.temporal_loop_dim_size.items():  # tl = temporal loop
            factors: dict[int, int] = factorint(tl_size)
            pfs = []
            counts = []
            for pf, multiplicity in factors.items():
                pfs.append(pf)
                counts.append(multiplicity)
                for _ in range(multiplicity):
                    lpfs.append((tl_dim, pf))
            temporal_loop_pfs[tl_dim] = tuple(pfs)
            temporal_loop_pf_counts[tl_dim] = tuple(counts)
            temporal_loop_pf_count_sums[tl_dim] = sum(counts)

        # If there are no temporal LPFs generated, i.e. all loops are unrolled spatially,
        # we manually insert a loop of size 1
        if lpfs == []:
            loop_dim = self.layer.layer_dims[0]
            temporal_loop_pfs = {loop_dim: (1,)}
            temporal_loop_pf_counts = {loop_dim: (1,)}
            temporal_loop_pf_count_sums = {loop_dim: 1}
            lpfs = [(loop_dim, 1)]

        logger.debug(f"Generated {len(lpfs)} LPFs for layer {self.layer}.")

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
        logger.debug(f"Launching {self.nb_permutations:,} temporal loop order permutations.")

    def limit_lpfs(self) -> None:
        """! Function to limit the total number of loop prime factors present in this instance.
        This function scans the lpfs and while the number of lpfs is greater than self.lpf_limit it:
        - picks the loop dimension that has the most lpfs
        - merges the smallest two lpfs of that loop dimension (multiplying their values)
        """
        n_pf = sum(self.temporal_loop_pf_count_sums.values())
        if self.lpf_limit is None or n_pf <= self.lpf_limit:
            logger.debug(f"No lpf limiting performed for layer {self.layer}")
            return
        while n_pf > self.lpf_limit:
            # Find the loop dimension with the most lpfs
            max_ld = max(self.temporal_loop_pf_count_sums.items(), key=operator.itemgetter(1))[0]
            # Get the prime factors of this loop dimension
            max_pfs: list[int] = list(self.temporal_loop_pfs[max_ld])
            # Get the multiplicity of these prime factors
            max_counts: list[int] = list(self.temporal_loop_pf_counts[max_ld])

            if max_counts[0] == 1:  # multiplicity of smallest pf is 1
                new_factor = max_pfs[0] * max_pfs[1]
                max_counts[0] -= 1
                max_counts[1] -= 1
            else:  # multiplicity of smalles pf is > 1
                new_factor = max_pfs[0] * max_pfs[0]
                max_counts[0] -= 2

            if new_factor in max_pfs:  # possible if not first iteration of while loop
                new_factor_idx = max_pfs.index(new_factor)
                max_counts[new_factor_idx] += 1
            else:  # the new factor is not yet present in the factors, insert so list remains sorted
                new_factor_idx = len([pf for pf in max_pfs if pf < new_factor])
                max_pfs.insert(new_factor_idx, new_factor)
                max_counts.insert(new_factor_idx, 1)  # first time this factor occured, count = 1

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

        logger.debug(f"Limited layer {self.layer} to {len(self.lpfs)} lpfs.")
        return

    def ordering_generator(self) -> Generator[list[tuple[LayerDim, int]], None, None]:
        """! Generator that yields all orderings of the temporal loops."""
        return permutations(self.lpfs)  # type:ignore
