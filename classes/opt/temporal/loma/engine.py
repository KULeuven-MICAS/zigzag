"""
This file contains the core code of the temporal mapping optimization method
called loma: loop order based memory allocation.

TODO: Get a layers' dimensions to generate the multiset permutations for all loop types
TODO: Write generator that takes loop-type-specific multiset permutations and generates loop order permutation
TODO: Write uneven memory allocator, that allocates the loops of the loop order bottom-up to the memories in the hierarchy
TODO: (optional) Write even memory allocator
TODO: Once we have allocated the loops to the different hierarchy levels, call the cost model to get energy, latency
TODO: Save the best found loop order (and its associated allocated mapping)
"""
import operator
from typing import Generator

import numpy as np
from sympy.ntheory import factorint
import logging

import classes.io.input_config as inputs
from classes.hardware.architecture.accelerator import Accelerator
from classes.mapping.temporal.temporal_mapping import TemporalMapping
from classes.workload.layer_node import LayerNode
from classes.mapping.spatial.spatial_mapping import SpatialMapping
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.opt.temporal.loma.multipermute import permutations
from classes.opt.temporal.loma.memory_allocator import MemHierarchyTooSmallException, MemoryAllocator
from classes.opt.temporal.loma.cost_model import CostModelCaller

logger = logging.getLogger(__name__)

class LomaEngine:
    """
    Class that handles optimization of temporal mapping given a:
    - layer
    - spatial mapping
    - a memory hierarchy
    This optimization is carried out through loop order based memory allocation.
    For each ordering of the temporal loops, they are allocated bottom-up to the
    levels in the memory hierarchy.
    See https://ieeexplore.ieee.org/document/9458493 for more details.
    """

    def __init__(self, *, accelerator, layer, spatial_mapping, loma_lpf_limit=np.inf,  **kwargs):
        """
        Initialize the engine with the given:
        - Accelerator
        - LayerNode
        - SpatialMapping

        The memory hierarchy from the correct core is extracted from the accelerator.
        :param accelerator: accelerator to use the memory hierarchy of
        :param layer: layer to generate temporal mappings for
        :param spatial_mapping: SpatialMapping to use
        :param loma_lpf_limit:
        :param kwargs: further unused, for ease of calling only
        """
        self.lpf_limit = loma_lpf_limit

        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping

        # Extract the memory hierarchy from the accelerator
        # TODO: Take into account that data might be stored in lower level,
        # TODO: thus adapt the memory hierarchy.
        # TODO: The fact that there is a global buffer above the cores requires attention.
        core_id = layer.core_allocation
        self.memory_hierarchy: MemoryHierarchy = accelerator.get_core(core_id).memory_hierarchy



    def run(self):
        """
        :returns : Generator that yields all temporal mappings
        TODO: add the criterion(s) as inputs to this function.
        """
        self.get_temporal_loops()  # get all the temporal loops
        self.get_prime_factors()  # convert these to LPFs (loop prime factors)
        yielded = False
        for ordering in self.og():  # ordering generator
            allocator = MemoryAllocator(self.accelerator, self.layer, self.spatial_mapping, ordering)
            # using try catch here because in the depth-first mode the highest level might not be big enough,
            # which
            try:
                temporal_mapping = allocator.run()  # allocate this ordering to the memories
                yielded = True
                yield temporal_mapping
            except MemHierarchyTooSmallException:
                pass
        if not yielded:
            raise MemHierarchyTooSmallException("No loop ordering was found that did not exceed memory capacity")


    def get_temporal_loops(self):
        """
        Get all loops that have to be temporally scheduled given layer and spatial mapping.
        """
        temporal_loop_dim_size = self.layer.loop_dim_size.copy()  # init with all loop sizes
        for spatial_loop in self.spatial_mapping.spatial_loop_dim_size:
            (spatial_loop_dim, spatial_loop_size) = spatial_loop
            # Allow greedy mapping. If the spatial unrolling is not a multiple of the layer dimension size,
            # we take the ceil of the division, so there can be one extra temporal iteration.
            q = int(np.ceil(temporal_loop_dim_size[spatial_loop_dim] / spatial_loop_size))
            # q, rem = divmod(temporal_loop_dim_size[spatial_loop_dim], spatial_loop_size)
            # assert rem == 0, "Division of dimension size by spatial unrolling size is not an integer"
            if q == 1:
                del temporal_loop_dim_size[spatial_loop_dim]
            else:
                temporal_loop_dim_size[spatial_loop_dim] = q
        self.temporal_loop_dim_size = temporal_loop_dim_size

    def get_prime_factors(self):
        """
        Get the prime factors for all temporal loops.

        This is saved in three separate class attributes:
        temporal_loop_pfs: a dict that for each temporal loop dimension contains the prime factors
        temporal_loop_pf_counts: a dict that for each temporal loop dimension contains the prime factor multiplicities
        temporal_loop_pf_count_sums: a dict that for each temporal loop dimension contains the total amount of prime factors
        """
        temporal_loop_pfs = {}
        temporal_loop_pf_counts = {}
        temporal_loop_pf_count_sums = {}
        lpfs = []
        for (tl_dim, tl_size) in self.temporal_loop_dim_size.items():  # tl = temporal loop
            factors = factorint(tl_size)
            pfs = []
            counts = []
            for pf, multiplicity in factors.items():
                pfs.append(pf)
                counts.append(multiplicity)
                for i in range(multiplicity):
                    lpfs.append((tl_dim, pf))
            temporal_loop_pfs[tl_dim] = tuple(pfs)
            temporal_loop_pf_counts[tl_dim] = tuple(counts)
            temporal_loop_pf_count_sums[tl_dim] = sum(counts)

        logger.info(f"Generated {len(lpfs)} LPFs for layer {self.layer}.")

        self.temporal_loop_pfs = temporal_loop_pfs
        self.temporal_loop_pf_counts = temporal_loop_pf_counts
        self.temporal_loop_pf_count_sums = temporal_loop_pf_count_sums
        self.lpfs = lpfs

        # Limit the number of lpfs (if this is set in the settings)
        self.limit_lpfs()

    def limit_lpfs(self):
        """
        Function to limit the total number of loop prime factors present in this instance.
        This function scans the lpfs and while the number of lpfs is greater than self.lpf_limit it:
        - picks the loop dimension that has the most lpfs
        - merges the smallest two lpfs of that loop dimension (multiplying their values)
        """
        n_pf = sum(self.temporal_loop_pf_count_sums.values())
        if n_pf <= self.lpf_limit:
            logger.info(f"No lpf limiting performed for layer {self.layer}")
            return
        while n_pf > self.lpf_limit:
            # Find the loop dimension with the most lpfs
            max_ld = max(self.temporal_loop_pf_count_sums.items(), key=operator.itemgetter(1))[0]
            # Get the prime factors of this loop dimension
            max_pfs = list(self.temporal_loop_pfs[max_ld])
            # Get the multiplicity of these prime factors
            max_counts = list(self.temporal_loop_pf_counts[max_ld])

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
        lpfs = []
        for dim in self.temporal_loop_pfs.keys():
            for (pf, count) in zip(self.temporal_loop_pfs[dim], self.temporal_loop_pf_counts[dim]):
                lpfs += list(((dim, pf),) * count)
        self.lpfs = lpfs

        logger.info(f"Limited layer {self.layer} to {len(self.lpfs)} lpfs.")
        return

    def og(self):
        """
        Generator that yields all orderings of the temporal loops.
        """
        # The lpfs are stored in self.lpfs
        return permutations(self.lpfs)


if __name__ == "__main__":
    from inputs.examples.workload1 import workload
    from classes.hardware.architecture.memory_hierarchy import memory_hierarchy_example1
    from classes.hardware.architecture.operational_array import multiplier_array_example1


    layer = workload[1]
    layer_node = LayerNode(layer)

    '''This dictionary should be generated automatically from the user provided spatial mapping if there is one'''
    spatial_mapping_dic = {'W': [[('OX', 14), ('OY', 7)], [('FX', 3)], [], []],
                           'I': [[], [('OX', 14), ('FX', 3)], [('OY', 7)], []],
                           'O': [[('FX', 3)], [('OX', 14), ('OY', 7)], []]}

    spatial_mapping = SpatialMapping(spatial_mapping_dict=spatial_mapping_dic, layer_node=layer_node)

    multiplier_array = multiplier_array_example1()
    memory_hierarchy = memory_hierarchy_example1(multiplier_array)

    loma_engine = LomaEngine(layer=layer_node,
                                    spatial_mapping=spatial_mapping,
                                    memory_hierarchy=memory_hierarchy)
    loma_engine.run()
    a = 1
