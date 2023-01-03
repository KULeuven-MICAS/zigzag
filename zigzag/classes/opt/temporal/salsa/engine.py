#   =====================================================================
#   Title:        engine.py
#   Description: This file contains the engine class that handles the 
#   optimization of temporal mapping of SALSA.
#  
#   Date:        02.01.2023
#  
#   =====================================================================
# 
#   Copyright (C) 2020 ETH Zurich and University of Bologna.
#  
#   Author: Victor Jung, ETH Zurich
#  
#   SPDX-License-Identifier: Apache-2.0
#  
#   Licensed under the Apache License, Version 2.0 (the License); you may
#   not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  
#   www.apache.org/licenses/LICENSE-2.0
#  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an AS IS BASIS, WITHOUT
#   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  

from copy import deepcopy
from sympy.ntheory import factorint
import numpy as np
import logging
import random

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.opt.temporal.loma.multipermute import permutations
from zigzag.classes.opt.temporal.loma.memory_allocator import MemoryAllocator
from zigzag.classes.opt.temporal.salsa.state import SalsaState

logger = logging.getLogger(__name__)


class SalsaEngine:
    """
    Class that handles optimization of temporal mapping given a:
    - layer
    - spatial mapping
    - memory hierarchy
    - number of iterations
    - start temperature
    This optimization is carried out through simulated annealing loop order based.
    Each loop is broken down to the smallest possible part (prime factors), then a runtime
    estimation is performed to choose the fastest engine to use (LOMA or SALSA).
    """

    def __init__(self, *, accelerator: Accelerator, layer: LayerNode, spatial_mapping: SpatialMapping, **kwargs):
    
    #iteration_number, start_temperature, opt_criterion_name
        """
        Initialize the engine with the given:
        - LayerNode
        - SpatialMapping
        - Accelerator
        - Number of iterations
        - Start temperature
        The memory hierarchy from the correct core is extracted from the accelerator.
        """
        # Hardware and mapping related inputs
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        #self.memory_hierarchy: MemoryHierarchy = self.accelerator.get_core(layer.core_allocation).memory_hierarchy

        # Algorithm related inputs
        self.iteration_number = kwargs.get("salsa_iteration_number", 1000)
        self.start_temperature = kwargs.get("salsa_start_temperature", 0.05)
        self.opt_criterion_name = kwargs.get("salsa_opt_criterion", "energy")
        self.lpf_limit = kwargs.get("loma_lpf_limit", 4)


    def run(self, cme_queue):
        """
        Call the necessary methods, start the processes and collect the best temporal mapping found during the run.
        """
        self.cme_queue = cme_queue  
        self.get_temporal_loops()
        self.get_prime_factors()
        self.run_simulated_annealing_opt(self.cme_queue)
    

    def run_simulated_annealing_opt(self, cme_queue):
        """
        Run a simulated annealing optimiation on the loop ordering using a loma memory allocation strategy.
        """
        temperature = self.start_temperature
        start_ordering = self.temporal_mapping_lpf # tmo stands for temporal mapping ordering

        # Initialize the algorithm with a random starting point
        random.shuffle(start_ordering)

        # Initialize variables to store current, next and best state
        best_state = SalsaState(self.accelerator, self.layer, self.spatial_mapping, start_ordering, self.opt_criterion_name)
        current_state = SalsaState(self.accelerator, self.layer, self.spatial_mapping, start_ordering, self.opt_criterion_name)
        next_state = SalsaState(self.accelerator, self.layer, self.spatial_mapping, start_ordering, self.opt_criterion_name)

        for it in range(self.iteration_number):

            temperature = self.start_temperature*(0.995**it)

            # Get the index of the loop to swap
            i = np.random.randint(0, len(current_state.ordering))
            j = np.random.randint(0, len(current_state.ordering))

            # Swap the loops
            next_state = current_state.swap(i,j)

            x = np.random.rand() # x belongs to [0, 1]
            p = np.exp(((current_state.opt_criterion / next_state.opt_criterion) - 1) / temperature) # probability of accepting the next state

            if x < p:
                # Replace the current state by the next state and compare the energy with the best state
                current_state = deepcopy(next_state)
                
                if current_state.opt_criterion< best_state.opt_criterion:
                    best_state = deepcopy(current_state)
        
        cme_queue.put(best_state.cme)


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

        # Remove all dimensions with a temporal loop size of 1
        temporal_loop_dim_size_no_1s = {key: val for (key, val) in temporal_loop_dim_size.items() if val > 1}

        self.temporal_loop_dim_size = temporal_loop_dim_size_no_1s
        min_nb_temporal_loops = len(self.temporal_loop_dim_size)
        if self.lpf_limit < min_nb_temporal_loops:
            logger.debug(f"Updated layer {self.layer}'s lpf limit from {self.lpf_limit} to {min_nb_temporal_loops} lpfs.")
            self.lpf_limit = min_nb_temporal_loops
    

    def get_prime_factors(self):
        """
        Get the prime factors for all temporal loops in the following format:
        [('C', 2), ('OY', 2), ('OX', 2), ('K', 7), ...]
        """
        temporal_loop_pfs = {}
        temporal_loop_pf_counts = {}
        temporal_loop_pf_count_sums = {}
        lpfs = []

        self.temporal_mapping_lpf = []

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

        #logger.info(f"Generated {len(lpfs)} LPFs for layer {self.layer}.")

        for loop_type in list(temporal_loop_pfs.keys()):
          for i in range(len(temporal_loop_pfs[loop_type])):
               loop_size = temporal_loop_pfs[loop_type]
               for number_of_loop in range(temporal_loop_pf_counts[loop_type][i]):
                    self.temporal_mapping_lpf.append((loop_type, loop_size[i]))