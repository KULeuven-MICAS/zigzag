import numpy as np
import logging
import random

from sympy.ntheory import factorint

import classes.io.input_config as inputs
from classes.hardware.architecture.accelerator import Accelerator
from classes.workload.layer_node import LayerNode
from classes.mapping.spatial.spatial_mapping import SpatialMapping
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from classes.opt.temporal.loma.multipermute import permutations
from classes.opt.temporal.loma.memory_allocator import MemoryAllocator
from classes.opt.temporal.loma.cost_model import CostModelCaller
from classes.opt.temporal.loma.engine import LomaEngine
from classes.stages.loma_pipeline import LomaPipeline
from classes.opt.temporal.salsa.state import SalsaState
from utils import pickle_deepcopy

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

    def __init__(self, main_inputs: inputs.MainInputs):
        """
        Initialize the engine with the given:
        - LayerNode
        - SpatialMapping
        - Accelerator
        - Number of iterations
        - Start temperature
        The memory hierarchy from the correct core is extracted from the accelerator.
        """
        self.__main_inputs = main_inputs
        self.best_state = None
        self.loma_pipeline:LomaPipeline = None

    def set_main_inputs(self, main_inputs: inputs.MainInputs):
        """
        Set the main inputs of this instance to the new main_inputs
        :param main_inputs: to be set. Is NOT copied
        :return:
        """
        self.__main_inputs = main_inputs

    def run(self):
        """
        Call the necessary methods and return the best temporal mapping found during the run.
        """

        self.get_temporal_loops()
        self.get_prime_factors()
        self.runtime_estimation()

        if self.salsa_fastest_engine:
            self.best_state = self.run_simulated_annealing_opt()
            best_energy = self.best_state.energy
            logger.info(f"Best found energy total for this spatial mapping: {best_energy:.4e}.")
            return self.best_state.cme
        else:
            # Call Loma Engine on the current input
            self.loma_pipeline = LomaPipeline(self.__main_inputs)
            self.loma_pipeline.run() 
            best_energy = self.loma_pipeline.best_cme.energy_total
            logger.info(f"Best found energy total for this spatial mapping: {best_energy:.4e}.")
            return self.loma_pipeline.best_cme

    def get_temporal_loops(self):
        """
        Get all loops that have to be temporally scheduled given layer and spatial mapping.
        """
        temporal_loop_dim_size = self.__main_inputs.layer.loop_dim_size.copy()  # init with all loop sizes
        for spatial_loop in self.__main_inputs.spatial_mapping.spatial_loop_dim_size:
            (spatial_loop_dim, spatial_loop_size) = spatial_loop
            q, rem = divmod(temporal_loop_dim_size[spatial_loop_dim], spatial_loop_size)
            assert rem == 0, "Division of dimension size by spatial unrolling size is not an integer"
            if q == 1:
                del temporal_loop_dim_size[spatial_loop_dim]
            else:
                temporal_loop_dim_size[spatial_loop_dim] = q
        self.temporal_loop_dim_size = temporal_loop_dim_size
    
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

        logger.info(f"Generated {len(lpfs)} LPFs for layer {self.layer}.")

        for loop_type in list(temporal_loop_pfs.keys()):
          for i in range(len(temporal_loop_pfs[loop_type])):
               loop_size = temporal_loop_pfs[loop_type]
               for number_of_loop in range(temporal_loop_pf_counts[loop_type][i]):
                    self.__main_inputs.temporal_mapping_lpf.append((loop_type, loop_size[i]))
        
    def runtime_estimation(self):
        """P
        Estimate the runtime of SALSA and LOMA and set self.salsa_fastest_engine according to the estimation.
        It is based on an approximation of the number of combination a core can evaluate per second, since the
        execution time of the simulated annealing optimization increase in a linear way we can get a good approximation.
        The combination_evaluation_per_second is fixed here and specific to my processor (Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz).
        """
        # TODO: Include the number of core used by Loma
        loma_engine = LomaEngine(main_inputs=self.__main_inputs)
        loma_engine.get_temporal_loops()
        loma_engine.get_prime_factors()
        ordering_counter = 0 

        for ordering in loma_engine.og():
            ordering_counter += 1
        
        # For 3000 iterations the simulated annealing optimizer takes 8 seconds,
        # more precise estimation will be done later.
        if ordering_counter / self.combination_evaluation_per_second > self.iteration_number * (8/3000):
            self.salsa_fastest_engine = True
            logger.info(f"Using SALSA engine")
        else:
            self.salsa_fastest_engine = False
            logger.info(f"Using LOMA engine")

    def run_simulated_annealing_opt(self):
        """
        Run a simulated annealing optimiation on the loop ordering using a loma memory allocation strategy.
        """
        starttemp = self.__main_inputs.settings.start_temperature
        iternb = self.__main_inputs.settings.iteration_number
        end_temperature = starttemp * (0.999**iternb)  # temperature at the end of the optimization
        temperature_linspace = np.flip(np.linspace(end_temperature, starttemp, iternb))
        start_ordering = self.temporal_mapping_lpf  # tmo stands for temporal mapping ordering

        # Initialize the algorithm with a random starting point
        random.shuffle(start_ordering)

        # Initialize variables to store current, next and best state
        best_state = SalsaState(self.__main_inputs, start_ordering)
        current_state = SalsaState(self.__main_inputs, start_ordering)
        next_state = SalsaState(self.__main_inputs, start_ordering)

        for temperature in temperature_linspace:
            # Get the index of the loop to swap
            i = np.random.randint(0, len(current_state.ordering))
            j = np.random.randint(0, len(current_state.ordering))

            # Swap the loops
            next_state = current_state.swap(i,j)

            x = np.random.rand() # x belongs to [0, 1]
            p = np.exp(((current_state.energy / next_state.energy) - 1) / temperature) # probability of accepting the next state

            if x < p:
                # Replace the current state by the next state and compare the energy with the best state
                current_state = pickle_deepcopy(next_state)
                
                if current_state.energy < best_state.energy:
                    best_state = pickle_deepcopy(current_state)

        return best_state

