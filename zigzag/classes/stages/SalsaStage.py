#   =====================================================================
#   Title:        SalsaStage.py
#   Description:
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

import multiprocessing_on_dill as multiprocessing
from sympy.ntheory import factorint
from copy import deepcopy
import logging

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.opt.temporal.salsa.engine import SalsaEngine
from zigzag.classes.workload.layer_node import LayerNode
from typing import Generator, Callable, List, Tuple, Any
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation

from zigzag.classes.stages.ReduceStages import MinimalEnergyStage
from zigzag.classes.stages.ReduceStages import MinimalLatencyStage

logger = logging.getLogger(__name__)

class SalsaStage(Stage):
    """
    Class that return the best temporal mapping found by the Simulated Annealing
    Loop-ordering Scheduler for Accelerators (SALSA) for a single layer.
    """
    def __init__(self, list_of_callables: List[Callable], *, accelerator, layer, spatial_mapping, **kwargs):
        """Initialize the LomaStage by setting the accelerator, layer, and spatial mapping.

        Args:
            list_of_callables (List[Callable]): List of substages to call with each generated temporal mapping.
            accelerator (Accelerator): The accelerator object.
            layer (Layer): The layer object.
            spatial_mapping (SpatialMapping): The spatial mapping object.
        """
        super().__init__(list_of_callables, **kwargs)
        self.accelerator, self.layer, self.spatial_mapping = accelerator, layer, spatial_mapping
        self.engine = None
        self.best_cme = None

        self.opt_criterion_name = kwargs.get("salsa_opt_criterion", "energy")
        self.number_of_core_allocated = kwargs.get("salsa_number_of_core", 1)
        
        # Multiprocessing parameters
        self.worker_list = []
        self.cme_queue = multiprocessing.Queue()

        if self.opt_criterion_name == "energy":
            self.compare_stage = self.compare_cme_energy
        elif self.opt_criterion_name == "latency":
            self.compare_stage = self.compare_cme_latency
        else:  
            raise Exception("Invalid optimization criterion for SALSA. Must be either 'energy' or 'latency'.")


    def run(self):
        """
        Set up and start salsa engine, then collect and return the best cost model evaluation
        """

        logger.info(f"Running SALSA Temporal Mapping Optimizer with {self.number_of_core_allocated} core(s).")

        self.engine = SalsaEngine(accelerator=self.accelerator, 
                                    layer=self.layer, 
                                    spatial_mapping=self.spatial_mapping, 
                                    **self.kwargs)

        # self.best_cme = self.engine.run(self.cme_queue)

        # Get the number of core the user wants to allocate
        if self.number_of_core_allocated <= multiprocessing.cpu_count():
            self.number_of_core = self.number_of_core_allocated
        else:
            self.number_of_core = multiprocessing.cpu_count()

        # Create processes
        for core_id in range(0,self.number_of_core):
            p = multiprocessing.Process(target=self.engine.run, args=(self.cme_queue,))
            self.worker_list.append(p)

        # Start the processes
        for core_id in range(0,self.number_of_core):
            logger.debug(f"Starting SALSA Process #{core_id}.")
            self.worker_list[core_id].start()

        # For every core we gather the ouput
        for core_id in range(0,self.number_of_core):
            cme = self.cme_queue.get()
            self.compare_stage(cme)

        # Then join them to make sure they all end before continuing the execution
        for core_id in range(0,self.number_of_core):
            self.worker_list[core_id].join()


        kwargs = self.kwargs.copy()
        kwargs['accelerator'] = self.accelerator
        kwargs['layer'] = self.layer
        kwargs['spatial_mapping'] = self.spatial_mapping
        kwargs['temporal_mapping'] = self.best_cme.mapping.temporal_mapping
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)

        for cme, extra_info in sub_stage.run():
            yield cme, (self.best_cme.mapping.temporal_mapping, extra_info)


    def compare_cme_latency(self, cme):
        """
        Compare the latency of the current cost model evaluation with the best latency found so far.
        Then replace the current best cme if the current cme has a lower latency.
        """

        if self.best_cme is None:
            self.best_cme = cme
        elif cme.latency_total2 == self.best_cme.latency_total2 and cme.energy_total < self.best_cme.energy_total:
            self.best_cme = cme
        elif cme.latency_total2 < self.best_cme.latency_total2:
            self.best_cme = cme


    def compare_cme_energy(self, cme):
        """
        Compare the energy of the current cost model evaluation with the best energy found so far.
        Then replace the best cme if the current cme has a lower energy.
        """
        if self.best_cme is None:
            self.best_cme = cme
        elif cme.energy_total == self.best_cme.energy_total and cme.latency_total2 < self.best_cme.latency_total2:
            self.best_cme = cme
        elif cme.energy_total < self.best_cme.energy_total:
            self.best_cme = cme