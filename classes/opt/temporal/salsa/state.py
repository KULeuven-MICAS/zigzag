from utils import pickle_deepcopy
from classes.opt.temporal.loma.memory_allocator import MemoryAllocator
from classes.cost_model.cost_model import CostModelEvaluation

import classes.io.input_config as inputs
from classes.hardware.architecture.accelerator import Accelerator
from classes.workload.layer_node import LayerNode
from classes.mapping.spatial.spatial_mapping import SpatialMapping
from classes.hardware.architecture.memory_hierarchy import MemoryHierarchy

class SalsaState:
    """
    State storing an ordering, his temporal mapping and his energy value.
    """

    def __init__(self, main_inputs: inputs.MainInputs, ordering):
        self.__main_inputs = main_inputs
        self.ordering = ordering
        self.memory_hierarchy: MemoryHierarchy = self.__main_inputs.accelerator.get_core(
                                                     self.__main_inputs.layer.core_allocation).memory_hierarchy

        allocator = MemoryAllocator(self.__main_inputs)

        self.temporal_mapping = allocator.run()  # allocate this ordering to the memories 

        self.cme = CostModelEvaluation(self.__main_inputs)

        # The optimization criterion will be minimized
        if self.opt_criterion == "energy":
            self.energy = self.cme.energy_total 
        else:
            self.energy = None

    def set_main_inputs(self, main_inputs: input.MainInputs):
        """
        Set the main inputs of this instance to the new main_inputs
        :param main_inputs: to be set. Is NOT copied
        :return:
        """
        self.__main_inputs = main_inputs
        allocator = MemoryAllocator(self.__main_inputs)
        self.temporal_mapping = allocator.run()  # allocate this ordering to the memories
        self.cme.set_main_inputs(self.__main_inputs)

    def swap(self, i, j):
        """
        Swap between the element at positon i and j in the ordering 
        and return the new resulting state.
        """

        swapped_ordering = pickle_deepcopy(self.ordering)
        swapped_ordering[i], swapped_ordering[j] = swapped_ordering[j], swapped_ordering[i]

        return SalsaState(self.__main_inputs, swapped_ordering)
