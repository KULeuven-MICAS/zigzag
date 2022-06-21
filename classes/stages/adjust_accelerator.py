import logging
from utils import pickle_deepcopy
from typing import Callable, Any
import classes.io.input_config as inputs
from classes.hardware.architecture.accelerator import Accelerator

logger = logging.getLogger(__name__)

class AdjustAcceleratorStage:
    """
    Class to insert as a stage that adjusts the memory hierarchy before next stages are called.
    """

    def __init__(self, main_inputs: inputs.MainInputs, change: Callable[[inputs.MainInputs], Any]):
        """

        :param main_inputs: MainInputs, NOT copied
        """
        self.__main_inputs = main_inputs
        self.change_f = change

    def set_main_inputs(self, main_inputs: inputs.MainInputs):
        """
        Set the main inputs of this instance to the new main_inputs
        :param main_inputs: to be set. Is NOT copied
        :return:
        """
        self.__main_inputs = main_inputs

    def set_change(self, change: Callable[[Accelerator], Any]):
        self.change_f = change

    def run(self):
        self.change_f(self.__main_inputs)



#inputs.origonal_accelerator = input.accelerator
#For layers
#
#    new_accelerator = adjust_stage(inputs.accelerator).run()
#    cost_stage(new_accelerator)
#    whatever pipeline


