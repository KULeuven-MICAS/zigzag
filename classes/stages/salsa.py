from classes.hardware.architecture.accelerator import Accelerator
from classes.mapping.spatial.spatial_mapping import SpatialMapping
from classes.workload.layer_node import LayerNode
import classes.io.input_config as inputs
from classes.opt.temporal.salsa.engine import SalsaEngine


class SalsaStage:
    """
    Class that return the best temporal mapping found by the simulated annealing
    loop ordering-based engine (salsa).
    """
    def __init__(self, main_inputs: inputs.MainInputs):
        """
        Initially the engine is set to None.
        When the stage is ran through the run() method, this will be feed
        to the salsa engine with parameters present in the inputs.
        :param main_inputs: MainInputs, NOT copied
        """
        self.__main_inputs = main_inputs
        self.engine: SalsaEngine = None

    def set_main_inputs(self, main_inputs):
        """
        Set the main inputs of this instance to the new main_inputs
        :param main_inputs: to be set. Is NOT copied
        :return:
        """
        self.__main_inputs = main_inputs

    def get_main_inputs(self, main_inputs):
        """
        :return: the main_inputs used by this instance. NOT a copy
        """
        return self.__main_inputs

    def run(self):
        """
        Run this stage by returning the generator returned by the loma engine.
        """

        self.engine = SalsaEngine(main_inputs=self.__main_inputs)
        ans = self.engine.run()
        self.engine = None
        return ans
