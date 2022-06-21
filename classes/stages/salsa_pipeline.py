import classes.io.input_config as inputs
from classes.cost_model.cost_model import CostModelEvaluation
from classes.stages.cost_model_pipeline import CostModelPipeline
from classes.stages.salsa import SalsaStage


class SalsaPipeline:
    """
    Class that runs a single layer node through the salsa search engine.
    """
    def __init__(self, main_inputs: inputs.MainInputs):
        """
        Initialize the pipeline by adding the salsa stage and the cost model pipeline.
        :param main_inputs: MainInputs, NOT copied
        """
        self.salsa_stage = SalsaStage(main_inputs)
        self.cost_model_pipeline = CostModelPipeline(main_inputs=main_inputs)
        self.best_cme: CostModelEvaluation = None
        self.__main_inputs = main_inputs

    def set_main_inputs(self, main_inputs: inputs.MainInputs):
        """
        Set the main inputs of this instance to the new main_inputs
        :param main_inputs: to be set. Is NOT copied
        :return:
        """
        self.salsa_stage.set_main_inputs(main_inputs)
        self.cost_model_pipeline.set_main_inputs(main_inputs)

    def run(self):
        """
        Initialize the input of the salsa engine and run it, then 
        gather the output temporal mapping.
        """
        # Set the correct inputs for the salsa stage as this might have changed/not been set yet.
        self.best_cme = self.salsa_stage.run()

