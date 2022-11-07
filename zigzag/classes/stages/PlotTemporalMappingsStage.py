import pickle
from typing import Generator, Any, Tuple
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown
import os

class PlotTemporalMappingsStage(Stage):
    """
    Class that passes through all results yielded by substages, but keeps the TMs cme's and saves a plot.
    """

    def __init__(self, list_of_callables, *, plot_filename_pattern, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.plot_filename_pattern = plot_filename_pattern

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the compare stage by comparing a new cost model output with the current best found result.
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        cmes = []
        filename = self.plot_filename_pattern
        for cme, extra_info in substage.run():
            cmes.append(cme)
            yield cme, extra_info
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        bar_plot_cost_model_evaluations_breakdown(cmes, filename)
