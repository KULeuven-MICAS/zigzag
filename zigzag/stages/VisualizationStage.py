import logging
from contextlib import redirect_stdout
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
from zigzag.visualization.results.print_mapping import print_mapping

logger = logging.getLogger(__name__)


class VisualizationStage(Stage):
    """! Class that passes through all results yielded by substages, and saves the visualizations of configurations
    and results.
    """

    def __init__(self, list_of_callables: list[StageCallable], *, dump_folder: str, **kwargs: Any):
        """
        @param dump_folder: Output folder for dumps
        @param kwargs: any kwargs, passed on to substages
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_folder = dump_folder
        self.loop_ordering_file = dump_folder + "/loop_ordering.txt"
        # Save global figures (i.e. not dependent on layer/CME) only once when iterating through CMEs
        self.figure_is_saved = False

    def run(self):
        self.kwargs["dump_folder"] = self.dump_folder
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            if isinstance(cme, CostModelEvaluation):
                # Save global figures
                if not self.figure_is_saved:
                    self.__save_mem_hierarchy(cme)

                self.__save_loop_ordering(cme)

            yield cme, extra_info

    def __save_loop_ordering(self, cme: CostModelEvaluation):
        # Save loop ordering for all CMEs to single file (append)
        with open(self.loop_ordering_file, "a", encoding="UTF-8") as f:
            with redirect_stdout(f):
                print_mapping(cme)

    def __save_mem_hierarchy(self, cme: CostModelEvaluation):
        visualize_memory_hierarchy_graph(
            cme.accelerator.cores[0].memory_hierarchy,
            save_path=self.dump_folder + "/mem_hierarchy.png",
        )
        self.figure_is_saved = True
