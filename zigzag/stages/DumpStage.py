import pickle
from typing import Any


from zigzag.stages.Stage import Stage, StageCallable
from zigzag.cost_model.cost_model import CostModelEvaluation
import os


class DumpStage(Stage):
    """! Class that passes through all results yielded by substages, but dumps the results as a pickled list to a file
    at the end of the iteration
    """

    def __init__(self, list_of_callables: list[StageCallable], *, dump_filename_pattern: str, **kwargs: Any):
        """
        @param list_of_callables: see Stage
        @param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        @param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self):
        """! Run the compare stage by comparing a new cost model output with the current best found result."""
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        dump_list: list[tuple[CostModelEvaluation, Any]] = []
        filename = self.dump_filename_pattern.format(**self.kwargs)
        for cme, extra_info in substage.run():
            dump_list.append((cme, extra_info))
            yield cme, extra_info
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(dump_list, f, -1)
