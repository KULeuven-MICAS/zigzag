from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.stages.Stage import StageCallable


class MainStage:
    """! Not actually a Stage, as running it does return (not yields!) a list of results instead of a generator
    Can be used as the main entry point
    """

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

    def run(self):
        answers: list[tuple[CostModelEvaluationABC, Any]] = []
        for cme, extra_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
            answers.append((cme, extra_info))
        return answers
