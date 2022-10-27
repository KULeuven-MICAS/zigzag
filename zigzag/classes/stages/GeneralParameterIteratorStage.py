import logging

from typing import Generator, Callable, List, Tuple, Any
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
logger = logging.getLogger(__name__)


class GeneralParameterIteratorStage(Stage):
    """
    General iterator over any parameter whose values can be set from a predetermined list
    """
    def __init__(self, list_of_callables, *, general_parameter_iterations, **kwargs):
        """

        :param list_of_callables: see Stage
        :param general_parameter_iterations: dictionary with:
                - keys: variables to iterate over, or tuples of variables to iterate over
                        With K1 and K2 both keys, all combinations of K1 values and K2 values are tried.
                        With K1 and K2 together in a tuple as key, their values are paired and the dictionary value
                        must be a list (or other iterable) with tuples containing the values for K1 and K2
               - values: a list of values (single arg key) or a list of tuples of values (multi arg keys)
        :param kwargs: see Stage
        """
        super().__init__(list_of_callables, **kwargs)
        self.param_iters = general_parameter_iterations

    def recursive_run(self, reduced_param_iters, runparams):
        if reduced_param_iters:
            key = next(iter(reduced_param_iters))
            reduced_param_iters_reduced = reduced_param_iters.copy()
            runparams = runparams.copy()
            del reduced_param_iters_reduced[key]
            for v in reduced_param_iters[key]:
                if isinstance(key, (list, tuple)):
                    for kk, vv in zip(key, v):
                        runparams[kk] = vv
                    iterable = True
                else:
                    runparams[key] = v
                    iterable = False
                for cme, extra_info in self.recursive_run( reduced_param_iters_reduced, runparams):
                    yield cme, ( (tuple((kk, vv) for kk, vv in zip(key, v)) + extra_info[0],) + extra_info[1:] \
                                 if iterable else \
                               (((key, v),) + extra_info[0],) + extra_info[1:] )
        else:
            # trivial case, no more extra parameters to iterate over
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **runparams)
            for cme, extra_info in sub_stage.run():
                yield cme, (tuple(), extra_info)

    def run(self):
        return self.recursive_run(self.param_iters, self.kwargs)


if __name__ == '__main__':
    class Dummy(Stage):
        def is_leaf(self):
            return True
        def run(self):
            yield None, self.kwargs
    from zigzag.classes.stages.Stage import MainStage
    DUT = MainStage([GeneralParameterIteratorStage, Dummy],
                       general_parameter_iterations = {
        ('arg1.1', 'arg1.2'): ((111, 121), (112, 122), (113, 123)),
        'arg2': (21, 22, 23, 24, 25),
        'arg3': (31, 32)
        })
    for l in DUT.run():
        print(l)