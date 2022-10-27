from typing import Generator, Callable, List, Tuple, Any
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation

import logging
import os
logger = logging.getLogger(__name__)


class RemoveExtraInfoStage(Stage):
    """
    Strips extra info for subcallables to save memory
    """
    def __init__(self, list_of_callables, **kwargs):
        """
        Initialize the remove extra info stage.
        """
        super().__init__(list_of_callables, **kwargs)

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the remove extra info stage by running the substage and discarding the extra_info.
        """
        sub_list_of_callables = self.list_of_callables[1:]
        substage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        for cme, extra_info in substage.run():
           yield cme, None

class CacheBeforeYieldStage(Stage):
    """
    Caches results in a list and then yields them.
    This breaks the yield flow from top to bottom.
    """
    def __init__(self, list_of_callables, **kwargs):
        """
        Initialize the cache before yield stage.
        """
        super().__init__(list_of_callables, **kwargs)

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the cache before yield stage by running the substage and caching everything it yields, then yielding everything.
        """
        sub_list_of_callables = self.list_of_callables[1:]
        substage = self.list_of_callables[0](sub_list_of_callables, **self.kwargs)

        to_yield = []
        for ty in substage.run():
           to_yield.append(ty)
        for ty in to_yield:
            yield ty

class SkipIfDumpExistsStage(Stage):
    """
    Check if the output file is already generated, skip the run if so.
    """
    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self):
        filename = self.dump_filename_pattern.format(**self.kwargs)
        if os.path.isfile(filename):
            print(f"==================================Dump {filename} already existed. Skip! ==================================")
            return
        substage = self.list_of_callables[0](self.list_of_callables[1:], dump_filename_pattern=self.dump_filename_pattern, **self.kwargs)
        for cme, extra in substage.run():
            yield cme, extra




import multiprocessing


threadpool = None

def get_threadpool(nb_threads_if_non_existent):
    global threadpool
    if threadpool is None:
        threadpool = multiprocessing.Pool(nb_threads_if_non_existent)
    return threadpool

def close_threadpool():
    global threadpool
    threadpool.close()
    threadpool = None

def terminate_threadpool():
    global threadpool
    threadpool.terminate()
    threadpool = None


def raise_exception(e):
    terminate_threadpool()
    raise e

class MultiProcessingSpawnStage(Stage):
    """
    Multiprocessing support stage.
    Warning: does not yield (CostModelEvaluation, extra_info) pairs.
    Use as follows in a list_of_callables:
    [..., ..., MultiProcessingGatherStage, some stage(s) that loop over stuff and just yield (cme, extra_info) pairs
     every iteration without postprocessing it, MultiProcessingSpawnStage, ..., ...]
    Note: list of callables may not contain lambda functions, as this will break pickling which is required for
          by multiprocessing
    Note: there is quite some overhead in spawning these parallel processes (python...; it needs to copy through pickle
          all variables), so best to do this at some high level loop (early in list of callables)
    """
    def __init__(self, list_of_callables, *, multiprocessing_callback, nb_multiprocessing_threads=multiprocessing.cpu_count(), **kwargs):
        """

        :param list_of_callables: may not contain lambda functions, as this will break pickling which is required for
          by multiprocessing.
        :param multiprocessing_callback: intended to be set by MultiProcessingGatherStage
        :param kwargs:
        """
        super().__init__(list_of_callables, **kwargs)
        self.nb_multiprocessing_threads = nb_multiprocessing_threads
        self.callback = multiprocessing_callback

    def _to_run(self):
        return list(self.sub_stage.run())



    def run(self):
        self.sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        get_threadpool(self.nb_multiprocessing_threads).apply_async(self._to_run, callback=self.callback, error_callback=raise_exception)
        yield None, None

class MultiProcessingGatherStage(Stage):
    """
    Multiprocessing support stage.
    Use as follows in a list_of_callables:
    [..., ..., MultiProcessingGatherStage, some stage(s) that loop over stuff and just yield (cme, extra_info) pairs
     every iteration without postprocessing it, MultiProcessingSpawnStage, ..., ...]
    Note: list of callables may not contain lambda functions, as this will break pickling which is required for
          by multiprocessing
    """
    def _callback(self, ans):
        self.queue.put(ans)

    def run(self):
        self.queue = multiprocessing.Manager().Queue()
        kwargs = self.kwargs.copy()
        kwargs['multiprocessing_callback'] = self._callback

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        count_to_get = 0
        for i in sub_stage.run():
            count_to_get += 1
        logger.info(f'Multiprocessing results to get: {count_to_get}')
        count = 0
        while count < count_to_get:
            for ans in self.queue.get(block=True):
                yield ans
            count += 1
            if count % (count_to_get // 10) == 0:
                logger.info(f'Multiprocessing results received: {count} of {count_to_get}')
        close_threadpool()


