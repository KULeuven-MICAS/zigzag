from typing import Generator, Any, Tuple
from classes.stages.Stage import Stage
from classes.cost_model.cost_model import CostModelEvaluation
import os
import json
from datetime import datetime
import numpy as np

import logging
logger = logging.getLogger(__name__)

class CompleteSaveStage(Stage):
    """
    Class that passes through all results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the complete save stage by running the substage and saving the CostModelEvaluation json representation.
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        
        for id, (cme, extra_info) in enumerate(substage.run()):
            cme: CostModelEvaluation
            filename = self.dump_filename_pattern.format(datetime=datetime.now().isoformat().replace(":", "-"))
            self.save_to_json(cme, filename=filename)
            logger.info(f"Saved CostModelEvaluation with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}")
            yield cme, extra_info

    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)

    @staticmethod
    def complexHandler(obj):
        # print(type(obj))
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if hasattr(obj, '__jsonrepr__'):
            return obj.__jsonrepr__()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable. Create a __jsonrepr__ method.")


class SimpleSaveStage(Stage):
    """
    Class that passes through results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    In this simple version, only the energy total and latency total are saved.
    """

    def __init__(self, list_of_callables, *, dump_filename_pattern, **kwargs):
        """
        :param list_of_callables: see Stage
        :param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables)
        :param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        """
        Run the simple save stage by running the substage and saving the CostModelEvaluation simple json representation.
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        
        for id, (cme, extra_info) in enumerate(substage.run()):
            cme: CostModelEvaluation
            filename = self.dump_filename_pattern.format(datetime=datetime.now().isoformat().replace(":", "-"))
            self.save_to_json(cme, filename=filename)
            logger.info(f"Saved CostModelEvaluation with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}")
            yield cme, extra_info


    def save_to_json(self, obj, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as fp:
            json.dump(obj, fp, default=self.complexHandler, indent=4)


    @staticmethod
    def complexHandler(obj):
        # print(type(obj))
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if hasattr(obj, '__simplejsonrepr__'):
            return obj.__simplejsonrepr__()
        else:
            raise TypeError(f"Object of type {type(obj)} is not serializable. Create a __simplejsonrepr__ method.")
