from typing import Any
import os
import pickle
import json
import yaml
import logging

from zigzag.cost_model.cost_model import CostModelEvaluation, CostModelEvaluationABC, CumulativeCME
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.utils import json_repr_handler

logger = logging.getLogger(__name__)


class CompleteSaveStage(Stage):
    """! Class that passes through all results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    """

    def __init__(self, list_of_callables: list[StageCallable], *, dump_filename_pattern: str, **kwargs: Any):
        """
        @param dump_filename_pattern: filename string formatting pattern, which can use named field whose values will be
        in kwargs (thus supplied by higher level runnables). Must contain `?`
        @param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_filename_pattern = dump_filename_pattern

    def run(self):
        """! Run the complete save stage by running the substage and saving the CostModelEvaluation json
        representation."""
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            if isinstance(cme, CumulativeCME):
                filename = self.dump_filename_pattern.replace("?", "overall_complete")
            elif isinstance(cme, CostModelEvaluation):
                filename = self.dump_filename_pattern.replace("?", f"{cme.layer}_complete")
            else:
                raise NotImplementedError

            self.save_to_json(cme, filename=filename)
            yaml_name = os.path.join(os.path.splitext(filename)[0], ".yml")
            self.save_to_yaml(json_name=filename, yaml_name=yaml_name)
            logger.info(
                f"Saved {cme} with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}"
            )
            yield cme, extra_info

    def save_to_json(self, obj: object, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as fp:
            json.dump(obj, fp, default=json_repr_handler, indent=4)

    def save_to_yaml(self, json_name: str, yaml_name: str):
        os.makedirs(os.path.dirname(yaml_name), exist_ok=True)
        with open(json_name, "r") as fp:
            res = json.load(fp)
        with open(yaml_name, "w") as fp:
            yaml.dump(res, fp, Dumper=yaml.SafeDumper)


class SimpleSaveStage(Stage):
    """! Class that passes through results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    In this simple version, only the energy total and latency total are saved.
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
        """! Run the simple save stage by running the substage and saving the CostModelEvaluation simple json
        representation."""
        self.kwargs["dump_filename_pattern"] = self.dump_filename_pattern
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            if isinstance(cme, CumulativeCME):
                filename = self.dump_filename_pattern.replace("?", "overall_simple")
            elif isinstance(cme, CostModelEvaluation):
                filename = self.dump_filename_pattern.replace("?", f"{cme.layer}_simple")
            else:
                raise NotImplementedError
            self.save_to_json(cme, filename=filename)
            logger.info(
                f"Saved {cme} with energy {cme.energy_total:.3e} and latency {cme.latency_total2:.3e} to {filename}"
            )
            yield cme, extra_info

    def save_to_json(self, obj: CostModelEvaluationABC, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="UTF-8") as fp:

            def handler(obj: object):
                return json_repr_handler(obj, simple=True)

            json.dump(obj, fp, default=handler, indent=4)


class PickleSaveStage(Stage):
    """! Class that dumps all received CMEs into a list and saves that list to a pickle file."""

    def __init__(self, list_of_callables: list[StageCallable], *, pickle_filename: str, **kwargs: Any):
        """
        @param list_of_callables: see Stage
        @param pickle_filename: output pickle filename
        @param kwargs: any kwargs, passed on to substages and can be used in dump_filename_pattern
        """
        super().__init__(list_of_callables, **kwargs)
        self.pickle_filename = pickle_filename

    def run(self):
        """! Run the simple save stage by running the substage and saving the CostModelEvaluation simple json representation.
        This should be placed above a ReduceStage such as the SumStage, as we assume the list of CMEs is passed as extra_info
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for cme, extra_info in substage.run():
            all_cmes: list[CostModelEvaluation] = [cme for (cme, _) in extra_info]
            yield cme, extra_info

        # After we have received all the CMEs, save them to the specified output location.
        dirname = os.path.dirname(self.pickle_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(self.pickle_filename, "wb") as handle:
            pickle.dump(all_cmes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved pickled list of {len(all_cmes)} CMEs to {self.pickle_filename}.")
