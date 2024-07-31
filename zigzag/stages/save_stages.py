import json
import logging
import os
import pickle
from typing import Any

from zigzag.cost_model.cost_model import (
    CostModelEvaluation,
    CostModelEvaluationABC,
    CumulativeCME,
)
from zigzag.stages.Stage import Stage, StageCallable
from zigzag.utils import json_repr_handler

logger = logging.getLogger(__name__)


class CompleteSaveStage(Stage):
    """! Class that passes through all results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    """

    def __init__(self, list_of_callables: list[StageCallable], *, dump_folder: str, **kwargs: Any):
        """
        @param dump_folder: Output folder for dumps
        @param kwargs: any kwargs, passed on to substages
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_folder = dump_folder

    def run(self):
        """! Run the complete save stage by running the substage and saving the CostModelEvaluation json
        representation."""
        self.kwargs["dump_folder"] = self.dump_folder
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            if isinstance(cme, CumulativeCME):
                json_filename = self.dump_folder + "/overall_complete.json"
            elif isinstance(cme, CostModelEvaluation):
                # Slashes are interpreted by subfolders and must be replaced in the file name
                layer_name = cme.layer.name.replace("/", "_")
                json_filename = self.dump_folder + f"/{layer_name}_complete.json"
            else:
                raise NotImplementedError

            self.save_to_json(cme, filename=json_filename)
            logger.info(
                "Saved %s with energy %s and latency %s to %s",
                cme,
                f"{cme.energy_total:.3e}",
                f"{cme.latency_total2:.3e}",
                json_filename,
            )
            yield cme, extra_info

    def save_to_json(self, obj: object, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="UTF-8") as fp:
            json.dump(obj, fp, default=json_repr_handler, indent=4)


class SimpleSaveStage(Stage):
    """! Class that passes through results yielded by substages, but saves the results as a json list to a file
    at the end of the iteration.
    In this simple version, only the energy total and latency total are saved.
    """

    def __init__(self, list_of_callables: list[StageCallable], *, dump_folder: str, **kwargs: Any):
        """
        @param list_of_callables: see Stage
        @param dump_folder: Output folder for dumps
        @param kwargs: any kwargs, passed on to substages
        """
        super().__init__(list_of_callables, **kwargs)
        self.dump_folder = dump_folder

    def run(self):
        """! Run the simple save stage by running the substage and saving the CostModelEvaluation simple json
        representation."""
        self.kwargs["dump_folder"] = self.dump_folder
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in substage.run():
            if isinstance(cme, CumulativeCME):
                json_filename = self.dump_folder + "/overall_simple.json"
            elif isinstance(cme, CostModelEvaluation):
                # Slashes are interpreted by subfolders and must be replaced in the file name
                layer_name = cme.layer.name.replace("/", "_")
                json_filename = self.dump_folder + f"/{layer_name}_simple.json"
            else:
                raise NotImplementedError

            self.save_to_json(cme, filename=json_filename)
            logger.info(
                "Saved %s with energy %s and latency %s to %s",
                cme,
                f"{cme.energy_total:.3e}",
                f"{cme.latency_total2:.3e}",
                json_filename,
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

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        pickle_filename: str,
        **kwargs: Any,
    ):
        """
        @param list_of_callables: see Stage
        @param pickle_filename: output pickle filename
        @param kwargs: any kwargs, passed on to substages
        """
        super().__init__(list_of_callables, **kwargs)
        self.pickle_filename = pickle_filename

    def run(self):
        """! Run the simple save stage by running the substage and saving the CostModelEvaluation simple json
        representation. This should be placed above a ReduceStage such as the SumStage, as we assume the list of CMEs is
        passed as extra_info
        """
        substage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        for cme, extra_info in substage.run():
            all_cmes: list[CostModelEvaluationABC] = [cme for (cme, _) in extra_info]
            yield cme, extra_info

        # After we have received all the CMEs, save them to the specified output location.
        dirname = os.path.dirname(self.pickle_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        try:
            with open(self.pickle_filename, "wb") as handle:
                pickle.dump(all_cmes, handle, protocol=pickle.HIGHEST_PROTOCOL)  # type: ignore
            logger.info(
                "Saved pickled list of %i CMEs to %s.",
                len(all_cmes),  # type: ignore
                self.pickle_filename,
            )
        except NameError:
            logger.warning("No CMEs found to save in PickleSaveStage")
