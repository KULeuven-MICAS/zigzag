import pickle
from copy import deepcopy
from typing import Dict, List, Tuple, TypeVar, overload
from typing import TYPE_CHECKING
from collections import defaultdict

from zigzag.classes.hardware.architecture.memory_level import ServedMemDimsUserFormat


if TYPE_CHECKING:
    from zigzag.classes.cost_model.cost_model import CostModelEvaluation


def pickle_deepcopy(to_copy):
    try:
        copy = pickle.loads(pickle.dumps(to_copy, -1))
        return copy
    except:
        return deepcopy(to_copy)


def pickle_save(to_save, path):
    with open(path, "wb") as fp:
        status = pickle.dump(to_save, fp)
    return status


def pickle_load(path):
    with open(path, "rb") as fp:
        obj = pickle.load(fp)
    return obj
