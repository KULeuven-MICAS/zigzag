import pickle
from copy import deepcopy
from typing import Dict, List, Tuple
from typing import TYPE_CHECKING
from collections import defaultdict


if TYPE_CHECKING:
    from zigzag.classes.cost_model.cost_model import CostModelEvaluation


def pickle_deepcopy(to_copy):
    copy = None
    copied = False
    try:
        copy = pickle.loads(pickle.dumps(to_copy, -1))
        return copy
    except:
        pass
        # fallback to other options

    if not copied:
        return deepcopy(to_copy)
