import logging
import pickle
from copy import deepcopy
from typing import Any

import numpy as np
import yaml


def pickle_deepcopy(to_copy: Any) -> Any:
    try:
        copy = pickle.loads(pickle.dumps(to_copy, -1))
        return copy
    except:  # noqa: E722 # pylint: disable=W0702
        return deepcopy(to_copy)


def pickle_save(to_save: str, path: str):
    with open(path, "wb") as fp:
        status = pickle.dump(to_save, fp)
    return status


def pickle_load(path: str):
    with open(path, "rb") as fp:
        obj = pickle.load(fp)
    return obj


def open_yaml(path: str):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def json_repr_handler(obj: Any, simple: bool = False) -> Any:
    """! Recursively converts objects into a json representation"""
    attr = "__simplejsonrepr__" if simple else "__jsonrepr__"

    # Recursive: catch end nodes
    if obj is None:
        return None
    if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool) or isinstance(obj, str):
        return obj
    if isinstance(obj, np.int32):  # type: ignore
        return int(obj)  # type: ignore
    if hasattr(obj, attr):
        return obj.__simplejsonrepr__() if simple else obj.__jsonrepr__()

    # Recursive calls
    if isinstance(obj, dict):
        return {json_repr_handler(k, simple): json_repr_handler(v, simple) for k, v in obj.items()}  # type: ignore
    if isinstance(obj, set):
        return json_repr_handler(list(obj), simple)  # type: ignore
    if isinstance(obj, list):
        return [json_repr_handler(x, simple) for x in obj]  # type: ignore
    if isinstance(obj, tuple):
        return tuple(json_repr_handler(x, simple) for x in obj)  # type: ignore

    raise TypeError(f"Object of type {type(obj)} is not serializable. Create a {attr} method.")


class UniqueMessageFilter(logging.Filter):
    """! Prevents the logger from filtering duplicate messages"""

    def __init__(self):
        super().__init__()
        self.recorded_messages: set[str] = set()

    def filter(self, record: logging.LogRecord):
        message = record.getMessage()
        if message in self.recorded_messages:
            return False  # Skip this message
        else:
            self.recorded_messages.add(message)
            return True
