from typing import Set
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance


class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    """

    def __init__(self, name, core_set: Set[Core]):
        self.name = name
        self.cores = sorted([core for core in core_set], key=lambda core: core.id)

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.cores}

    def get_core(self, core_id: int or str) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        core = next((core for core in self.cores if core.id == core_id), None)
        if not core:
            raise ValueError(
                f"Requested core with id {core_id} is not present in accelerator."
            )
        return core
