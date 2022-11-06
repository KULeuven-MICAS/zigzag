from typing import Set
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance


class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    """
    def __init__(self, name, core_set: Set[Core], global_buffer: MemoryInstance or None):
        self.name = name
        self.cores = sorted([core for core in core_set], key=lambda core: core.id)
        self.global_buffer = global_buffer

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, 
                "cores": self.cores}

    def get_core(self, core_id: int or str) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        core = next((core for core in self.cores if core.id == core_id), None)
        if not core:
            raise ValueError(f"Requested core with id {core_id} is not present in accelerator.")
        return core

def accelerator_example():
    from zigzag.classes.hardware.architecture.core import core_example
    core1, core2 = core_example()
    cores = {core1, core2}
    global_buffer = MemoryInstance(name="sram_256KB_BW_384b", size=2097152, bw=(384, 384), cost=(10, 15), area=25, bank=4,
                                   random_bank_access=True, rd_port=1, wr_port=1, rd_wr_port=0, latency=1)
    accelerator = Accelerator("example", cores, global_buffer)
    return accelerator


if __name__ == "__main__":
    accelerator = accelerator_example()
    pass