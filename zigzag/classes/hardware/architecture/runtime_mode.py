from zigzag.classes.hardware.architecture.operational_array import OperationalArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy


class RuntimeMode:
    """
    This class holds a single runtime mode of an accelerator. A mode is defined by:
     - an operand sharing (embedded in the operational array
     - a memory hierarchy?
    """
    def __init__(self, id: int, operational_array: OperationalArray, memory_hierarchy: MemoryHierarchy):
        self.id = id
        self.operational_array = operational_array
        self.memory_hierarchy = memory_hierarchy

    def __str__(self):
        return f"mode {self.id}"

    def __repr__(self):
        return str(self)

if __name__ == "__main__":
    from zigzag.classes.hardware.architecture.operational_array import multiplier_array_example1, multiplier_array_example2, OperationalArray
    from zigzag.classes.hardware.architecture.memory_hierarchy import memory_hierarchy_example1, memory_hierarchy_example2
    multiplier_array1 = multiplier_array_example1()
    memory_hierarchy1 = memory_hierarchy_example1(multiplier_array1)

    multiplier_array2 = multiplier_array_example2()
    memory_hierarchy2 = memory_hierarchy_example2(multiplier_array2)

    runtime_mode1 = RuntimeMode(id=1, operational_array=multiplier_array1, memory_hierarchy=memory_hierarchy1)
    runtime_mode2 = RuntimeMode(id=2, operational_array=multiplier_array2, memory_hierarchy=memory_hierarchy2)
    runtime_modes = [runtime_mode1, runtime_mode2]
    for rt_mode in runtime_modes:
        print(rt_mode, rt_mode.operational_array.operand_spatial_sharing)