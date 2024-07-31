from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.operational_array import OperationalArrayABC
from zigzag.mapping.spatial_mapping import SpatialMapping
from zigzag.utils import json_repr_handler


class Core:
    """! The Core class houses the array of multipliers and the attached memory hierarchy.
    This class supports a singular multiplier array and memory hierarchy, runtime flexibility should be implemented
    on top.
    """

    def __init__(
        self,
        core_id: int,
        operational_array: OperationalArrayABC,
        memory_hierarchy: MemoryHierarchy,
        dataflows: SpatialMapping | None = None,
    ):
        self.id = core_id
        self.id = core_id
        self.operational_array = operational_array
        self.memory_hierarchy = memory_hierarchy
        self.mem_hierarchy_dict: dict[MemoryOperand, list[MemoryLevel]] = {}

        self.dataflows = dataflows
        self.mem_hierarchy_dict: dict[MemoryOperand, list[MemoryLevel]] = {}

        self.dataflows = dataflows
        self.recalculate_memory_hierarchy_information()

    def get_memory_level(self, mem_op: MemoryOperand, mem_lv: int) -> MemoryLevel:
        """! Returns a specific memory level in the memory hierarchy for the memory operand"""
        # Sort the nodes topologically and filter out all memories that don't store mem_op
        memory = [node for node in self.memory_hierarchy.topological_sort() if mem_op in node.operands]
        return memory[mem_lv]

    def recalculate_memory_hierarchy_information(self):
        self.__generate_memory_hierarchy_dict()
        self.__generate_memory_sharing_list()

    def __generate_memory_hierarchy_dict(self):
        mem_operands = self.memory_hierarchy.nb_levels.keys()
        mem_hierarchy_dict: dict[MemoryOperand, list[MemoryLevel]] = {}
        mem_size_dict: dict[MemoryOperand, list[int]] = {}
        mem_r_bw_dict: dict[MemoryOperand, list[int]] = {}
        mem_w_bw_dict: dict[MemoryOperand, list[int]] = {}
        mem_r_bw_min_dict: dict[MemoryOperand, list[int]] = {}
        mem_w_bw_min_dict: dict[MemoryOperand, list[int]] = {}
        for mem_op in mem_operands:
            mem_hierarchy_dict[mem_op] = [
                node for node in self.memory_hierarchy.topological_sort() if mem_op in node.operands
            ]
            mem_size_dict[mem_op] = [
                node.memory_instance.size
                for node in self.memory_hierarchy.topological_sort()
                if mem_op in node.operands
            ]
            mem_r_bw_dict[mem_op] = [
                node.memory_instance.r_bw
                for node in self.memory_hierarchy.topological_sort()
                if mem_op in node.operands
            ]
            mem_w_bw_dict[mem_op] = [
                node.memory_instance.w_bw
                for node in self.memory_hierarchy.topological_sort()
                if mem_op in node.operands
            ]
            mem_r_bw_min_dict[mem_op] = [
                node.memory_instance.r_bw_min
                for node in self.memory_hierarchy.topological_sort()
                if mem_op in node.operands
            ]
            mem_w_bw_min_dict[mem_op] = [
                node.memory_instance.w_bw_min
                for node in self.memory_hierarchy.topological_sort()
                if mem_op in node.operands
            ]
        self.mem_hierarchy_dict = mem_hierarchy_dict
        self.mem_size_dict = mem_size_dict
        self.mem_r_bw_dict = mem_r_bw_dict
        self.mem_w_bw_dict = mem_w_bw_dict
        self.mem_r_bw_min_dict = mem_r_bw_min_dict
        self.mem_w_bw_min_dict = mem_w_bw_min_dict

    def __generate_memory_sharing_list(self):
        """! Generates a list of dictionary that indicates which operand's which memory levels are sharing the same
        physical memory"""
        memory_sharing_list: list[dict[MemoryOperand, int]] = []
        for mem_lv in self.mem_hierarchy_dict.values():
            for mem in mem_lv:
                operand_mem_share = mem.mem_level_of_operands
                if len(operand_mem_share) > 1 and operand_mem_share not in memory_sharing_list:
                    memory_sharing_list.append(operand_mem_share)

        self.mem_sharing_list = memory_sharing_list

    def get_top_memory_instance(self, mem_op: MemoryOperand) -> MemoryInstance:
        if mem_op not in self.memory_hierarchy.get_operands():
            raise ValueError(f"Memory operand {mem_op} not in {self}.")
        mem_level = self.memory_hierarchy.get_operand_top_level(mem_op)
        mem_instance = mem_level.memory_instance
        return mem_instance

    def get_memory_bw_dict(self):
        return self.mem_r_bw_dict, self.mem_w_bw_dict

    def get_memory_bw_min_dict(self):
        return self.mem_r_bw_min_dict, self.mem_w_bw_min_dict

    def __str__(self) -> str:
        return f"Core({self.id})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self):
        return json_repr_handler(self.__dict__)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Core)
            and self.id == other.id
            and self.operational_array == other.operational_array
            and self.memory_hierarchy == other.memory_hierarchy
        )
