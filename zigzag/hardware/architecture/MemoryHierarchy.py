from collections import defaultdict
from typing import Any, Iterator

import networkx as nx
from networkx import DiGraph

from zigzag.datatypes import MemoryOperand
from zigzag.hardware.architecture.memory_level import MemoryLevel, ServedMemDimensions
from zigzag.hardware.architecture.memory_port import PortAllocation
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.operational_array import OperationalArrayABC
from zigzag.utils import json_repr_handler


class MemoryHierarchy(DiGraph):
    """! Class that represents a memory hierarchy as a directed networkx graph.
    The memory hierarchy graph is directed, with the root nodes representing the lowest level
    in the memory hierarchy.
    """

    def __init__(
        self,
        operational_array: OperationalArrayABC,
        name: str = "Memory Hierarchy",
        **attr: Any,
    ):
        """
        Initialize the memory hierarchy graph.
        The initialization sets the operational array this memory hierarchy will connect to.
        The graph nodes are the given nodes. The edges are extracted from the operands the memory levels store.
        @param nodes: a list of MemoryLevels. Entries need to be provided from lowest to highest memory level.
        """
        super().__init__(**attr)  # type: ignore
        self.name: str = name
        self.operational_array = operational_array
        # Initialize the set that will store all memory operands
        self.operands: set[MemoryOperand] = set()
        # Initialize the dict that will store how many memory levels an operand has
        self.nb_levels: dict[MemoryOperand, int] = {}
        self.mem_level_list: list[MemoryLevel] = []
        self.memory_level_id = 0

    def add_memory(
        self,
        memory_instance: MemoryInstance,
        operands: list[MemoryOperand],
        port_alloc: PortAllocation,
        served_dimensions: ServedMemDimensions,
    ):
        """! Adds a memory to the memory hierarchy graph.
        NOTE: memory level need to be added from bottom level (e.g., Reg) to top level (e.g., DRAM) for each operand !!!

        Internally a MemoryLevel object is built, which represents the memory node.

        Edges are added from all sink nodes in the graph to this node if the memory operands match
        @param memory_instance: The MemoryInstance containing the different memory characteristics.
        @param operands: The memory operands the memory level stores.
        @param served_dimensions: The operational array dimensions this memory level serves. Default: no served
            dimensions -> unroll over
        """

        # Add the memory operands to the self.operands set attribute that stores all memory operands.
        for mem_op in operands:
            if mem_op not in self.operands:
                self.nb_levels[mem_op] = 1
                self.operands.add(mem_op)
            else:
                self.nb_levels[mem_op] += 1
            self.operands.add(mem_op)

        # Compute which memory level this is for all the operands
        mem_level_of_operands: dict[MemoryOperand, int] = {}
        for mem_op in operands:
            nb_levels_so_far = len([node for node in self.memory_nodes if mem_op in node.operands])
            mem_level_of_operands[mem_op] = nb_levels_so_far

        memory_level = MemoryLevel(
            memory_instance=memory_instance,
            operands=operands,
            mem_level_of_operands=mem_level_of_operands,
            port_alloc=port_alloc,
            served_dimensions=served_dimensions,
            operational_array=self.operational_array,
            identifier=self.memory_level_id,
        )
        self.mem_level_list.append(memory_level)
        self.memory_level_id += 1

        # Pre-compute appropriate edges
        to_edge_from: set[MemoryLevel] = set()
        for mem_op in operands:
            # Find top level memories of the operands
            for m in self.get_operator_top_level(mem_op)[0]:
                to_edge_from.add(m)

        # Add the node to the graph
        self.__add_node(memory_level)

        for sink_node in to_edge_from:
            # Add an edge from this sink node to the current node
            self.__add_edge(sink_node, memory_level)

    def get_memory_levels(self, mem_op: MemoryOperand) -> list[MemoryLevel]:
        """! Returns a list of memories in the memory hierarchy for the memory operand.
        The first entry in the returned list is the innermost memory level.
        """
        # Sort the nodes topologically and filter out all memories that don't store mem_op
        return [node for node in self.topological_sort() if mem_op in node.operands]

    def get_operands(self) -> set[MemoryOperand]:
        """! Returns all the memory operands this memory hierarchy graph contains as a set."""
        return self.operands

    def get_inner_memories(self) -> list[MemoryLevel]:
        """! Returns the inner-most memory levels for all memory operands."""
        return [node for node, in_degree in self.in_degree() if in_degree == 0]  # type: ignore

    def get_outer_memories(self) -> list[MemoryLevel]:
        """! Returns the outer-most memory levels for all memory operands."""
        return [node for node, out_degree in self.out_degree() if out_degree == 0]  # type: ignore

    def get_top_memories(self) -> tuple[list[MemoryLevel], int]:
        """! Returns the 'top'-most MemoryLevels, where 'the' level of MemoryLevel is considered to be the largest
        level it has across its assigned operands
        @return (list_of_memories_on_top_level, top_level)
        """
        level_to_mems: defaultdict[int, list[MemoryLevel]] = defaultdict(lambda: [])
        for node in self.memory_nodes:
            level_to_mems[max(node.mem_level_of_operands.values())].append(node)
        top_level = max(level_to_mems.keys())
        return level_to_mems[top_level], top_level

    def get_operator_top_level(self, operand: MemoryOperand) -> tuple[list[MemoryLevel], int]:
        """! Finds the highest level of memories that have the given operand assigned to it, and returns the MemoryLevel
        instance on this level that have the operand assigned to it.
        'The' level of a MemoryLevel is considered to be the largest level it has across its assigned operands.
        """
        level_to_mems: dict[int, list[MemoryLevel]] = defaultdict(lambda: [])
        for node in self.memory_nodes:
            if operand in node.operands:
                level_to_mems[max(node.mem_level_of_operands.values())].append(node)
        top_level = max(level_to_mems.keys()) if level_to_mems else -1
        return level_to_mems[top_level], top_level

    def get_operand_top_level(self, operand: MemoryOperand) -> MemoryLevel:
        """! Finds the highest level of memory that have the given operand assigned to, and returns the MemoryLevel"""
        top_lv = self.nb_levels[operand] - 1
        for mem in reversed(self.mem_level_list):
            if operand in mem.mem_level_of_operands.keys():
                if mem.mem_level_of_operands[operand] == top_lv:
                    return mem
        raise ValueError(f"Operand {operand} not found in any of the memory instances.")

    def topological_sort(self) -> Iterator[MemoryLevel]:
        """! Wrap `DiGraph.topological_sort` with correct type annotation"""
        return nx.topological_sort(self)  # type: ignore

    def __add_node(self, node: MemoryLevel) -> None:
        """! Wrap `DiGraph.add_node` with correct type annotation"""
        self.add_node(node)  # type: ignore

    def __add_edge(self, sink_node: MemoryLevel, source_node: MemoryLevel):
        """! Wrap `DiGraph.add_edge` with correct type annotation"""
        self.add_edge(sink_node, source_node)  # type: ignore

    @property
    def memory_nodes(self) -> list[MemoryLevel]:
        """! Wrap `DiGraph.nodes()` with custom type annotation"""
        return list(self.nodes())  # type: ignore

    def __jsonrepr__(self):
        """! JSON Representation of this object to save it to a json file."""
        return json_repr_handler(list(self.topological_sort()))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MemoryHierarchy)
            and self.nb_levels == other.nb_levels
            and all([self_ml == other_ml for self_ml, other_ml in zip(self.memory_nodes, other.memory_nodes)])
        )
