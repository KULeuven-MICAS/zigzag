from collections import defaultdict
import networkx as nx
from networkx import DiGraph

from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel, ServedMemDimensions, ServedMemDimsUserFormat
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
from zigzag.classes.workload.layer_node import MemOperandStr


class MemoryHierarchy(DiGraph):
    """!  Class that represents a memory hierarchy as a directed networkx graph.
    The memory hierarchy graph is directed, with the root nodes representing the lowest level
    in the memory hierarchy.
    """

    def __init__(
        self,
        operational_array: OperationalArray,
        name: str = "Memory Hierarchy",
        **attr,
    ):
        """!  The class constructor
        Initialize the memory hierarchy graph.
        The initialization sets the operational array this memory hierarchy will connect to.
        The graph nodes are the given nodes. The edges are extracted from the operands the memory levels store.
        @param nodes: a list of MemoryLevels. Entries need to be provided from lowest to highest memory level.
        """
        super().__init__(**attr)
        self.name = name
        self.operational_array = operational_array
        # Initialize the set that will store all memory operands
        self.operands: set[MemOperandStr] = set()
        # Initialize the dict that will store how many memory levels an operand has
        self.nb_levels: dict[MemOperandStr, int] = {}
        self.mem_level_list: list[MemoryLevel] = []
        self.memory_level_id = 0

    def __jsonrepr__(self):
        """!  JSON Representation of this object to save it to a json file."""
        return {"memory_levels": [node for node in nx.topological_sort(self)]}

    def __eq__(self, other) -> bool:
        if not isinstance(other, MemoryHierarchy):
            return False
        return self.nb_levels == other.nb_levels and all(
            [self_ml == other_ml for (self_ml, other_ml) in zip(self.nodes(), other.nodes())]
        )

    def add_memory(
        self,
        memory_instance: MemoryInstance,
        operands: tuple[str, ...],
        port_alloc: tuple[dict, ...] = (),
        # Default setting: no served dimensions = unroll over every Operational Array unit
        served_dimensions: ServedMemDimsUserFormat = (),
    ):
        """!  Adds a memory to the memory hierarchy graph.
        NOTE: memory level need to be added from bottom level (e.g., Reg) to top level (e.g., DRAM) for each operand !!!

        Internally a MemoryLevel object is built, which represents the memory node.

        Edges are added from all sink nodes in the graph to this node if the memory operands match
        @param memory_instance: The MemoryInstance containing the different memory characteristics.
        @param operands: The memory operands the memory level stores.
        @param served_dimensions: The operational array dimensions this memory level serves.
        Each vector in the set is a direction that is served.
        Use 'all' to represent all dimensions (i.e. the memory level is not unrolled).
        """
        served_dims_parsed = ServedMemDimensions.parse_user_format(served_dimensions)

        if port_alloc == ():
            # Define the standard port allocation scheme (this assumes one read port and one write port)
            if not (memory_instance.r_port == 1 and memory_instance.w_port == 1 and memory_instance.rw_port == 0):
                raise ValueError(
                    f"No port allocation was provided for memory level of instance {memory_instance} and doesn't match with standard port allocation generation of 1 read and 1 write port."
                )
            for operand in operands:
                if operand == "O":
                    port_alloc += (
                        {
                            "fh": "w_port_1",
                            "tl": "r_port_1",
                            "fl": "w_port_1",
                            "th": "r_port_1",
                        },
                    )
                else:
                    port_alloc += (({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None}),)

        # # Assert that if served_dimensions is a string, it is "all"
        # if isinstance(served_dimensions, str):
        #     assert served_dimensions == "all", "Served dimensions is a string, but is not all."

        # Add the memory operands to the self.operands set attribute that stores all memory operands.
        for mem_op in operands:
            if mem_op not in self.operands:
                self.nb_levels[mem_op] = 1
                self.operands.add(mem_op)
            else:
                self.nb_levels[mem_op] += 1
            self.operands.add(mem_op)

        # # Parse the served_dimensions by replicating it into a tuple for each memory operand
        # # as the MemoryLevel constructor expects this.
        # # TODO what is this ?
        # served_dimensions_repl = tuple([served_dimensions for _ in range(len(operands))])

        # Compute which memory level this is for all the operands
        mem_level_of_operands = {}
        for operand in operands:
            nb_levels_so_far = len([node for node in self.nodes() if operand in node.operands])
            mem_level_of_operands[operand] = nb_levels_so_far

        memory_level = MemoryLevel(
            memory_instance=memory_instance,
            operands=operands,
            mem_level_of_operands=mem_level_of_operands,
            port_alloc=port_alloc,
            served_dimensions=served_dims_parsed,
            operational_array=self.operational_array,
            id=self.memory_level_id,
        )
        self.mem_level_list.append(memory_level)
        self.memory_level_id += 1

        # Precompute appropriate edges
        to_edge_from = set()
        for mem_op in operands:
            # Find top level memories of the operands
            for m in self.get_operator_top_level(mem_op)[0]:
                to_edge_from.add(m)

        # Add the node to the graph
        self.add_node(memory_level)

        for sink_node in to_edge_from:
            # Add an edge from this sink node to the current node
            self.add_edge(sink_node, memory_level)

    def get_memory_levels(self, mem_op: str):
        """!  Returns a list of memories in the memory hierarchy for the memory operand.
        The first entry in the returned list is the innermost memory level.
        """
        # Sort the nodes topologically and filter out all memories that don't store mem_op
        memories = [node for node in nx.topological_sort(self) if mem_op in node.operands]
        return memories

    def get_operands(self):
        """!  Returns all the memory operands this memory hierarchy graph contains as a set."""
        return self.operands

    def get_inner_memories(self) -> list[MemoryLevel]:
        """!  Returns the inner-most memory levels for all memory operands."""
        memories = [node for node, in_degree in self.in_degree() if in_degree == 0]
        return memories

    def get_outer_memories(self) -> list[MemoryLevel]:
        """!  Returns the outer-most memory levels for all memory operands."""
        memories = [node for node, out_degree in self.out_degree() if out_degree == 0]
        return memories

    def get_top_memories(self) -> tuple[list[MemoryLevel], int]:
        """!  Returns the 'top'-most MemoryLevels, where 'the' level of MemoryLevel is considered to be the largest
        level it has across its assigned operands
        @return (list_of_memories_on_top_level, top_level)
        """
        level_to_mems = defaultdict(lambda: [])
        node: MemoryLevel
        for node in self.nodes():
            level_to_mems[max(node.mem_level_of_operands.values())].append(node)
        top_level = max(level_to_mems.keys())
        return level_to_mems[top_level], top_level

    # def remove_top_level(self) -> tuple[list[MemoryLevel], int]:
    #     """!  Removes the top level of this memory hierarchy.
    #     'The' level of MemoryLevel instance is considered to be the largest level it has across its assigned operands,
    #     and those with the highest appearing level will be removed from this MemoryHierarchy instance
    #     @return (removed_MemoryLevel_instances, new_number_of_levels_in_the_hierarchy)
    #     """
    #     to_remove, top_level = self.get_top_memories()
    #     for tr in to_remove:
    #         self.mem_level_list.remove(tr)
    #         self.remove_node(tr)

    #     for k in self.nb_levels:
    #         self.nb_levels[k] = len(
    #             set(node.mem_level_of_operands.get(k) for node in self.nodes() if k in node.mem_level_of_operands)
    #         )
    #     return to_remove, max(self.nb_levels.keys())

    def get_operator_top_level(self, operand) -> tuple[list[MemoryLevel], int]:
        """!  Finds the highest level of memories that have the given operand assigned to it, and returns the MemoryLevel
        instance on this level that have the operand assigned to it.
        'The' level of a MemoryLevel is considered to be the largest
        level it has across its assigned operands.
        @param operand
        @return level_to_mems[top_level], top_level
        """
        level_to_mems = defaultdict(lambda: [])
        node: MemoryLevel
        for node in self.nodes():
            if operand in node.operands[:]:
                level_to_mems[max(node.mem_level_of_operands.values())].append(node)
        top_level = max(level_to_mems.keys()) if level_to_mems else -1
        return level_to_mems[top_level], top_level

    def get_operand_top_level(self, operand) -> MemoryLevel:
        """!  Finds the highest level of memory that have the given operand assigned to, and returns the MemoryLevel
        @param operand
        """
        top_lv = self.nb_levels[operand] - 1
        for mem in reversed(self.mem_level_list):
            if operand in mem.mem_level_of_operands.keys():
                if mem.mem_level_of_operands[operand] == top_lv:
                    return mem
        raise ValueError(f"Operand {operand} not found in any of the memory instances.")

    # def remove_operator_top_level(self, operand):
    #     """!  Finds the highest level of memories that have the given operand assigned to it, and returns the MemoryLevel
    #     instance on this level that have the operand assigned to it AFTER removing the operand from its operands.
    #     'The' level of a MemoryLevel is considered to be the largest
    #     level it has across its assigned operands.
    #     If a memory has no operands left, it is removed altogether.
    #     @param operand
    #     @return list of MemoryLevel instance that have the operand removed, new top_level of the operand
    #     """
    #     to_remove, top_level = self.get_operator_top_level(operand)

    #     served_dimensions = []
    #     for tr in to_remove:
    #         del tr.mem_level_of_operands[operand]
    #         tr.operands.remove(operand)
    #         for p in tr.port_list:
    #             for so in p.served_op_lv_dir[:]:
    #                 if so[0] == operand:
    #                     p.served_op_lv_dir.remove(so)
    #         if len(tr.mem_level_of_operands) == 0:
    #             self.mem_level_list.remove(tr)
    #             self.remove_node(tr)

    #     for k in self.nb_levels:
    #         self.nb_levels[k] = len(
    #             set(node.mem_level_of_operands.get(k) for node in self.nodes() if k in node.mem_level_of_operands)
    #         )

    #     return to_remove, self.nb_levels[operand]
