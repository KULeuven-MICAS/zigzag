# multipermute - permutations of a multiset
# Github: https://github.com/ekg/multipermute
# Erik Garrison <erik.garrison@bc.edu> 2010
# This module encodes functions to generate the permutations of a multiset
# following this algorithm:
# Algorithm 1
# Visits the permutations of multiset E. The permutations are stored
# in a singly-linked list pointed to by head pointer h. Each node in the linked
# list has a value field v and a next field n. The init(E) call creates a
# singly-linked list storing the elements of E in non-increasing order with h, i,
# and j pointing to its first, second-last, and last nodes, respectively. The
# null pointer is given by φ. Note: If E is empty, then init(E) should exit.
# Also, if E contains only one element, then init(E) does not need to provide a
# value for i.
# [h, i, j] ← init(E)
# visit(h)
# while j.n ≠ φ orj.v <h.v do
#     if j.n ≠    φ and i.v ≥ j.n.v then
#         s←j
#     else
#         s←i
#     end if
#     t←s.n
#     s.n ← t.n
#     t.n ← h
#     if t.v < h.v then
#         i←t
#     end if
#     j←i.n
#     h←t
#     visit(h)
# end while
# ... from "Loopless Generation of Multiset Permutations using a Constant Number
# of Variables by Prefix Shifts."  Aaron Williams, 2009


from abc import ABC, abstractmethod
from typing import Any

from zigzag.datatypes import LayerDim


class ListElement:
    def __init__(self, value: Any, next_elem: Any):
        self.value = value
        self.next_elem = next_elem

    def nth(self, n: int):
        o = self
        i = 0
        while i < n and o.next_elem is not None:
            o = o.next_elem
            i += 1
        return o


class PermutationConstraint(ABC):
    """! An abstract class to represent a constraint on a permutation."""

    @abstractmethod
    def is_valid(self, permutation: list[Any]) -> bool:
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        ...


class StaticPositionsConstraint(PermutationConstraint):
    """! A class to represent a constraint on a permutation that requires
    a predefined order for some or all elements."""

    static_positions: dict[int, LayerDim]

    def __init__(self, static_positions: dict[int, LayerDim]):
        self.static_positions = static_positions

    def is_valid(self, permutation: list[Any]) -> bool:
        return all(permutation[position][0] == item for position, item in self.static_positions.items())

    def is_empty(self):
        return not self.static_positions or len(self.static_positions) == 0


class StaticPositionsAndSizesConstraint(PermutationConstraint):
    """! A class to represent a constraint on a permutation
    that requires a predefined order and size for some or all elements."""

    static_positions_and_sizes: dict[int, tuple[LayerDim, int]]

    def __init__(self, static_positions_and_sizes: dict[int, tuple[LayerDim, int]]):
        self.static_positions_and_sizes = static_positions_and_sizes

    def is_valid(self, permutation: list[Any]) -> bool:
        return all(
            permutation[position][0] == item and permutation[position][1] == size
            for position, (item, size) in self.static_positions_and_sizes.items()
        )

    def is_empty(self):
        return not self.static_positions_and_sizes or len(self.static_positions_and_sizes) == 0


def init(multiset: list[Any]):
    multiset.sort()  # ensures proper non-increasing order
    h = ListElement(multiset[0], None)
    for item in multiset[1:]:
        h = ListElement(item, h)
    return h, h.nth(len(multiset) - 2), h.nth(len(multiset) - 1)


def visit(h: ListElement) -> list[Any]:
    """! Converts our bespoke linked list to a python list."""
    o = h
    this_list: list[Any] = []
    while o is not None:
        this_list.append(o.value)
        o = o.next_elem
    return this_list


def constrainded_permutations(multiset: list[Any], constraints: list[PermutationConstraint]):
    """! Generator providing all multiset permutations of a multiset with constraints."""
    h, i, j = init(multiset)
    if all(constr.is_valid(visit(h)) for constr in constraints):
        yield visit(h)
    while j.next_elem is not None or j.value < h.value:
        if j.next_elem is not None and i.value >= j.next_elem.value:
            s = j
        else:
            s = i
        t = s.next_elem
        s.next_elem = t.next_elem
        t.next_elem = h
        if t.value < h.value:
            i = t
        j = i.next_elem
        h = t
        if all(constr.is_valid(visit(h)) for constr in constraints):
            yield visit(h)


def permutations(multiset: list[Any]):
    """! Generator providing all multiset permutations of a multiset."""
    h, i, j = init(multiset)
    yield visit(h)
    while j.next_elem is not None or j.value < h.value:
        if j.next_elem is not None and i.value >= j.next_elem.value:
            s = j
        else:
            s = i
        t = s.next_elem
        s.next_elem = t.next_elem
        t.next_elem = h
        if t.value < h.value:
            i = t
        j = i.next_elem
        h = t
        yield visit(h)
