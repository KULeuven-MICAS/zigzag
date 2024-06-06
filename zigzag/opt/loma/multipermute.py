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


from typing import Any


class ListElement:

    def __init__(self, value: Any, next: Any):
        self.value = value
        self.next = next

    def nth(self, n: int):
        o = self
        i = 0
        while i < n and o.next is not None:
            o = o.next
            i += 1
        return o


def init(multiset: list[Any]):
    multiset.sort()  # ensures proper non-increasing order
    h = ListElement(multiset[0], None)
    for item in multiset[1:]:
        h = ListElement(item, h)
    return h, h.nth(len(multiset) - 2), h.nth(len(multiset) - 1)


def visit(h: ListElement) -> list[Any]:
    """! Converts our bespoke linked list to a python list."""
    o = h
    l: list[Any] = []
    while o is not None:
        l.append(o.value)
        o = o.next
    return l


def permutations(multiset: list[Any]):
    """! Generator providing all multiset permutations of a multiset."""
    h, i, j = init(multiset)
    yield visit(h)
    while j.next is not None or j.value < h.value:
        if j.next is not None and i.value >= j.next.value:
            s = j
        else:
            s = i
        t = s.next
        s.next = t.next
        t.next = h
        if t.value < h.value:
            i = t
        j = i.next
        h = t
        yield visit(h)
