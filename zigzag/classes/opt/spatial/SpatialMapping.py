from functools import reduce
from typing import TypeAlias
from copy import deepcopy
from dataclasses import dataclass
import itertools
import logging
import math
from pydantic import BaseModel

from zigzag.classes.hardware.architecture.dimension import Dimension


logger = logging.getLogger(__name__)


"""
Type aliases for legacy data structures
"""
UnrollFactor: TypeAlias = int | float
LayerDimStr: TypeAlias = str
LimitedUSM: TypeAlias = dict[Dimension, list[tuple[LayerDimStr, UnrollFactor]]]
UserSpatialMappingLegacy: TypeAlias = dict[Dimension, dict[LayerDimStr, UnrollFactor]]


@dataclass
class LayerDim:
    """! (for-loop) dimension of a workload layer (e.g. `K`, `C`)"""

    name: str

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class MappingSingleOADim:
    """! Spatial unrolling for a single Operational Array Dimension"""

    def __init__(self, data: dict[LayerDim, UnrollFactor]):
        self.data: dict[LayerDim, UnrollFactor] = data

    def get_nb_unrolled_dims(self):
        """! Return the number of different layer dimensions unrolled (unroll factor > 1) over this
        Operational Array (spatial) Dimension"""
        return len([x for x in self.data.values() if x > 1])

    def is_nested(self):
        """! Return True iff multiple layer dimensions are unrolled over this MappingSingleOADim"""
        return self.get_nb_unrolled_dims() > 1

    def get_utilization(self):
        """! Returns the `hardware utilization`, i.e. the product of all unrolled dimensions"""
        return math.prod([factor for factor in self.data.values()])

    def layer_dims(self) -> set[LayerDim]:
        return set(self.keys())

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key: LayerDim):
        assert isinstance(key, LayerDim)
        return self.data[key]

    def __delitem__(self, key: LayerDim):
        del self.data[key]

    def __contains__(self, key: LayerDim):
        return self.data.__contains__(key)

    def items(self):
        return self.data.items()

    def __setitem__(self, key: LayerDim, value: UnrollFactor):
        self.data[key] = value

    def __str__(self):
        return str(self.data)

    def __jsonrepr__(self):
        return {layer_dim.name: str(unroll_factor) for layer_dim, unroll_factor in self.items()}


class SpatialMapping:
    """! Spatial unrollings defined for every operational array dimension"""

    def __init__(self, data: dict[Dimension, MappingSingleOADim]):
        assert isinstance(data, dict)
        self.data = data

    # TODO make layer_dim_size a property of LayerDim
    def is_valid(
        self, max_unrollings: "SpatialMapping", oa_dims: list[Dimension], layer_dim_sizes: dict[LayerDim, int]
    ):
        """! Return True iff
        1) the utilization at each OA Dimension does not exceed the size of the Dimension
        2) the instance does not contain LayerDims that are not bounded by `max_unrollings`
        3) each LayerDim unrolling does not exceed the unrolling prescribed in max_unrollings
        4) the instance does not contain OA Dimensions that are not part of the given list
        5) the total unrolling of each contained LayerDim does not exceed the LayerDim size
        6) the instance does not contain LayerDims that are not bounded by the given layer_dim_sizes
        @param max_unrollings a SpatialMapping instance that contains the maximally allowed
        unrolling for each Layer Dimension in each OA Dimension individually
        @param oa_dims list of Operational Array Dimensions that should be included in this instance
        @param layer_dim_sizes dict that contains the size of each LayerDim
        """
        if not all(oa_dim in self for oa_dim in oa_dims):
            return False

        for oa_dim in oa_dims:
            for layer_dim, unrolling in self[oa_dim].items():
                if layer_dim not in max_unrollings[oa_dim]:
                    return False
                if unrolling > max_unrollings[oa_dim][layer_dim]:
                    return False

            if self[oa_dim].get_utilization() > oa_dim.size:
                return False

        if not all(oa_dim in oa_dims for oa_dim in self.data):
            return False

        if not all([layer_dim in layer_dim_sizes for layer_dim in self.get_all_contained_layer_dims()]):
            return False

        if not all(
            [self.get_total_unrolling_of_layer_dim(layer_dim) <= size for layer_dim, size in layer_dim_sizes.items()]
        ):
            return False

        return True

    def check_and_reduce(
        self, max_unrollings: "SpatialMapping", oa_dims: list[Dimension], layer_dim_sizes: dict[LayerDim, int]
    ):
        """! Verify
        - that the utilization at each OA Dimension does not exceed the size of the Dimension
        - that each LayerDim unrolling does not exceed the unrolling prescribed in max_unrollings
        Reduce the unrollings otherwise.
        # TODO maybe it is better to throw an error when the user-provided spatial mapping is invalid?
        """
        assert all([oa_dim in oa_dims for oa_dim in self.oa_dims()]), "Unexpected OA Dim found"
        assert all(
            [layer_dim in layer_dim_sizes for layer_dim in self.get_all_contained_layer_dims()]
        ), "Unexpected LayerDim found"

        # Check every OA Dim separately
        for oa_dim, mapping_this_oa_dim in self.items():
            # First check individual LayerDim unrollings
            for layer_dim, unrolling in mapping_this_oa_dim.items():
                max_unrolling = max_unrollings[oa_dim][layer_dim]
                if unrolling > max_unrolling:
                    logger.info(
                        "User provided spatial unrolling of %i for %s in Dimension %s exceeded maximally allowed \
                            unrolling of %i. Reducing unrolling to this value.",
                        unrolling,
                        str(layer_dim),
                        str(oa_dim),
                        max_unrolling,
                    )
                    self[oa_dim][layer_dim] = max_unrolling

            # Check full OA Dimension
            if mapping_this_oa_dim.get_utilization() > oa_dim.size:
                logger.info(
                    "User provided spatial unrolling of for Dimension %s exceeded maximally allowed unrolling of %i. \
                        Removing arbitrary Layer unrollings to meet this constraint",
                    str(oa_dim),
                    oa_dim.size,
                )
                while mapping_this_oa_dim.get_utilization() > oa_dim.size:
                    # Remove any LayerDim
                    some_layer_dim = mapping_this_oa_dim.layer_dims().pop()
                    del mapping_this_oa_dim[some_layer_dim]

        # Check every LayerDim separately, over all OA Dims
        for layer_dim in self.get_all_contained_layer_dims():
            layer_size = layer_dim_sizes[layer_dim]
            if self.get_total_unrolling_of_layer_dim(layer_dim) > layer_size:
                logger.info(
                    "User provided spatial unrolling of for Layer Dimension %s exceeded layer size of %i. Removing \
                        unrollings for %s in arbitrary Dimensions to meet this constraint",
                    layer_dim.name,
                    layer_size,
                    layer_dim.name,
                )
                while self.get_total_unrolling_of_layer_dim(layer_dim) > layer_size:
                    some_oa_dim = set(
                        filter(lambda x: layer_dim in self[x], self.oa_dims())  # pylint: disable=W0640
                    ).pop()
                    del self[some_oa_dim][layer_dim]

    # TODO make the return type clearer
    def get_utilization(self) -> UnrollFactor:
        """! Returns the `hardware utilization`, i.e. the product of all unrolled dimensions"""
        return math.prod([x.get_utilization() for x in self.values()])

    def get_all_contained_layer_dims(self) -> set[LayerDim]:
        """! Return a set containing all the LayerDims contained in the mapping at any OA Dim"""
        return set([layer_dim for layer_dim, _ in self.flatten_unrollings()])

    def get_total_unrolling_of_layer_dim(self, layer_dim: LayerDim) -> UnrollFactor:
        """! Return the total unroll factor of a given Layer Dimension, over all Operational Array Dimensions"""
        return math.prod([v for k, v in self.flatten_unrollings() if k == layer_dim])

    def flatten_unrollings(self) -> list[tuple[LayerDim, UnrollFactor]]:
        """! Convert all unrollings (pair of LayerDim and UnrollFactor) at all Operational Array Dimension to a single
        list of tuples.
        e.g. -> [('K', 4), ('C', 2), ('K', 8)]"""
        return reduce(
            lambda a, b: a + b, [list(mapping_single_dim.items()) for mapping_single_dim in self.mappings()], []
        )

    def oa_dims(self) -> set[Dimension]:
        return set(self.keys())

    def mappings(self) -> set[MappingSingleOADim]:
        return set(self.values())

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def __getitem__(self, key: Dimension) -> MappingSingleOADim:
        assert isinstance(key, Dimension)
        return self.data[key]

    def __setitem__(self, key: Dimension, value: MappingSingleOADim):
        self.data[key] = value

    def __delitem__(self, b):
        pass

    def __contains__(self, key: Dimension):
        return self.data.__contains__(key)

    def __len__(self):
        return len(self.data)

    def copy(self):
        return SpatialMapping(deepcopy(self.data))

    def __str__(self):
        return str(self.__jsonrepr__())

    def __jsonrepr__(self):
        return {oa_dim.name: mapping.__jsonrepr__() for oa_dim, mapping in self.items()}

    # def convert_to_legacy(self) -> UserSpatialMappingLegacy:
    #     """! Convert this instance to the format used throughout older parts of the code base"""
    #     return {
    #         oa_dim: {layer_dim.name: unrolling for layer_dim, unrolling in mapping_single_oa_dim.items()}
    #         for oa_dim, mapping_single_oa_dim in self.items()
    #     }

    @staticmethod
    def parse_user_input(x: dict[Dimension, tuple[str, UnrollFactor] | tuple[tuple]]):
        """! Parse legacy notation
        Example input: {"D1": ("OX", 25), "D2": (("FX", 3), ("FY", 3))}
        """
        if x is None:
            return SpatialMapping({})

        if isinstance(x, list):
            raise NotImplementedError("No support for multiple provided spatial mappings by user")

        data: dict[Dimension, MappingSingleOADim] = {}
        for k, v in x.items():
            mapping_single_dim_dict: dict[LayerDim, UnrollFactor] = {}

            if isinstance(v[0], tuple):
                # v: tuple[tuple[str, UnrollFactor]]
                # Nested layer dimensions
                for layer_dim, factor in v:  # type: ignore
                    mapping_single_dim_dict[LayerDim(layer_dim)] = factor  # type: ignore
            else:
                # v : tuple[str, UnrollFactor]
                layer_dim, factor = v  # type: ignore
                mapping_single_dim_dict[LayerDim(layer_dim)] = factor  # type: ignore

            data[k] = MappingSingleOADim(mapping_single_dim_dict)
        return SpatialMapping(data)


class SpatialMappingHint:
    """! Suggested LayerDims to be unrolled for every Operational Array Dimension"""

    def __init__(self, data: dict[Dimension, set[LayerDim]]):
        self.data = data

    def complete_with_defaults(self, oa_dims: list[Dimension], layer_dims: set[LayerDim]):
        for oa_dim in filter(lambda x: x not in self, oa_dims):
            self.data[oa_dim] = layer_dims

    def __getitem__(self, key: Dimension):
        return self.data[key]

    # def items(self):
    #     return self.data.items()

    # def __setitem__(self, key: Dimension, value: set[LayerDim]):
    #     self.data[key] = value

    def __contains__(self, key: Dimension):
        return self.data.__contains__(key)

    @staticmethod
    def parse_user_input(x: dict[str, list]):
        if x is None:
            return SpatialMappingHint({})
        return SpatialMappingHint(
            {Dimension.parse_user_input(k): {LayerDim(layer_dim) for layer_dim in v} for k, v in x.items()}
        )
