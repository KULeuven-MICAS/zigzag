import copy
import logging
import math
from typing import Any

from zigzag.datatypes import LayerDim, OADimension, UnrollFactor, UnrollFactorInt
from zigzag.utils import UniqueMessageFilter, json_repr_handler
from zigzag.workload.LayerAttribute import LayerAttribute

logger = logging.getLogger(__name__)
logger.addFilter(UniqueMessageFilter())


class MappingSingleOADim:
    """! Spatial unrolling for a single OADimension"""

    def __init__(self, data: dict[LayerDim, UnrollFactor]):
        # float type is used in `SpatialMappingConversionStage`
        self.__data: dict[LayerDim, UnrollFactor] = data

    @property
    def utilization(self):
        """! Returns the `hardware utilization`, i.e. the product of all unrolled dimensions"""
        return math.prod([factor for factor in self.__data.values()])

    @property
    def layer_dims(self) -> set[LayerDim]:
        return set(self.keys())

    @property
    def unroll_sizes(self) -> list[UnrollFactor]:
        return list(self.__data.values())

    def keys(self):
        return self.__data.keys()

    def get_data(self):
        return self.__data

    def __getitem__(self, key: LayerDim):
        return self.__data[key]

    def __delitem__(self, key: LayerDim):
        del self.__data[key]

    def __contains__(self, key: LayerDim):
        return self.__data.__contains__(key)

    def items(self):
        return self.__data.items()

    def update(self, other: "MappingSingleOADim"):
        self.__data.update(other.get_data())

    def __setitem__(self, key: LayerDim, value: UnrollFactor | float):
        self.__data[key] = value  # type: ignore

    def __str__(self):
        return str({str(k): str(v) for k, v in self.items()}).replace("'", "")

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        return json_repr_handler(self.__data)

    def __eq__(self, other: Any) -> bool:
        """! Return true iff the contained LayerDims are the same and all unrollings are the same"""
        return (
            isinstance(other, MappingSingleOADim)
            and all([layer_dim in other for layer_dim in self.layer_dims])
            and all([layer_dim in self for layer_dim in other.layer_dims])
            and all([self[layer_dim] == other[layer_dim] for layer_dim in self.layer_dims])
        )

    def __hash__(self):
        return hash(frozenset(self.__data))


class SpatialMapping(LayerAttribute):
    """! Spatial unrollings defined for every operational array dimension"""

    def __init__(self, data: dict[OADimension, MappingSingleOADim]):
        self.data = data
        self.oa_dim_sizes: dict[OADimension, int] | None = None

    def initialize_oa_dims(self, oa_dim_sizes: dict[OADimension, int]) -> None:
        """! Initialize the OA Dimensions's sizes in this instance.
        When the SpatialMapping is parsed from the workload definition, the OADimension sizes are unknown and given
        in the hardware architecture definition instead. To come to a valid SpatialMapping, the OADimension sizes
        must be updated.
        @param oa_dim_sizes Mapping of OA Dimensions and corresponding sizes from the hardware architecture definition.
        """
        assert self.oa_dim_sizes is None, "OA Dimensions were already initialized"
        self.oa_dim_sizes = oa_dim_sizes

    def is_valid(
        self,
        max_unrollings: dict[OADimension, dict[LayerDim, UnrollFactorInt]],
        layer_dim_sizes: dict[LayerDim, UnrollFactor],
    ):
        """! Return True iff
        1) the instance's OA Dimensions have been initialized to the size from the hardware architecture definition
        2) the instance does not contain OA Dimensions that are not defined in the given list (no rogue OA Dim)
        3) all OA Dimensions defined in the given list are contained within this instance
        4) the instance does not contain LayerDims that are not bounded by `max_unrollings` (no rogue LayerDim)
        5) each LayerDim unrolling does not exceed the unrolling prescribed in max_unrollings
        6) the utilization at each OADimension does not exceed the size of the OADimension
        7) the instance does not contain LayerDims that are not bounded by the given layer_dim_sizes
        8) the total unrolling of each contained LayerDim does not exceed the LayerDim size
        @param max_unrollings a SpatialMapping instance that contains the maximally allowed unrolling for each
        Layer Dimension in each OADimension individually
        @param dict of `LayerDimSizes`. `LayerDimSizes` instance cannot be used due to circular import.
        @param oa_dim_sizes List of OA Dimensions from hardware architecture definition to compare to
        """
        # 1)
        assert self.oa_dim_sizes is not None, "Initialize OA Dimensions first"
        # 2)
        assert all(self_oa_dim in self.oa_dim_sizes for self_oa_dim in self.oa_dims), "Illegal OADimension found"
        # 3)
        assert all(given_oa_dim in self for given_oa_dim in self.oa_dim_sizes), "SpatialMapping misses OADimension"

        for oa_dim in self.oa_dims:
            for layer_dim, unrolling in self[oa_dim].items():
                # 4)
                if layer_dim not in max_unrollings[oa_dim]:
                    return False
                # 5)
                if unrolling > max_unrollings[oa_dim][layer_dim]:
                    return False
            # 6)
            if self[oa_dim].utilization > self.oa_dim_sizes[oa_dim]:
                return False

        # 7)
        if not all([layer_dim in layer_dim_sizes for layer_dim in self.all_contained_layer_dims]):
            return False

        # 8)
        if not all(
            [self.get_total_unrolling_of_layer_dim(layer_dim) <= size for layer_dim, size in layer_dim_sizes.items()]
        ):
            return False

        return True

    def check_and_reduce(
        self,
        max_unrollings: dict[OADimension, dict[LayerDim, UnrollFactorInt]],
        layer_dim_sizes: dict[LayerDim, UnrollFactor],
    ):
        """! Verify
        - that the utilization at each OADimension does not exceed the size of the OADimension
        - that each LayerDim unrolling does not exceed the unrolling prescribed in max_unrollings
        Reduce the unrollings otherwise.
        @param dict of `LayerDimSizes`. `LayerDimSizes` instance cannot be used due to circular import.

        """
        assert self.oa_dim_sizes is not None, "Initialize OA Dimensions first"

        # Remove LayerDim if not listed in `layer_dim_sizes`
        for layer_dim in self.all_contained_layer_dims:
            if layer_dim not in layer_dim_sizes:
                logger.warning(
                    "User provided spatial unrolling %s is not defined in Loop Dimension Sizes %s. Removing %s.",
                    layer_dim,
                    layer_dim_sizes,
                    layer_dim,
                )
                self.delete_layer_dim(layer_dim)

        # Check every OA Dim separately
        for oa_dim, mapping_this_oa_dim in self.items():
            # Limit unrolling to maximally allowed per LayerDim
            for layer_dim, unrolling in mapping_this_oa_dim.items():
                max_unrolling = max_unrollings[oa_dim][layer_dim]
                if unrolling > max_unrolling:
                    logger.warning(
                        "User provided spatial unrolling (%s:%i) in Dimension %s exceeded maximally allowed unrolling "
                        "of %i. Reducing unrolling to this value.",
                        layer_dim,
                        unrolling,
                        oa_dim,
                        max_unrolling,
                    )
                    self[oa_dim][layer_dim] = max_unrolling

            # Check full OADimension
            if mapping_this_oa_dim.utilization > self.oa_dim_sizes[oa_dim]:
                logger.info(
                    """User provided spatial unrolling of for Dimension %s exceeded maximally allowed unrolling of %i.
                    Removing arbitrary Layer unrollings to meet this constraint""",
                    oa_dim,
                    self.oa_dim_sizes[oa_dim],
                )
                while mapping_this_oa_dim.utilization > self.oa_dim_sizes[oa_dim]:
                    # Remove arbitrary LayerDim
                    some_layer_dim = next(iter(mapping_this_oa_dim.layer_dims))
                    del mapping_this_oa_dim[some_layer_dim]

        # Check every LayerDim separately, over all OA Dims
        for layer_dim in self.all_contained_layer_dims:
            layer_size = layer_dim_sizes[layer_dim]
            if self.get_total_unrolling_of_layer_dim(layer_dim) > layer_size:
                logger.warning(
                    """User provided spatial unrolling of for Layer Dimension %s exceeded layer size of %i. Removing
                    unrollings for %s in arbitrary Dimensions to meet this constraint""",
                    layer_dim.name,
                    layer_size,
                    layer_dim.name,
                )
                while self.get_total_unrolling_of_layer_dim(layer_dim) > layer_size:
                    some_oa_dim = set(
                        filter(lambda x: layer_dim in self[x], self.oa_dims)  # pylint: disable=W0640
                    ).pop()
                    del self[some_oa_dim][layer_dim]

    @property
    def hw_utilization(self) -> UnrollFactor:
        """! Returns the `hardware utilization`, i.e. the product of all unrolled dimensions"""
        return int(math.prod([x.utilization for x in self.values()]))

    @property
    def all_contained_layer_dims(self) -> set[LayerDim]:
        """! Return a set containing all the LayerDims contained in the mapping at any OA Dim"""
        return set([layer_dim for layer_dim, _ in self.flatten_unrollings()])

    def get_total_unrolling_of_layer_dim(self, layer_dim: LayerDim) -> UnrollFactor:
        """! Return the total unroll factor of a given Layer Dimension, over all Operational Array Dimensions"""
        return math.prod([v for k, v in self.flatten_unrollings() if k == layer_dim])

    def get_performance_indicator(self) -> float:
        """! Return a value that indicates how well this SpatialMapping is expected to perform when used in ZigZag,
        compared to other SpatialMappings. The mapping with the highest value is expected to give the best results.
        Note: it is expected that the SpatialMapping is valid for the considered architecture and workload.
        The performance indicator consists of a term to represent the hardware representation and a term
        to indicate the diversity in LayerDims. Mappings with a higher hardware utilization should always have a higher
        performance indicator regardless of the diversity.
        Diversity is of importance because a mapping that unrolls the same LayerDim over all OA Dims will perform poorly
        due to the limited bandwidth of the memory that associated with that LayerDim.
        # TODO this is a rather naive metric and doesn't consider that multiple LayerDims may use the same memory
        The rational behind this function is that it costs less computation time to estimate the performance than
        actually computing the CMEs for each SpatialMapping
        """
        hw_utilization = float(self.hw_utilization)
        unrolling_per_layer_dim = [self.get_total_unrolling_of_layer_dim(dim) for dim in self.all_contained_layer_dims]
        diversity_indicator = hw_utilization / max(unrolling_per_layer_dim) - 1
        assert diversity_indicator < hw_utilization, "error in the performance indicator formula"
        return hw_utilization + diversity_indicator

    def flatten_unrollings(self) -> list[tuple[LayerDim, UnrollFactor]]:
        """! Convert all unrollings (pair of LayerDim and UnrollFactor) at all OADimension to a single
        list of tuples.
        e.g. -> [('K', 4), ('C', 2), ('K', 8)]"""
        result: list[tuple[LayerDim, UnrollFactor]] = []
        for mapping_single_dim in self.mappings:
            result += list(mapping_single_dim.items())
        return result

    def delete_layer_dim(self, layer_dim: LayerDim) -> None:
        assert layer_dim in self.all_contained_layer_dims
        for oa_dim in self.oa_dims:
            if layer_dim in self[oa_dim]:
                del self[oa_dim][layer_dim]

    @property
    def oa_dims(self) -> set[OADimension]:
        return set(self.keys())

    @property
    def mappings(self) -> list[MappingSingleOADim]:
        """! Return a list with all of the MappingSingleOADims contained in this instance.
        Note: converting this to a set may cause problems since MappingSingleOADim objects with identical unrollings
        will be mapped on the same set element."""
        return list(self.values())

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def __getitem__(self, key: OADimension) -> MappingSingleOADim:
        assert isinstance(key, OADimension)
        return self.data[key]

    def __setitem__(self, key: OADimension, value: MappingSingleOADim):
        self.data[key] = value

    def copy(self) -> "SpatialMapping":
        return copy.deepcopy(self)

    def __str__(self):
        return str({str(k): str(v) for k, v in self.items()}).replace('"', "").replace("'", "")

    def __eq__(self, other: Any) -> bool:
        """! Return true if the contained dimensions are the same and all MappingSingleOADims are the same"""
        return (
            isinstance(other, SpatialMapping)
            and all([oa_dim in other for oa_dim in self.oa_dims])
            and all([oa_dim in self for oa_dim in other.oa_dims])
            and all([self[oa_dim] == other[oa_dim] for oa_dim in self.oa_dims])
        )

    def __hash__(self):
        return hash(frozenset(map(lambda x: (x[0], hash(x[1])), self.items())))

    @staticmethod
    def empty() -> "SpatialMapping":
        return SpatialMapping({})


class SpatialMappingHint(LayerAttribute):
    """! Suggested LayerDims to be unrolled for every OADimension"""

    def __init__(self, data: dict[OADimension, set[LayerDim]]):
        self.data = data

    def clear_invalid_hits(self, valid_layer_dims: list[LayerDim]):
        """Check the hints at all contained OADimension. If the OADimension doesn't contain a single LayerDim that is
        also present in the given list `valid_layer_dims`, remove the OADimension from this instance.
        """
        invalid_oa_dims: list[OADimension] = []
        for oa_dim, hints in self.data.items():
            if not any([layer_dim in valid_layer_dims for layer_dim in hints]):
                logger.warning(
                    "Spatial mapping hint %s at OADimension %s doesn't contain a single layer dimension that is "
                    "in part of this layer: %s. Removing spatial mapping hints at %s.",
                    self,
                    oa_dim,
                    valid_layer_dims,
                    oa_dim,
                )
                invalid_oa_dims.append(oa_dim)

        for oa_dim in invalid_oa_dims:
            del self.data[oa_dim]

    def complete_with_defaults(self, oa_dim_sizes: dict[OADimension, int], layer_dims: list[LayerDim]):
        """For all OADimensions in `oa_dim_sizes` that are not already in this SpatialMappingHint, fill the hints for
        that OADimension with all LayerDims from `layer_dims`."""
        for oa_dim in filter(lambda x: x not in self, oa_dim_sizes):
            self.data[oa_dim] = set(layer_dims)

    def __getitem__(self, key: OADimension):
        return self.data[key]

    @staticmethod
    def empty() -> "SpatialMappingHint":
        return SpatialMappingHint({})
