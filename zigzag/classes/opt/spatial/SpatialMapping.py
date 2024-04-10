from copy import deepcopy
from dataclasses import dataclass
import itertools
import logging
import math
from matplotlib.hatch import SmallCircles
from pydantic import BaseModel

from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.dimension import Dimension
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
from zigzag.classes.workload.layer_node import LayerNode


logger = logging.getLogger(__name__)


@dataclass
class LayerDim:
    """! (for-loop) dimension of a workload layer (e.g. `K`, `C`)"""

    name: str

    def __hash__(self):
        return hash(self.name)


class MappingSingleOADim:
    """! Spatial unrolling for a single Operational Array Dimension"""

    # TODO right now, mappings are defined as {"OA dim: (("layerdim1", i), ("layerdim2", j))
    def __init__(self, data: dict[LayerDim, int]):
        self.data = data

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

    def __getitem__(self, key: LayerDim):
        return self.data[key]

    def __contains__(self, key: LayerDim):
        return self.data.__contains__(key)

    def items(self):
        return self.data.items()

    def __setitem__(self, key: LayerDim, value: int):
        self.data[key] = value


class SpatialMapping:
    """! Spatial unrollings defined for every operational array dimension"""

    def __init__(self, data: dict[Dimension, MappingSingleOADim]):
        self.data = data

    def is_valid(self, max_unrollings: "SpatialMapping", oa_dims: list[Dimension]):
        """! Return True iff
        - the utilization at each OA Dimension does not exceed the size of the Dimension
        - the instance does not contain LayerDims that are not bounded by `max_unrollings`
        - each LayerDim unrolling does not exceed the unrolling prescribed in max_unrollings
        - the instance does not contain OA Dimensions that are not part of the given list
        """
        for oa_dim in oa_dims:
            if oa_dim not in self:
                return False
            for layer_dim, unrolling in self[oa_dim].items():
                if layer_dim not in max_unrollings[oa_dim]:
                    return False
                if unrolling > max_unrollings[oa_dim][layer_dim]:
                    return False

            if self[oa_dim].get_utilization() > oa_dim.size:
                return False

        for oa_dim in self.data.keys():
            if oa_dim not in oa_dims:
                return False

        return True

    def check_and_reduce(self, max_unrollings: "SpatialMapping", oa_dims: list[Dimension]):
        """! Verify
        - that the utilization at each OA Dimension does not exceed the size of the Dimension
        - that each LayerDim unrolling does not exceed the unrolling prescribed in max_unrollings
        Reduce the unrollings otherwise.
        """
        for oa_dim in filter(lambda x: x in self, oa_dims):
            assert oa_dim in self
            # First check individual LayerDim unrollings
            for layer_dim, unrolling in self[oa_dim].items():
                assert layer_dim in max_unrollings[oa_dim], "Unexpected LayerDim found"
                max_unrolling = max_unrollings[oa_dim][layer_dim]
                if unrolling > max_unrolling:
                    logger.info(
                        "User provided spatial unrolling of %i for %s in Dimension %s exceeded maximally allowed unrolling of %i. Reducing unrolling to this value.",
                        unrolling,
                        str(layer_dim),
                        str(oa_dim),
                        max_unrolling,
                    )
                    self[oa_dim][layer_dim] = max_unrolling

            # Check full OA Dimension
            if self[oa_dim].get_utilization() > oa_dim.size:
                logger.info(
                    "User provided spatial unrolling of for Dimension %s exceeded maximally allowed unrolling of %i. Removing Layer Dimension unrollings to meet this constraint",
                    str(oa_dim),
                    oa_dim.size,
                )
                while self[oa_dim].get_utilization() > oa_dim.size:
                    # Remove any LayerDim
                    some_layer_dim = list(self[oa_dim].data.keys()).pop()
                    del self[oa_dim].data[some_layer_dim]

    def get_utilization(self):
        """! Returns the `hardware utilization`, i.e. the product of all unrolled dimensions"""
        return math.prod([x.get_utilization() for x in self.data.values()])

    def __getitem__(self, key: Dimension):
        return self.data[key]

    def items(self):
        return self.data.items()

    def __setitem__(self, key: Dimension, value: MappingSingleOADim):
        self.data[key] = value

    def __contains__(self, key: Dimension):
        return self.data.__contains__(key)

    def __len__(self):
        return len(self.data)

    def copy(self):
        return SpatialMapping(deepcopy(self.data))

    @staticmethod
    def parse_spatial_mapping(x: dict[Dimension, tuple[str, int] | tuple[tuple]]):
        """! Parse legacy notation
        Example:
        x = {"D1": ("OX", 25), "D2": (("FX", 3), ("FY", 3))}
        return SpatialMapping({"D1": {"OX": 25}, "D2": {"FX":3, "FY": 3}})
        """
        spatial_mapping_dict = {}  # operational array dimension : MappingSingleOADim
        for k, v in x.items():
            mapping_single_dim_dict: dict[LayerDim, int] = {}

            if isinstance(v[0], tuple):
                # v: tuple[tuple[str, int]]
                # Nested layer dimensions
                for layer_dim, factor in v:  # type: ignore
                    mapping_single_dim_dict[LayerDim(layer_dim)] = factor  # type: ignore
            else:
                # v : tuple[str, int]
                layer_dim, factor = v  # type: ignore
                mapping_single_dim_dict[LayerDim(layer_dim)] = factor  # type: ignore

            spatial_mapping_dict[k] = MappingSingleOADim(mapping_single_dim_dict)
        return SpatialMapping(spatial_mapping_dict)


@dataclass
class SpatialMappingHint:
    """! Suggested LayerDims to be unrolled for every Operational Array Dimension"""

    def __init__(self, data: dict):
        self.data: dict[Dimension, set[LayerDim]]

        if isinstance(next(iter(data.keys())), Dimension):
            self.data = data
        # Legacy notation: dict[str, list[str]]
        elif isinstance(next(iter(data.keys())), str):
            self.data = {
                Dimension(name=oa_dim_name): {LayerDim(layer_dim_name) for layer_dim_name in hints}
                for oa_dim_name, hints in data.items()
            }

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
