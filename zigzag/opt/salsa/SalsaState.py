#   =====================================================================
#   Title:        state.py
#   Description: This file contains the state class that represents a
#   state of SALSA's makrov chain.
#
#   Date:        02.01.2023
#
#   =====================================================================
#
#   Copyright (C) 2020 ETH Zurich and University of Bologna.
#
#   Author: Victor Jung, ETH Zurich
#
#   SPDX-License-Identifier: Apache-2.0
#
#   Licensed under the Apache License, Version 2.0 (the License); you may
#   not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an AS IS BASIS, WITHOUT
#   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

from copy import deepcopy

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import LayerDim, UnrollFactorInt
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.opt.loma.MemoryAllocator import MemoryAllocator
from zigzag.workload.layer_node import LayerNode


class SalsaState:
    """! State of SALSA, storing an ordering, his temporal mapping and his energy value."""

    def __init__(
        self,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        ordering: list[tuple[LayerDim, UnrollFactorInt]],
        opt_criterion_name: str,
    ):
        assert opt_criterion_name in ("energy", "latency")  # TODO make this an enum?
        self.ordering = ordering
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.memory_hierarchy: MemoryHierarchy = self.accelerator.get_core(layer.core_allocation[0]).memory_hierarchy
        self.opt_criterion_name = opt_criterion_name

        allocator = MemoryAllocator(self.accelerator, self.layer, self.spatial_mapping, ordering)

        self.temporal_mapping = allocator.run()  # allocate this ordering to the memories

        self.cme = CostModelEvaluation(
            accelerator=self.accelerator,
            layer=self.layer,
            spatial_mapping=self.spatial_mapping,
            spatial_mapping_int=self.spatial_mapping,  # TODO the int version is missing?
            temporal_mapping=self.temporal_mapping,
        )

        # The optimization criterion will be minimized
        if self.opt_criterion_name == "energy":
            self.opt_criterion = self.cme.energy_total
        elif self.opt_criterion_name == "latency":
            self.opt_criterion = self.cme.latency_total0

    def swap(self, i: int, j: int) -> "SalsaState":
        """! Swap between the element at position i and j in the ordering and return the new resulting state."""

        swapped_ordering = deepcopy(self.ordering)
        temp = swapped_ordering[i]
        swapped_ordering[i] = swapped_ordering[j]
        swapped_ordering[j] = temp

        return SalsaState(
            self.accelerator,
            self.layer,
            self.spatial_mapping,
            swapped_ordering,
            self.opt_criterion_name,
        )
