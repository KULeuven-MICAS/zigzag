"""
# TODO this file needs to be reworked
"""

from math import ceil
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.memory_port import PortAllocation
from zigzag.hardware.architecture.operational_array import OperationalArray
from zigzag.hardware.architecture.operational_unit import OperationalUnit
from zigzag.utils import pickle_deepcopy
from zigzag.stages.Stage import Stage, StageCallable

import logging

logger = logging.getLogger(__name__)


class PEArrayScalingStage(Stage):
    """! This stage scales the PE array of the given accelerator.
    Because the user-defined spatial mapping resides in the different workload layer nodes,
    We also have to modify those to scale accordingly
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload,
        accelerator: Accelerator,
        pe_array_scaling: int,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)

        # SANITY CHECKS
        # Only allow scaling factors that are a power of 2
        assert pe_array_scaling in [2**i for i in range(-3, 3)]
        # Make sure there's only one core so that the correct one is scaled
        # If your accelerator has more cores, modify the function below
        assert len(accelerator.cores) == 1

        self.workload = workload
        self.accelerator = accelerator
        self.pe_array_scaling = pe_array_scaling

    def run(self):
        scaled_accelerator = self.generate_scaled_accelerator()
        modified_workload = self.scale_workload_spatial_mapping()
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            workload=modified_workload,
            accelerator=scaled_accelerator,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def generate_scaled_accelerator(self):
        """
        Recreate the Accelerator with PE array dimension scaling in all dimensions.
        The elements required for this recreation are:
        - accelerator
            - name
            - cores
                - operational array
                    - operational unit
                    - dimension sizes
                - memory hierarchy
                    - name
                    - memory levels
                        - memory instance
                        - operands
                        - port allocation
                        - served dimensions
        """
        # Get the relevant accelerator attributes
        core = next(iter(self.accelerator.cores))
        operational_array: OperationalArray = core.operational_array
        operational_unit = operational_array.unit
        dimension_sizes = operational_array.oa_dim_sizes
        memory_hierarchy = core.memory_hierarchy

        # Create new operational array
        new_operational_unit: OperationalUnit = pickle_deepcopy(operational_unit)
        new_dimension_sizes = [ceil(self.pe_array_scaling * dim_size) for dim_size in dimension_sizes]
        new_dimensions = {f"D{i}": new_dim_size for i, new_dim_size in enumerate(new_dimension_sizes, start=1)}
        new_operational_array = OperationalArray(new_operational_unit, new_dimensions)

        # Initialize the new memory hierarchy
        mh_name = memory_hierarchy.name
        new_mh_name = mh_name + "-scaled"
        new_memory_hierarchy = MemoryHierarchy(new_operational_array, new_mh_name)

        # Add memories to the new memory hierarchy with the correct attributes
        for memory_level in memory_hierarchy.mem_level_list:
            memory_instance = memory_level.memory_instance
            operands = tuple(memory_level.operands)
            port_alloc = memory_level.port_alloc_raw
            served_dimensions_vec = memory_level.served_dimensions_vec
            assert len(served_dimensions_vec) >= 1
            served_dimensions = served_dimensions_vec[0]

            new_memory_instance: MemoryInstance = pickle_deepcopy(memory_instance)
            new_operands: tuple[str] = pickle_deepcopy(operands)
            new_port_alloc: PortAllocation = pickle_deepcopy(port_alloc)
            new_served_dimensions = pickle_deepcopy(served_dimensions)
            new_memory_hierarchy.add_memory(
                memory_instance=new_memory_instance,
                operands=new_operands,
                port_alloc=new_port_alloc,
                served_dimensions=new_served_dimensions,
            )

        # Create the new core
        id = core.id
        new_id = id
        new_core = Core(
            id=new_id,
            operational_array=new_operational_array,
            memory_hierarchy=new_memory_hierarchy,
        )

        # Create the new accelerator
        name = self.accelerator.name
        new_name = name + "-scaled"
        new_cores = {new_core}
        new_accelerator = Accelerator(
            name=new_name,
            core_set=new_cores,
        )

        return new_accelerator

    def scale_workload_spatial_mapping(self):
        """
        Scale the user-defined mappings for each layer.
        """
        modified_workload = pickle_deepcopy(self.workload)
        for node in modified_workload.nodes():
            if hasattr(node, "user_spatial_mapping") and node.user_spatial_mapping:
                for array_dim, (layer_dim, size) in node.user_spatial_mapping.items():
                    if size != 1:
                        node.user_spatial_mapping[array_dim] = (
                            layer_dim,
                            self.pe_array_scaling * size,
                        )
        return modified_workload
