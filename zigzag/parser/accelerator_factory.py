from typing import Any
from zigzag.datatypes import Constants, LayerDim, MemoryOperand, OADimension, UnrollFactor
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.memory_level import ServedMemDimensions
from zigzag.hardware.architecture.memory_port import DataDirection, PortAllocation
from zigzag.hardware.architecture.operational_array import MultiplierArray, OperationalArray
from zigzag.hardware.architecture.operational_unit import Multiplier
from zigzag.mapping.spatial_mapping import MappingSingleOADim, SpatialMapping


class AcceleratorFactory:
    """! Converts valid user-provided accelerator data into an `Accelerator` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Accelerator` instance from the validated user-provided data."""
        self.data = data

    def create(self) -> Accelerator:
        """! Create an Accelerator instance from the user-provided data.
        NOTE the memory instances must be defined from lowest to highest.
        """
        core_factory = CoreFactory(self.data)
        core = core_factory.create()
        return Accelerator(name=self.data["name"], core_set={core})


class CoreFactory:
    """! Converts valid user-provided accelerator data into a `Core` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Core` instance from the validated user-provided data."""
        self.data = data

    def create(self, core_id: int = 1) -> Core:
        """! Create an Core instance from the user-provided data.
        NOTE the memory instances must be defined from lowest to highest.
        """
        operational_array = self.create_operational_array()
        mem_graph = MemoryHierarchy(operational_array)
        dataflows = self.create_dataflows()

        for mem_name in self.data["memories"]:
            memory_factory = MemoryFactory(mem_name, self.data["memories"][mem_name])
            memory_factory.add_memory_to_graph(mem_graph)

        return Core(
            core_id=core_id, operational_array=operational_array, memory_hierarchy=mem_graph, dataflows=dataflows
        )

    def create_operational_array(self) -> OperationalArray:
        mul_data: dict[str, Any] = self.data["multipliers"]
        multiplier = Multiplier(
            input_precision=mul_data["input_precision"],
            energy_cost=mul_data["multiplier_energy"],
            area=mul_data["multiplier_area"],
        )

        oa_dims: list[str] = mul_data["dimensions"]
        dimension_sizes: dict[OADimension, int] = {
            OADimension(oa_dim): mul_data["sizes"][i] for i, oa_dim in enumerate(oa_dims)
        }
        multiplier_array = MultiplierArray(multiplier, dimension_sizes)
        return multiplier_array

    def create_dataflows(self) -> SpatialMapping | None:
        if "dataflows" not in self.data:
            return None
        if self.data["dataflows"] is None:
            return None

        user_data: dict[str, list[str]] = self.data["dataflows"]
        spatial_mapping_dict: dict[OADimension, MappingSingleOADim] = {}

        for oa_dim_str, unrolling_list in user_data.items():
            oa_dim = OADimension(oa_dim_str)
            mapping_this_oa_dim = self.__create_dataflow_single_oa_dim(unrolling_list)
            spatial_mapping_dict[oa_dim] = mapping_this_oa_dim

        return SpatialMapping(spatial_mapping_dict)

    def __create_dataflow_single_oa_dim(self, mapping_data: list[str]) -> MappingSingleOADim:
        mapping_dict: dict[LayerDim, UnrollFactor] = {}

        for single_unrolling in mapping_data:
            layer_dim_str = single_unrolling.split(",")[0]
            unrolling = int(single_unrolling.split(",")[-1])
            layer_dim = LayerDim(layer_dim_str)
            mapping_dict[layer_dim] = unrolling

        return MappingSingleOADim(mapping_dict)


class MemoryFactory:
    """! Create MemoryInstances and adds them to memory hierarchy."""

    def __init__(self, name: str, mem_data: dict[str, Any]):
        self.data = mem_data
        self.name = name

    def create_memory_instance(self) -> MemoryInstance:
        return MemoryInstance(
            name=self.name,
            size=self.data["size"],
            r_bw=self.data["r_bw"],
            w_bw=self.data["w_bw"],
            r_cost=self.data["r_cost"],
            w_cost=self.data["w_cost"],
            area=self.data["area"],
            r_port=self.data["r_port"],
            w_port=self.data["w_port"],
            rw_port=self.data["rw_port"],
            latency=self.data["latency"],
            min_r_granularity=self.data["min_r_granularity"],
            min_w_granularity=self.data["min_w_granularity"],
        )

    def add_memory_to_graph(self, mem_graph: MemoryHierarchy) -> None:
        """Create a new MemoryInstance and add it to the given MemoryHierarchy"""
        instance = self.create_memory_instance()

        operands: list[MemoryOperand] = [MemoryOperand(x) for x in self.data["operands"]]
        port_allocation = self.create_port_allocation()
        served_dimensions = self.create_served_mem_dimensions()

        mem_graph.add_memory(
            memory_instance=instance,
            operands=operands,
            port_alloc=port_allocation,
            served_dimensions=served_dimensions,
        )

    def create_served_mem_dimensions(self) -> ServedMemDimensions:
        data = {OADimension(oa_dim_str) for oa_dim_str in self.data["served_dimensions"]}
        return ServedMemDimensions(data)

    def create_port_allocation(self) -> PortAllocation:
        """The order of the port allocations matches the order of the MemoryOperands.
        # TODO support empty allocation -> return default configuration
        """
        port_data: list[dict[str, str]] = self.data["ports"]

        data: dict[MemoryOperand, dict[DataDirection, str]] = {
            MemoryOperand(mem_op_str): {
                self.translate_to_data_direction(direction): port_name
                for direction, port_name in port_data[idx].items()
            }
            for idx, mem_op_str in enumerate(self.data["operands"])
        }
        return PortAllocation(data)

    def create_default_port_allocation(self) -> PortAllocation:
        data: dict[MemoryOperand, dict[DataDirection, str]] = dict()
        for mem_op_str in self.data["operands"]:
            mem_op = MemoryOperand(mem_op_str)
            if mem_op == Constants.OUTPUT_MEM_OP:
                data[mem_op] = {
                    DataDirection.WR_IN_BY_HIGH: "w_port_1",
                    DataDirection.WR_IN_BY_LOW: "w_port_1",
                    DataDirection.RD_OUT_TO_HIGH: "r_port_1",
                    DataDirection.RD_OUT_TO_LOW: "r_port_1",
                }
            else:
                data[mem_op] = {
                    DataDirection.WR_IN_BY_HIGH: "w_port_1",
                    DataDirection.RD_OUT_TO_LOW: "r_port_1",
                }
        return PortAllocation(data)

    def translate_to_data_direction(self, x: str) -> DataDirection:
        match x:
            case "fh":
                return DataDirection.WR_IN_BY_HIGH
            case "fl":
                return DataDirection.WR_IN_BY_LOW
            case "th":
                return DataDirection.RD_OUT_TO_HIGH
            case "tl":
                return DataDirection.RD_OUT_TO_LOW
            case _:
                raise ValueError(f"Data direction must be either `fh`, `th`, `fl`, or `tl`. Not {x}")
