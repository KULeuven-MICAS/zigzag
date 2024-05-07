from typing import Any
from zigzag.datatypes import Constants, MemoryOperand, OADimension
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.Core import Core
from zigzag.hardware.architecture.MemoryHierarchy import MemoryHierarchy
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.hardware.architecture.memory_level import ServedMemDimensions
from zigzag.hardware.architecture.memory_port import DataDirection, PortAllocation
from zigzag.hardware.architecture.operational_array import MultiplierArray, OperationalArray
from zigzag.hardware.architecture.operational_unit import Multiplier


class AcceleratorFactory:
    """! Converts valid user-provided accelerator data into an `Accelerator` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Accelerator` instance from the validated user-provided data."""
        self.data = data

    def create(self) -> Accelerator:
        """! Create an Accelerator instance from the user-provided data.
        NOTE the memory instances must be defined from lowest to highest.
        """
        operational_array = self.create_operational_array()
        mem_graph = MemoryHierarchy(operational_array)

        for mem_name in self.data["memories"]:
            memory_factory = MemoryFactory(mem_name, self.data["memories"][mem_name])
            memory_factory.add_memory_to_graph(mem_graph)

        core = Core(id=1, operational_array=operational_array, memory_hierarchy=mem_graph)
        return Accelerator(name=self.data["name"], core_set={core})

    def create_operational_array(self) -> OperationalArray:
        mul_data: dict[str, Any] = self.data["multipliers"]
        multiplier = Multiplier(
            input_precision=mul_data["input_precision"],
            energy_cost=mul_data["multiplier_energy"],
            area=mul_data["multiplier_area"],
        )

        oa_dims: list[str] = mul_data["dimensions"]
        dimension_sizes: dict[OADimension, int] = {
            self.create_oa_dim(oa_dim): mul_data["sizes"][i] for i, oa_dim in enumerate(oa_dims)
        }
        multiplier_array = MultiplierArray(multiplier, dimension_sizes)
        return multiplier_array

    @staticmethod
    def create_oa_dim(name: str) -> OADimension:
        return OADimension(name)

    @staticmethod
    def create_memory_operand(name: str) -> MemoryOperand:
        return MemoryOperand(name)


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
        data = {AcceleratorFactory.create_oa_dim(oa_dim_str) for oa_dim_str in self.data["served_dimensions"]}
        return ServedMemDimensions(data)

    def create_port_allocation(self) -> PortAllocation:
        """The order of the port allocations matches the order of the MemoryOperands.
        # TODO support empty allocation -> return default configuration
        """
        port_data: list[dict[str, str]] = self.data["ports"]

        data: dict[MemoryOperand, dict[DataDirection, str]] = {
            AcceleratorFactory.create_memory_operand(mem_op_str): {
                self.translate_to_data_direction(direction): port_name
                for direction, port_name in port_data[idx].items()
            }
            for idx, mem_op_str in enumerate(self.data["operands"])
        }
        return PortAllocation(data)

    def create_default_port_allocation(self) -> PortAllocation:
        data: dict[MemoryOperand, dict[DataDirection, str]] = dict()
        for mem_op_str in self.data["operands"]:
            mem_op = AcceleratorFactory.create_memory_operand(mem_op_str)
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
