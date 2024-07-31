import logging

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.ImcArray import ImcArray
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping
from zigzag.utils import json_repr_handler
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class CostModelEvaluationForIMC(CostModelEvaluation):
    """! Class that stores inputs and runs them through the zigzag cost model.

    Initialize the cost model evaluation with the following inputs:
    - accelerator: the accelerator that includes the core on which to run the layer
    - layer: the layer to run
    - spatial_mapping: the spatial mapping
    - temporal_mapping: the temporal mapping

    From these parameters, the following attributes are computed:
    * core: The core on which the layer is ran. This should be specified in the LayerNode attributes.
    * mapping: The combined spatial and temporal mapping object where access patterns are computed.

    The following cost model attributes are also initialized:
    - mem_energy_breakdown: The energy breakdown for all operands
    - energy: The total energy

    After initialization, the cost model evaluation is run.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        spatial_mapping_int: SpatialMappingInternal,
        temporal_mapping: TemporalMapping,
        access_same_data_considered_as_no_access: bool = True,
    ):
        self.is_imc = True
        self.core = next(iter(accelerator.cores))
        assert isinstance(self.core.operational_array, ImcArray)
        self.operational_array: ImcArray = self.core.operational_array
        super().__init__(
            accelerator=accelerator,
            layer=layer,
            spatial_mapping=spatial_mapping,
            spatial_mapping_int=spatial_mapping_int,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access,
        )

    def run(self) -> None:
        """! Run the cost model evaluation."""
        super().calc_memory_utilization()
        super().calc_memory_word_access()
        self.calc_energy()
        self.calc_latency()
        self.collect_area_data()

    def collect_area_data(self):
        # get imc area
        self.imc_area_breakdown = self.operational_array.area_breakdown
        self.imc_area = self.operational_array.area
        # get mem area
        self.mem_area = 0
        self.mem_area_breakdown: dict[str, float] = {}
        for mem in self.mem_level_list:
            memory_instance = mem.memory_instance
            memory_instance_name = memory_instance.name
            self.mem_area += memory_instance.area
            self.mem_area_breakdown[memory_instance_name] = memory_instance.area
        # get total area
        self.area_total = self.imc_area + self.mem_area

    def calc_mac_energy_cost(self):
        """Calculate the dynamic MAC energy
        Overrides superclass' method
        """
        self.mac_energy_breakdown = self.operational_array.get_energy_for_a_layer(self.layer, self.mapping)
        self.mac_energy = sum([energy for energy in self.mac_energy_breakdown.values()])

    def calc_latency(self):
        """!  Calculate latency in 4 steps

        1) As we already calculated the ideal data transfer rate in combined_mapping.py (in the Mapping class),
        here we start with calculating the required (or allowed) memory updating window by comparing the effective
        data size with the physical memory size at each level. If the effective data size is smaller than 50%
        of the physical memory size, then we take the whole period as the allowed memory updating window (double buffer
        effect);
        otherwise we take the the period divided by the top_ir_loop as the allowed memory updating window.

        2) Then, we compute the real data transfer rate given the actual memory bw per functional port pair,
        assuming we have enough memory ports.

        3) In reality, there is no infinite memory port to use. So, as the second step, we combine the real
        data transfer attributes per physical memory port.

        4) Finally, we combine the stall/slack of each memory port to get the final latency.
        """
        super().calc_double_buffer_flag()
        super().calc_allowed_and_real_data_transfer_cycle_per_data_transfer_link()
        # Update the latency model to fit IMC requirement
        super().combine_data_transfer_rate_per_physical_port()
        self.update_tclk()
        super().calc_data_loading_offloading_latency()
        # find the cycle count per mac
        cycles_per_mac = self.operational_array.activation_precision / self.operational_array.bit_serial_precision
        super().calc_overall_latency(cycles_per_mac=cycles_per_mac)

    def update_tclk(self):
        """! This function calculate the Tclk for IMC (In-Memory-Computing)"""
        self.tclk = self.operational_array.tclk
        self.tclk_breakdown = self.operational_array.tclk_breakdown

    # JSON representation used for saving this object to a json file.
    def __jsonrepr__(self):
        # latency_total0 breakdown
        computation_breakdown = {
            "mac_computation": self.ideal_temporal_cycle,
            "memory_stalling": self.stall_slack_comb,
        }

        return json_repr_handler(
            {
                "outputs": {
                    "memory": {
                        "utilization": (self.mem_utili_shared if hasattr(self, "mem_utili_shared") else None),
                        "word_accesses": self.memory_word_access,
                    },
                    "energy": {
                        "energy_total": self.energy_total,
                        "operational_energy": self.mac_energy,
                        "operational_energy_breakdown": self.mac_energy_breakdown,
                        "memory_energy": self.mem_energy,
                        "memory_energy_breakdown_per_level": self.mem_energy_breakdown,
                        "memory_energy_breakdown_per_level_per_operand": self.mem_energy_breakdown_further,
                    },
                    "latency": {
                        "data_onloading": self.latency_total1 - self.latency_total0,
                        "computation": self.latency_total0,
                        "data_offloading": self.latency_total2 - self.latency_total1,
                        "computation_breakdown": computation_breakdown,
                    },
                    "clock": {
                        "tclk (ns)": self.tclk,
                        "tclk_breakdown (ns)": self.tclk_breakdown,
                    },
                    "area (mm^2)": {
                        "total_area": self.area_total,
                        "total_area_breakdown:": {
                            "imc_area": self.imc_area,
                            "mem_area": self.mem_area,
                        },
                        "total_area_breakdown_further": {
                            "imc_area_breakdown": self.imc_area_breakdown,
                            "mem_area_breakdown": self.mem_area_breakdown,
                        },
                    },
                    "spatial": {
                        "mac_utilization": {
                            "ideal": self.mac_spatial_utilization,
                            "stalls": self.mac_utilization0,
                            "stalls_onloading": self.mac_utilization1,
                            "stalls_onloading_offloading": self.mac_utilization2,
                        }
                    },
                },
                "inputs": {
                    "accelerator": self.accelerator,
                    "layer": self.layer,
                    "spatial_mapping": (self.spatial_mapping_int if hasattr(self, "spatial_mapping_int") else None),
                    "temporal_mapping": (self.temporal_mapping if hasattr(self, "temporal_mapping") else None),
                },
            }
        )

    def __simplejsonrepr__(self):
        """! Simple JSON representation used for saving this object to a simple json file."""
        return {
            "energy": self.energy_total,
            "latency": self.latency_total2,
            "tclk (ns)": self.tclk,
            "area": self.area_total,
        }
