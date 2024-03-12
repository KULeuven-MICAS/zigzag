import logging
from typing import Dict, List, Tuple
from math import ceil
import numpy as np

from zigzag.classes.mapping.combined_mapping import Mapping
from zigzag.classes.mapping.combined_mapping import FourWayDataMoving
from zigzag.utils import pickle_deepcopy
from zigzag.classes.cost_model.cost_model import (
    PortActivity, PortBeginOrEndActivity, get_shared_mem_list, spatial_mapping_fractional_to_int, calc_MUW_union,)

logger = logging.getLogger(__name__)

## Class that stores inputs and runs them through the zigzag cost model.
#
# Initialize the cost model evaluation with the following inputs:
# - accelerator: the accelerator that includes the core on which to run the layer
# - layer: the layer to run
# - spatial_mapping: the spatial mapping
# - temporal_mapping: the temporal mapping
#
# From these parameters, the following attributes are computed:
# * core: The core on which the layer is ran. This should be specified in the LayerNode attributes.
# * mapping: The combined spatial and temporal mapping object where access patterns are computed.
#
# The following cost model attributes are also initialized:
# - mem_energy_breakdown: The energy breakdown for all operands
# - energy: The total energy
#
# After initialization, the cost model evaluation is run.
class CostModelEvaluationForIMC:

    ## The class constructor
    # After initialization, the cost model evaluation is run
    # @param accelerator the accelerator that includes the core on which to run the
    # @param layer the layer to run
    # @param spatial_mapping the spatial mapping
    # @param temporal_mapping the temporal mapping
    # @param access_same_data_considered_as_no_access (optional)
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        spatial_mapping_int,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = spatial_mapping_int
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = (
            access_same_data_considered_as_no_access
        )

        self.core_id = layer.core_allocation
        self.mem_level_list = (
            accelerator.get_core(self.core_id).get_memory_hierarchy().mem_level_list
        )
        self.mem_hierarchy_dict = accelerator.get_core(
            self.core_id
        ).get_memory_hierarchy_dict()
        self.mem_size_dict = accelerator.get_core(self.core_id).get_memory_size_dict()
        self.mem_r_bw_dict, self.mem_w_bw_dict = accelerator.get_core(
            self.core_id
        ).get_memory_bw_dict()
        self.mem_r_bw_min_dict, self.mem_w_bw_min_dict = accelerator.get_core(
            self.core_id
        ).get_memory_bw_min_dict()
        self.mem_sharing_list = accelerator.get_core(
            self.core_id
        ).get_memory_sharing_list()
        self.layer_op_to_mem_op = layer.memory_operand_links
        self.mem_op_to_layer_op = dict(
            [(value, key) for key, value in self.layer_op_to_mem_op.items()]
        )

        """ generate the integer spatial mapping from fractional spatial mapping (due to greedy mapping support).
        Later the fractional one is used for calculating energy, and the integer one is used for calculating latency"""
        # self.spatial_mapping_dict_int = spatial_mapping_fractional_to_int(
        #     self.spatial_mapping.mapping_dict_origin
        # )
        self.spatial_mapping_dict_int = self.spatial_mapping_int.mapping_dict_origin

        # For constructing Mapping object,  the last parameter "self.access_same_data_considered_as_no_access" is optional
        self.mapping = Mapping(
            self.accelerator,
            self.spatial_mapping,
            self.temporal_mapping,
            self.layer,
            self.access_same_data_considered_as_no_access,
        )
        self.mapping_int = Mapping(
            self.accelerator,
            self.spatial_mapping_dict_int,
            self.temporal_mapping,
            self.layer,
            self.access_same_data_considered_as_no_access,
        )

        self.active_mem_level = self.mapping.mem_level

        # TODO: [jiacong] [ADD]
        self.MAC_energy_breakdown = None
        # TODO: [jiacong] [FINISH]

        # Run the cost model evaluation
        self.run()

    def __str__(self):
        return f"CostModelEvaluation(layer={self.layer}, core={self.core_id})"

    def __repr__(self):
        return str(self)

    # JSON representation used for saving this object to a json file.
    def __jsonrepr__(self):
        # latency_total0 breakdown
        computation_breakdown = {
            "mac_computation": self.ideal_temporal_cycle,
            "weight_loading": self.SS_comb,
        }

        return {
            "outputs": {
                "memory": {
                    "utilization": self.mem_utili_shared
                    if hasattr(self, "mem_utili_shared")
                    else None,
                    "word_accesses": self.memory_word_access,
                },
                "energy": {
                    "energy_total": self.energy_total,
                    "operational_energy": self.MAC_energy,
                    "operational_energy_breakdown": self.MAC_energy_breakdown,
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
                        "ideal": self.MAC_spatial_utilization,
                        "stalls": self.MAC_utilization0,
                        "stalls_onloading": self.MAC_utilization1,
                        "stalls_onloading_offloading": self.MAC_utilization2,
                    }
                },
            },
            "inputs": {
                "accelerator": self.accelerator,
                "layer": self.layer,
                "spatial_mapping": self.spatial_mapping
                if hasattr(self, "spatial_mapping")
                else None,
                "temporal_mapping": self.temporal_mapping
                if hasattr(self, "temporal_mapping")
                else None,
            },
        }

    ## Simple JSON representation used for saving this object to a simple json file.
    def __simplejsonrepr__(self):
        return {"energy": self.energy_total, "latency": self.latency_total2}

    ## Run the cost model evaluation.
    def run(self):
        # - TODO: Latency calculation
        self.calc_memory_utilization()
        self.calc_memory_word_access()
        self.calc_energy()
        self.calc_latency()
        self.collect_area_data()

    def collect_area_data(self):
        # get imc area
        operational_array = self.accelerator.get_core(self.core_id).operational_array
        self.imc_area = operational_array.total_area
        self.imc_area_breakdown = operational_array.area_breakdown
        # get mem area
        self.mem_area = 0
        self.mem_area_breakdown = {}
        for mem in self.mem_level_list:
            memory_instance = mem.memory_instance
            memory_instance_name = memory_instance.name
            self.mem_area += memory_instance.area
            self.mem_area_breakdown[memory_instance_name] = memory_instance.area
        # get total area
        self.area_total = self.imc_area + self.mem_area

    ## Calculate occupancy for each physical memory based on the mapping.
    def calc_memory_utilization(self):
        # mem_utili_individual: the memory utilization of each operand individually.
        # mem_utili_shared: the memory utilization taking operand memory sharing into consideration.
        mem_utili_individual = {}
        effective_mem_utili_individual = {}
        for layer_op in self.layer.operand_list:
            mem_utili_individual[layer_op] = []
            effective_mem_utili_individual[layer_op] = []
            for mem_lv in range(self.active_mem_level[layer_op]):
                mem_utilization = (
                    self.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1]
                    / self.mem_size_dict[self.layer_op_to_mem_op[layer_op]][mem_lv]
                )
                assert mem_utilization <= 1, (
                    f"Operand {layer_op} memory level {mem_lv}'s individual memory utilization is "
                    f"{mem_utilization}, which is larger than 1 "
                    f"(memory level starts from 0)"
                )
                mem_utili_individual[layer_op].append(mem_utilization)

                # if we do not count copied data in parallel memories as effective, what is the utilization then? =>
                effective_mem_utilization = (
                    self.mapping.effective_data_bit[layer_op][mem_lv + 1]
                    / self.mem_size_dict[self.layer_op_to_mem_op[layer_op]][mem_lv]
                )
                effective_mem_utili_individual[layer_op].append(
                    effective_mem_utilization
                )

        mem_utili_shared = pickle_deepcopy(mem_utili_individual)
        effective_mem_utili_shared = pickle_deepcopy(effective_mem_utili_individual)
        for mem_share_dict in self.mem_sharing_list:
            mem_utilization = 0
            effective_mem_utilization = 0
            for mem_op, mem_lv in mem_share_dict.items():
                try:
                    layer_op = self.mem_op_to_layer_op[mem_op]
                except:  # mem to layer op might not contain this mem op (e.g. pooling layer)
                    continue
                mem_utilization += mem_utili_individual[layer_op][mem_lv]
                effective_mem_utilization += effective_mem_utili_individual[layer_op][
                    mem_lv
                ]
            assert mem_utilization <= 1, (
                f"Memory shared by {mem_share_dict} (memory operand, memory level) has shared utilization of "
                f"{mem_utilization}, which is > 1 "
                f"(memory level starts from 0)."
            )
            for mem_op, mem_lv in mem_share_dict.items():
                try:
                    layer_op = self.mem_op_to_layer_op[mem_op]
                except:  # mem to layer op might not contain this mem op (e.g. pooling layer)
                    continue
                mem_utili_shared[layer_op][mem_lv] = mem_utilization
                effective_mem_utili_shared[layer_op][mem_lv] = effective_mem_utilization

        self.mem_utili_individual = mem_utili_individual
        self.mem_utili_shared = mem_utili_shared
        self.effective_mem_utili_individual = effective_mem_utili_individual
        self.effective_mem_utili_shared = effective_mem_utili_shared

    ## Calculates the memory word access based on unit memory's data element move count and the physical memory bw.
    def calc_memory_word_access(self):
        memory_word_access = {}
        for layer_op in self.layer.operand_list:
            memory_word_access[layer_op] = []
            for mem_lv in range(self.mapping.mem_level[layer_op]):
                """wr_in_by_low"""
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.wr_in_by_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    wr_in_by_low = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_low
                    max_bw = self.mem_w_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_w_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    if mem_lv > 0:
                        another_side_bw = self.mem_r_bw_dict[
                            self.layer_op_to_mem_op[layer_op]
                        ][mem_lv - 1] * (
                            self.spatial_mapping.unit_unique[layer_op][mem_lv]
                            / self.spatial_mapping.unit_unique[layer_op][mem_lv + 1]
                        )
                        data_elem_move_per_cycle_in_a_period = min(
                            (another_side_bw / data_precision),
                            (max_bw / data_precision),
                            data_elem_move_per_period,
                        )
                        cycle_in_a_period = ceil(
                            data_elem_move_per_period
                            / data_elem_move_per_cycle_in_a_period
                        )
                    else:
                        data_elem_move_per_cycle_in_a_period = data_elem_move_per_period
                        cycle_in_a_period = 1
                    # wr_in_by_low = (
                    #    ceil(
                    #        (data_elem_move_per_cycle_in_a_period * data_precision)
                    #        / min_bw
                    #    )
                    #    * (min_bw / max_bw)
                    #    * total_period_count
                    #    * cycle_in_a_period
                    #    * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    # )

                    # 2023/06/30, solve the memory access granuarity issue - Jiacong Sun, Linyan Mei
                    # Originally we used the cycle_in_a_period to compute the memory word access.
                    # This neglected the finer-grained memory access possibility (the min_bw, the minimal memory access granuarity, like half-word access).
                    # Now we changed to calculation based on min_bw.
                    wr_in_by_low = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ rd_out_to_low """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.rd_out_to_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    rd_out_to_low = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_low
                    max_bw = self.mem_r_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_r_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    if mem_lv > 0:
                        another_side_bw = self.mem_w_bw_dict[
                            self.layer_op_to_mem_op[layer_op]
                        ][mem_lv - 1] * (
                            self.spatial_mapping.unit_unique[layer_op][mem_lv]
                            / self.spatial_mapping.unit_unique[layer_op][mem_lv + 1]
                        )
                        data_elem_move_per_cycle_in_a_period = min(
                            (another_side_bw / data_precision),
                            (max_bw / data_precision),
                            data_elem_move_per_period,
                        )
                        cycle_in_a_period = ceil(
                            data_elem_move_per_period
                            / data_elem_move_per_cycle_in_a_period
                        )
                        # rd_out_to_low = (
                        #    ceil(
                        #        (data_elem_move_per_cycle_in_a_period * data_precision)
                        #        / min_bw
                        #    )
                        #    * (min_bw / max_bw)
                        #    * total_period_count
                        #    * cycle_in_a_period
                        #    * self.mapping.spatial_mapping.unit_count[layer_op][
                        #        mem_lv + 1
                        #    ]
                        # )
                    # else:

                    # 2023/06/30, solve the memory access granuarity issue - Jiacong Sun, Linyan Mei
                    # Originally we used the cycle_in_a_period to compute the memory word access.
                    # This neglected the finer-grained memory access possibility (the min_bw, the minimal memory access granuarity, like half-word access).
                    # Now we changed to calculation based on min_bw.
                    rd_out_to_low = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ rd_out_to_high """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.rd_out_to_high
                if data_elem_move_per_period == 0:
                    rd_out_to_high = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_precision.rd_out_to_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_high
                    max_bw = self.mem_r_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_r_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    rd_out_to_high = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ wr_in_by_high """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.wr_in_by_high
                if data_elem_move_per_period == 0:
                    wr_in_by_high = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_precision.wr_in_by_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_high
                    max_bw = self.mem_w_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_w_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    wr_in_by_high = (
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ All """
                memory_word_access_single = FourWayDataMoving(
                    rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high
                )
                memory_word_access[layer_op].append(memory_word_access_single)

        self.memory_word_access = memory_word_access

    ## Calculates the energy cost of this cost model evaluation by calculating the memory reading/writing energy.
    def calc_energy(self):
        # - TODO: Interconnection energy
        self.calc_MAC_energy_cost()
        self.calc_memory_energy_cost()

    ## Calculate the dynamic MAC energy
    def calc_MAC_energy_cost(self):
        core = self.accelerator.get_core(self.core_id)
        # TODO: [jiacong] [MODIFY]
        self.MAC_energy_breakdown = core.operational_array.unit.get_energy_for_a_layer(self.layer, self.mapping)
        self.MAC_energy = sum([energy for energy in self.MAC_energy_breakdown.values()])
        # TODO: [jiacong] [FINISH]

    ## Computes the memories reading/writing energy by converting the access patterns in self.mapping to
    # energy breakdown using the memory hierarchy of the core on which the layer is mapped.
    #
    # The energy breakdown is saved in self.mem_energy_breakdown.
    #
    # The energy total consumption is saved in self.energy_total.
    def calc_memory_energy_cost(self):
        core = self.accelerator.get_core(self.core_id)
        mem_hierarchy = core.memory_hierarchy

        mem_energy_breakdown = {}
        mem_energy_breakdown_further = {}
        energy_total = 0
        for (layer_op, mem_access_list_per_op) in self.memory_word_access.items():
            """Retrieve the memory levels in the hierarchy for this memory operand"""
            mem_op = self.layer_op_to_mem_op[layer_op]
            memory_levels = mem_hierarchy.get_memory_levels(mem_op=mem_op)

            breakdown = (
                []
            )  # Stores the energy breakdown of a single layer operand (W, I, ...)
            breakdown_further = []  # Stores
            for (access_count, memory_level) in zip(
                mem_access_list_per_op, memory_levels
            ):
                energy_cost_per_read_out = memory_level.read_energy
                energy_cost_per_write_in = memory_level.write_energy
                read_out_energy_to_above = access_count.get_total_read_outs_to_above(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_above = (
                    access_count.get_total_write_ins_from_above(
                        scaling=energy_cost_per_write_in
                    )
                )
                read_out_energy_to_below = access_count.get_total_read_outs_to_below(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_below = (
                    access_count.get_total_write_ins_from_below(
                        scaling=energy_cost_per_write_in
                    )
                )
                total_read_out_energy = (
                    read_out_energy_to_above + read_out_energy_to_below
                )
                total_write_in_energy = (
                    write_in_energy_from_above + write_in_energy_from_below
                )
                total_energy_cost_memory = total_read_out_energy + total_write_in_energy
                breakdown.append(
                    total_energy_cost_memory
                )  # Here the breakdown only saves the total energy cost per memory level
                breakdown_further.append(
                    FourWayDataMoving(
                        read_out_energy_to_below,
                        write_in_energy_from_below,
                        read_out_energy_to_above,
                        write_in_energy_from_above,
                    )
                )  # here it contains the full split
                energy_total += total_energy_cost_memory
            mem_energy_breakdown[layer_op] = breakdown
            mem_energy_breakdown_further[layer_op] = breakdown_further
        self.mem_energy_breakdown = mem_energy_breakdown
        self.mem_energy_breakdown_further = mem_energy_breakdown_further
        self.mem_energy = energy_total
        self.energy_total = self.mem_energy + self.MAC_energy
        logger.debug(f"Ran {self}. Total energy = {self.energy_total}")

    ##  Calculate latency in 4 steps
    #
    # 1) As we already calculated the ideal data transfer rate in combined_mapping.py (in the Mapping class),
    # here we start with calculating the required (or allowed) memory updating window by comparing the effective
    # data size with the physical memory size at each level. If the effective data size is smaller than 50%
    # of the physical memory size, then we take the whole period as the allowed memory updating window (double buffer effect);
    # otherwise we take the the period divided by the top_ir_loop as the allowed memory updating window.
    #
    # 2) Then, we compute the real data transfer rate given the actual memory bw per functional port pair,
    # assuming we have enough memory ports.
    #
    # 3) In reality, there is no infinite memory port to use. So, as the second step, we combine the real
    # data transfer attributes per physical memory port.
    #
    # 4) Finally, we combine the stall/slack of each memory port to get the final latency.
    def calc_latency(self):
        self.calc_double_buffer_flag()
        self.calc_allowed_and_real_data_transfer_cycle_per_DTL()

        # TODO: [jiacong] [MODIFY] update the latency model to fit IMC requirement
        self.combine_data_transfer_rate_per_physical_port_imc()
        # TODO: [jiacong] [FINISH]

        self.calc_data_loading_offloading_latency()
        self.calc_overall_latency()



    ## This function checks the double-buffer possibility for each operand at each memory level
    # (minimal memory BW requirement case) by comparing the physical memory size with the effective
    # data size, taking into account the memory sharing between operands.
    def calc_double_buffer_flag(self):
        double_buffer_true = {}
        for layer_op in self.layer.operand_list:
            mem_op = self.layer_op_to_mem_op[layer_op]
            """ start with False for each operand at the lowest arch level (MAC array level) """
            double_buffer_true[layer_op] = [False]
            for mem_lv in range(0, self.mapping_int.mem_level[layer_op]):
                if self.effective_mem_utili_shared[layer_op][mem_lv] <= 0.5:
                    double_buffer_true[layer_op].append(True)
                elif (
                    self.effective_mem_utili_individual[layer_op][mem_lv]
                    <= 1 - self.effective_mem_utili_shared[layer_op][mem_lv]
                ):
                    double_buffer_true[layer_op].append(True)
                    shared_mem_list = get_shared_mem_list(
                        mem_op, mem_lv, self.mem_sharing_list
                    )
                    """ When one of the operand in the shared memory get the "double-buffer" chance, 
                    all operands of that shared memory level need to update the memory utilization 
                    for later memory free space evaluation """
                    for shared_mem_op, shared_mem_lv in shared_mem_list:
                        try:
                            shared_layer_op = self.mem_op_to_layer_op[shared_mem_op]
                        except:  # mem op to layer op might not have this mem op (e.g. pooling layer)
                            continue
                        self.effective_mem_utili_shared[shared_layer_op][
                            shared_mem_lv
                        ] += self.effective_mem_utili_individual[layer_op][mem_lv]
                else:
                    double_buffer_true[layer_op].append(False)

        self.double_buffer_true = double_buffer_true

    ## Construct a 4-way data transfer pattern for each unit mem, calculate
    # {allowed_mem_updating_cycle, real_data_trans_cycle, DTL_SS_cycle} per period
    def calc_allowed_and_real_data_transfer_cycle_per_DTL(self):
        allowed_mem_updat_cycle = {}
        real_data_trans_cycle = {}
        """ stall (+) or slack (-) cycle within each period per virtual data transfer link (DTL) """
        DTL_SS_cycle = {}

        for layer_op in self.layer.operand_list:
            allowed_mem_updat_cycle[layer_op] = []
            real_data_trans_cycle[layer_op] = []
            DTL_SS_cycle[layer_op] = []
            mem_op = self.layer_op_to_mem_op[layer_op]
            for mem_lv in range(self.mapping_int.mem_level[layer_op]):
                """======================================allowed_mem_updating_cycle(below)====================================="""
                """ wr_in_by_low & rd_out_to_low"""
                if self.double_buffer_true[layer_op][mem_lv]:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.rd_out_to_low
                else:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.rd_out_to_low

                """ wr_in_by_high & rd_out_to_high """
                if self.double_buffer_true[layer_op][mem_lv + 1]:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.rd_out_to_high
                else:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.rd_out_to_high

                """ All """
                updating_window = FourWayDataMoving(
                    rd_out_to_low_allowed,
                    wr_in_by_low_allowed,
                    rd_out_to_high_allowed,
                    wr_in_by_high_allowed,
                )
                allowed_mem_updat_cycle[layer_op].append(updating_window)
                """ ======================================allowed_mem_updating_cycle(above)===================================== """

                """ =========================================real_data_trans_cycle(below)======================================== """
                """ wr_in_by_low """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_low
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ rd_out_to_low """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_low
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ rd_out_to_high """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_high
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ wr_in_by_high """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_high
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ All """
                real_data_trans = FourWayDataMoving(
                    rd_out_to_low_real,
                    wr_in_by_low_real,
                    rd_out_to_high_real,
                    wr_in_by_high_real,
                )
                real_data_trans_cycle[layer_op].append(real_data_trans)
                """ =========================================real_data_trans_cycle(above)======================================= """

        self.allowed_mem_updat_cycle = allowed_mem_updat_cycle
        self.real_data_trans_cycle = real_data_trans_cycle

    ## This function calculate the stalling cycles for IMC (In-Memory-Computing) hardware template
    # Consider memory sharing and port sharing, combine the data transfer activity
    # Step 1: collect port activity per memory instance per physical memory port
    # Step 2: calculate SS combine and MUW union parameters per physical memory port
    # Note: this calculation is incorrect when following conditions are ALL true:
    # (1) there are more than two mem levels for storing weights, e.g. dram -> cache -> IMC cells
    # (2) extra stalling is introduced due to the intermediate mem levels (e.g. due to insifficuent bw of cache)
    def combine_data_transfer_rate_per_physical_port_imc(self):
        # Step 1: collect port activity per memory instance per physical memory port
        port_activity_collect = []
        for mem_instance in self.mem_level_list:
            port_activity_single = {}
            port_list = mem_instance.port_list
            for port in port_list:
                port_activity_single[str(port)] = []
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    try:
                        layer_op = self.mem_op_to_layer_op[mem_op]
                    except:  # mem op to layer might not have this mem op (e.g. pooling layer)
                        continue
                    period_count = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip the inactive data movement activities because they won't impact SS
                        continue
                    period = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period,
                        mov_dir,
                    )
                    real_cycle = getattr(
                        self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                    )
                    allowed_cycle = getattr(
                        self.allowed_mem_updat_cycle[layer_op][mem_lv], mov_dir
                    )
                    port_activity = PortActivity(
                        real_cycle,
                        allowed_cycle,
                        period,
                        period_count,
                        layer_op,
                        mem_lv,
                        mov_dir,
                    )
                    port_activity_single[str(port)].append(port_activity)
            port_activity_collect.append(port_activity_single)
        self.port_activity_collect = port_activity_collect

        # TODO: [jiacong] [MODIFY]
        # Step 2: calculate weight loading cycles
        layer_const_operand = self.layer.constant_operands[0] # e.g. "W"
        # get spatial mapping in a macro
        core = next(iter(self.accelerator.cores))
        operational_array = core.operational_array
        memory_hierarchy = core.mem_hierarchy_dict
        hd_param = operational_array.unit.hd_param
        wl_dim = hd_param["wordline_dimension"]
        bl_dim = hd_param["bitline_dimension"]
        spatial_mapping_in_macro = []
        for layer_dim, loop in self.layer.user_spatial_mapping.items():
            if layer_dim in [wl_dim, bl_dim]:  # serve the dimension inside the macro
                if isinstance(loop[0], str):  # single layer_dim unrolling
                    spatial_mapping_in_macro.append(loop)
                else:  # mix layer_dim unrolling
                    for element in loop:
                        spatial_mapping_in_macro.append(element)
        # check if there is only one mem level for weight in accelerator. No weight loading required if that is the case.
        weight_mem_op = self.layer_op_to_mem_op[layer_const_operand]
        weight_mem_hierarchy: list = memory_hierarchy[weight_mem_op]
        if len(weight_mem_hierarchy) == 1: # there is only one mem level for weight
            require_weight_loading = False
        else:
            require_weight_loading = True
        # check how many times of weight reloading is required
        # here assume imc cells is the lowest mem level for weight and rw_port
        for imc_port, imc_ports in port_activity_collect[0].items(): # 0: the lowest mem node in the graph
            for port in imc_ports:
                if port.served_op_lv_dir[2] == "wr_in_by_high":
                    nb_of_weight_reload_periods = port.period_count

        # get the number of mapped rows in a macro
        imc_macro = operational_array.unit
        mapped_rows_total = imc_macro.mapped_rows_total

        # get the number of weights stored in each cell group
        mapped_group_depth = imc_macro.mapped_group_depth

        # calculate the total number of weight loading cycles
        if require_weight_loading:
            weight_loading_cycles = nb_of_weight_reload_periods * mapped_rows_total * mapped_group_depth
        else:
            weight_loading_cycles = 0

        self.SS_comb = weight_loading_cycles

        # Step 3: fetch tclk information
        self.tclk = operational_array.tclk
        self.tclk_breakdown = operational_array.tclk_breakdown
        # TODO: [jiacong] [FINISH]

    ## Calculate the initial/final data loading/off-loading cycle by separating out
    # the first-time input operands' / the last-time output operand's data movement
    # on corresponding ports.
    def calc_data_loading_offloading_latency(self):
        # Collect ports' initial data-loading and final data-offloading activities
        data_loading_per_mem_inst = []
        data_loading_cc_per_op = {op: {} for op in self.layer.input_operands}
        data_offloading_per_mem_inst = []
        data_offloading_cc_per_op = {}
        for mem_inst_idx, mem_instance in enumerate(self.mem_level_list):
            data_loading_single = {}
            data_offloading_single = {}
            port_list = mem_instance.port_list
            for port in port_list:
                data_loading_single[str(port)] = []
                data_offloading_single[str(port)] = []
                served_operands = set(
                    s[0] for s in port.served_op_lv_dir if s[0] in ["I1", "I2"]
                )
                port_is_shared_by_two_input_operands = len(served_operands) > 1
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    try:
                        layer_op = self.mem_op_to_layer_op[mem_op]
                    except:  # mem op to layer op might not have this mem op (e.g. pooling layer)
                        continue
                    period_count = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip for the inactive data movement
                        continue
                    if mem_op in ["I1", "I2"]:
                        real_cycle = getattr(
                            self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                        )
                        data_in_charge = getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period,
                            mov_dir,
                        ) * getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_precision,
                            mov_dir,
                        )
                        if mov_dir[:2] == "rd":
                            mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                        else:
                            mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                        port_activity = PortBeginOrEndActivity(
                            real_cycle,
                            data_in_charge,
                            mem_bw,
                            layer_op,
                            mem_lv,
                            mov_dir,
                        )
                        data_loading_single[str(port)].append(port_activity)
                        data_loading_cc_per_op[layer_op][
                            layer_op + str(mem_lv) + "_" + mov_dir
                        ] = (real_cycle, port_is_shared_by_two_input_operands)
                    else:
                        if mov_dir in ["rd_out_to_low", "wr_in_by_high"]:
                            # don't consider partial sum flowing in the final data off-loading stage
                            continue
                        real_cycle = getattr(
                            self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                        )
                        data_in_charge = getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period,
                            mov_dir,
                        ) * getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_precision,
                            mov_dir,
                        )
                        if mov_dir[:2] == "rd":
                            mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                        else:
                            mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                        port_activity = PortBeginOrEndActivity(
                            real_cycle,
                            data_in_charge,
                            mem_bw,
                            layer_op,
                            mem_lv,
                            mov_dir,
                        )
                        data_offloading_single[str(port)].append(port_activity)
                        data_offloading_cc_per_op[
                            layer_op + str(mem_lv) + "_" + mov_dir
                        ] = real_cycle

            data_loading_per_mem_inst.append(data_loading_single)
            data_offloading_per_mem_inst.append(data_offloading_single)
        self.data_loading_per_mem_inst = data_loading_per_mem_inst
        self.data_loading_cc_per_op = data_loading_cc_per_op
        self.data_offloading_per_mem_inst = data_offloading_per_mem_inst
        self.data_offloading_per_op = data_offloading_cc_per_op

        # Combine ports' initial data-loading activities to get the data loading cycle amount
        data_loading_cc_pair_combined_per_op = {
            op: [] for op in self.layer.input_operands
        }
        data_loading_individual_part = {op: 0 for op in self.layer.input_operands}
        data_loading_half_shared_part = {op: 0 for op in self.layer.input_operands}
        data_loading_shared_part = {op: 0 for op in self.layer.input_operands}
        for layer_op in self.layer.input_operands:
            for mem_lv in range(self.active_mem_level[layer_op] - 1):
                elem1 = data_loading_cc_per_op[layer_op][
                    layer_op + str(mem_lv) + "_" + "wr_in_by_high"
                ]
                elem2 = data_loading_cc_per_op[layer_op][
                    layer_op + str(mem_lv + 1) + "_" + "rd_out_to_low"
                ]
                completely_shared = elem1[1] and elem2[1]
                completely_separate = not (elem1[1]) and not (elem2[1])
                longest_loading_cc = max(elem1[0], elem2[0])
                # for the ports that serve the same data movement purpose, take the longest data loading cycle
                data_loading_cc_pair_combined = longest_loading_cc
                data_loading_cc_pair_combined_per_op[layer_op].append(
                    data_loading_cc_pair_combined
                )
                if completely_separate:
                    data_loading_individual_part[layer_op] += longest_loading_cc
                elif completely_shared:
                    data_loading_shared_part[layer_op] += longest_loading_cc
                else:
                    # the data transfer link between two memory levels is half-shared,
                    # i.e. on one memory side, the port is shared, while on another memory side,
                    # there are different memories with separate ports
                    data_loading_half_shared_part[layer_op] = longest_loading_cc

        if len(self.layer.input_operands) == 1:
            data_loading_cycle = data_loading_individual_part[
                self.layer.input_operands[0]
            ]
        else:
            op1 = self.layer.input_operands[0]
            op2 = self.layer.input_operands[1]
            possible1 = data_loading_shared_part[op1] + max(
                data_loading_shared_part[op2]
                + data_loading_half_shared_part[op2]
                + data_loading_individual_part[op2],
                data_loading_half_shared_part[op1] + data_loading_individual_part[op1],
            )
            possible2 = data_loading_shared_part[op2] + max(
                data_loading_shared_part[op1]
                + data_loading_half_shared_part[op1]
                + data_loading_individual_part[op1],
                data_loading_half_shared_part[op2] + data_loading_individual_part[op2],
            )
            data_loading_cycle = min(possible1, possible2)

        self.data_loading_cc_pair_combined_per_op = data_loading_cc_pair_combined_per_op
        self.data_loading_individual_part = data_loading_individual_part
        self.data_loading_half_shared_part = data_loading_half_shared_part
        self.data_loading_shared_part = data_loading_shared_part
        self.data_loading_cycle = data_loading_cycle

        # Combine ports' final data-offloading activities to get the data offloading cycle amount
        # TODO Only considered the worst case for now
        #  (assumed that all the ports are working in series during the final data off-loading phase)
        data_offloading_cc_pair_combined = []
        layer_op = self.layer.output_operand
        for mem_lv in range(self.active_mem_level[layer_op] - 1):
            elem1 = data_offloading_cc_per_op[
                layer_op + str(mem_lv) + "_" + "rd_out_to_high"
            ]
            elem2 = data_offloading_cc_per_op[
                layer_op + str(mem_lv + 1) + "_" + "wr_in_by_low"
            ]
            longest_offloading_cc = max(elem1, elem2)
            # for the ports that serve the same data movement purpose, take the longest data loading cycle
            data_offloading_cc_pair_combined.append(longest_offloading_cc)
        data_offloading_cycle = sum(data_offloading_cc_pair_combined)

        self.data_offloading_cc_pair_combined = data_offloading_cc_pair_combined
        self.data_offloading_cycle = data_offloading_cycle

    ## This function integrates the previous calculated SScomb, data loading and off-loading cycle to get the overall latency
    def calc_overall_latency(self):
        # the ideal cycle count assuming the MAC array is 100% utilized
        ideal_cycle = ceil(
            self.layer.total_MAC_count
            / self.accelerator.get_core(self.core_id).operational_array.total_unit_count
        )

        # the ideal temporal cycle count given the spatial mapping (the spatial mapping can be non-ideal)
        ideal_temporal_cycle = self.mapping_int.temporal_mapping.total_cycle

        # TODO: [jiacong] [ADD]
        # scale ideal_temporal_cycle considering bit-serial input of imc
        operational_array = self.accelerator.get_core(self.core_id).operational_array
        hd_param = operational_array.unit.hd_param
        nb_of_cycles_per_mac = hd_param["input_precision"] / hd_param["input_bit_per_cycle"]
        ideal_cycle *= nb_of_cycles_per_mac
        ideal_temporal_cycle *= nb_of_cycles_per_mac
        # TODO: [jiacong] [FINISH]

        MAC_spatial_utilization = ideal_cycle / ideal_temporal_cycle

        # Total latency without the initial data loading and the final data off-loading
        latency_total0 = ideal_temporal_cycle + self.SS_comb
        MAC_utilization0 = ideal_cycle / latency_total0

        # Total latency with the initial data loading, but without the final data off-loading
        latency_total1 = ideal_temporal_cycle + self.SS_comb + self.data_loading_cycle
        MAC_utilization1 = ideal_cycle / latency_total1

        # Total latency with both the initial data loading and the final data off-loading
        latency_total2 = (
            ideal_temporal_cycle
            + self.SS_comb
            + self.data_loading_cycle
            + self.data_offloading_cycle
        )
        MAC_utilization2 = ideal_cycle / latency_total2

        self.ideal_cycle = ideal_cycle
        self.ideal_temporal_cycle = ideal_temporal_cycle
        self.MAC_spatial_utilization = MAC_spatial_utilization
        self.latency_total0 = latency_total0
        self.latency_total1 = latency_total1
        self.latency_total2 = latency_total2
        self.MAC_utilization0 = MAC_utilization0
        self.MAC_utilization1 = MAC_utilization1
        self.MAC_utilization2 = MAC_utilization2

    def __add__(self, other):
        sum = pickle_deepcopy(self)

        ## Energy
        sum.MAC_energy += other.MAC_energy
        sum.mem_energy += other.mem_energy
        for op in sum.MAC_energy_breakdown.keys():
            if op in other.MAC_energy_breakdown.keys():
                sum.MAC_energy_breakdown[op] = self.MAC_energy_breakdown[op] + other.MAC_energy_breakdown[op]

        for op in sum.mem_energy_breakdown.keys():
            if op in other.mem_energy_breakdown.keys():
                l = []
                for i in range(
                    min(len(self.mem_energy_breakdown[op]), len(other.mem_energy_breakdown[op]))
                ):
                    l.append(
                        self.mem_energy_breakdown[op][i] + other.mem_energy_breakdown[op][i]
                    )
                i = min(len(self.mem_energy_breakdown[op]), len(other.mem_energy_breakdown[op]))
                l += self.mem_energy_breakdown[op][i:]
                l += other.mem_energy_breakdown[op][i:]
                sum.mem_energy_breakdown[op] = l

        for op in sum.mem_energy_breakdown_further.keys():
            if op in other.mem_energy_breakdown_further.keys():
                l = []
                for i in range(
                    min(
                        len(self.mem_energy_breakdown_further[op]),
                        len(other.mem_energy_breakdown_further[op]),
                    )
                ):
                    l.append(
                        self.mem_energy_breakdown_further[op][i]
                        + other.mem_energy_breakdown_further[op][i]
                    )
                i = min(
                    len(self.mem_energy_breakdown_further[op]),
                    len(other.mem_energy_breakdown_further[op]),
                )
                l += self.mem_energy_breakdown_further[op][i:]
                l += other.mem_energy_breakdown_further[op][i:]
                sum.mem_energy_breakdown_further[op] = l

        # Get all the operands from other that are not in self and add them to the energy breakdown as well
        op_diff = set(other.mem_energy_breakdown.keys()) - set(self.mem_energy_breakdown.keys())
        for op in op_diff:
            sum.mem_energy_breakdown[op] = other.mem_energy_breakdown[op]
            sum.mem_energy_breakdown_further[op] = other.mem_energy_breakdown_further[op]

        op_diff = set(other.MAC_energy_breakdown.keys()) - set(self.MAC_energy_breakdown.keys())
        for op in op_diff:
            sum.MAC_energy_breakdown[op] = other.MAC_energy_breakdown[op]

        sum.energy_total += other.energy_total

        ## Memory access
        for op in sum.memory_word_access.keys():
            if op in other.memory_word_access.keys():
                l = []
                for i in range(
                    min(
                        len(self.memory_word_access[op]),
                        len(other.memory_word_access[op]),
                    )
                ):
                    l.append(
                        self.memory_word_access[op][i] + other.memory_word_access[op][i]
                    )
                i = min(
                    len(self.memory_word_access[op]), len(other.memory_word_access[op])
                )
                l += self.memory_word_access[op][i:]
                l += other.memory_word_access[op][i:]
                sum.memory_word_access[op] = l
        for op in op_diff:
            sum.memory_word_access[op] = other.memory_word_access[op]

        ## Latency
        sum.data_loading_cycle += other.data_loading_cycle
        sum.data_offloading_cycle += other.data_offloading_cycle
        sum.ideal_cycle += other.ideal_cycle
        sum.SS_comb += other.SS_comb  # stalling cycles
        sum.ideal_temporal_cycle += other.ideal_temporal_cycle  # ideal computation cycles without stalling
        sum.latency_total0 += other.latency_total0
        sum.latency_total1 += other.latency_total1
        sum.latency_total2 += other.latency_total2

        ## MAC utilization
        sum.MAC_spatial_utilization = sum.ideal_cycle / sum.ideal_temporal_cycle
        sum.MAC_utilization0 = sum.ideal_cycle / sum.latency_total0
        sum.MAC_utilization1 = sum.ideal_cycle / sum.latency_total1
        sum.MAC_utilization2 = sum.ideal_cycle / sum.latency_total2

        ## layer
        if type(sum.layer) != list:
            sum.layer = [sum.layer.id]
        if type(other.layer) != list:
            other_layer = [other.layer.id]
        sum.layer += other_layer

        ## core_id
        if type(sum.core_id) != list:
            sum.core_id = [sum.core_id]
        if type(other.layer) != list:
            other_core_id = [other.core_id]
        sum.core_id += other_core_id

        ## Not addable
        func = [
            "calc_allowed_and_real_data_transfer_cycle_per_DTL",
            "calc_data_loading_offloading_latency",
            "calc_double_buffer_flag",
            "calc_overall_latency",
            "calc_MAC_energy_cost",
            "calc_energy",
            "calc_latency",
            "calc_memory_energy_cost",
            "calc_memory_utilization",
            "calc_memory_word_access",
            "combine_data_transfer_rate_per_physical_port_imc",
            "collect_area_data",
            "run",
        ]
        add_attr = [
            "MAC_energy",
            "mem_energy",
            "MAC_energy_breakdown",
            "mem_energy_breakdown",
            "mem_energy_breakdown_further",
            "energy_total",
            "memory_word_access",
            "data_loading_cycle",
            "data_offloading_cycle",
            "ideal_cycle",
            "ideal_temporal_cycle",
            "SS_comb",
            "latency_total0",
            "latency_total1",
            "latency_total2",
            "tclk",
            "tclk_breakdown",
            "MAC_spatial_utilization",
            "MAC_utilization0",
            "MAC_utilization1",
            "MAC_utilization2",
            "area_total",
            "imc_area",
            "mem_area",
            "imc_area_breakdown",
            "mem_area_breakdown",
            "layer",
            "core_id",
        ]

        if hasattr(self, "accelerator") and hasattr(other, "accelerator"):
            if self.accelerator.name.startswith(other.accelerator.name):
                sum.accelerator = other.accelerator
                add_attr.append("accelerator")
            elif other.accelerator.name.startswith(self.accelerator.name):
                add_attr.append("accelerator")
        else:
            pass

        for attr in dir(sum):
            if attr not in (func + add_attr) and attr[0] != "_":
                delattr(sum, attr)

        return sum

    # def __mul__(self, number):
    #     mul = pickle_deepcopy(self)
    #
    #     # Energy
    #     mul.MAC_energy *= number
    #     mul.mem_energy *= number
    #     mul.mem_energy_breakdown = {
    #         op: [
    #             mul.mem_energy_breakdown[op][i] * number
    #             for i in range(len(mul.mem_energy_breakdown[op]))
    #         ]
    #         for op in mul.mem_energy_breakdown.keys()
    #     }
    #     mul.mem_energy_breakdown_further = {
    #         op: [
    #             mul.mem_energy_breakdown_further[op][i] * number
    #             for i in range(len(mul.mem_energy_breakdown_further[op]))
    #         ]
    #         for op in mul.mem_energy_breakdown_further.keys()
    #     }
    #     mul.energy_total *= number
    #
    #     # Memory access
    #     mul.memory_word_access = {
    #         op: [
    #             mul.memory_word_access[op][i] * number
    #             for i in range(len(mul.memory_word_access[op]))
    #         ]
    #         for op in mul.memory_word_access.keys()
    #     }
    #
    #     # Latency
    #     mul.data_loading_cycle *= number
    #     mul.data_offloading_cycle *= number
    #     mul.ideal_cycle *= number
    #     mul.ideal_temporal_cycle *= number
    #     mul.latency_total0 *= number
    #     mul.latency_total1 *= number
    #     mul.latency_total2 *= number
    #
    #     # MAC utilization
    #     mul.MAC_spatial_utilization = mul.ideal_cycle / mul.ideal_temporal_cycle
    #     mul.MAC_utilization0 = mul.ideal_cycle / mul.latency_total0
    #     mul.MAC_utilization1 = mul.ideal_cycle / mul.latency_total1
    #     mul.MAC_utilization2 = mul.ideal_cycle / mul.latency_total2
    #
    #     # Not addable
    #     func = [
    #         "calc_allowed_and_real_data_transfer_cycle_per_DTL",
    #         "calc_data_loading_offloading_latency",
    #         "calc_double_buffer_flag",
    #         "calc_overall_latency",
    #         "calc_MAC_energy_cost",
    #         "calc_energy",
    #         "calc_latency",
    #         "calc_memory_energy_cost",
    #         "calc_memory_utilization",
    #         "calc_memory_word_access",
    #         "combine_data_transfer_rate_per_physical_port",
    #         "run",
    #     ]
    #     mul_attr = [
    #         "MAC_energy",
    #         "mem_energy",
    #         "mem_energy_breakdown",
    #         "mem_energy_breakdown_further",
    #         "energy_total",
    #         "memory_word_access",
    #         "data_loading_cycle",
    #         "data_offloading_cycle",
    #         "ideal_cycle",
    #         "ideal_temporal_cycle",
    #         "latency_total0",
    #         "latency_total1",
    #         "latency_total2",
    #         "MAC_spatial_utilization",
    #         "MAC_utilization0",
    #         "MAC_utilization1",
    #         "MAC_utilization2",
    #         "layer",
    #         "accelerator",
    #     ]
    #
    #     for attr in dir(mul):
    #         if attr not in (func + mul_attr) and attr[0] != "_":
    #             delattr(mul, attr)
    #
    #     return mul
