import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import lru_cache
from math import ceil
from typing import TypedDict

import numpy as np

from zigzag.cost_model.port_activity import PortActivity, PortBeginOrEndActivity
from zigzag.datatypes import ArrayType, Constants, LayerOperand, MemoryOperand
from zigzag.hardware.architecture.accelerator import Accelerator
from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.hardware.architecture.memory_port import MemoryPort
from zigzag.hardware.architecture.operational_array import OperationalArray
from zigzag.mapping.data_movement import AccessEnergy, DataDirection, FourWayDataMoving, MemoryAccesses
from zigzag.mapping.mapping import Mapping
from zigzag.mapping.spatial_mapping_internal import SpatialMappingInternal
from zigzag.mapping.temporal_mapping import TemporalMapping
from zigzag.utils import json_repr_handler, pickle_deepcopy
from zigzag.workload.layer_node import LayerNode

logger = logging.getLogger(__name__)


class MemoryUtilization(TypedDict):
    individual: dict[LayerOperand, list[float]]
    shared: dict[LayerOperand, list[float]]


class CostModelEvaluationABC(metaclass=ABCMeta):
    """! Superclass for CostModelEvaluation and CumulativeCME"""

    accelerator: Accelerator

    @abstractmethod
    def __init__(self) -> None:
        # Attributes that all subclasses should define
        self.mem_energy_breakdown: dict[LayerOperand, list[float]]
        self.mem_energy_breakdown_further: dict[LayerOperand, list[AccessEnergy]]
        self.memory_word_access: dict[LayerOperand, list[MemoryAccesses]]
        self.mac_energy: float
        self.mem_energy: float
        self.energy_total: float
        self.data_onloading_cycle: float
        self.data_offloading_cycle: float
        self.ideal_cycle: float
        self.ideal_temporal_cycle: float
        self.latency_total0: float
        self.latency_total1: float
        self.latency_total2: float
        self.mac_spatial_utilization: float
        self.mac_utilization0: float
        self.mac_utilization1: float
        self.mac_utilization2: float

        self.accelerator: Accelerator | None

    @property
    def core(self):
        return self.accelerator

    def __add__(self, other: "CostModelEvaluationABC") -> "CumulativeCME":
        result = CumulativeCME()

        # Energy
        result.mac_energy = self.mac_energy + other.mac_energy
        result.mem_energy = self.mem_energy + other.mem_energy
        result.energy_total = self.energy_total + other.energy_total

        for layer_op, breakdown_this in self.mem_energy_breakdown.items():
            if layer_op in other.mem_energy_breakdown.keys():
                breakdown_other = other.mem_energy_breakdown[layer_op]
                list_temp: list[float] = []
                breakdown_len = min(len(breakdown_this), len(breakdown_other))
                for i in range(breakdown_len):
                    list_temp.append(breakdown_this[i] + breakdown_other[i])

                list_temp += breakdown_this[breakdown_len:]
                list_temp += breakdown_other[breakdown_len:]
                result.mem_energy_breakdown[layer_op] = list_temp

        for layer_op, breakdown_this in self.mem_energy_breakdown_further.items():
            if layer_op in other.mem_energy_breakdown_further.keys():
                breakdown_other = other.mem_energy_breakdown_further[layer_op]
                list_temp2: list[AccessEnergy] = []
                breakdown_len = min(len(breakdown_this), len(breakdown_other))
                for i in range(breakdown_len):
                    list_temp2.append(breakdown_this[i] + breakdown_other[i])
                list_temp2 += breakdown_this[breakdown_len:]
                list_temp2 += breakdown_other[breakdown_len:]
                result.mem_energy_breakdown_further[layer_op] = list_temp2

        # Add the operands from other that are either in `self` or `other`, but not yet in `result`
        op_diff_self = set(self.mem_energy_breakdown.keys()) - set(result.mem_energy_breakdown.keys())
        op_diff_other = set(other.mem_energy_breakdown.keys()) - set(result.mem_energy_breakdown.keys())
        for layer_op in op_diff_self:
            result.mem_energy_breakdown[layer_op] = self.mem_energy_breakdown[layer_op]
            result.mem_energy_breakdown_further[layer_op] = self.mem_energy_breakdown_further[layer_op]
        for layer_op in op_diff_other:
            result.mem_energy_breakdown[layer_op] = other.mem_energy_breakdown[layer_op]
            result.mem_energy_breakdown_further[layer_op] = other.mem_energy_breakdown_further[layer_op]

        # Memory access
        for layer_op, accesses_this in self.memory_word_access.items():
            if layer_op in other.memory_word_access.keys():
                accesses_other = other.memory_word_access[layer_op]
                length = min(len(accesses_this), len(accesses_other))
                list_temp3: list[MemoryAccesses] = []
                for i in range(length):
                    list_temp3.append(accesses_this[i] + accesses_other[i])
                list_temp3 += accesses_this[length:]
                list_temp3 += accesses_other[length:]
                result.memory_word_access[layer_op] = list_temp3

        for layer_op in op_diff_self:
            result.memory_word_access[layer_op] = self.memory_word_access[layer_op]
        for layer_op in op_diff_other:
            result.memory_word_access[layer_op] = other.memory_word_access[layer_op]

        # Latency
        result.data_onloading_cycle = self.data_onloading_cycle + other.data_onloading_cycle
        result.data_offloading_cycle = self.data_offloading_cycle + other.data_offloading_cycle
        result.ideal_cycle = self.ideal_cycle + other.ideal_cycle
        result.ideal_temporal_cycle = self.ideal_temporal_cycle + other.ideal_temporal_cycle
        result.latency_total0 = self.latency_total0 + other.latency_total0
        result.latency_total1 = self.latency_total1 + other.latency_total1
        result.latency_total2 = self.latency_total2 + other.latency_total2

        # MAC utilization
        result.mac_spatial_utilization = result.ideal_cycle / result.ideal_temporal_cycle
        result.mac_utilization0 = result.ideal_cycle / result.latency_total0
        result.mac_utilization1 = result.ideal_cycle / result.latency_total1
        result.mac_utilization2 = result.ideal_cycle / result.latency_total2

        for cme in (self, other):
            if isinstance(cme, CostModelEvaluation):
                result.layer_ids.append(cme.layer.id)
            elif isinstance(cme, CumulativeCME):
                result.layer_ids += cme.layer_ids

        # Select accelerator for result
        if isinstance(self, CumulativeCME) and self.accelerator is None:
            assert other.accelerator is not None, "Accelerator undefined on both CMEs being added"
            result.accelerator = other.accelerator
        elif isinstance(other, CumulativeCME) and other.accelerator is None:
            assert self.accelerator is not None, "Accelerator undefined on both CMEs being added"
            result.accelerator = self.accelerator
        elif self.accelerator is not None and other.accelerator is not None:
            if self.accelerator.name.startswith(other.accelerator.name):
                result.accelerator = other.accelerator
            elif other.accelerator.name.startswith(self.accelerator.name):
                result.accelerator = self.accelerator
            else:
                raise ValueError("Adding CMEs of unrelated accelerators")

        return result

    def __mul__(self, number: int):
        result: "CostModelEvaluationABC" = pickle_deepcopy(self)

        # Energy
        result.mac_energy *= number
        result.mem_energy *= number
        result.energy_total *= number

        result.mem_energy_breakdown = {
            op: [result.mem_energy_breakdown[op][i] * number for i in range(len(result.mem_energy_breakdown[op]))]
            for op in result.mem_energy_breakdown.keys()
        }
        result.mem_energy_breakdown_further = {
            op: [
                result.mem_energy_breakdown_further[op][i] * number
                for i in range(len(result.mem_energy_breakdown_further[op]))
            ]
            for op in result.mem_energy_breakdown_further.keys()
        }

        # Memory access
        result.memory_word_access = {
            op: [result.memory_word_access[op][i] * number for i in range(len(result.memory_word_access[op]))]
            for op in result.memory_word_access.keys()
        }

        # Latency
        result.data_onloading_cycle *= number
        result.data_offloading_cycle *= number
        result.ideal_cycle *= number
        result.ideal_temporal_cycle *= number
        result.latency_total0 *= number
        result.latency_total1 *= number
        result.latency_total2 *= number

        # MAC utilization
        result.mac_spatial_utilization = result.ideal_cycle / result.ideal_temporal_cycle
        result.mac_utilization0 = result.ideal_cycle / result.latency_total0
        result.mac_utilization1 = result.ideal_cycle / result.latency_total1
        result.mac_utilization2 = result.ideal_cycle / result.latency_total2

        return result

    def __simplejsonrepr__(self) -> dict[str, float]:
        """! Simple JSON representation used for saving this object to a simple json file."""
        return {"energy": self.energy_total, "latency": self.latency_total2}

    def __jsonrepr__(self):
        """! JSON representation used for saving this object to a json file."""
        return json_repr_handler(
            {
                "outputs": {
                    "memory": {
                        "utilization": (self.mem_utili_shared if isinstance(self, CostModelEvaluation) else None),
                        "word_accesses": self.memory_word_access,
                    },
                    "energy": {
                        "energy_total": self.energy_total,
                        "operational_energy": self.mac_energy,
                        "memory_energy": self.mem_energy,
                        "memory_energy_breakdown_per_level": self.mem_energy_breakdown,
                        "memory_energy_breakdown_per_level_per_operand": self.mem_energy_breakdown_further,
                    },
                    "latency": {
                        "data_onloading": self.latency_total1 - self.latency_total0,
                        "computation": self.latency_total0,
                        "data_offloading": self.latency_total2 - self.latency_total1,
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
                    "layer": (
                        self.layer
                        if isinstance(self, CostModelEvaluation)
                        else self.layer_ids if isinstance(self, CumulativeCME) else None
                    ),
                    "spatial_mapping": (self.spatial_mapping_int if isinstance(self, CostModelEvaluation) else None),
                    "temporal_mapping": (self.temporal_mapping if isinstance(self, CostModelEvaluation) else None),
                },
            }
        )


class CumulativeCME(CostModelEvaluationABC):
    """! Represents the sum of multiple CMEs. This class only contains attributes that make sense for cumulated CMEs"""

    def __init__(self):
        self.mem_energy_breakdown: dict[LayerOperand, list[float]] = dict()
        self.mem_energy_breakdown_further: dict[LayerOperand, list[AccessEnergy]] = dict()
        self.memory_word_access: dict[LayerOperand, list[MemoryAccesses]] = dict()

        self.layer_ids: list[int] = []

        self.mac_energy: float = 0.0
        self.mem_energy: float = 0.0
        self.energy_total: float = 0.0
        self.data_onloading_cycle: float = 0.0
        self.data_offloading_cycle: float = 0.0
        self.ideal_cycle: float = 0.0
        self.ideal_temporal_cycle: float = 0.0
        self.latency_total0: float = 0.0
        self.latency_total1: float = 0.0
        self.latency_total2: float = 0.0

        self.accelerator = None

    def __str__(self):
        return "CumulativeCME"


class CostModelEvaluation(CostModelEvaluationABC):
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
        *,
        accelerator: Accelerator,
        layer: LayerNode,
        spatial_mapping: SpatialMappingInternal,
        spatial_mapping_int: SpatialMappingInternal,
        temporal_mapping: TemporalMapping,
        access_same_data_considered_as_no_access: bool = True,
    ):
        """
        After initialization, the cost model evaluation is run
        @param accelerator the accelerator that includes the core on which to run the
        @param layer the layer to run
        @param access_same_data_considered_as_no_access (optional)
        """
        self.accelerator = accelerator
        self.layer: LayerNode = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = spatial_mapping_int  # the original spatial mapping without decimal
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = access_same_data_considered_as_no_access

        self.mem_level_list = accelerator.memory_hierarchy.mem_level_list
        self.mem_hierarchy_dict = accelerator.mem_hierarchy_dict
        self.mem_size_dict = accelerator.mem_size_dict
        self.mem_r_bw_dict, self.mem_w_bw_dict = accelerator.get_memory_bw_dict()
        self.mem_r_bw_min_dict, self.mem_w_bw_min_dict = accelerator.get_memory_bw_min_dict()
        self.mem_sharing_tuple = tuple(tuple(i.items()) for i in accelerator.mem_sharing_list)
        self.memory_operand_links = layer.memory_operand_links

        self.cumulative_layer_ids: list[int] = []  # In case the CME results from adding other CMEs together

        # generate the integer spatial mapping from fractional spatial mapping (due to greedy mapping support).
        # Later the fractional one is used for calculating energy, and the integer one is used for calculating latency
        self.spatial_mapping_dict_int = self.spatial_mapping_int.mapping_dict_origin

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

        # Run the cost model evaluation
        self.run()

    def run(self) -> None:
        """! Run the cost model evaluation."""
        self.calc_memory_utilization()
        self.calc_memory_word_access()
        self.calc_energy()
        self.calc_latency()

    @lru_cache(maxsize=512)
    def __get_shared_mem_list(
        self,
        mem_op: MemoryOperand,
        mem_lv: int,
    ) -> tuple[tuple[MemoryOperand, int], ...] | None:
        """! Given a certain operand's storage level (for example (A,1): operand A's 1st memory level),
        return a list of the rest operand's storage levels that share physical memory with the former one (A1)
        """
        for mem_share_group in self.mem_sharing_tuple:
            if (mem_op, mem_lv) in mem_share_group:
                return mem_share_group
        return None

    def __calc_mem_updating_window_union(self, port_duty_list: list[PortActivity]) -> int:
        """!  This function calculates the union length of all the share-port MUW (memory updating window).
        The following encoding has to be used:
        - 'P' for single period length
        - 'A' for allowed MUW per period
        - 'PC' for period count within the whole layer computation

        Pre-process the port_duty_list to generate input_dict, which looks like:
        - input_dict = {'O1': {'P': 3, 'A': 1, 'PC': 8}, 'O2': {'P': 6, 'A': 2, 'PC': 4},
        'O3': {'P': 12, 'A': 4, 'PC': 2}}
        # TODO clean up
        """

        input_dict: dict[str, dict[str, int]] = {}
        # As long as one of the port duty can make use of the whole computation time, the MUW union is
        # set to the whole computation time
        for port_duty in port_duty_list:
            if port_duty.period == port_duty.allowed_cycle:
                return port_duty.period * port_duty.period_count
            key = str(port_duty.served_op_lv_dir)
            input_dict[key] = {
                "P": int(port_duty.period),
                "A": int(port_duty.allowed_cycle),
                "PC": int(port_duty.period_count),
            }

        max_period = 0
        max_period_operand: str = ""
        for op, values in input_dict.items():
            if values["P"] > max_period:
                max_period = values["P"]
                max_period_operand = op

        indicators: ArrayType = np.zeros((len(input_dict), max_period))
        for i, op in enumerate(input_dict):
            # reshape to period of this operand
            indicators_reshape: ArrayType = indicators.reshape((len(input_dict), -1, input_dict[op]["P"]))
            # fill in first few time units as used
            indicators_reshape[i, :, : input_dict[op]["A"]] = 1

        union = max_period - int((~indicators.any(axis=0)).sum())

        # Multiply with number of periods of largest period (as it was normalized to largest period)
        return union * input_dict[max_period_operand]["PC"]

    def calc_memory_utilization(self) -> None:
        mem_utilization: MemoryUtilization = {
            "individual": defaultdict(list),
            "shared": defaultdict(list),
        }
        effective_mem_utilization: MemoryUtilization = {
            "individual": defaultdict(list),
            "shared": defaultdict(list),
        }

        for layer_op in self.layer.layer_operands:
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            for mem_lv in range(self.active_mem_level[layer_op]):
                mem_utilization["individual"][layer_op].append(
                    self.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1] / self.mem_size_dict[mem_op][mem_lv]
                )
                effective_mem_utilization["individual"][layer_op].append(
                    self.mapping.effective_data_bit[layer_op][mem_lv + 1] / self.mem_size_dict[mem_op][mem_lv]
                )

        mem_utili_shared = pickle_deepcopy(mem_utilization["individual"])
        effective_mem_utilization_shared = pickle_deepcopy(effective_mem_utilization["individual"])

        for mem_share_dict in self.mem_sharing_tuple:
            mem_utilization_sum = sum(
                mem_utilization["individual"][self.memory_operand_links.mem_to_layer_op(mem_op)][mem_lv]
                for mem_op, mem_lv in mem_share_dict
                if self.memory_operand_links.contains_mem_op(mem_op)
            )
            effective_mem_utilization_sum = sum(
                effective_mem_utilization["individual"][self.memory_operand_links.mem_to_layer_op(mem_op)][mem_lv]
                for mem_op, mem_lv in mem_share_dict
                if self.memory_operand_links.contains_mem_op(mem_op)
            )

            for mem_op, mem_lv in mem_share_dict:
                if self.memory_operand_links.contains_mem_op(mem_op):
                    layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
                    mem_utili_shared[layer_op][mem_lv] = mem_utilization_sum
                    effective_mem_utilization_shared[layer_op][mem_lv] = effective_mem_utilization_sum

        self.mem_utili_individual = mem_utilization["individual"]
        self.mem_utili_shared = mem_utili_shared
        self.effective_mem_utili_individual = effective_mem_utilization["individual"]
        self.effective_mem_utili_shared = effective_mem_utilization_shared

    def calc_memory_word_access(self) -> None:
        memory_word_access: dict[LayerOperand, list[MemoryAccesses]] = defaultdict(list)

        for layer_op in self.layer.layer_operands:
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            for mem_lv in range(self.mapping.mem_level[layer_op]):
                data_elem_move = self.mapping.unit_mem_data_movement[layer_op][mem_lv]
                max_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                min_bw = self.mem_w_bw_min_dict[mem_op][mem_lv]

                wr_in_by_low = self._calc_memory_access(
                    data_elem_move.data_trans_amount_per_period.wr_in_by_low,
                    data_elem_move.data_precision.wr_in_by_low,
                    data_elem_move.data_trans_period_count.wr_in_by_low,
                    max_bw,
                    min_bw,
                    layer_op,
                    mem_lv,
                )

                rd_out_to_low = self._calc_memory_access(
                    data_elem_move.data_trans_amount_per_period.rd_out_to_low,
                    data_elem_move.data_precision.rd_out_to_low,
                    data_elem_move.data_trans_period_count.rd_out_to_low,
                    self.mem_r_bw_dict[mem_op][mem_lv],
                    self.mem_r_bw_min_dict[mem_op][mem_lv],
                    layer_op,
                    mem_lv,
                )

                rd_out_to_high = self._calc_memory_access(
                    data_elem_move.data_trans_amount_per_period.rd_out_to_high,
                    data_elem_move.data_precision.rd_out_to_high,
                    data_elem_move.data_trans_period_count.rd_out_to_high,
                    self.mem_r_bw_dict[mem_op][mem_lv],
                    self.mem_r_bw_min_dict[mem_op][mem_lv],
                    layer_op,
                    mem_lv,
                )

                wr_in_by_high = self._calc_memory_access(
                    data_elem_move.data_trans_amount_per_period.wr_in_by_high,
                    data_elem_move.data_precision.wr_in_by_high,
                    data_elem_move.data_trans_period_count.wr_in_by_high,
                    max_bw,
                    min_bw,
                    layer_op,
                    mem_lv,
                )

                memory_word_access[layer_op].append(
                    MemoryAccesses(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)
                )

        self.memory_word_access = memory_word_access

    def _calc_memory_access(
        self,
        data_elem_move_per_period: int,
        data_precision: int,
        total_period_count: int,
        max_bw: int,
        min_bw: int,
        layer_op: LayerOperand,
        mem_lv: int,
    ) -> int:
        if data_elem_move_per_period == 0 or data_precision == 0:
            return 0
        return int(
            ceil((data_elem_move_per_period * data_precision) / min_bw)
            * (min_bw / max_bw)
            * total_period_count
            * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
        )

    def calc_energy(self) -> None:
        """! Calculates the energy cost of this cost model evaluation by calculating the memory reading/writing
        energy."""
        # TODO: Interconnection energy
        self.calc_mac_energy_cost()
        self.calc_memory_energy_cost()

    def calc_mac_energy_cost(self) -> None:
        """! Calculate the dynamic MAC energy"""
        operational_array = self.accelerator.operational_array
        assert isinstance(
            operational_array, OperationalArray
        ), "This method expects an OperationalArray instance. Otherwise, the method should be overridden in a subclass."
        single_mac_energy = operational_array.unit.energy_cost
        self.mac_energy = single_mac_energy * self.layer.total_mac_count

    def calc_memory_energy_cost(self):
        """! Computes the memories reading/writing energy by converting the access patterns in self.mapping to
        energy breakdown using the memory hierarchy of the core on which the layer is mapped.

        The energy breakdown is saved in self.mem_energy_breakdown.

        The energy total consumption is saved in self.energy_total.
        """
        mem_hierarchy = self.accelerator.memory_hierarchy

        mem_energy_breakdown: dict[LayerOperand, list[float]] = {}
        mem_energy_breakdown_further: dict[LayerOperand, list[AccessEnergy]] = {}
        energy_total = 0

        # Retrieve the memory levels in the hierarchy for this memory operand
        for layer_op, mem_access_list_per_op in self.memory_word_access.items():
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            memory_levels = mem_hierarchy.get_memory_levels(mem_op=mem_op)

            breakdown: list[float] = []  # Stores the energy breakdown of a single layer operand (W, I, ...)
            breakdown_further: list[AccessEnergy] = []
            for access_count, memory_level in zip(mem_access_list_per_op, memory_levels):
                energy_cost_per_read_out = memory_level.read_energy
                energy_cost_per_write_in = memory_level.write_energy
                read_out_energy_to_above = access_count.get_total_read_outs_to_above() * energy_cost_per_read_out
                write_in_energy_from_above = access_count.get_total_write_ins_from_above() * energy_cost_per_write_in
                read_out_energy_to_below = access_count.get_total_read_outs_to_below() * energy_cost_per_read_out
                write_in_energy_from_below = access_count.get_total_write_ins_from_below() * energy_cost_per_write_in
                total_read_out_energy = read_out_energy_to_above + read_out_energy_to_below
                total_write_in_energy = write_in_energy_from_above + write_in_energy_from_below
                total_energy_cost_memory = total_read_out_energy + total_write_in_energy
                # Here the breakdown only saves the total energy cost per memory level
                breakdown.append(total_energy_cost_memory)
                breakdown_further.append(
                    AccessEnergy(
                        read_out_energy_to_below,
                        write_in_energy_from_below,
                        read_out_energy_to_above,
                        write_in_energy_from_above,
                    )
                )
                energy_total += total_energy_cost_memory
            mem_energy_breakdown[layer_op] = breakdown
            mem_energy_breakdown_further[layer_op] = breakdown_further
        self.mem_energy_breakdown = mem_energy_breakdown
        self.mem_energy_breakdown_further = mem_energy_breakdown_further
        self.mem_energy = energy_total
        self.energy_total: float = self.mem_energy + self.mac_energy
        logger.debug("Ran %s. Total energy = %f", self, self.energy_total)

    def calc_latency(self) -> None:
        """!  Calculate latency in 4 steps

        1) As we already calculated the ideal data transfer rate in combined_mapping.py (in the Mapping class),
        here we start with calculating the required (or allowed) memory updating window by comparing the effective
        data size with the physical memory size at each level. If the effective data size is smaller than 50%
        of the physical memory size, then we take the whole period as the allowed memory updating window
        (double buffer effect);
        otherwise we take the the period divided by the top_ir_loop as the allowed memory updating window.

        2) Then, we compute the real data transfer rate given the actual memory bw per functional port pair,
        assuming we have enough memory ports.

        3) In reality, there is no infinite memory port to use. So, as the second step, we combine the real
        data transfer attributes per physical memory port.

        4) Finally, we combine the stall/slack of each memory port to get the final latency.
        """
        self.calc_double_buffer_flag()
        self.calc_allowed_and_real_data_transfer_cycle_per_data_transfer_link()
        self.combine_data_transfer_rate_per_physical_port()
        self.calc_data_loading_latency()
        self.calc_overall_latency()

    def calc_double_buffer_flag(self) -> None:
        """! This function checks the double-buffer possibility for each operand at each memory level
        (minimal memory BW requirement case) by comparing the physical memory size with the effective
        data size, taking into account the memory sharing between operands.
        """
        double_buffer_true: dict[LayerOperand, list[bool]] = {}
        for layer_op in self.layer.layer_operands:
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            # start with False for each operand at the lowest arch level (MAC array level)
            double_buffer_true[layer_op] = [False]
            for mem_lv in range(0, self.mapping_int.mem_level[layer_op]):
                if self.effective_mem_utili_shared[layer_op][mem_lv] <= 0.5:
                    double_buffer_true[layer_op].append(True)
                elif (
                    self.effective_mem_utili_individual[layer_op][mem_lv]
                    <= 1 - self.effective_mem_utili_shared[layer_op][mem_lv]
                ):
                    double_buffer_true[layer_op].append(True)
                    shared_mem_list = self.__get_shared_mem_list(mem_op, mem_lv)
                    # When one of the operand in the shared memory get the "double-buffer" chance,
                    # all operands of that shared memory level need to update the memory utilization
                    # for later memory free space evaluation
                    if shared_mem_list is not None:
                        for shared_mem_op, shared_mem_lv in shared_mem_list:
                            if self.memory_operand_links.contains_mem_op(shared_mem_op):
                                shared_layer_op = self.memory_operand_links.mem_to_layer_op(shared_mem_op)
                                self.effective_mem_utili_shared[shared_layer_op][
                                    shared_mem_lv
                                ] += self.effective_mem_utili_individual[layer_op][mem_lv]
                else:
                    double_buffer_true[layer_op].append(False)

        self.double_buffer_true = double_buffer_true

    def calc_allowed_and_real_data_transfer_cycle_per_data_transfer_link(self):
        """! Construct a 4-way data transfer pattern for each unit mem, calculate
        {allowed_mem_updating_cycle, real_data_trans_cycle, DTL_SS_cycle} per period
        # TODO cleanup
        """
        allowed_mem_update_cycle: dict[LayerOperand, list[MemoryAccesses]] = {}
        real_data_trans_cycle: dict[LayerOperand, list[MemoryAccesses]] = {}

        for layer_op in self.layer.layer_operands:
            allowed_mem_update_cycle[layer_op] = []
            real_data_trans_cycle[layer_op] = []
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            for mem_lv in range(self.mapping_int.mem_level[layer_op]):
                #  wr_in_by_low & rd_out_to_low
                if self.double_buffer_true[layer_op][mem_lv]:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period.rd_out_to_low
                else:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].inst_data_trans_window.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].inst_data_trans_window.rd_out_to_low

                # #  wr_in_by_high & rd_out_to_high
                if self.double_buffer_true[layer_op][mem_lv + 1]:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period.rd_out_to_high
                else:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].inst_data_trans_window.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].inst_data_trans_window.rd_out_to_high

                # All
                updating_window = MemoryAccesses(
                    rd_out_to_low_allowed,
                    wr_in_by_low_allowed,
                    rd_out_to_high_allowed,
                    wr_in_by_high_allowed,
                )
                allowed_mem_update_cycle[layer_op].append(updating_window)

                # wr_in_by_low
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][mem_lv].data_precision.wr_in_by_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_low
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                #  rd_out_to_low
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][mem_lv].data_precision.rd_out_to_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_low
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                #  rd_out_to_high
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][mem_lv].data_precision.rd_out_to_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_high
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                #  wr_in_by_high
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][mem_lv].data_precision.wr_in_by_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_high
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                # All
                real_data_trans = MemoryAccesses(
                    rd_out_to_low_real,
                    wr_in_by_low_real,
                    rd_out_to_high_real,
                    wr_in_by_high_real,
                )
                real_data_trans_cycle[layer_op].append(real_data_trans)

        self.allowed_mem_update_cycle = allowed_mem_update_cycle
        self.real_data_trans_cycle = real_data_trans_cycle

    def combine_data_transfer_rate_per_physical_port(self) -> None:
        """! Consider memory sharing and port sharing, combine the data transfer activity
        Step 1: collect port activity per memory instance per physical memory port
        Step 2: calculate SS combine and MUW union parameters per physical memory port
        """
        # Step 1: collect port activity per memory instance per physical memory port
        port_activity_collect: list[dict[str, list[PortActivity]]] = []
        for mem_instance in self.mem_level_list:
            port_activity_single: dict[str, list[PortActivity]] = {}
            port_list = mem_instance.port_list
            for port in port_list:
                port_activity_single[str(port)] = []
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    if self.memory_operand_links.contains_mem_op(mem_op):
                        layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
                        period_count = self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count.get_single_dir_data(mov_dir)
                        if period_count == 0:
                            # skip the inactive data movement activities because they won't impact SS
                            continue
                        period = self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period.get_single_dir_data(mov_dir)
                        real_cycle = self.real_data_trans_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
                        allowed_cycle = self.allowed_mem_update_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
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

        # Step 2: calculate SS combine and MUW union parameters per physical memory port
        stall_slack_comb_collect: list[dict[str, int | float]] = [
            {port: 0 for port in mem_ports} for mem_ports in port_activity_collect
        ]
        stall_slack_comb_list: list[int | float] = [0]
        # intermediate parameters saved for debugging purpose
        mem_updating_window_union_collect: list[dict[str, int | float]] = [{} for _ in port_activity_collect]

        for idx, mem_ports in enumerate(port_activity_collect):
            for port_name, port_activity in mem_ports.items():
                if len(port_activity) == 1:
                    mem_updating_window_union_collect[idx][port_name] = port_activity[0].allowed_cycle
                    stall_slack_comb_collect[idx][port_name] = port_activity[0].stall_or_slack
                    stall_slack_comb_list.append(port_activity[0].stall_or_slack)
                elif len(port_activity) != 0:
                    mem_updating_window_union_collect[idx][port_name] = self.__calc_mem_updating_window_union(
                        port_activity
                    )
                    stall_slack_positive_sum = 0
                    stall_slack_negative_sum = 0
                    mem_updating_window_sum = 0
                    for port_d in port_activity:
                        if port_d.stall_or_slack > 0:
                            stall_slack_positive_sum += port_d.stall_or_slack
                        else:
                            stall_slack_negative_sum += port_d.stall_or_slack
                        mem_updating_window_sum += port_d.mem_updating_window
                    stall_slack_comb = stall_slack_positive_sum + max(
                        0,
                        stall_slack_negative_sum
                        + mem_updating_window_sum
                        - mem_updating_window_union_collect[idx][port_name],
                    )
                    stall_slack_comb_collect[idx][port_name] = stall_slack_comb
                    stall_slack_comb_list.append(stall_slack_comb)

        self.mem_updating_window_union_collect = mem_updating_window_union_collect
        self.stall_slack_comb_collect = stall_slack_comb_collect
        # Assuming all the memory ports can work in parallel
        self.stall_slack_comb = max(stall_slack_comb_list)

    def calc_loading_single_port_period_count_1(
        self,
        port: MemoryPort,
        mem_op_level_direction_combs: list[tuple[MemoryOperand, int, DataDirection]],
        total_req_bw_aver_computation: float,
    ) -> list[PortBeginOrEndActivity]:
        """! Calculate the loading activities for the memory port movement directions with period count of 1"""
        # Get the minimal and total amount of cycles for each operand in mem_op_level_direction_combs
        # This is the amount of data required for a single period of the lowest memory level
        # This is entry 1 in the mapping object as entry represents the operational array
        min_required_cycles: list[float] = []
        total_required_cycles: list[float] = []
        for mem_op, mem_lv, mov_dir in mem_op_level_direction_combs:
            layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
            minimal_bits = self.mapping_int.data_bit_per_level_unrolled[layer_op][1]
            minimal_cycles_op = ceil(minimal_bits / port.bw)
            min_required_cycles.append(minimal_cycles_op)
            # self.real_data_trans_cycle takes into account that the MemoryLevel might be unrolled
            # This increases the 'effective bandwidth' for the WR_IN_BY_HIGH direction
            # and thus reduces the total required cycles for loading
            total_cycles_op = self.real_data_trans_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
            total_required_cycles.append(total_cycles_op)

        # Get the total amount of cycles of computation (corresponds to latency_total0)
        total_computation_cycles = self.temporal_mapping.total_cycle + self.stall_slack_comb
        # Get the total amount of cycles that we can use for loading during computation
        # TODO: make this more accurate by subtracting the final iteration cycles
        cycles_surplus = max(0, ((port.bw - total_req_bw_aver_computation) / port.bw) * total_computation_cycles)
        # Reduce the loading cycles during loading with the surplus
        reduced_loading_cycles = self.reduce_balanced(total_required_cycles, min_required_cycles, cycles_surplus)
        port_activities: list[PortBeginOrEndActivity] = []
        for (mem_op, mem_lv, mov_dir), cycles in zip(mem_op_level_direction_combs, reduced_loading_cycles):
            layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
            data_loaded = cycles * port.bw
            port_activity = PortBeginOrEndActivity(
                cycles,
                data_loaded,
                port.bw,
                layer_op,
                mem_lv,
                mov_dir,
            )
            port_activities.append(port_activity)
            # Update class variable
            class_var_to_update = (
                self.data_loading_cc_per_op
                if mem_op in [Constants.MEM_OP_1, Constants.MEM_OP_2]
                else self.data_offloading_cc_per_op
            )
            str_identifier = f"{layer_op.name}{mem_lv}_{mov_dir}"
            class_var_to_update[layer_op][str_identifier] = (
                cycles,
                port.port_is_shared_by_two_input_operands,
            )
        # Save the amount of cycles and average bandwidth that were borrowed from the computation phase
        borrowed_cycles_from_computation = sum(total_required_cycles) - sum(reduced_loading_cycles)
        self.total_loading_cycles_during_computation[port] = borrowed_cycles_from_computation
        self.total_loading_bw_during_computation[port] = (
            borrowed_cycles_from_computation / total_computation_cycles
        ) * port.bw
        # Create the PortBeginOrEndActivity objects with the reduced cycles and data
        return port_activities

    def calc_loading_single_port_period_count_greater_than_1(
        self, port: MemoryPort, mem_op_level_direction_combs: list[tuple[MemoryOperand, int, DataDirection]]
    ) -> list[PortBeginOrEndActivity]:
        """! Calculate the loading and offloading activities with mem port movement directions period > 1."""
        port_activities: list[PortBeginOrEndActivity] = []
        for mem_op, mem_lv, mov_dir in mem_op_level_direction_combs:
            layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
            unit_mem_data_movement = self.mapping_int.unit_mem_data_movement[layer_op][mem_lv]
            period_count = unit_mem_data_movement.data_trans_period_count.get_single_dir_data(mov_dir)

            # skip for the inactive data movement
            if period_count == 0:
                continue

            real_cycle = self.real_data_trans_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
            data_in_charge = unit_mem_data_movement.data_trans_amount_per_period.get_single_dir_data(
                mov_dir
            ) * unit_mem_data_movement.data_precision.get_single_dir_data(mov_dir)

            mem_bw = port.bw

            port_activity = PortBeginOrEndActivity(
                real_cycle,
                data_in_charge,
                mem_bw,
                layer_op,
                mem_lv,
                mov_dir,
            )
            # Update class variable
            class_var_to_update = (
                self.data_loading_cc_per_op
                if mem_op in [Constants.MEM_OP_1, Constants.MEM_OP_2]
                else self.data_offloading_cc_per_op
            )
            str_identifier = f"{layer_op.name}{mem_lv}_{mov_dir}"
            class_var_to_update[layer_op][str_identifier] = (
                real_cycle,
                port.port_is_shared_by_two_input_operands,
            )
            port_activities.append(port_activity)
        return port_activities

    def calc_loading_single_port(self, port: MemoryPort):
        """! Calculate the loading and offloading activities for a single memory port"""
        # Initialize the data loading list
        data_loading: list[PortBeginOrEndActivity] = []
        # Abbreviate the unit memory data movement from the mapping
        unit_mem_data_movement = self.mapping_int.unit_mem_data_movement
        # Get the output operand
        output_mem_op = self.memory_operand_links.layer_to_mem_op(self.layer.output_operand)

        # Get the operand - movement direction combinations with period count of 1
        # A period count of 1 indicates that all data of that operand is loaded before computation
        combs_period_count_1: list[tuple[MemoryOperand, int, DataDirection]] = []
        combs_period_count_greater_than_1: list[tuple[MemoryOperand, int, DataDirection]] = []
        for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
            # don't consider partial sum flowing in the final data off-loading stage
            if mem_op == output_mem_op and (
                mov_dir == DataDirection.RD_OUT_TO_LOW or mov_dir == DataDirection.WR_IN_BY_HIGH
            ):
                continue
            layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
            period_count = unit_mem_data_movement[layer_op][mem_lv].data_trans_period_count.get_single_dir_data(mov_dir)
            if period_count == 0:
                continue
            elif period_count == 1:
                combs_period_count_1.append((mem_op, mem_lv, mov_dir))
            else:
                combs_period_count_greater_than_1.append((mem_op, mem_lv, mov_dir))
        # Obtain total required average bandwidth during computation
        # This is the sum of all bandwidths for the operand - movement directions that have > 1 period
        total_req_bw_aver_computation = 0
        for mem_op, mem_lv, mov_dir in combs_period_count_greater_than_1:
            layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
            req_bw_aver = unit_mem_data_movement[layer_op][mem_lv].req_mem_bw_aver.get_single_dir_data(mov_dir)
            total_req_bw_aver_computation += req_bw_aver

        data_loading += self.calc_loading_single_port_period_count_1(
            port, combs_period_count_1, total_req_bw_aver_computation
        )

        data_loading += self.calc_loading_single_port_period_count_greater_than_1(
            port, combs_period_count_greater_than_1
        )

        return data_loading

    def calc_onloading_combined(self):
        # Combine ports' initial data-loading activities to get the data loading cycle amount
        data_loading_cc_pair_combined_per_op: dict[LayerOperand, list[int | float]] = {
            op: [] for op in self.layer.input_operands
        }
        data_loading_individual_part: dict[LayerOperand, int | float] = {op: 0 for op in self.layer.input_operands}
        data_loading_half_shared_part: dict[LayerOperand, int | float] = {op: 0 for op in self.layer.input_operands}
        data_loading_shared_part: dict[LayerOperand, int | float] = {op: 0 for op in self.layer.input_operands}

        for layer_op in self.layer.input_operands:
            for mem_lv in range(self.active_mem_level[layer_op] - 1):
                elem1 = self.data_loading_cc_per_op[layer_op][
                    layer_op.name + str(mem_lv) + "_" + str(DataDirection.WR_IN_BY_HIGH)
                ]
                elem2 = self.data_loading_cc_per_op[layer_op][
                    layer_op.name + str(mem_lv + 1) + "_" + str(DataDirection.RD_OUT_TO_LOW)
                ]
                completely_shared = elem1[1] and elem2[1]
                completely_separate = not elem1[1] and not elem2[1]
                longest_loading_cc = max(elem1[0], elem2[0])
                # for the ports that serve the same data movement purpose, take the longest data loading cycle
                data_loading_cc_pair_combined = longest_loading_cc
                data_loading_cc_pair_combined_per_op[layer_op].append(data_loading_cc_pair_combined)
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
            data_onloading_cycle = data_loading_individual_part[self.layer.input_operands[0]]
        else:
            op1 = self.layer.input_operands[0]
            op2 = self.layer.input_operands[1]
            possible1 = data_loading_shared_part[op1] + max(
                data_loading_shared_part[op2] + data_loading_half_shared_part[op2] + data_loading_individual_part[op2],
                data_loading_half_shared_part[op1] + data_loading_individual_part[op1],
            )
            possible2 = data_loading_shared_part[op2] + max(
                data_loading_shared_part[op1] + data_loading_half_shared_part[op1] + data_loading_individual_part[op1],
                data_loading_half_shared_part[op2] + data_loading_individual_part[op2],
            )
            data_onloading_cycle = min(possible1, possible2)

        self.data_loading_cc_pair_combined_per_op = data_loading_cc_pair_combined_per_op
        self.data_loading_individual_part = data_loading_individual_part
        self.data_loading_half_shared_part = data_loading_half_shared_part
        self.data_loading_shared_part = data_loading_shared_part
        return data_onloading_cycle

    def calc_offloading_combined(self):
        # Combine ports' final data-offloading activities to get the data offloading cycle amount
        # TODO Only considered the worst case for now
        #  (assumed that all the ports are working in series during the final data off-loading phase)
        data_offloading_cc_pair_combined: list[int | float] = []
        layer_op = self.layer.output_operand
        for mem_lv in range(self.active_mem_level[layer_op] - 1):
            elem1 = self.data_offloading_cc_per_op[layer_op][
                layer_op.name + str(mem_lv) + "_" + str(DataDirection.RD_OUT_TO_HIGH)
            ][0]
            elem2 = self.data_offloading_cc_per_op[layer_op][
                layer_op.name + str(mem_lv + 1) + "_" + str(DataDirection.WR_IN_BY_LOW)
            ][0]
            longest_offloading_cc = max(elem1, elem2)
            # for the ports that serve the same data movement purpose, take the longest data loading cycle
            data_offloading_cc_pair_combined.append(longest_offloading_cc)
        data_offloading_cycle = sum(data_offloading_cc_pair_combined)
        self.data_offloading_cc_pair_combined = data_offloading_cc_pair_combined
        return data_offloading_cycle

    def calc_borrowed_loading_cycles_and_bandwidth(self) -> tuple[float, float]:
        """! Calculate the amount of cycles and bandwidth that are borrowed from the computation phase."""
        borrowed_cycles_ports = self.total_loading_cycles_during_computation
        borrowed_bw_ports = self.total_loading_bw_during_computation
        return max(borrowed_cycles_ports.values(), default=0), max(borrowed_bw_ports.values(), default=0)

    def calc_data_loading_latency(self):
        """! Calculate the number of data onloading/offloading cycles.
        This function first gathers the correct cycles per port, then combines them.
        """
        self.data_loading_per_mem_inst: list[dict[MemoryPort, list[PortBeginOrEndActivity]]] = []
        self.data_offloading_per_mem_inst: list[dict[MemoryPort, list[PortBeginOrEndActivity]]] = []
        self.data_loading_cc_per_op: dict[LayerOperand, dict[str, tuple[int | float, bool]]] = {
            op: {} for op in self.layer.input_operands
        }
        self.data_offloading_cc_per_op: dict[LayerOperand, dict[str, tuple[int | float, bool]]] = {
            self.layer.output_operand: {}
        }
        self.total_loading_cycles_during_computation: dict[MemoryPort, int | float] = {}
        self.total_loading_bw_during_computation: dict[MemoryPort, int | float] = {}

        for mem_level in self.mem_level_list:
            data_loading_per_port = {port: self.calc_loading_single_port(port) for port in mem_level.port_list}
            self.data_loading_per_mem_inst.append(data_loading_per_port)

        # Combine the onloading of all ports to get the data onloading cycle amount
        data_onloading_cycle = self.calc_onloading_combined()
        self.data_onloading_cycle = data_onloading_cycle

        # Combine the offloading of all ports to get the data offloading cycle amount
        data_offloading_cycle = self.calc_offloading_combined()
        self.data_offloading_cycle = data_offloading_cycle

        # Combine the borrowed cycles and bandwidth for loading and offloading during computation
        borrowed_cycles, borrowed_bandwidth = self.calc_borrowed_loading_cycles_and_bandwidth()
        self.loading_offloading_cycles_borrowed_from_computation = borrowed_cycles
        self.loading_offloading_bandwidth_borrowed_from_computation = borrowed_bandwidth

    def calc_overall_latency(self, cycles_per_mac: float = 1) -> None:
        """! This function integrates the previous calculated SScomb, data loading and off-loading cycle to get the
        overall latency
        @param cycles_per_mac: cycle counts per mac operand (>1 for bit-serial computation)
        """
        # the ideal cycle count assuming the MAC array is 100% utilized
        ideal_cycle = int(
            ceil(self.layer.total_mac_count / self.accelerator.operational_array.total_unit_count) * cycles_per_mac
        )

        # the ideal temporal cycle count given the spatial mapping (the spatial mapping can be non-ideal)
        ideal_temporal_cycle = self.mapping_int.temporal_mapping.total_cycle * cycles_per_mac
        mac_spatial_utilization = ideal_cycle / ideal_temporal_cycle

        # Total latency without the initial data loading and the final data off-loading
        latency_total0 = ideal_temporal_cycle + self.stall_slack_comb
        mac_utilization0 = ideal_cycle / latency_total0

        # Total latency with the initial data loading, but without the final data off-loading
        latency_total1 = ideal_temporal_cycle + self.stall_slack_comb + self.data_onloading_cycle
        mac_utilization1 = ideal_cycle / latency_total1

        # Total latency with both the initial data loading and the final data off-loading
        latency_total2 = (
            ideal_temporal_cycle + self.stall_slack_comb + self.data_onloading_cycle + self.data_offloading_cycle
        )
        mac_utilization2 = ideal_cycle / latency_total2

        self.ideal_cycle = ideal_cycle
        self.ideal_temporal_cycle = ideal_temporal_cycle
        self.mac_spatial_utilization = mac_spatial_utilization
        self.latency_total0 = latency_total0
        self.latency_total1 = latency_total1
        self.latency_total2 = latency_total2
        self.mac_utilization0 = mac_utilization0
        self.mac_utilization1 = mac_utilization1
        self.mac_utilization2 = mac_utilization2

    def get_total_inst_bandwidth(self, memory_level: MemoryLevel) -> FourWayDataMoving[int]:
        """Given a cost model evaluation and a memory level, compute the memory's total instantaneous bandwidth
        required throughout the execution of the layer that corresponds to this CME. Raises an assertion error
        if the given memory level is not included in this CME's memory hierarchy.
        NOTE: this function is used in Stream
        """
        # Check that the given MemoryInstance is part of the memory hierarchy of this CME
        assert (
            memory_level in self.mem_level_list
        ), f"Memory level {memory_level} is not part of the memory hierarchy of this cost model evaluation."
        # Obtain the required instantaneous bandwidth to/from the memory for its operands
        total_inst_bw = FourWayDataMoving(0, 0, 0, 0)
        for mem_op in memory_level.operands:
            layer_op = self.layer.memory_operand_links.mem_to_layer_op(mem_op)
            req_elem_per_cc_4way = self.mapping.unit_mem_data_movement[layer_op][-1].req_mem_bw_inst
            precision_4way = self.mapping.unit_mem_data_movement[layer_op][-1].data_precision
            inst_bw_4way = FourWayDataMoving(
                req_elem_per_cc_4way.rd_out_to_low * precision_4way.rd_out_to_low,
                req_elem_per_cc_4way.wr_in_by_low * precision_4way.wr_in_by_low,
                req_elem_per_cc_4way.rd_out_to_high * precision_4way.rd_out_to_high,
                req_elem_per_cc_4way.wr_in_by_high * precision_4way.wr_in_by_high,
            )
            total_inst_bw += inst_bw_4way
        return total_inst_bw

    def __str__(self):
        return f"CostModelEvaluation({self.layer}, {self.accelerator})"

    def __repr__(self):
        return str(self)

    @staticmethod
    def reduce_balanced(c_list: list[float], m_list: list[float], s: float) -> list[float]:
        """Balance c_list towards minimums m_list with a total maximum reduction of s."""
        c_result = c_list.copy()
        if s <= 0 or not c_list:
            return c_list

        # Sort indices by c_list values in descending order
        indices = sorted(range(len(c_list)), key=lambda i: c_list[i], reverse=True)
        c_sorted = [c_list[i] for i in indices]
        m_sorted = [m_list[i] for i in indices]

        n = len(c_sorted)
        for i in range(n):
            max_reducible = (c_sorted[i] - m_sorted[i]) * (i + 1)
            if s >= max_reducible:
                reduction = c_sorted[i] - m_sorted[i]
                for j in range(i + 1):
                    c_sorted[j] -= reduction
                s -= max_reducible
            else:
                reduction = s / (i + 1)
                for j in range(i + 1):
                    c_sorted[j] -= reduction
                s = 0
                break

        # Return the result in the original order
        for i, idx in enumerate(indices):
            c_result[idx] = c_sorted[i]

        return c_result
