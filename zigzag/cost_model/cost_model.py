from abc import ABCMeta, abstractmethod
import logging
from math import ceil
import numpy as np
from zigzag.cost_model.port_activity import PortActivity, PortBeginOrEndActivity
from zigzag.datatypes import Constants, LayerOperand, MemoryOperand
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.MemoryInstance import MemoryInstance
from zigzag.mapping.Mapping import Mapping
from zigzag.mapping.data_movement import DataDirection, FourWayDataMoving
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping
from zigzag.workload.layer_node import LayerNode
from zigzag.utils import json_repr_handler, pickle_deepcopy

logger = logging.getLogger(__name__)


class CostModelEvaluationABC(metaclass=ABCMeta):
    """! Superclass for CostModelEvaluation and CumulativeCME"""

    @abstractmethod
    def __init__(self) -> None:
        # Attributes that all subclasses should define
        self.mem_energy_breakdown: dict[LayerOperand, list[float]]
        self.mem_energy_breakdown_further: dict[LayerOperand, list[FourWayDataMoving]]
        self.memory_word_access: dict[LayerOperand, list[FourWayDataMoving]]
        self.MAC_energy: float
        self.mem_energy: float
        self.energy_total: float
        self.data_loading_cycle: float
        self.data_offloading_cycle: float
        self.ideal_cycle: float
        self.ideal_temporal_cycle: float
        self.latency_total0: float
        self.latency_total1: float
        self.latency_total2: float
        self.MAC_spatial_utilization: float
        self.MAC_utilization0: float
        self.MAC_utilization1: float
        self.MAC_utilization2: float

        self.accelerator: Accelerator | None

    def __add__(self, other: "CostModelEvaluationABC") -> "CumulativeCME":
        result = CumulativeCME()

        # Energy
        result.MAC_energy = self.MAC_energy + other.MAC_energy
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
                list_temp2: list[FourWayDataMoving] = []
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
                list_temp3: list[FourWayDataMoving] = []
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
        result.data_loading_cycle = self.data_loading_cycle + other.data_loading_cycle
        result.data_offloading_cycle = self.data_offloading_cycle + other.data_offloading_cycle
        result.ideal_cycle = self.ideal_cycle + other.ideal_cycle
        result.ideal_temporal_cycle = self.ideal_temporal_cycle + other.ideal_temporal_cycle
        result.latency_total0 = self.latency_total0 + other.latency_total0
        result.latency_total1 = self.latency_total1 + other.latency_total1
        result.latency_total2 = self.latency_total2 + other.latency_total2

        # MAC utilization
        result.MAC_spatial_utilization = result.ideal_cycle / result.ideal_temporal_cycle
        result.MAC_utilization0 = result.ideal_cycle / result.latency_total0
        result.MAC_utilization1 = result.ideal_cycle / result.latency_total1
        result.MAC_utilization2 = result.ideal_cycle / result.latency_total2

        for cme in (self, other):
            if isinstance(cme, CostModelEvaluation):
                result.layer_ids.append(cme.layer.id)
                result.core_ids.append(cme.core_id)
            elif isinstance(cme, CumulativeCME):
                result.layer_ids += cme.layer_ids
                result.core_ids += cme.core_ids

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

    def __mul__(self, number: float):
        result: "CostModelEvaluationABC" = pickle_deepcopy(self)

        # Energy
        result.MAC_energy *= number
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
        result.data_loading_cycle *= number
        result.data_offloading_cycle *= number
        result.ideal_cycle *= number
        result.ideal_temporal_cycle *= number
        result.latency_total0 *= number
        result.latency_total1 *= number
        result.latency_total2 *= number

        # MAC utilization
        result.MAC_spatial_utilization = result.ideal_cycle / result.ideal_temporal_cycle
        result.MAC_utilization0 = result.ideal_cycle / result.latency_total0
        result.MAC_utilization1 = result.ideal_cycle / result.latency_total1
        result.MAC_utilization2 = result.ideal_cycle / result.latency_total2

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
                        "operational_energy": self.MAC_energy,
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
                            "ideal": self.MAC_spatial_utilization,
                            "stalls": self.MAC_utilization0,
                            "stalls_onloading": self.MAC_utilization1,
                            "stalls_onloading_offloading": self.MAC_utilization2,
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
        self.mem_energy_breakdown_further: dict[LayerOperand, list[FourWayDataMoving]] = dict()
        self.memory_word_access: dict[LayerOperand, list[FourWayDataMoving]] = dict()

        self.layer_ids: list[int] = []
        self.core_ids: list[int] = []

        self.MAC_energy: float = 0.0
        self.mem_energy: float = 0.0
        self.energy_total: float = 0.0
        self.data_loading_cycle: float = 0.0
        self.data_offloading_cycle: float = 0.0
        self.ideal_cycle: float = 0.0
        self.ideal_temporal_cycle: float = 0.0
        self.latency_total0: float = 0.0
        self.latency_total1: float = 0.0
        self.latency_total2: float = 0.0

        self.accelerator = None


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
        self.accelerator: Accelerator = accelerator  # type: ignore
        self.layer: LayerNode = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = spatial_mapping_int  # the original spatial mapping without decimal
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = access_same_data_considered_as_no_access

        self.core_id = layer.core_allocation[0]
        core = accelerator.get_core(self.core_id)
        self.mem_level_list = core.memory_hierarchy.mem_level_list
        self.mem_hierarchy_dict = core.mem_hierarchy_dict
        self.mem_size_dict = core.mem_size_dict
        self.mem_r_bw_dict, self.mem_w_bw_dict = core.get_memory_bw_dict()
        self.mem_r_bw_min_dict, self.mem_w_bw_min_dict = core.get_memory_bw_min_dict()
        self.mem_sharing_list = core.mem_sharing_list
        self.memory_operand_links = layer.memory_operand_links

        self.cumulative_layer_ids: list[int] = []  # In case the CME results from adding other CMEs together
        self.cumulative_core_ids: list[int] = []

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

    def __get_shared_mem_list(
        self,
        mem_op: MemoryOperand,
        mem_lv: int,
        memory_sharing_list: list[dict[MemoryOperand, int]],
    ) -> list[tuple[MemoryOperand, int]] | None:
        """! Given a certain operand's storage level (for example (A,1): operand A's 1st memory level),
        return a list of the rest operand's storage levels that share physical memory with the former one (A1)
        """
        for mem_share_group in memory_sharing_list:
            mem_share_grp = list(mem_share_group.items())
            mem_target = (mem_op, mem_lv)
            if mem_target in mem_share_grp:
                return mem_share_grp

    def __calc_MUW_union(self, port_duty_list: list[PortActivity]) -> int | float:
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

        input_dict: dict[str, dict[str, int | float]] = {}
        # As long as one of the port duty can make use of the whole computation time, the MUW union is
        # set to the whole computation time
        for port_duty in port_duty_list:
            if port_duty.period == port_duty.allowed_cycle:
                return port_duty.period * port_duty.period_count
            key = str(port_duty.served_op_lv_dir)
            input_dict[key] = {
                "P": port_duty.period,
                "A": port_duty.allowed_cycle,
                "PC": port_duty.period_count,
            }

        max_period = 0
        max_period_operand: str = ""
        for op, values in input_dict.items():
            if values["P"] > max_period:
                max_period = values["P"]
                max_period_operand = op

        indicators: np.ndarray = np.zeros((len(input_dict), max_period), dtype=np.int8)
        for i, op in enumerate(input_dict):
            # reshape to period of this operand
            indicators_reshape: np.ndarray = indicators.reshape((len(input_dict), -1, input_dict[op]["P"]))
            # fill in first few time units as used
            indicators_reshape[i, :, : input_dict[op]["A"]] = 1

        union = max_period - (~indicators.any(0)).sum(dtype=np.uint64)

        # take sum across operands => how many operand need memory for every time unit
        # Subtract 1 => number of stalls
        # Clip by 0 (-1 is not -1 stall)
        # Sum across time units (only remaining axis)
        # stall = (indicators.sum(0, dtype=np.int8) - 1).clip(min=0).sum()

        # Multiply with number of periods of largest period (as it was normalized to largest period)
        return union * input_dict[max_period_operand]["PC"]

    def calc_memory_utilization(self) -> None:
        """! Calculate occupancy for each physical memory based on the mapping."""
        # mem_utili_individual: the memory utilization of each operand individually.
        # mem_utili_shared: the memory utilization taking operand memory sharing into consideration.
        mem_utili_individual: dict[LayerOperand, list[float]] = {}
        effective_mem_utili_individual: dict[LayerOperand, list[float]] = {}
        for layer_op in self.layer.layer_operands:
            mem_utili_individual[layer_op] = []
            effective_mem_utili_individual[layer_op] = []
            for mem_lv in range(self.active_mem_level[layer_op]):
                mem_utilization = (
                    self.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1]
                    / self.mem_size_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
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
                    / self.mem_size_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                )
                effective_mem_utili_individual[layer_op].append(effective_mem_utilization)

        mem_utili_shared: dict[LayerOperand, list[float]] = pickle_deepcopy(mem_utili_individual)
        effective_mem_utili_shared: dict[LayerOperand, list[float]] = pickle_deepcopy(effective_mem_utili_individual)
        for mem_share_dict in self.mem_sharing_list:
            mem_utilization = 0
            effective_mem_utilization = 0
            for mem_op, mem_lv in mem_share_dict.items():
                # mem to layer op might not contain this mem op (e.g. pooling layer)
                if self.memory_operand_links.contains_mem_op(mem_op):
                    layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
                    mem_utilization += mem_utili_individual[layer_op][mem_lv]
                    effective_mem_utilization += effective_mem_utili_individual[layer_op][mem_lv]
            assert mem_utilization <= 1, (
                f"Memory shared by {mem_share_dict} (memory operand, memory level) has shared utilization of "
                f"{mem_utilization}, which is > 1 "
                f"(memory level starts from 0)."
            )
            for mem_op, mem_lv in mem_share_dict.items():
                if self.memory_operand_links.contains_mem_op(mem_op):
                    layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
                    mem_utili_shared[layer_op][mem_lv] = mem_utilization
                    effective_mem_utili_shared[layer_op][mem_lv] = effective_mem_utilization

        self.mem_utili_individual = mem_utili_individual
        self.mem_utili_shared = mem_utili_shared
        self.effective_mem_utili_individual = effective_mem_utili_individual
        self.effective_mem_utili_shared = effective_mem_utili_shared

    def calc_memory_word_access(self) -> None:
        """! Calculates the memory word access based on unit memory's data element move count and the physical
        memory bw."""
        memory_word_access: dict[LayerOperand, list[FourWayDataMoving]] = {}
        for layer_op in self.layer.layer_operands:
            memory_word_access[layer_op] = []
            for mem_lv in range(self.mapping.mem_level[layer_op]):
                # wr_in_by_low
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][mem_lv].data_precision.wr_in_by_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    wr_in_by_low: int = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_low
                    max_bw = self.mem_w_bw_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    min_bw = self.mem_w_bw_min_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    wr_in_by_low = int(
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                # rd_out_to_low
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][mem_lv].data_precision.rd_out_to_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    rd_out_to_low: int = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_low
                    max_bw = self.mem_r_bw_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    min_bw = self.mem_r_bw_min_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    rd_out_to_low = int(
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                # rd_out_to_high
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_high
                if data_elem_move_per_period == 0:
                    rd_out_to_high: int = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][mem_lv].data_precision.rd_out_to_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_high
                    max_bw = self.mem_r_bw_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    min_bw = self.mem_r_bw_min_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    rd_out_to_high = int(
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                # wr_in_by_high
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_high
                if data_elem_move_per_period == 0:
                    wr_in_by_high: int = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][mem_lv].data_precision.wr_in_by_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_high
                    max_bw = self.mem_w_bw_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    min_bw = self.mem_w_bw_min_dict[self.memory_operand_links.layer_to_mem_op(layer_op)][mem_lv]
                    wr_in_by_high = int(
                        ceil((data_elem_move_per_period * data_precision) / min_bw)
                        * (min_bw / max_bw)
                        * total_period_count
                        * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                # All
                memory_word_access_single = FourWayDataMoving(
                    rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high
                )
                memory_word_access[layer_op].append(memory_word_access_single)

        self.memory_word_access = memory_word_access

    def calc_energy(self) -> None:
        """! Calculates the energy cost of this cost model evaluation by calculating the memory reading/writing
        energy."""
        # TODO: Interconnection energy
        self.calc_MAC_energy_cost()
        self.calc_memory_energy_cost()

    def calc_MAC_energy_cost(self) -> None:
        """! Calculate the dynamic MAC energy"""
        core = self.accelerator.get_core(self.core_id)
        single_MAC_energy = core.operational_array.unit.cost
        self.MAC_energy = single_MAC_energy * self.layer.total_MAC_count

    def calc_memory_energy_cost(self):
        """! Computes the memories reading/writing energy by converting the access patterns in self.mapping to
        energy breakdown using the memory hierarchy of the core on which the layer is mapped.

        The energy breakdown is saved in self.mem_energy_breakdown.

        The energy total consumption is saved in self.energy_total.
        """
        core = self.accelerator.get_core(self.core_id)
        mem_hierarchy = core.memory_hierarchy

        mem_energy_breakdown: dict[LayerOperand, list[float]] = {}
        mem_energy_breakdown_further: dict[LayerOperand, list[FourWayDataMoving]] = {}
        energy_total = 0

        # Retrieve the memory levels in the hierarchy for this memory operand
        for layer_op, mem_access_list_per_op in self.memory_word_access.items():
            mem_op = self.memory_operand_links.layer_to_mem_op(layer_op)
            memory_levels = mem_hierarchy.get_memory_levels(mem_op=mem_op)

            breakdown: list[float] = []  # Stores the energy breakdown of a single layer operand (W, I, ...)
            breakdown_further: list[FourWayDataMoving] = []
            for access_count, memory_level in zip(mem_access_list_per_op, memory_levels):
                energy_cost_per_read_out = memory_level.read_energy
                energy_cost_per_write_in = memory_level.write_energy
                read_out_energy_to_above: float = access_count.get_total_read_outs_to_above(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_above: float = access_count.get_total_write_ins_from_above(
                    scaling=energy_cost_per_write_in
                )
                read_out_energy_to_below: float = access_count.get_total_read_outs_to_below(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_below: float = access_count.get_total_write_ins_from_below(
                    scaling=energy_cost_per_write_in
                )
                total_read_out_energy = read_out_energy_to_above + read_out_energy_to_below
                total_write_in_energy = write_in_energy_from_above + write_in_energy_from_below
                total_energy_cost_memory = total_read_out_energy + total_write_in_energy
                # Here the breakdown only saves the total energy cost per memory level
                breakdown.append(total_energy_cost_memory)
                breakdown_further.append(
                    FourWayDataMoving(
                        read_out_energy_to_below,
                        write_in_energy_from_below,
                        read_out_energy_to_above,
                        write_in_energy_from_above,
                    )
                )
                energy_total += total_energy_cost_memory
            mem_energy_breakdown[layer_op] = breakdown
            mem_energy_breakdown_further[layer_op] = breakdown_further
        self.mem_energy_breakdown: dict[LayerOperand, list[float]] = mem_energy_breakdown
        self.mem_energy_breakdown_further: dict[LayerOperand, list[FourWayDataMoving]] = mem_energy_breakdown_further
        self.mem_energy = energy_total
        self.energy_total: float = self.mem_energy + self.MAC_energy
        logger.debug(f"Ran {self}. Total energy = {self.energy_total}")

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
        self.__calc_allowed_and_real_data_transfer_cycle_per_DTL()
        self.__combine_data_transfer_rate_per_physical_port()
        self.calc_data_loading_offloading_latency()
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
                    shared_mem_list = self.__get_shared_mem_list(mem_op, mem_lv, self.mem_sharing_list)
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

    def __calc_allowed_and_real_data_transfer_cycle_per_DTL(self):
        """! Construct a 4-way data transfer pattern for each unit mem, calculate
        {allowed_mem_updating_cycle, real_data_trans_cycle, DTL_SS_cycle} per period
        # TODO cleanup
        """
        allowed_mem_updat_cycle: dict[LayerOperand, list[FourWayDataMoving]] = dict()
        real_data_trans_cycle: dict[LayerOperand, list[FourWayDataMoving]] = dict()
        # stall (+) or slack (-) cycle within each period per virtual data transfer link (DTL)
        DTL_SS_cycle = {}

        for layer_op in self.layer.layer_operands:
            allowed_mem_updat_cycle[layer_op] = []
            real_data_trans_cycle[layer_op] = []
            DTL_SS_cycle[layer_op] = []
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
                updating_window = FourWayDataMoving(
                    rd_out_to_low_allowed,
                    wr_in_by_low_allowed,
                    rd_out_to_high_allowed,
                    wr_in_by_high_allowed,
                )
                allowed_mem_updat_cycle[layer_op].append(updating_window)

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
                real_data_trans = FourWayDataMoving(
                    rd_out_to_low_real,
                    wr_in_by_low_real,
                    rd_out_to_high_real,
                    wr_in_by_high_real,
                )
                real_data_trans_cycle[layer_op].append(real_data_trans)

        self.allowed_mem_updat_cycle = allowed_mem_updat_cycle
        self.real_data_trans_cycle = real_data_trans_cycle

    def __combine_data_transfer_rate_per_physical_port(self) -> None:
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
                        allowed_cycle = self.allowed_mem_updat_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
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
        SS_comb_collect: list[dict[str, int | float]] = [
            {port: 0 for port in mem_ports} for mem_ports in port_activity_collect
        ]
        SS_comb_list: list[int | float] = [0]
        # intermediate parameters saved for debugging purpose
        MUW_union_collect: list[dict[str, int | float]] = [{} for _ in port_activity_collect]

        for idx, mem_ports in enumerate(port_activity_collect):
            for port_name, port_activity in mem_ports.items():
                if len(port_activity) == 1:
                    MUW_union_collect[idx][port_name] = port_activity[0].allowed_cycle
                    SS_comb_collect[idx][port_name] = port_activity[0].SS
                    SS_comb_list.append(port_activity[0].SS)
                elif len(port_activity) != 0:
                    MUW_union_collect[idx][port_name] = self.__calc_MUW_union(port_activity)
                    SS_positive_sum = 0
                    SS_negative_sum = 0
                    MUW_sum = 0
                    for port_d in port_activity:
                        if port_d.SS > 0:
                            SS_positive_sum += port_d.SS
                        else:
                            SS_negative_sum += port_d.SS
                        MUW_sum += port_d.MUW
                    SS_comb = SS_positive_sum + max(0, SS_negative_sum + MUW_sum - MUW_union_collect[idx][port_name])
                    SS_comb_collect[idx][port_name] = SS_comb
                    SS_comb_list.append(SS_comb)

        self.MUW_union_collect = MUW_union_collect
        self.SS_comb_collect = SS_comb_collect
        # Assuming all the memory ports can work in parallel
        self.SS_comb = max(SS_comb_list)

    def calc_data_loading_offloading_latency(self):
        """! Calculate the initial/final data loading/off-loading cycle by separating out
        the first-time input operands' / the last-time output operand's data movement
        on corresponding ports.
        """
        # Collect ports' initial data-loading and final data-offloading activities
        data_loading_per_mem_inst: list[dict[str, list[PortBeginOrEndActivity]]] = []
        data_loading_cc_per_op: dict[LayerOperand, dict[str, tuple[int | float, bool]]] = {
            op: {} for op in self.layer.input_operands
        }
        data_offloading_per_mem_inst: list[dict[str, list[PortBeginOrEndActivity]]] = []
        data_offloading_cc_per_op: dict[str, int | float] = {}
        for mem_instance in self.mem_level_list:
            data_loading_single: dict[str, list[PortBeginOrEndActivity]] = {}
            data_offloading_single: dict[str, list[PortBeginOrEndActivity]] = {}
            port_list = mem_instance.port_list
            for port in port_list:
                data_loading_single[str(port)] = []
                data_offloading_single[str(port)] = []
                served_operands = set(
                    s[0] for s in port.served_op_lv_dir if s[0] in [Constants.MEM_OP_1, Constants.MEM_OP_2]
                )
                port_is_shared_by_two_input_operands = len(served_operands) > 1
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    if self.memory_operand_links.contains_mem_op(mem_op):
                        layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
                        period_count = self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count.get_single_dir_data(mov_dir)
                        if period_count == 0:
                            # skip for the inactive data movement
                            continue
                        if mem_op in [Constants.MEM_OP_1, Constants.MEM_OP_2]:
                            real_cycle = self.real_data_trans_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
                            data_in_charge = self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period.get_single_dir_data(
                                mov_dir
                            ) * self.mapping_int.unit_mem_data_movement[
                                layer_op
                            ][
                                mem_lv
                            ].data_precision.get_single_dir_data(
                                mov_dir
                            )
                            if mov_dir == DataDirection.RD_OUT_TO_HIGH or mov_dir == DataDirection.RD_OUT_TO_LOW:
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
                            data_loading_cc_per_op[layer_op][layer_op.name + str(mem_lv) + "_" + str(mov_dir)] = (
                                real_cycle,
                                port_is_shared_by_two_input_operands,
                            )
                        else:
                            if mov_dir == DataDirection.RD_OUT_TO_LOW or mov_dir == DataDirection.WR_IN_BY_HIGH:
                                # don't consider partial sum flowing in the final data off-loading stage
                                continue
                            real_cycle = self.real_data_trans_cycle[layer_op][mem_lv].get_single_dir_data(mov_dir)
                            data_in_charge = self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period.get_single_dir_data(
                                mov_dir
                            ) * self.mapping_int.unit_mem_data_movement[
                                layer_op
                            ][
                                mem_lv
                            ].data_precision.get_single_dir_data(
                                mov_dir
                            )
                            if mov_dir == DataDirection.RD_OUT_TO_HIGH:
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
                            data_offloading_cc_per_op[layer_op.name + str(mem_lv) + "_" + str(mov_dir)] = real_cycle

            data_loading_per_mem_inst.append(data_loading_single)
            data_offloading_per_mem_inst.append(data_offloading_single)
        self.data_loading_per_mem_inst = data_loading_per_mem_inst
        self.data_loading_cc_per_op = data_loading_cc_per_op
        self.data_offloading_per_mem_inst = data_offloading_per_mem_inst
        self.data_offloading_per_op = data_offloading_cc_per_op

        # Combine ports' initial data-loading activities to get the data loading cycle amount
        data_loading_cc_pair_combined_per_op: dict[LayerOperand, list[int | float]] = {
            op: [] for op in self.layer.input_operands
        }
        data_loading_individual_part: dict[LayerOperand, int | float] = {op: 0 for op in self.layer.input_operands}
        data_loading_half_shared_part: dict[LayerOperand, int | float] = {op: 0 for op in self.layer.input_operands}
        data_loading_shared_part: dict[LayerOperand, int | float] = {op: 0 for op in self.layer.input_operands}
        for layer_op in self.layer.input_operands:
            for mem_lv in range(self.active_mem_level[layer_op] - 1):
                elem1 = data_loading_cc_per_op[layer_op][
                    layer_op.name + str(mem_lv) + "_" + str(DataDirection.WR_IN_BY_HIGH)
                ]
                elem2 = data_loading_cc_per_op[layer_op][
                    layer_op.name + str(mem_lv + 1) + "_" + str(DataDirection.RD_OUT_TO_LOW)
                ]
                completely_shared = elem1[1] and elem2[1]
                completely_separate = not (elem1[1]) and not (elem2[1])
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
            data_loading_cycle = data_loading_individual_part[self.layer.input_operands[0]]
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
            data_loading_cycle = min(possible1, possible2)

        self.data_loading_cc_pair_combined_per_op = data_loading_cc_pair_combined_per_op
        self.data_loading_individual_part = data_loading_individual_part
        self.data_loading_half_shared_part = data_loading_half_shared_part
        self.data_loading_shared_part = data_loading_shared_part
        self.data_loading_cycle = data_loading_cycle

        # Combine ports' final data-offloading activities to get the data offloading cycle amount
        # TODO Only considered the worst case for now
        #  (assumed that all the ports are working in series during the final data off-loading phase)
        data_offloading_cc_pair_combined: list[int | float] = []
        layer_op = self.layer.output_operand
        for mem_lv in range(self.active_mem_level[layer_op] - 1):
            elem1 = data_offloading_cc_per_op[layer_op.name + str(mem_lv) + "_" + str(DataDirection.RD_OUT_TO_HIGH)]
            elem2 = data_offloading_cc_per_op[layer_op.name + str(mem_lv + 1) + "_" + str(DataDirection.WR_IN_BY_LOW)]
            longest_offloading_cc = max(elem1, elem2)
            # for the ports that serve the same data movement purpose, take the longest data loading cycle
            data_offloading_cc_pair_combined.append(longest_offloading_cc)
        data_offloading_cycle = sum(data_offloading_cc_pair_combined)

        self.data_offloading_cc_pair_combined = data_offloading_cc_pair_combined
        self.data_offloading_cycle = data_offloading_cycle

    def calc_overall_latency(self, cycles_per_mac: int = 1) -> None:
        """! This function integrates the previous calculated SScomb, data loading and off-loading cycle to get the
        overall latency"""
        # @param cycles_per_mac: cycle counts per mac operand (>1 for bit-serial computation)
        # the ideal cycle count assuming the MAC array is 100% utilized
        ideal_cycle = int(
            ceil(
                self.layer.total_MAC_count / self.accelerator.get_core(self.core_id).operational_array.total_unit_count
            )
            * cycles_per_mac
        )

        # the ideal temporal cycle count given the spatial mapping (the spatial mapping can be non-ideal)
        ideal_temporal_cycle = self.mapping_int.temporal_mapping.total_cycle * cycles_per_mac
        MAC_spatial_utilization = ideal_cycle / ideal_temporal_cycle

        # Total latency without the initial data loading and the final data off-loading
        latency_total0 = ideal_temporal_cycle + self.SS_comb
        MAC_utilization0 = ideal_cycle / latency_total0

        # Total latency with the initial data loading, but without the final data off-loading
        latency_total1 = ideal_temporal_cycle + self.SS_comb + self.data_loading_cycle
        MAC_utilization1 = ideal_cycle / latency_total1

        # Total latency with both the initial data loading and the final data off-loading
        latency_total2 = ideal_temporal_cycle + self.SS_comb + self.data_loading_cycle + self.data_offloading_cycle
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

    def get_total_inst_bandwidth(self, memory_instance: MemoryInstance) -> FourWayDataMoving:
        """Given a cost model evaluation and a memory instance, compute the memory's total instantaneous bandwidth
        required throughout the execution of the layer that corresponds to this CME. Returns empty bandwidth
        requirements if the given memory instance is not included in this CME's memory hierarchy.
        NOTE: this function is used in Stream
        """
        # Check which operands require offchip memory throughout the computation
        offchip_mem_operands: list[MemoryOperand] = []
        for op, memory_levels in self.mem_hierarchy_dict.items():
            last_mem_level = memory_levels[-1]
            if last_mem_level.memory_instance == memory_instance:
                offchip_mem_operands.append(op)
        # Obtain the required instantaneous bandwidth to/from offchip for these operands
        total_inst_bw = FourWayDataMoving(0, 0, 0, 0)
        for mem_op in offchip_mem_operands:
            layer_op = self.layer.memory_operand_links.mem_to_layer_op(mem_op)
            inst_bw_4way = self.mapping.unit_mem_data_movement[layer_op][-1].req_mem_bw_inst
            total_inst_bw += inst_bw_4way
        return total_inst_bw

    def __str__(self):
        return f"CostModelEvaluation({self.layer}, core {self.core_id})"

    def __repr__(self):
        return str(self)
