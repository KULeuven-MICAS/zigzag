import math

import zigzag.mapping.mapping_assist_funcs as mapping_assist_funcs
from zigzag.datatypes import Constants, LayerDim, LayerOperand, UnrollFactor
from zigzag.hardware.architecture.Accelerator import Accelerator
from zigzag.hardware.architecture.memory_port import DataDirection
from zigzag.mapping.data_movement import DataMovePattern
from zigzag.mapping.mapping_assist_funcs import SpatialMappingPerMemLvl
from zigzag.mapping.SpatialMappingInternal import SpatialMappingInternal
from zigzag.mapping.TemporalMapping import TemporalMapping, TemporalMappingDict
from zigzag.workload.layer_node import LayerNode


class Mapping:
    """! Collect information of a complete mapping (spatial and temporal)

    NOTE: Mapping is HW-unaware, i.e. Mapping doesn't take in HW information
    like memory bw, access cost, size and so on.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        spatial_mapping: SpatialMappingPerMemLvl | SpatialMappingInternal,
        temporal_mapping: TemporalMappingDict | TemporalMapping,
        layer_node: LayerNode,
        access_same_data_considered_as_no_access: bool = False,
    ):
        # Mapping object can be initialized with separate spatial and temporal mappings
        self.accelerator = accelerator
        if isinstance(spatial_mapping, SpatialMappingInternal):
            self.spatial_mapping = spatial_mapping
        else:
            self.spatial_mapping = SpatialMappingInternal(spatial_mapping, layer_node)
        if isinstance(temporal_mapping, TemporalMapping):
            self.temporal_mapping = temporal_mapping
        else:
            self.temporal_mapping = TemporalMapping(temporal_mapping, layer_node)
        self.layer_node = layer_node
        self.operand_list = layer_node.layer_operands
        self.access_same_data_considered_as_no_access = access_same_data_considered_as_no_access
        self.mem_level = self.temporal_mapping.mem_level
        # Initialize unit_mem_data_movement, which collects all the important data movement info
        # related to each unit memory, such as data access count, data precision, required memory BW to
        # prevent stall, data transfer rate, etc.
        self.unit_mem_data_movement: dict[LayerOperand, list[DataMovePattern]] = {  # type: ignore
            op: [None for _ in range(self.mem_level[op])] for op in self.operand_list
        }

        # Combine spatial and temporal mapping dictionary into "joint_mapping_dict" in order to
        # enable decoupling pr loops into r and ir loops in one go
        self.combine_spatial_temporal_mapping_dict()

        # Decouple pr loops into r and ir loops, preparing for later mapping info extraction
        self.combined_mapping_dict_1s1t_reform = mapping_assist_funcs.decouple_pr_loop(
            self.combined_mapping_dict_1s1t, layer_node
        )
        self.combined_mapping_dict_1s2t_reform = mapping_assist_funcs.decouple_pr_loop(
            self.combined_mapping_dict_1s2t, layer_node
        )

        # Distinguish final output from partial output: "psum_flag"
        self.psum_flag = self.get_psum_flags()

        # Generate a dictionary that collect data precision for each operand at each arch level
        self.gen_data_precision_dict()

        # Generate r/ir loop size list at each level for each operand
        self.gen_r_ir_loop_list()

        # Calculate data size at each memory level, including total data size and
        # unrolled data size (data at each unrolled memory unit). Unit used: # element
        self.calc_data_size()

        # Calculate the effective data size at each unrolled memory unit.
        # Effective data size: the unrolled data size divided by all top r loops at that level.
        # Unit used: # element
        self.calc_effective_data_size()

        # Calculate data access count at each memory level. Unit used: # element
        # NOTE: this data access is not memory word access!
        self.calc_data_access()

        # Calculate required memory bandwidth and the periodic data transfer pattern
        self.calc_req_mem_bw_and_data_transfer_rate()

        # Ignore the data traffic between the top level memory and the external world
        self.disable_data_traffic_external()

    def combine_spatial_temporal_mapping_dict(self):
        """! Combine spatial and temporal mapping dictionary into combined_mapping_dict by
        inserting spatial loops above temporal loops at each level.

        - combined_mapping_dict_1s1t: corresponding level's spatial mapping and temporal mapping are merged together.
        Each level's data size is the total data size.
        - combined_mapping_dict_1s2t: each level's spatial mapping is merged to level+1's temporal mapping.
        Each level's data size is the unrolled data size.
        """

        # Initialization
        combined_mapping_dict_1s1t: dict[LayerOperand, list[list[tuple[LayerDim, UnrollFactor]]]] = {
            op: [[] for _ in range(self.spatial_mapping.arch_level[op])] for op in self.operand_list
        }
        combined_mapping_dict_1s2t: dict[LayerOperand, list[list[tuple[LayerDim, UnrollFactor]]]] = {
            op: [[] for _ in range(self.spatial_mapping.arch_level[op] + 1)] for op in self.operand_list
        }
        su_dict_seed = self.spatial_mapping.mapping_dict_origin
        # Add an empty innermost level and an empty outermost level
        tm_dict_seed: dict[LayerOperand, list[list[tuple[LayerDim, UnrollFactor]]]] = {
            op: [[]] + tm_list + [[]] for op, tm_list in self.temporal_mapping.mapping_dic_stationary.items()
        }

        # Combining
        for operand in self.operand_list:
            for level, current_level_su_loops in enumerate(su_dict_seed[operand]):
                current_level_tm_loops = tm_dict_seed[operand][level]
                above_level_tm_loops = tm_dict_seed[operand][level + 1]
                combined_mapping_dict_1s1t[operand][level] = current_level_tm_loops + current_level_su_loops
                combined_mapping_dict_1s2t[operand][level + 1] = above_level_tm_loops + current_level_su_loops

        self.combined_mapping_dict_1s1t = combined_mapping_dict_1s1t
        self.combined_mapping_dict_1s2t = combined_mapping_dict_1s2t

    def get_psum_flags(self) -> list[bool]:
        """! This function generates an list "psum_flag" that identify whether an output memory
        level holds partial or final output.
        E.g., psum_flag = [True, True, False] means that there are 3 memory levels for output and only the outermost
        memory level hold the final output, the 1st and 2nd memory levels need to store partial output for some time.
        For indexing convenience, we add an extra False to the end of the psum_flag list.
        # TODO cleanup
        """
        output_operand = self.layer_node.output_operand
        loop_dim_relevancy = self.layer_node.pr_decoupled_relevancy_info
        # output_ir_flag indicate whether at an architectural level, there is output's ir loop in it.
        # True for yes, there is.
        output_arch_level = self.spatial_mapping.arch_level[output_operand]
        output_ir_flag = [False] * output_arch_level
        for level, current_level_loops in enumerate(self.combined_mapping_dict_1s1t_reform[output_operand]):
            for loop_type, loop_dim in current_level_loops:
                if loop_type in loop_dim_relevancy.get_ir_layer_dims(output_operand) and loop_dim > 1:
                    output_ir_flag[level] = True
                    break
        # reversely check from current level to the top level whether there is ir loop shows up in the middle,
        # False means final output is present at current level
        psum_flag_h2l = [any(output_ir_flag[lv:output_arch_level]) for lv in reversed(range(output_arch_level))]
        psum_flag_l2h = list(reversed(psum_flag_h2l))
        return psum_flag_l2h[1:] + [False]  # add an extra False on top for later indexing convenience

    def gen_data_precision_dict(self):
        """! This function generates a dictionary that collect data precision for each operand at each arch level"""
        input_operands = self.layer_node.input_operands
        output_operand = self.layer_node.output_operand
        data_precision_dict: dict[LayerOperand, list[int]] = {
            op: [self.layer_node.operand_precision[op]] * (self.mem_level[op] + 1) for op in input_operands
        }
        data_precision_dict[output_operand] = []
        for i in range(self.mem_level[output_operand] + 1):
            if self.psum_flag[i]:
                data_precision_dict[output_operand].append(self.layer_node.operand_precision[output_operand])
            else:
                data_precision_dict[output_operand].append(
                    self.layer_node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
                )
        self.data_precision_dict = data_precision_dict

    def gen_r_ir_loop_list(self):
        """! Given the combined mapping, generate r/ir loop size list at each level for each operand
        # TODO cleanup
        """
        combined_mapping = self.combined_mapping_dict_1s1t_reform
        combined_mapping2 = self.combined_mapping_dict_1s2t_reform
        relevancy_table = self.layer_node.pr_decoupled_relevancy_info
        r_loop_size_per_level = {
            op: [
                math.prod(
                    [
                        lp_dim
                        for lp_type, lp_dim in combined_mapping[op][lv]
                        if lp_type in relevancy_table.get_r_layer_dims(op)
                    ]
                )
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        r_loop_size_per_level2 = {
            op: [
                math.prod(
                    [
                        lp_dim
                        for lp_type, lp_dim in combined_mapping2[op][lv]
                        if lp_type in relevancy_table.get_r_layer_dims(op)
                    ]
                )
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        ir_loop_size_per_level = {
            op: [
                math.prod(
                    [
                        lp_dim
                        for lp_type, lp_dim in combined_mapping[op][lv]
                        if lp_type in relevancy_table.get_ir_layer_dims(op)
                    ]
                )
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        ir_loop_size_per_level2 = {
            op: [
                math.prod(
                    [
                        lp_dim
                        for lp_type, lp_dim in combined_mapping2[op][lv]
                        if lp_type in relevancy_table.get_ir_layer_dims(op)
                    ]
                )
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }

        # current and below levels (cabl) r loop size
        r_loop_size_cabl = {
            op: [
                round(math.prod(r_loop_size_per_level[op][0 : lv + 1]))
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        r_loop_size_cabl2 = {
            op: [
                round(math.prod(r_loop_size_per_level2[op][0 : lv + 1]))
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        # current and below levels (cabl) ir loop size
        ir_loop_size_cabl = {
            op: [math.prod(ir_loop_size_per_level[op][0 : lv + 1]) for lv in range(self.spatial_mapping.arch_level[op])]
            for op in self.operand_list
        }
        # current and below levels (cabl) ir loop size
        ir_loop_size_cabl2 = {
            op: [
                math.prod(ir_loop_size_per_level2[op][0 : lv + 1]) for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        # current and above levels (caal) ir loop size, only for output operand for calculating psum backflow access
        # count
        output_operand = self.layer_node.output_operand
        output_ir_loop_size_caal = [
            math.prod(ir_loop_size_per_level[output_operand][lv : self.spatial_mapping.arch_level[output_operand]])
            for lv in range(self.spatial_mapping.arch_level[output_operand])
        ]
        # append two extra 1 to the list to facilitate the psum backflow access calculation later
        # We can see it as adding two output memory levels on top with no data reuse.
        output_ir_loop_size_caal.extend([1, 1])

        self.r_loop_size_per_level = r_loop_size_per_level
        self.r_loop_size_per_level2 = r_loop_size_per_level2
        self.ir_loop_size_per_level = ir_loop_size_per_level
        self.r_loop_size_cabl = r_loop_size_cabl
        self.r_loop_size_cabl2 = r_loop_size_cabl2
        self.ir_loop_size_cabl = ir_loop_size_cabl
        self.ir_loop_size_cabl2 = ir_loop_size_cabl2
        self.output_ir_loop_size_caal = output_ir_loop_size_caal

    def calc_data_size(self):
        """! Based on the r loop size list, calculate the data size held by each architectural level."""
        # data_elem_per_level_unrolled: data size held inside of each unrolled unit at each architectural level
        # data_elem_per_level: total data size at each architectural level (= data_elem_per_level_unrolled * unique
        # unit count)
        data_elem_per_level_unrolled = {
            op: [round(self.r_loop_size_cabl2[op][lv]) for lv in range(self.spatial_mapping.arch_level[op])]
            for op in self.operand_list
        }

        data_bit_per_level_unrolled = {
            op: [
                round(self.r_loop_size_cabl2[op][lv]) * self.data_precision_dict[op][lv]
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }

        data_elem_per_level = {
            op: [
                round(data_elem_unrolled * self.spatial_mapping.unit_unique[op][lv])
                for lv, data_elem_unrolled in enumerate(data_elem_per_level_unrolled[op])
            ]
            for op in self.operand_list
        }

        data_bit_per_level = {
            op: [
                round(data_elem_unrolled * self.spatial_mapping.unit_unique[op][lv]) * self.data_precision_dict[op][lv]
                for lv, data_elem_unrolled in enumerate(data_elem_per_level_unrolled[op])
            ]
            for op in self.operand_list
        }

        self.data_elem_per_level_unrolled = data_elem_per_level_unrolled
        self.data_bit_per_level_unrolled = data_bit_per_level_unrolled
        self.data_elem_per_level = data_elem_per_level
        self.data_bit_per_level = data_bit_per_level

    def calc_effective_data_size(self):
        """! Calculate the effective data size for getting the allowed memory updating window in latency calculation.
        The effective data size is calculated by using data_elem_per_level_unrolled divided by the top r loops.
        """
        effective_data_elem = {
            op: [
                data_elem_unrolled // self.temporal_mapping.top_r_loop_size[op][lv]
                for lv, data_elem_unrolled in enumerate(self.data_elem_per_level_unrolled[op])
            ]
            for op in self.operand_list
        }

        effective_data_bit = {
            op: [
                data_bit_unrolled // self.temporal_mapping.top_r_loop_size[op][lv]
                for lv, data_bit_unrolled in enumerate(self.data_bit_per_level_unrolled[op])
            ]
            for op in self.operand_list
        }
        self.effective_data_elem = effective_data_elem
        self.effective_data_bit = effective_data_bit

    def calc_data_access(self):
        """! Based on the ir loop size list and the total MAC Op count, calculate the data access
        at each memory level in a bottom-up way.
        """
        total_mac_count = self.layer_node.total_mac_count

        # data_access_raw doesn't distinguish read and write, doesn't distinguish input operands from output operand
        # data_access_raw: each memory levels' spatial loops and temporal loops are put together
        # (combined_mapping_dict_1s1t)
        # data_access_raw2: each memory levels' spatial loops are put to memory level + 1s' temporal loops location
        # (combined_mapping_dict_1s2t)
        data_access_raw = {
            op: [
                round(total_mac_count / self.ir_loop_size_cabl[op][lv])
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        data_access_raw2 = {
            op: [
                round(total_mac_count / self.ir_loop_size_cabl2[op][lv])
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        self.data_access_raw = data_access_raw
        self.data_access_raw2 = data_access_raw2

        # Distinguish read and write, unify input operands and output operand
        # For input operands
        for operand in self.layer_node.input_operands:
            for mem_level in range(self.mem_level[operand]):
                unit_mem_data_movement = DataMovePattern(operand, mem_level)

                # data access count
                if (
                    self.access_same_data_considered_as_no_access
                    and mem_level == 0
                    and self.accelerator.get_core(self.layer_node.core_allocation[0]).mem_r_bw_dict[
                        self.layer_node.memory_operand_links[operand]
                    ][mem_level]
                    >= self.data_bit_per_level[operand][mem_level]
                    // self.spatial_mapping.unit_unique[operand][mem_level + 1]
                ):
                    # If we need access the same input data multiple times from the innermost memory level and the data
                    ##size is smaller than the memory read bw, take into account only one-time access cost (assume the
                    # data can stay at the output pins of the memory as long as it is needed).
                    rd_out_to_low = int(
                        data_access_raw[operand][mem_level]
                        / self.temporal_mapping.mac_level_data_stationary_cycle[operand]
                    )
                else:
                    rd_out_to_low = data_access_raw[operand][mem_level]
                wr_in_by_low = 0
                rd_out_to_high = 0
                wr_in_by_high = data_access_raw2[operand][mem_level + 1]
                unit_mem_data_movement.set_data_elem_move_count(
                    rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high
                )
                # data precision
                rd_out_to_low_pre = self.layer_node.operand_precision[operand]
                wr_in_by_low_pre = 0
                rd_out_to_high_pre = 0
                wr_in_by_high_pre = self.layer_node.operand_precision[operand]
                unit_mem_data_movement.set_data_precision(
                    rd_out_to_low_pre,
                    wr_in_by_low_pre,
                    rd_out_to_high_pre,
                    wr_in_by_high_pre,
                )

                self.unit_mem_data_movement[operand][mem_level] = unit_mem_data_movement

        # For output operand
        output_operand = self.layer_node.output_operand
        for mem_level in range(self.mem_level[output_operand]):
            unit_mem_data_movement = DataMovePattern(output_operand, mem_level)

            # Note that the index for data_access_raw is arch_level, which is one level more than mem_level.
            # the first arch_level means the operational array level (e.g. MAC array level);
            # the first mem_level means the innermost memory level (e.g. register file level.

            # data access count
            wr_in_by_low = data_access_raw[output_operand][mem_level]
            rd_out_to_low = int(
                self.layer_node.operand_size_elem[output_operand] * (self.output_ir_loop_size_caal[mem_level + 1] - 1)
            )

            rd_out_to_high = data_access_raw2[output_operand][mem_level + 1]
            wr_in_by_high = int(
                self.layer_node.operand_size_elem[output_operand] * (self.output_ir_loop_size_caal[mem_level + 2] - 1)
            )

            unit_mem_data_movement.set_data_elem_move_count(rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high)

            # data precision
            if rd_out_to_low != 0:
                # partial output data precision
                wr_in_by_low_pre = self.layer_node.operand_precision[output_operand]
                rd_out_to_low_pre = self.layer_node.operand_precision[output_operand]
            else:
                # final output data precision
                wr_in_by_low_pre = self.layer_node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
                rd_out_to_low_pre = 0
            if wr_in_by_high != 0:
                # partial output data precision
                wr_in_by_high_pre = self.layer_node.operand_precision[output_operand]
                rd_out_to_high_pre = self.layer_node.operand_precision[output_operand]
            else:
                # final output data precision
                wr_in_by_high_pre = 0
                rd_out_to_high_pre = self.layer_node.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]

            unit_mem_data_movement.set_data_precision(
                rd_out_to_low_pre,
                wr_in_by_low_pre,
                rd_out_to_high_pre,
                wr_in_by_high_pre,
            )

            self.unit_mem_data_movement[output_operand][mem_level] = unit_mem_data_movement

    def calc_req_mem_bw_and_data_transfer_rate(self):
        """! This function calculates the average & instant required memory bw and the periodic data transfer
        pattern."""

        if self.access_same_data_considered_as_no_access:
            # For input operands, add operational array level's 'mac_level_data_stationary_cycle' cycle in the below to
            # align with the list length of data_each_level
            cycle_each_level = {
                op: [self.temporal_mapping.mac_level_data_stationary_cycle[op]]
                + self.temporal_mapping.cycle_cabl_level[op]
                for op in self.layer_node.input_operands
            }
            # For output operands, add operational array level's 1 cycle in the below to align with the list length of
            # data_each_level
            cycle_each_level[self.layer_node.output_operand] = [1] + self.temporal_mapping.cycle_cabl_level[
                self.layer_node.output_operand
            ]
        else:
            cycle_each_level = {op: [1] + self.temporal_mapping.cycle_cabl_level[op] for op in self.operand_list}
        data_each_level_unrolled = self.data_elem_per_level_unrolled

        # Add the mem BW boost factor 1 on the top (the memory BW boost factor from outside to top memory)
        # to align with the list length of data_each_level
        mem_bw_boost_factor = {op: self.spatial_mapping.mem_bw_boost[op] + [1] for op in self.operand_list}

        # req_mem_bw_raw doesn't distinguish read and write, doesn't distinguish input operands from output operand
        # "_L/_H" indicates for each data transfer link (DTL), the lower/higher memory level's required BW,
        # e.g. for the DTL of Global Buffer (Weight) talking to spatially unrolled Weight Reg File,
        # each Weight Reg File's required write BW is indicated by "_L",
        # while Global Buffer (Weight)'s required read BW is indicate by "_H"
        req_mem_bw_low_raw = {
            op: [
                data_each_level_unrolled[op][lv] / cycle_each_level[op][lv]
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }
        req_mem_bw_high_raw = {
            op: [
                req_mem_bw_low_raw[op][lv] * mem_bw_boost_factor[op][lv]
                for lv in range(self.spatial_mapping.arch_level[op])
            ]
            for op in self.operand_list
        }

        # Calculates the average required memory bw.

        # Distinguish read and write, unify input operands and output operand
        # For input operands
        for operand in self.layer_node.input_operands:
            for mem_level in range(self.mem_level[operand]):
                # average required memory BW
                rd_out_to_low_bw = int(req_mem_bw_high_raw[operand][mem_level])
                wr_in_by_low_bw = 0
                rd_out_to_high_bw = 0
                wr_in_by_high_bw = int(req_mem_bw_low_raw[operand][mem_level + 1])
                self.unit_mem_data_movement[operand][mem_level].set_req_mem_bw_aver(
                    rd_out_to_low_bw,
                    wr_in_by_low_bw,
                    rd_out_to_high_bw,
                    wr_in_by_high_bw,
                )
                # data transfer period
                rd_out_to_low_pd = int(cycle_each_level[operand][mem_level])
                wr_in_by_low_pd = 0
                rd_out_to_high_pd = 0
                wr_in_by_high_pd = int(cycle_each_level[operand][mem_level + 1])
                self.unit_mem_data_movement[operand][mem_level].set_data_trans_period(
                    rd_out_to_low_pd,
                    wr_in_by_low_pd,
                    rd_out_to_high_pd,
                    wr_in_by_high_pd,
                )
                # data transfer period count
                rd_out_to_low_pc = int(self.temporal_mapping.total_cycle // cycle_each_level[operand][mem_level])
                wr_in_by_low_pc = 0
                rd_out_to_high_pc = 0
                wr_in_by_high_pc = int(self.temporal_mapping.total_cycle // cycle_each_level[operand][mem_level + 1])
                self.unit_mem_data_movement[operand][mem_level].set_data_trans_period_count(
                    rd_out_to_low_pc,
                    wr_in_by_low_pc,
                    rd_out_to_high_pc,
                    wr_in_by_high_pc,
                )
                # per-period data transfer amount
                rd_out_to_low_da = (
                    data_each_level_unrolled[operand][mem_level] * mem_bw_boost_factor[operand][mem_level]
                )
                wr_in_by_low_da = 0
                rd_out_to_high_da = 0
                wr_in_by_high_da = data_each_level_unrolled[operand][mem_level + 1]

                self.unit_mem_data_movement[operand][mem_level].set_data_trans_amount_per_period(
                    rd_out_to_low_da,
                    wr_in_by_low_da,
                    rd_out_to_high_da,
                    wr_in_by_high_da,
                )

        # For output operand
        output_operand = self.layer_node.output_operand
        for mem_level in range(self.mem_level[output_operand]):
            wr_in_by_low_bw = int(req_mem_bw_high_raw[output_operand][mem_level])
            rd_out_to_high_bw = int(req_mem_bw_low_raw[output_operand][mem_level + 1])
            wr_in_by_low_pd = int(cycle_each_level[output_operand][mem_level])
            rd_out_to_high_pd = int(cycle_each_level[output_operand][mem_level + 1])
            wr_in_by_low_pc = int(self.temporal_mapping.total_cycle // cycle_each_level[output_operand][mem_level])
            rd_out_to_high_pc = int(
                self.temporal_mapping.total_cycle // cycle_each_level[output_operand][mem_level + 1]
            )
            wr_in_by_low_da = int(
                data_each_level_unrolled[output_operand][mem_level] * mem_bw_boost_factor[output_operand][mem_level]
            )
            rd_out_to_high_da = data_each_level_unrolled[output_operand][mem_level + 1]

            if self.psum_flag[mem_level]:
                rd_out_to_low_bw = wr_in_by_low_bw
                rd_out_to_low_pd = wr_in_by_low_pd
                rd_out_to_low_pc = wr_in_by_low_pc
                rd_out_to_low_da = wr_in_by_low_da
            else:
                rd_out_to_low_bw = 0
                rd_out_to_low_pd = 0
                rd_out_to_low_pc = 0
                rd_out_to_low_da = 0

            if self.psum_flag[mem_level + 1]:
                wr_in_by_high_bw = rd_out_to_high_bw
                wr_in_by_high_pd = rd_out_to_high_pd
                wr_in_by_high_pc = rd_out_to_high_pc
                wr_in_by_high_da = rd_out_to_high_da
            else:
                wr_in_by_high_bw = 0
                wr_in_by_high_pd = 0
                wr_in_by_high_pc = 0
                wr_in_by_high_da = 0

            # average required memory BW
            self.unit_mem_data_movement[output_operand][mem_level].set_req_mem_bw_aver(
                rd_out_to_low_bw, wr_in_by_low_bw, rd_out_to_high_bw, wr_in_by_high_bw
            )
            # data transfer period
            self.unit_mem_data_movement[output_operand][mem_level].set_data_trans_period(
                rd_out_to_low_pd, wr_in_by_low_pd, rd_out_to_high_pd, wr_in_by_high_pd
            )
            # data transfer period count
            self.unit_mem_data_movement[output_operand][mem_level].set_data_trans_period_count(
                rd_out_to_low_pc, wr_in_by_low_pc, rd_out_to_high_pc, wr_in_by_high_pc
            )
            # per-period data transfer amount
            self.unit_mem_data_movement[output_operand][mem_level].set_data_trans_amount_per_period(
                rd_out_to_low_da, wr_in_by_low_da, rd_out_to_high_da, wr_in_by_high_da
            )

        # Calculate the instant memory updating behavior.
        top_ir_loop_size = self.temporal_mapping.top_ir_loop_size
        for operand in self.operand_list:
            for level, data_movement_item in enumerate(self.unit_mem_data_movement[operand]):
                req_mem_bw_aver = data_movement_item.req_mem_bw_aver
                # calculate "instant required memory BW" based on "average required memory BW"
                rd_out_to_low_bw = int(req_mem_bw_aver.rd_out_to_low * top_ir_loop_size[operand][level])
                wr_in_by_low_bw = int(req_mem_bw_aver.wr_in_by_low * top_ir_loop_size[operand][level])
                rd_out_to_high_bw = int(req_mem_bw_aver.rd_out_to_high * top_ir_loop_size[operand][level + 1])
                wr_in_by_high_bw = int(req_mem_bw_aver.wr_in_by_high * top_ir_loop_size[operand][level + 1])
                data_movement_item.set_req_mem_bw_inst(
                    rd_out_to_low_bw,
                    wr_in_by_low_bw,
                    rd_out_to_high_bw,
                    wr_in_by_high_bw,
                )

                data_trans_period = data_movement_item.data_trans_period
                # calculate "instant data transferring window", assuming non-double buffered memory
                rd_out_to_low_wd = int(data_trans_period.rd_out_to_low / top_ir_loop_size[operand][level])
                wr_in_by_low_wd = int(data_trans_period.wr_in_by_low / top_ir_loop_size[operand][level])
                rd_out_to_high_wd = int(data_trans_period.rd_out_to_high / top_ir_loop_size[operand][level + 1])
                wr_in_by_high_wd = int(data_trans_period.wr_in_by_high / top_ir_loop_size[operand][level + 1])
                data_movement_item.set_inst_data_trans_window(
                    rd_out_to_low_wd,
                    wr_in_by_low_wd,
                    rd_out_to_high_wd,
                    wr_in_by_high_wd,
                )

    def disable_data_traffic_external(self):
        """! This function set all the data traffic between the top level memory and the external world to 0
        in unit_mem_data_movement.
        """
        for operand in self.operand_list:
            mem_level = self.mem_level[operand] - 1
            self.unit_mem_data_movement[operand][mem_level].update_single_dir_data(DataDirection.RD_OUT_TO_HIGH, 0)
            self.unit_mem_data_movement[operand][mem_level].update_single_dir_data(DataDirection.WR_IN_BY_HIGH, 0)
