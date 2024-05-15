import logging
from termios import ICANON
from zigzag.hardware.architecture.ImcArray import ImcArray
from zigzag.utils import pickle_deepcopy
from zigzag.cost_model.cost_model import CostModelEvaluation, PortActivity

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

    def __init__(self):
        super().__init__(...)  # TODO
        operational_array = self.accelerator.get_core(self.core_id).operational_array
        self.imc_area = operational_array.total_area
        assert isinstance(operational_array, ImcArray)
        self.operational_array: ImcArray = operational_array

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

    def calc_energy(self):
        """! Calculates the energy cost of this cost model evaluation by calculating the memory reading/writing energy."""
        # - TODO: Interconnection energy
        self.calc_MAC_energy_cost()
        super().calc_memory_energy_cost()

    def calc_MAC_energy_cost(self):
        """! Calculate the dynamic MAC energy"""
        core = self.accelerator.get_core(self.core_id)
        self.MAC_energy_breakdown = core.operational_array.unit.get_energy_for_a_layer(self.layer, self.mapping)
        self.MAC_energy = sum([energy for energy in self.MAC_energy_breakdown.values()])

    def calc_latency(self):
        """!  Calculate latency in 4 steps

        1) As we already calculated the ideal data transfer rate in combined_mapping.py (in the Mapping class),
        here we start with calculating the required (or allowed) memory updating window by comparing the effective
        data size with the physical memory size at each level. If the effective data size is smaller than 50%
        of the physical memory size, then we take the whole period as the allowed memory updating window (double buffer effect);
        otherwise we take the the period divided by the top_ir_loop as the allowed memory updating window.

        2) Then, we compute the real data transfer rate given the actual memory bw per functional port pair,
        assuming we have enough memory ports.

        3) In reality, there is no infinite memory port to use. So, as the second step, we combine the real
        data transfer attributes per physical memory port.

        4) Finally, we combine the stall/slack of each memory port to get the final latency.
        """
        super().calc_double_buffer_flag()
        super().calc_allowed_and_real_data_transfer_cycle_per_DTL()
        # Update the latency model to fit IMC requirement
        self.combine_data_transfer_rate_per_physical_port()
        super().calc_data_loading_offloading_latency()
        # find the cycle count per mac
        operational_array = self.accelerator.get_core(self.core_id).operational_array
        hd_param = operational_array.unit.hd_param
        cycles_per_mac = hd_param["input_precision"] / hd_param["input_bit_per_cycle"]
        super().calc_overall_latency(cycles_per_mac=cycles_per_mac)

    def combine_data_transfer_rate_per_physical_port(self):
        """! This function calculate the stalling cycles for IMC (In-Memory-Computing) hardware template
        Consider memory sharing and port sharing, combine the data transfer activity
        Step 1: collect port activity per memory instance per physical memory port
        Step 2: calculate SS combine and MUW union parameters per physical memory port
        Note: this calculation is incorrect when following conditions are ALL true:
        (1) there are more than two mem levels for storing weights, e.g. dram -> cache -> IMC cells
        (2) extra stalling is introduced due to the intermediate mem levels (e.g. due to insifficuent bw of cache)
        """
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
                        self.mapping_int.unit_mem_data_movement[layer_op][mem_lv].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip the inactive data movement activities because they won't impact SS
                        continue
                    period = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][mem_lv].data_trans_period,
                        mov_dir,
                    )
                    real_cycle = getattr(self.real_data_trans_cycle[layer_op][mem_lv], mov_dir)
                    allowed_cycle = getattr(self.allowed_mem_updat_cycle[layer_op][mem_lv], mov_dir)
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

        # Step 2: calculate weight loading cycles
        layer_const_operand = self.layer.constant_operands[0]  # e.g. "W"
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
        if len(weight_mem_hierarchy) == 1:  # there is only one mem level for weight
            require_weight_loading = False
        else:
            require_weight_loading = True
        # check how many times of weight reloading is required
        # here assume imc cells is the lowest mem level for weight and rw_port
        for imc_port, imc_ports in port_activity_collect[0].items():  # 0: the lowest mem node in the graph
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
                    min(
                        len(self.mem_energy_breakdown[op]),
                        len(other.mem_energy_breakdown[op]),
                    )
                ):
                    l.append(self.mem_energy_breakdown[op][i] + other.mem_energy_breakdown[op][i])
                i = min(
                    len(self.mem_energy_breakdown[op]),
                    len(other.mem_energy_breakdown[op]),
                )
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
                    l.append(self.mem_energy_breakdown_further[op][i] + other.mem_energy_breakdown_further[op][i])
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
                    l.append(self.memory_word_access[op][i] + other.memory_word_access[op][i])
                i = min(len(self.memory_word_access[op]), len(other.memory_word_access[op]))
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
            "combine_data_transfer_rate_per_physical_port",
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
                    "utilization": (self.mem_utili_shared if hasattr(self, "mem_utili_shared") else None),
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
                "spatial_mapping": (self.spatial_mapping_int if hasattr(self, "spatial_mapping_int") else None),
                "temporal_mapping": (self.temporal_mapping if hasattr(self, "temporal_mapping") else None),
            },
        }

    def __simplejsonrepr__(self):
        """! Simple JSON representation used for saving this object to a simple json file."""
        return {
            "energy": self.energy_total,
            "latency": self.latency_total2,
            "tclk": self.tclk,
            "area": self.area_total,
        }
