from typing import Dict
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zigzag.classes.workload.layer_node import LayerNode
import zigzag.classes.mapping.mapping_assist_funcs as mapping_assist_funcs


class SpatialMapping:
    """
    Class that collect all the info related to spatial mapping.
    """

    def __init__(self, spatial_mapping_dict: Dict, layer_node: 'LayerNode'):
        self.mapping_dict_origin = spatial_mapping_dict
        self.mapping_dict_reform = mapping_assist_funcs.decouple_pr_loop(spatial_mapping_dict, layer_node)
        self.layer_node = layer_node
        self.operand_list = layer_node.operand_list

        ''' Extract architecture level count for each operand from spatial mapping definition, starting from MAC level '''
        self.arch_level = {op: len(smap) for (op, smap) in spatial_mapping_dict.items()}

        ''' Calculate unrolled loop size for different loop types (r/ir/total) '''
        self.calc_unroll_size()

        ''' Calculate total/unique/duplicate unit count '''
        self.calc_unit_count()

        ''' Calculate data serve scope: each data element serves/(is served by) how many unit at below level 
        NOTE: data_serve_scope doesn't include MAC level, thus is one level less than other spatial mapping attributes. '''
        self.calc_data_serve_scope()

        ''' Calculate memory bandwidth incremental factor between architectural levels
        NOTE: mem_bw_boost_factor doesn't include MAC level, thus is one level less than other spatial mapping attributes. '''
        self.calc_mem_bw_boost_factor()

        ''' Added for loma: Get list of the spatially unrolled loops, without any information about arch levels'''
        self.save_spatial_loop_dim_size()

    def __str__(self):
        return f"SpatialMapping({self.mapping_dict_origin})"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """
        JSON representation of this object to save it to a file.
        """
        return {"spatial_mapping": self.mapping_dict_origin}

    def get_unrolling(self, op: str, level: int):
        """
        Return the unrolled loops for operand 'op' at level 'level'.
        'level' = 0 would signify the operational level.
        """
        return self.mapping_dict_origin[op][level]

    def calc_unroll_size(self):
        """
        Calculate unrolled loop size for different loop types (r/ir/total) per operand per architecture level
        """
        ''' Initialization '''
        unroll_size_r = {op: [1] * arch_lv for (op, arch_lv) in self.arch_level.items()}
        unroll_size_ir = {op: [1] * arch_lv for (op, arch_lv) in self.arch_level.items()}
        unroll_size_total = {op: [1] * arch_lv for (op, arch_lv) in self.arch_level.items()}

        ''' Go through the reformed spatial mapping and extract the unroll size '''
        for operand in self.operand_list:
            for level, current_level_loops in enumerate(self.mapping_dict_reform[operand]):
                for loop_type, loop_dim in current_level_loops:
                    if loop_type in self.layer_node.operand_loop_dim_reform[operand]['r']:
                        unroll_size_r[operand][level] *= loop_dim
                    else:
                        unroll_size_ir[operand][level] *= loop_dim
                    unroll_size_total[operand][level] *= loop_dim

        self.unroll_size_r = unroll_size_r
        self.unroll_size_ir = unroll_size_ir
        self.unroll_size_total = unroll_size_total

    def calc_unit_count(self):
        """
        Calculate total/unique/duplicate unit count per operand per architecture level
        """
        ''' Number of unit at each level (for each operand) '''
        # Added round call as number doesn't remain integer due to self.mapping_dict_reform number instability
        unit_count = {op: [
            round(prod(self.unroll_size_total[op][lv:self.arch_level[op]])) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        ''' ASSERT: The bottom level (MAC level) unit count must be the same for all operand '''
        bottom_unit_count = [unit_count[op][0] for op in unit_count.keys()]
        assert all(x == bottom_unit_count[0] for x in bottom_unit_count), \
            f"The MAC level unit count is not the same for all operand {bottom_unit_count}, please correct the spatial mapping."

        ''' Number of unit at each level that hold unique data (for each operand) '''
        unit_unique = {op: [
            prod(self.unroll_size_r[op][lv:self.arch_level[op]]) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        ''' Number of unit at each level that hold the same data (for each operand) '''
        unit_duplicate = {op: [
            prod(self.unroll_size_ir[op][lv:self.arch_level[op]]) for lv in range(self.arch_level[op])
        ] for op in self.operand_list}

        self.unit_count = unit_count
        self.unit_unique = unit_unique
        self.unit_duplicate = unit_duplicate

    def calc_data_serve_scope(self):
        """
        Calculate data serve scope, i.e.,
        for input operands, it means that each data element is broadcast to how many unit at below level;
        for output operand, it means that how many unit add/collect their output values to one result, and push it to above level '''

        NOTE: data_serve_scope doesn't include MAC level, thus is one level less than other spatial mapping attributes.
        """
        ''' data_serve_scope is calculated by dividing unit_duplicate at current level by unit_count at one level above. '''
        data_serve_scope = {op: [
            self.unit_duplicate[op][lv]/self.unit_duplicate[op][lv+1] for lv in range(self.arch_level[op]-1)
        ] for op in self.operand_list}

        self.data_serve_scope = data_serve_scope

    def calc_mem_bw_boost_factor(self):
        """
        Calculate memory bandwidth incremental factor between architectural levels.

        NOTE: mem_bw_boost doesn't include MAC level, thus is one level less than other spatial mapping attributes.
        """
        ''' mem_bw_boost can calculated by either dividing unit_unique at current level by unit_count at one level above. '''
        mem_bw_boost = {op: [
            int(self.unit_unique[op][lv]/self.unit_unique[op][lv+1]) for lv in range(self.arch_level[op]-1)
        ] for op in self.operand_list}

        self.mem_bw_boost = mem_bw_boost

    def save_spatial_loop_dim_size(self):
        """
        Save the loops that were unrolled spatially in a list without any arch level information for easy access in loma.
        """
        # We take one of the input operands and go through the spatial mapping dict for that operand.
        # Which operand shouldn't matter as all operands store the same loops, but possibly at different arch levels.
        op = self.layer_node.input_operands[0]
        self.spatial_loop_dim_size = [loop for spatial_loops in self.mapping_dict_origin[op] for loop in spatial_loops]
