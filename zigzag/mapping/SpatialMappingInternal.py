from math import prod

from zigzag.datatypes import LayerDim, LayerOperand, UnrollFactor
from zigzag.mapping.mapping_assist_funcs import (
    SpatialMappingPerMemLvl,
    decouple_pr_loop,
)
from zigzag.utils import json_repr_handler
from zigzag.workload.layer_node import LayerNode


class SpatialMappingInternal:
    """! Class that collect all the info related to spatial mapping."""

    def __init__(self, spatial_mapping_dict: SpatialMappingPerMemLvl, layer_node: "LayerNode"):
        self.mapping_dict_origin = spatial_mapping_dict
        self.mapping_dict_reform: SpatialMappingPerMemLvl = decouple_pr_loop(spatial_mapping_dict, layer_node)
        self.layer_node = layer_node
        self.layer_operands: list[LayerOperand] = layer_node.layer_operands

        # Extract architecture level count for each operand from spatial mapping definition, starting from MAC level
        self.arch_level = {op: len(spatial_mapping) for op, spatial_mapping in spatial_mapping_dict.items()}

        # Calculate unrolled loop size for different loop types (r/ir/total)
        self.calc_unroll_size()

        # Calculate total/unique/duplicate unit count
        self.calc_unit_count()

        # Calculate data serve scope: each data element serves/(is served by) how many unit at below level
        # NOTE: data_serve_scope doesn't include MAC level, thus is one level less than other spatial mapping
        # attributes.
        self.calc_data_serve_scope()

        # Calculate memory bandwidth incremental factor between architectural levels
        # mem_bw_boost_factor doesn't include MAC level, thus is one level less than other spatial mapping attributes.
        self.calc_mem_bw_boost_factor()

        # Added for loma: Get list of the spatially unrolled loops, without any information about arch levels
        self.save_spatial_loop_dim_size()

    def __str__(self):
        return f"SpatialMapping({self.mapping_dict_origin})"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """! JSON representation of this object to save it to a file."""
        return json_repr_handler(
            {
                layer_op: [[str(loop_factor_pair) for loop_factor_pair in mem_level] for mem_level in mapping_layer_op]
                for layer_op, mapping_layer_op in self.mapping_dict_origin.items()
            }
        )

    def get_unrolling(self, op: LayerOperand, level: int):
        """! Return the unrolled loops for operand 'op' at level 'level'.
        'level' = 0 would signify the operational level.
        """
        return self.mapping_dict_origin[op][level]

    def get_unrolling_all(self, op: LayerOperand, min_level: int) -> list[tuple[LayerDim, UnrollFactor]]:
        """Return all the spatial loops at a given level and above for a given operand.

        Args:
            op (str): The layer operand for which to return the spatial loops.
            min_level (int): The lowest level.

        Returns:
            list: A list of all spatial loops at given level and above.
        # TODO this has the same functionality as SpatialMapping.flatten_unrollings()
        """
        spatial_loops: list[tuple[LayerDim, UnrollFactor]] = []
        for level in range(min_level, self.arch_level[op]):
            spatial_loops += self.get_unrolling(op, level)
        return spatial_loops

    def calc_unroll_size(self) -> None:
        """! Calculate unrolled loop size for different loop types (r/ir/total) per operand per architecture level"""

        # Initialization
        unroll_size_r: dict[LayerOperand, list[UnrollFactor]] = {
            op: [1] * arch_lv for op, arch_lv in self.arch_level.items()
        }
        unroll_size_ir: dict[LayerOperand, list[UnrollFactor]] = {
            op: [1] * arch_lv for op, arch_lv in self.arch_level.items()
        }
        unroll_size_total: dict[LayerOperand, list[UnrollFactor]] = {
            op: [1] * arch_lv for op, arch_lv in self.arch_level.items()
        }

        # Go through the reformed spatial mapping and extract the unroll size
        for layer_op in self.layer_operands:
            for level, current_level_loops in enumerate(self.mapping_dict_reform[layer_op]):
                for layer_dim, loop_dim in current_level_loops:
                    # TODO use pr_decoupled_relevancy_info
                    if layer_dim in self.layer_node.pr_decoupled_relevancy_info.get_r_layer_dims(layer_op):
                        unroll_size_r[layer_op][level] *= loop_dim
                    else:
                        unroll_size_ir[layer_op][level] *= loop_dim
                    unroll_size_total[layer_op][level] *= loop_dim

        self.unroll_size_r = unroll_size_r
        self.unroll_size_ir = unroll_size_ir
        self.unroll_size_total = unroll_size_total

    def calc_unit_count(self):
        """! Calculate total/unique/duplicate unit count per operand per architecture level"""
        # Number of unit at each level (for each operand)
        # Added round call as number doesn't remain integer due to self.mapping_dict_reform number instability
        unit_count = {
            op: [
                round(round(prod(self.unroll_size_total[op][lv : self.arch_level[op]]), 3))
                for lv in range(self.arch_level[op])
            ]
            for op in self.layer_operands
        }

        #  ASSERT: The bottom level (MAC level) unit count must be the same for all operand
        bottom_unit_count = [unit_count[op][0] for op in unit_count.keys()]
        assert all(x == bottom_unit_count[0] for x in bottom_unit_count), (
            f"The MAC level unit count is not the same for all operand {bottom_unit_count}, "
            f"please correct the spatial mapping."
        )

        #  Number of unit at each level that hold unique data (for each operand)
        unit_unique = {
            op: [prod(self.unroll_size_r[op][lv : self.arch_level[op]]) for lv in range(self.arch_level[op])]
            for op in self.layer_operands
        }

        #  Number of unit at each level that hold the same data (for each operand)
        unit_duplicate = {
            op: [prod(self.unroll_size_ir[op][lv : self.arch_level[op]]) for lv in range(self.arch_level[op])]
            for op in self.layer_operands
        }

        self.unit_count = unit_count
        self.unit_unique = unit_unique
        self.unit_duplicate = unit_duplicate

    def calc_data_serve_scope(self):
        """! Calculate data serve scope, i.e., for input operands, it means that each data element
        is broadcast to how many unit at below level; for output operand, it means that how
        many unit add/collect their output values to one result, and push it to above level

        NOTE: data_serve_scope doesn't include MAC level, thus is one level less than other spatial mapping attributes.

        data_serve_scope is calculated by dividing unit_duplicate at current level by unit_count at one level above.
        """
        data_serve_scope = {
            op: [self.unit_duplicate[op][lv] / self.unit_duplicate[op][lv + 1] for lv in range(self.arch_level[op] - 1)]
            for op in self.layer_operands
        }

        self.data_serve_scope = data_serve_scope

    def calc_mem_bw_boost_factor(self):
        """! Calculate memory bandwidth incremental factor between architectural levels.

        NOTE: mem_bw_boost doesn't include MAC level, thus is one level less than other spatial mapping attributes.

        mem_bw_boost can calculated by either dividing unit_unique at current level by unit_count at one level above.
        """
        mem_bw_boost = {
            op: [
                round(self.unit_unique[op][lv] / self.unit_unique[op][lv + 1]) for lv in range(self.arch_level[op] - 1)
            ]
            for op in self.layer_operands
        }

        self.mem_bw_boost = mem_bw_boost

    def save_spatial_loop_dim_size(self) -> None:
        """! Save the loops that were unrolled spatially in a list without any arch level information for easy access in
        loma. We take one of the input operands and go through the spatial mapping dict for that operand.
        Which operand shouldn't matter as all operands store the same loops, but possibly at different arch levels.
        """

        op = self.layer_node.input_operands[0]
        self.spatial_loop_dim_size: list[tuple[LayerDim, UnrollFactor]] = [
            loop for spatial_loops in self.mapping_dict_origin[op] for loop in spatial_loops
        ]
