from math import ceil
from typing import Any

from onnx import ModelProto, NodeProto

from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.workload.layer_node import LayerNode


class ConvParser(ONNXOperatorParser):
    """! Parser for ONNX Conv and QLinearConv nodes into LayerNode."""

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        mapping_data: list[dict[str, Any]],
        onnx_model: ModelProto,
    ) -> None:
        super().__init__(node_id, node, nodes_outputs, onnx_model)
        self.mapping_data = mapping_data
        self.onnx_model = onnx_model

    def run(self) -> LayerNode:
        """! Run the parser and return the created LayerNode object"""
        return self.generate_layer_node_for_conv()

    def get_layer_node_user_format(
        self,
        kernel_shape: list[int],
        strides: list[int],
        dilations: list[int],
        group_size: int,
        padding: list[int],
        ia_shape: list[int],
        oa_shape: list[int],
    ) -> dict[str, Any]:
        """! Generate the necessary dictionary items required for the LayerNode creation. If there is no data for a
        given Layer Attribute, the Layer Attribute is not included in the returned dict.
        """

        predecessors = self.get_node_predecessors()
        act_precision = self.get_activation_precision()
        weight_precision = self.get_weight_precision()
        intermediate_output_precision = self.get_intermediate_output_precision()

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        # IMPORTANT: If any of the input loops require padding, they should be defined as the rightmost dimensions
        # in the equation.  This is because we construct the dimensionality order and then add the padding to those last
        # dimensions in the order
        data["equation"] = "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]"
        data["operand_source"] = self.get_operand_source_user_format(predecessors)

        # Get dimension sizes from input parameters
        assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
        batch_size = oa_shape[0] if oa_shape[0] > 0 else 1
        size_k = ceil(oa_shape[1] / group_size)
        size_ox = oa_shape[3]
        size_oy = oa_shape[2]
        size_c = ceil(ia_shape[1] / group_size)
        size_ix = ia_shape[3]
        size_iy = ia_shape[2]
        size_fx = kernel_shape[0]
        size_fy = kernel_shape[1]

        data["loop_dims"] = ["B", "K", "G", "OX", "OY", "C", "FX", "FY"]
        data["loop_sizes"] = [
            batch_size,
            size_k,
            group_size,
            size_ox,
            size_oy,
            size_c,
            size_fx,
            size_fy,
        ]
        data["dimension_relations"] = [
            f"ix={strides[0]}*ox+{dilations[0]}*fx",
            f"iy={strides[1]}*oy+{dilations[1]}*fy",
        ]

        data["operand_precision"] = {
            "W": weight_precision,
            "I": act_precision,
            "O_final": act_precision,
            "O": intermediate_output_precision,
        }

        # Add padding information
        data["pr_loop_dims"] = ["IX", "IY"]
        data["pr_loop_sizes"] = [size_ix, size_iy]
        data["padding"] = [
            [padding[0], padding[2]],
            [padding[1], padding[3]],
        ]

        return data

    def generate_layer_node_for_conv(self):
        attrs = self.node.attribute
        kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type: ignore
        strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type: ignore
        dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type: ignore
        group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type: ignore
        padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type: ignore

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # Create LayerNode
        layer_data = self.get_layer_node_user_format(
            kernel_shape,
            strides,
            dilations,
            group_size,
            padding,
            ia_dimension_shape,
            oa_dimension_shape,
        )
        factory = LayerNodeFactory(layer_data, self.mapping_data)
        layer_node = factory.create()

        return layer_node
