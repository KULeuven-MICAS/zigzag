from math import ceil
from typing import Any


from zigzag.parser.onnx.Parser import Parser
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
    get_onnx_tensor_type,
)
from zigzag.workload.layer_attributes import LayerAttributes
from zigzag.workload.layer_node import LayerNode
from zigzag.utils import pickle_deepcopy

import logging

logger = logging.getLogger(__name__)


class ConvParser(Parser):
    """! Parser for ONNX Conv and QLinearConv nodes into LayerNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:

        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self) -> LayerNode:
        """! Run the parser and return the created LayerNode object"""
        layer_node = self.generate_layer_node_for_conv()
        return layer_node

    def generate_layer_node_for_conv(self):
        def get_weight_name(node):
            """! Return the name of the weight input of this node depending on its operator type
            @param node (NodeProto): The node
            """
            op_type = node.op_type  # 'Conv', 'QLinearConv', ...
            if op_type == "Conv":
                return node.input[1]
            elif op_type == "QLinearConv":
                return node.input[3]
            else:
                raise NotImplementedError(f"Retrieving weight name for onnx node of type {op_type} is not supported.")

        def get_input_output_weight_data_type(node, model):
            """! Return the data type of the input, output and weight tensors of this node.
            @param node
            @param model
            """
            input_name = node.input[0]
            output_name = node.output[0]
            weight_name = get_weight_name(node)

            input_elem_type = get_onnx_tensor_type(input_name, model).elem_type
            output_elem_type = get_onnx_tensor_type(output_name, model).elem_type
            weight_elem_type = get_onnx_tensor_type(weight_name, model).elem_type

            return input_elem_type, output_elem_type, weight_elem_type

        def get_layer_node_input_format(
            kernel_shape,
            strides,
            dilations,
            groups,
            padding,
            ia_shape,
            oa_shape,
            node_mapping,
        ) -> dict[str, Any]:
            """! Generate the necessary dictionary items required for the LayerNode creation. If there is no data for a given Layer Attribute, the Layer Attribute is not included in the returned dict."""
            # convert the data types to precisions based on the onnx definition

            # Equation
            d = {}
            # IMPORTANT: If any of the input loops require padding, they should be defined as the rightmost dimensions in the equation
            # This is because we construct the dimensionality order and then add the padding to those last dimensions in the order
            d["equation"] = "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]"

            # Get dimension sizes from input parameters
            assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
            B = oa_shape[0]
            if B == 0:
                B = 1
            G = groups
            K = ceil(oa_shape[1] / G)
            OX = oa_shape[3]
            OY = oa_shape[2]
            C = ceil(ia_shape[1] / G)
            IX = ia_shape[3]
            IY = ia_shape[2]
            FX = kernel_shape[0]
            FY = kernel_shape[1]
            d["loop_dim_size"] = {
                "B": B,
                "K": K,
                "G": G,
                "OX": OX,
                "OY": OY,
                "C": C,
                "FX": FX,
                "FY": FY,
            }
            d["pr_loop_dim_size"] = {"IX": IX, "IY": IY}
            d["dimension_relations"] = [
                f"ix={strides[0]}*ox+{dilations[0]}*fx",
                f"iy={strides[1]}*oy+{dilations[1]}*fy",
            ]
            d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
            # d["operand_source"] =  {'W': [], 'I': []}
            d["constant_operands"] = ["W"]
            d["core_allocation"] = node_mapping["core_allocation"]
            d["memory_operand_links"] = node_mapping["memory_operand_links"]

            if "temporal_ordering" in node_mapping:
                d["temporal_ordering"] = node_mapping["temporal_ordering"]
            if "spatial_mapping" in node_mapping:
                d["spatial_mapping"] = node_mapping["spatial_mapping"]
            if "spatial_mapping_hint" in node_mapping:
                d["spatial_mapping_hint"] = node_mapping["spatial_mapping_hint"]

                # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {"I": preds}

            # Add padding information

            d["padding"] = {
                "IY": (padding[0], padding[2]),
                "IX": (padding[1], padding[3]),
            }

            return d

        attrs = self.node.attribute
        # Find kernel shape in attrs
        kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
        # Find strides in attrs
        strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])
        # Find dilation rate in attrs
        dilations = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])
        # Find number of groups in attrs
        groups = get_attribute_ints_with_name("group", attrs, default=1)
        # Find padding in attrs
        padding = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # Get the input and output activation and weight data type (precision)
        ia_data_type, oa_data_type, w_data_type = get_input_output_weight_data_type(self.node, self.onnx_model)

        # Get the hw mapping of this node.
        if self.node.name in self.mapping:
            node_mapping = self.mapping[self.node.name]
        else:
            try:
                node_mapping = self.mapping["default"]
            except:
                raise ValueError(f"There is no mapping provided for node {self.node.name}, nor a default one.")

        # Take a deepcopy of the mapping, otherwise it will be changed for other layers if using default
        node_mapping = pickle_deepcopy(node_mapping)

        node_attrs = get_layer_node_input_format(
            kernel_shape,
            strides,
            dilations,
            groups,
            padding,
            ia_dimension_shape,
            oa_dimension_shape,
            node_mapping,
        )

        node_obj = LayerNode(
            self.node_id,
            # NOTE we first generate the layer attributes in user input format and then parse to `LayerAttributes`. This is redundant
            LayerAttributes.parse_user_input(node_attrs),
            node_name=self.node.name,
            layer_type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Conv node {self.node.name}")

        return node_obj
