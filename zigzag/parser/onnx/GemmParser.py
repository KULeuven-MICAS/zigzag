from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_node_input_output_dimension_shapes,
    get_attribute_ints_with_name,
)
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.workload.layer_node import LayerNode


class GemmParser(ONNXOperatorParser):
    """! Parses an ONNX Gemm operator into a LayerNode"""

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
        """! Run the parser"""
        return self.generate_layer_node()

    def generate_layer_node(self):
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        transpose_first_input = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        if transpose_first_input:
            assert len(ia_dimension_shape) == 2, "Transpose only supported for GEMMs with two input dimensions"
            ia_dimension_shape = [ia_dimension_shape[1], ia_dimension_shape[0]]

        # In case input tensor shape is unknown
        if not ia_dimension_shape:
            ia_dimension_shape = self.infer_input_activation_shape(oa_dimension_shape)

        assert len(ia_dimension_shape) == len(oa_dimension_shape), "Input and output size expected to be the same"
        assert ia_dimension_shape[0] == oa_dimension_shape[0], "Batch size should be the same for input and output"

        batch_size = 1 if ia_dimension_shape[0] == 0 else ia_dimension_shape[0]

        match len(ia_dimension_shape):
            # TODO: is it I*W->O or W*I->O?
            case 2:
                size_in = ia_dimension_shape[1]
                size_out = oa_dimension_shape[1]
                # No reduction dimension
                size_shared = 1
            case 3:
                # Input: batch_size x size_shared x size_in
                size_shared = ia_dimension_shape[1]
                size_in = ia_dimension_shape[2]
                # Output: batch_size x size_shared x size_out
                size_out = oa_dimension_shape[2]
                assert oa_dimension_shape[1] == size_shared
            case _:
                raise ValueError("Input size of Matmul ONNX node must be either 2 or 3.")

        # Create LayerNode
        layer_data = self.get_layer_node_user_format_gemm(
            batch_size=batch_size,
            size_in=size_in,
            size_out=size_out,
            size_shared=size_shared,
        )
        factory = LayerNodeFactory(layer_data, self.mapping_data)
        layer_node = factory.create()

        return layer_node

    def infer_input_activation_shape(self, oa_dimension_shape: list[int]) -> list[int]:
        """
        In case the input activations are empty (which can happen if there is a shape operator in the path):
        extract the weights from the model graph initializer to get the correct input activation size.
        NOTE: only implemented for tensors with dimension size 2.
        # TODO having a shape operator in the ONNX graph should be dealt with at a higher level
        """
        weight_name = self.node.input[1]
        initializer_names = [i.name for i in self.onnx_model.graph.initializer]
        weight_name_index = initializer_names.index(weight_name)
        # Get the weight dimensions
        weights = self.onnx_model.graph.initializer[weight_name_index]
        weight_dims = list(weights.dims)
        assert len(weight_dims) == 2, f"There are {len(weight_dims)} weight dimensions for Gemm node {self.node.name}"
        # Check if the weights are transposed
        transpose_second_input = get_attribute_ints_with_name("transB", self.node.attribute, default=0)
        if transpose_second_input:
            weight_dims = [weight_dims[1], weight_dims[0]]
        assert len(oa_dimension_shape) == 2, "Can't infer ia_dimension_shape if oa_dimension_shape is also not known."
        size_b = oa_dimension_shape[0]
        size_c = weight_dims[0]
        ia_dimension_shape = [size_b, size_c]
        return ia_dimension_shape
