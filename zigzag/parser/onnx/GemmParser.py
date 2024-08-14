from typing import Any

from onnx import ModelProto, NodeProto

from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
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

    def run(self) -> LayerNode:
        """! Run the parser"""
        return self.generate_layer_node()

    def get_layer_node_user_format(
        self,
        input_shape: list[int],
        output_shape: list[int],
    ) -> dict[str, Any]:
        """! Generate layer data in user input format for MatMul or GEMM ONNX nodes.
        I[B]   [D][C] * W([B])   [C][K]-> O [B]   [D][K] or
        I[B][H][D][C] * W([B][H])[C][K]-> O [B][H][D][K]

        """
        predecessors = self.get_node_predecessors()
        act_precision = self.get_activation_precision()
        weight_precision = self.get_weight_precision()
        intermediate_output_precision = self.get_intermediate_output_precision()
        # If there are 2 input nodes, `weights` represents a variable input
        weights_are_constant = len(predecessors) < 2

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["dimension_relations"] = []
        data["operand_source"] = self.get_operand_source_user_format(predecessors)
        data["operand_precision"] = {
            "W": weight_precision if weights_are_constant else act_precision,
            "I": act_precision,
            "O_final": act_precision,
            "O": intermediate_output_precision,
        }

        # I[B,H][D][C] * W[B,H][C][K] -> O[B,H][D][K]
        assert input_shape[-2] == output_shape[-2], "First dimension of input and output matrix must be the same"
        size_d = input_shape[-2]
        size_c = input_shape[-1]
        size_k = output_shape[-1]
        size_h = 0  # In transformers, this dimension represents the number of heads

        match len(input_shape):
            case 2:
                batch_size = 0
            case 3:
                assert input_shape[0] == output_shape[0], "Batch size should be the same for input and output"
                batch_size = 1 if input_shape[0] == 0 else input_shape[0]
            case 4:
                assert input_shape[0] == output_shape[0], "Batch size should be the same for input and output"
                assert input_shape[1] == output_shape[1], "Batch size (axis=1) should be the same for input and output"
                batch_size = 1 if input_shape[0] == 0 else input_shape[0]
                size_h = 0 if input_shape[1] == 0 else input_shape[1]
            case _:
                raise ValueError("Input size of GeMM or Matmul ONNX node must be either 2, 3 or 4.")

        # Construct sizes
        loop_dim_b_h = (["B"] if batch_size else []) + (["H"] if size_h else [])
        loop_size_b_h = ([batch_size] if batch_size else []) + ([size_h] if size_h else [])
        data["loop_dims"] = ["C", "D", "K"] + loop_dim_b_h
        data["loop_sizes"] = [size_c, size_d, size_k] + loop_size_b_h

        # Construct equation
        layer_dim_b_h = ("[b]" if batch_size else "") + ("[h]" if size_h else "")
        part_out = f"O{layer_dim_b_h}[d][k]"
        part_in = f"I{layer_dim_b_h}[d][c]"
        # No batch dimensions (B or H) if the weights are constant
        part_weight = f"W{'' if weights_are_constant else layer_dim_b_h}[c][k]"
        data["equation"] = f"{part_out}+={part_in}*{part_weight}"

        return data

    def generate_layer_node(self):
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)
        assert len(input_shape) == len(output_shape), "Input and output size expected to be the same"

        transpose_first_input = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        if transpose_first_input:
            assert len(input_shape) == 2, "Transpose only supported for GEMMs with two input dimensions"
            input_shape = [input_shape[1], input_shape[0]]

        # In case input tensor shape is unknown
        if not input_shape:
            input_shape = self.infer_input_activation_shape(output_shape)

        # Create LayerNode
        layer_data = self.get_layer_node_user_format(
            input_shape,
            output_shape,
        )
        factory = LayerNodeFactory(layer_data, self.mapping_data)
        layer_node = factory.create()

        return layer_node

    def infer_input_activation_shape(self, output_shape: list[int]) -> list[int]:
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
        assert len(output_shape) == 2, "Can't infer input_shape if output_shape is also not known."
        size_b = output_shape[0]
        size_c = weight_dims[0]
        input_shape = [size_b, size_c]
        return input_shape
