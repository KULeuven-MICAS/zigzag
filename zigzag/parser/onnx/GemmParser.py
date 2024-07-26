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

    def run(self) -> LayerNode:
        """! Run the parser"""
        return self.generate_layer_node()

    def get_layer_node_user_format(
        self,
        batch_size: int,
        size_c: int,
        size_k: int,
        size_d: int,
    ) -> dict[str, Any]:
        """! Generate layer data in user input format for MatMul or GEMM ONNX nodes.
        I[B][K][C} * W[1 or B][C][D]-> O [B][K][D]
        """

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["loop_dims"] = ["B", "C", "K", "D"]
        data["loop_sizes"] = [batch_size, size_c, size_k, size_d]
        data["dimension_relations"] = []

        predecessors = self.get_node_predecessors()
        act_precision = self.get_activation_precision()
        weight_precision = self.get_weight_precision()
        intermediate_output_precision = self.get_intermediate_output_precision()

        # If there are 2 input nodes, `weights` represents a variable input
        weights_are_constant = len(predecessors) < 2

        data["equation"] = f"O[b][k][d]+=I[b][k][c]*W{'' if weights_are_constant else '[b]'}[c][d]"
        data["operand_precision"] = {
            "W": weight_precision if weights_are_constant else act_precision,
            "I": act_precision,
            "O_final": act_precision,
            "O": intermediate_output_precision,
        }

        match len(predecessors):
            case 0:
                # No source operands -> assume one is constant
                data["operand_source"] = {"W": self.node_id}
            case 1:
                # One source operand, one constant
                data["operand_source"] = {"W": self.node_id, "I": predecessors[0]}
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                data["operand_source"] = {"W": predecessors[1], "I": predecessors[0]}
            case _:
                raise ValueError("No more than 2 layer predecessors expected")

        return data

    def generate_layer_node(self):
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        transpose_first_input = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        if transpose_first_input:
            assert len(input_shape) == 2, "Transpose only supported for GEMMs with two input dimensions"
            input_shape = [input_shape[1], input_shape[0]]

        # In case input tensor shape is unknown
        if not input_shape:
            input_shape = self.infer_input_activation_shape(output_shape)

        assert len(input_shape) == len(output_shape), "Input and output size expected to be the same"
        assert input_shape[0] == output_shape[0], "Batch size should be the same for input and output"

        # I[B][K][C] * W[B][C][D] -> O[B][K][D]
        match len(input_shape):
            case 2:
                size_k = input_shape[0]
                size_c = input_shape[1]
                size_d = output_shape[1]
                batch_size = 1
            case 3:
                assert input_shape[1] == output_shape[1], "First dimension of input and output matrix must be the same"
                batch_size = 1 if input_shape[0] == 0 else input_shape[0]
                size_k = input_shape[1]
                size_c = input_shape[2]
                size_d = output_shape[2]
            case _:
                raise ValueError("Input size of Matmul ONNX node must be either 2 or 3.")

        # Create LayerNode
        layer_data = self.get_layer_node_user_format(
            batch_size=batch_size,
            size_c=size_c,
            size_k=size_k,
            size_d=size_d,
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
