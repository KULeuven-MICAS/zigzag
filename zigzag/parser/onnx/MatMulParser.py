from typing import Any
from onnx import ModelProto
from onnx import NodeProto

from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)


class MatMulParser(ONNXOperatorParser):
    """! Parses an ONNX MatMul operator into a LayerNode.
    # TODO this is identical to GemmParser
    """

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
        """Run the parser"""
        return self.generate_layer_node()

    def generate_layer_node(self):
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)
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
