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
        layer_node = self.generate_layer_node_for_matmul()
        return layer_node

    def get_layer_node_input_format(
        self,
        batch_size: int,
        size_in: int,
        size_out: int,
        prev_node_id: int | None = None,
    ) -> dict[str, Any]:
        """! Generate the necessary dictionary items required for the Node creation.
        # TODO this is identical to the one from `GemmParser`
        """

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node_name
        data["operator_type"] = self.node.op_type
        data["equation"] = "O[b][k]+=W[k][c]*I[b][c]"
        data["loop_dims"] = ["B", "C", "K"]
        data["loop_sizes"] = [batch_size, size_in, size_out]

        data["dimension_relations"] = []
        data["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
        # Constant operand
        data["operand_source"] = {"W": self.node_id}
        if prev_node_id is not None:
            data["operand_source"]["I"] = prev_node_id

        return data

    def generate_layer_node_for_matmul(self):

        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # TODO it should be able to deal with tensors
        # First element is batch size, second is input/output channel
        assert len(ia_dimension_shape) == len(oa_dimension_shape) == 2
        # Batch size should be the same for input and output
        assert ia_dimension_shape[0] == oa_dimension_shape[0]
        # If the batch size is 0, we discard it by setting it to 1 internally inside ZigZag
        batch_size = ia_dimension_shape[0]
        size_b = 1 if batch_size == 0 else batch_size
        size_c = ia_dimension_shape[1]
        size_k = oa_dimension_shape[1]

        # Compute node input source
        predecessors: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)
        assert len(predecessors) <= 1, "Only a single layer operand source expected"
        prev_node_id = None if len(predecessors) == 0 else predecessors.pop()

        # Create LayerNode
        layer_data = self.get_layer_node_input_format(size_b, size_c, size_k, prev_node_id)
        factory = LayerNodeFactory(layer_data, self.mapping_data)
        layer_node = factory.create()

        return layer_node
