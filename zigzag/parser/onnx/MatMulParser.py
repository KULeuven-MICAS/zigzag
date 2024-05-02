from typing import Any
import onnx
from onnx import ModelProto
from onnx import NodeProto

from zigzag.parser.onnx.Parser import Parser
from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.workload.layer_attributes import LayerAttributes
from zigzag.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)


class MatMulParser(Parser):
    """
    Parses an ONNX MatMul operator into a LayerNode
    """

    def run(self) -> LayerNode:
        """Run the parser"""
        layer_node = self.generate_layer_node_for_matmul()
        return layer_node

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        mapping: dict[str, dict[str, Any]] | None,
        onnx_model: ModelProto | None,
    ) -> None:
        assert mapping is not None
        assert onnx_model is not None
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def generate_layer_node_for_matmul(self):
        def get_layer_node_input_format(
            B: int, C: int, K: int, node_mapping: dict[str, Any], nodes_outputs: dict[int, Any]
        ) -> dict[str, Any]:
            """! Generate the necessary dictionary items required for the Node creation."""
            # convert the data types to precisions based on the onnx definition

            # Equation
            d: dict[str, Any] = {}
            d["equation"] = "O[b][k]+=B[k][c]*A[b][c]"

            # Get dimension sizes from input parameters
            K = K
            C = C
            B = B  # Not to be confused with operand 'B' which is the weights
            d["loop_dim_size"] = {"K": K, "C": C, "B": B}
            d["dimension_relations"] = []
            d["operand_precision"] = {"O": 16, "O_final": 8, "B": 8, "A": 8}
            d["operand_source"] = {"B": [], "A": []}
            d["constant_operands"] = ["B"]

            d["core_allocation"] = node_mapping["core_allocation"]
            d["memory_operand_links"] = {"O": "O", "B": "I2", "A": "I1"}

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
                for n in nodes_outputs:
                    if node_input in nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {"A": preds}

            return d

        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # TODO it should be able to deal with tensors
        assert (
            len(ia_dimension_shape) == len(oa_dimension_shape) == 2
        )  # First element is batch size, second is input/output channel
        assert ia_dimension_shape[0] == oa_dimension_shape[0]  # Batch size should be the same for input and output
        # If the batch size is 0, we discard it by setting it to 1 internally inside ZigZag
        batch_size = ia_dimension_shape[0]
        if batch_size == 0:
            B = 1
        else:
            B = batch_size
        C = ia_dimension_shape[1]
        K = oa_dimension_shape[1]

        # Get the hw mapping of this node.
        if self.node.name in self.mapping:
            node_mapping = self.mapping[self.node.name]
        else:
            try:
                node_mapping = self.mapping["default"]
            except:
                raise ValueError(f"There is no mapping provided for node {self.node.name}, nor a default one.")

        node_attrs = get_layer_node_input_format(B, C, K, node_mapping, self.nodes_outputs)
        node_obj = LayerNode(
            self.node_id,
            # NOTE we first generate the layer attributes in user input format and then parse to `LayerAttributes`. This is redundant
            LayerAttributes.parse_user_input(node_attrs),
            node_name=self.node.name,
            layer_type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed MatMul node {self.node.name}")  # pylint disable=W1203

        return node_obj
