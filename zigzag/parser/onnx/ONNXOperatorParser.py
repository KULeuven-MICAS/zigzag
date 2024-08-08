from abc import ABCMeta, abstractmethod
from typing import Any

from onnx import ModelProto, NodeProto

from zigzag.parser.onnx.utils import get_attribute_ints_with_name, get_onnx_tensor_type
from zigzag.workload.LayerNodeABC import LayerNodeABC


class ONNXOperatorParser(metaclass=ABCMeta):
    """! Abstract base class that represents a parser of an onnx operator. Example: Conv, MatMul, etc."""

    # These attribute names can be added to an ONNX node to change the data precision used in ZigZag
    CUSTOM_WEIGHT_SIZE_ATTR = "weight_size"
    CUSTOM_ACT_SIZE_ATTR = "act_size"
    CUSTOM_OUTPUT_SIZE_ATTR = "output_size"

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
    ) -> None:
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model

    @abstractmethod
    def run(self) -> LayerNodeABC:
        ...

    def get_input_output_weight_data_type(self):
        """! Return the data type of the input, output and weight tensors of this node."""
        input_name = self.node.input[0]
        output_name = self.node.output[0]
        weight_name = self.get_weight_name(self.node)

        input_elem_type = get_onnx_tensor_type(input_name, self.onnx_model).elem_type
        output_elem_type = get_onnx_tensor_type(output_name, self.onnx_model).elem_type
        weight_elem_type = get_onnx_tensor_type(weight_name, self.onnx_model).elem_type

        return input_elem_type, output_elem_type, weight_elem_type

    def get_weight_name(self, node: NodeProto):
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

    def get_node_predecessors(self) -> list[int]:
        """Compute node input sources"""
        predecessors: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)
        return predecessors

    def get_operand_source_user_format(self, predecessors: list[int]):
        """Set input source and indicate constant operands"""
        match len(predecessors):
            case 0:
                # No source operands -> all are considered constant
                return {"W": self.node_id, "I": self.node_id}
            case 1:
                # One source operand, one constant
                return {"W": self.node_id, "I": predecessors[0]}
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                return {"W": predecessors[1], "I": predecessors[0]}
            case _:
                raise ValueError("No more than 2 layer predecessors expected")

    def get_weight_precision(self):
        """Return the weight precision for this node.
        The weight precision of ONNX nodes can be customized by manually adding the attribute `CUSTOM_WEIGHT_SIZE_ATTR`
        to the node."""
        default = 8
        try:
            return get_attribute_ints_with_name(
                name=ONNXOperatorParser.CUSTOM_WEIGHT_SIZE_ATTR, attrs=self.node.attribute, default=default
            )
        except NotImplementedError as exc:
            raise ValueError("Custom weight size attribute must be an integer.") from exc

    def get_activation_precision(self):
        """Return the activation precision for this node.
        The activation precision of ONNX nodes can be customized by manually adding the attribute
         `CUSTOM_WEIGHT_SIZE_ATTR` to the node."""
        default = 8
        try:
            return get_attribute_ints_with_name(
                name=ONNXOperatorParser.CUSTOM_ACT_SIZE_ATTR, attrs=self.node.attribute, default=default
            )
        except NotImplementedError as exc:
            raise ValueError("Custom activation size attribute must be an integer.") from exc

    def get_intermediate_output_precision(self):
        """Return the intermediate output precision for this node.
        The intermediate output precision of ONNX nodes can be customized by manually adding the attribute
         `CUSTOM_OUTPUT_SIZE_ATTR` to the node."""
        default = 2 * self.get_activation_precision()
        try:
            return get_attribute_ints_with_name(
                name=ONNXOperatorParser.CUSTOM_OUTPUT_SIZE_ATTR, attrs=self.node.attribute, default=default
            )
        except NotImplementedError as exc:
            raise ValueError("Custom intermediate output size attribute must be an integer.") from exc
