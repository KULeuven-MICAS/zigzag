from abc import ABCMeta, abstractmethod
from typing import Any

from onnx import ModelProto, NodeProto

from zigzag.parser.onnx.utils import get_onnx_tensor_type
from zigzag.workload.LayerNodeABC import LayerNodeABC


class ONNXOperatorParser(metaclass=ABCMeta):
    """! Abstract base class that represents a parser of an onnx operator. Example: Conv, MatMul, etc."""

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
    def run(self) -> LayerNodeABC: ...

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

    def get_node_name(self) -> str:
        if self.node.name != "":
            return self.node.name
        return f"Layer{self.node_id}"

    def get_node_predecessors(self) -> list[int]:
        """Compute node input source"""
        predecessors: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)
        # assert len(predecessors) <= 2, f"Unexpected number of layer node predecessors: {len(predecessors)}"
        return predecessors

    def get_layer_node_user_format_gemm(
        self,
        batch_size: int,
        size_in: int,
        size_out: int,
        size_shared: int,
    ) -> dict[str, Any]:
        """! Generate layer data in user input format for MatMul or GEMM ONNX nodes.
        W[size_out][size_in] * I[batch_size][size_in][size_shared] -> O [batch_size][size_out][size_shared]
        """

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = f"Layer{self.node_id}"
        data["operator_type"] = self.node.op_type
        data["equation"] = "O[b][k][d]+=W[k][c]*I[b][c][d]"
        data["loop_dims"] = ["B", "C", "K", "D"]
        data["loop_sizes"] = [batch_size, size_in, size_out, size_shared]

        data["dimension_relations"] = []
        data["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}

        # Operand sources
        predecessors = self.get_node_predecessors()
        match len(predecessors):
            case 0:
                # No source operands -> assume one is constant
                # TODO should this be 2?
                data["operand_source"] = {"W": self.node_id}
            case 1:
                # One source operand, one constant
                data["operand_source"] = {"W": self.node_id, "I": predecessors[0]}
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                data["operand_source"] = {"W": predecessors[0], "I": predecessors[1]}
            case _:
                raise ValueError("No more than 2 layer predecessors expected")

        return data
