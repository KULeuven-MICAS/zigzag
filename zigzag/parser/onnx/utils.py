import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List

import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TypeProto, compose, helper, numpy_helper

logger = logging.getLogger(__name__)

BRANCH_ATTRIBUTE = "branch"


def parse_onnx_model_from_path(onnx_model_path: str) -> ModelProto:
    return onnx.load(onnx_model_path, load_external_data=False)  # type: ignore


def add_attribute(node: NodeProto, name: str, value: Any):
    attr = helper.make_attribute(name, value)
    if hasattr(node, "attribute"):
        node.attribute.append(attr)
    else:
        raise ValueError(f"{node} doesn't have an attribute field.")


def add_branch_attribute(graph: GraphProto, branch: int = 0):
    for node in graph.node:
        add_attribute(node, BRANCH_ATTRIBUTE, branch)
        if node.op_type in ["If"]:
            g0 = node.attribute[0].g  # then or else branch
            g1 = node.attribute[1].g  # then or else branch
            add_branch_attribute(g0, branch + 1)
            add_branch_attribute(g1, branch + 1)


def unroll_branches(graph: GraphProto) -> GraphProto:
    seen_if = False
    new_graph = graph
    for node in graph.node:
        if node.op_type in ["If"]:
            if seen_if:
                raise ValueError("Unrolling only implemented for single If operator in model.")
            seen_if = True
            g0 = node.attribute[0].g
            g1 = node.attribute[1].g
            if len(g1.node) > len(g0.node):
                g0, g1 = g1, g0
            # Recursively unroll any 'If' operator in the subgraph
            g0 = unroll_branches(g0)
            # Assume the first node in g0 is the only source node
            g0_source = g0.node[0]
            input_name = g0_source.input[0]
            value_info = next(i for i in graph.value_info if i.name == input_name)
            # Add value info to original graph output if it's not present
            if input_name not in [vi.name for vi in graph.output]:
                graph.output.extend([value_info])
            # Add value info to subgraph input if it's not present
            if input_name not in [vi.name for vi in g0.input]:
                g0.input.extend([value_info])
            # g0 is the graph we will combine with the original one
            new_graph = compose.merge_graphs(graph, g0, io_map=[(input_name, input_name)])
            break
    return new_graph


def is_dynamic(model: ModelProto):
    return "If" in [n.op_type for n in model.graph.node]


def parse_dynamic_onnx_model(model: ModelProto) -> ModelProto:
    """! Modifies the given onnx model if there's dynamic behavior in terms of an 'If' operator.
    All nodes are assigned a 'branch' attribute which specifies in which branch they live.
    The branch attribute starts from 0 and increases for each seen If operator.
    The nested graphs of the 'If' operators are then unrolled into a planar onnx model.
    """
    new_model = model
    if is_dynamic(model):
        add_branch_attribute(model.graph)
        pass
        graph = unroll_branches(model.graph)
        new_model = helper.make_model(graph)
    return new_model


def get_attribute_ints_with_name(name: str, attrs: Any, default: list[int] | int | None = None) -> list[int] | int:
    """! Return the value of an attribute of given name from the given attributes
    If name does not exist in attrs, the default provided by the caller is used.
    If the caller doesn't supply a default, an error is thrown.

    """
    attrs_names = [attr.name for attr in attrs]
    try:
        name_idx = attrs_names.index(name)
        value = attrs[name_idx]
        attr_type = value.type
        if attr_type == AttributeProto.AttributeType.INT:  # type: ignore
            return int(value.i)
        elif attr_type == AttributeProto.AttributeType.INTS:  # type: ignore
            return list(value.ints)
        elif attr_type == AttributeProto.AttributeType.TENSOR:  # type: ignore
            return list(numpy_helper.to_array(value.t).tolist())  # type: ignore
        else:
            raise NotImplementedError(f"Attribute extraction of type {attr_type} not supported.")
    except ValueError as exc:
        if default is not None:
            return default
        else:
            raise ValueError(
                f"attrs has no attribute called {name} and no default was given. Names = {attrs_names}."
            ) from exc


class OnnxTensorCategory(Enum):
    """Internal representation of ONNX tensor category"""

    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()
    CONSTANT = auto()

    @property
    def is_output(self):
        return self == OnnxTensorCategory.OUTPUT

    @property
    def is_input(self):
        return self == OnnxTensorCategory.INPUT

    @property
    def is_hidden(self):
        return self == OnnxTensorCategory.HIDDEN

    @property
    def is_constant(self):
        return self == OnnxTensorCategory.CONSTANT


@dataclass
class OnnxTensorType:
    shape: List[int]
    elem_type: int
    category: OnnxTensorCategory

    @staticmethod
    def from_tensor_type(tensor_type: TypeProto.Tensor, category: OnnxTensorCategory):
        shape = [d.dim_value for d in tensor_type.shape.dim]
        elem_type = tensor_type.elem_type

        return OnnxTensorType(shape, elem_type, category)


def get_onnx_tensor_type(name: str, model: ModelProto):
    for input_value in model.graph.input:
        if input_value.name == name:
            return OnnxTensorType.from_tensor_type(input_value.type.tensor_type, OnnxTensorCategory.INPUT)

    for output in model.graph.output:
        if output.name == name:
            return OnnxTensorType.from_tensor_type(output.type.tensor_type, OnnxTensorCategory.OUTPUT)

    for value_info in model.graph.value_info:
        if value_info.name == name:
            return OnnxTensorType.from_tensor_type(value_info.type.tensor_type, OnnxTensorCategory.HIDDEN)

    for init in model.graph.initializer:
        if init.name == name:
            # initializers are represented a bit differently from other tensors
            return OnnxTensorType(list(init.dims), init.data_type, OnnxTensorCategory.CONSTANT)

    raise KeyError(
        f""
        f"Could not find type for value {name} in model. "
        f"Make sure you are loading in an inferred model, "
        f"see https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model"
    )


def get_node_input_output_dimension_shapes(node: NodeProto, model: ModelProto):
    # assumed it is the first input, don't see a way to otherwise know
    input_name = node.input[0]
    input_shape = get_onnx_tensor_type(input_name, model).shape

    output_name = node.output[0]
    output_shape = get_onnx_tensor_type(output_name, model).shape

    return input_shape, output_shape
