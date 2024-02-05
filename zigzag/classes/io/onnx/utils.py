import enum
import importlib
import logging
from dataclasses import dataclass
from enum import auto
from os import path
from typing import List
import pickle

import onnx
from onnx import AttributeProto, helper, compose

logger = logging.getLogger(__name__)

BRANCH_ATTRIBUTE = "branch"

## Parse the input accelerator residing in accelerator_path.
# @param mapping_path
def parse_mapping_from_path(mapping_path):
    # Sanity check on mapping_path
    if mapping_path is None:
        # Update the mapping_path to the default mapping file
        if path.exists("inputs/examples/mapping/default.py"):
            mapping_path = "zigzag.inputs.examples.mapping.default"
        else:
            raise ValueError(
                "No mapping path/dict provided, and default was not found."
            )
    if "/" in mapping_path and mapping_path.split(".")[-1] in [
        "pickle",
        "pkl",
        "mapping",
    ]:
        # Load in the pickle mapping file
        with open(mapping_path, "rb") as fp:
            mapping = pickle.load(fp)
    else:
        global module
        module = importlib.import_module(mapping_path)
        mapping = module.mapping
    if "default" in mapping:
        default_present = "\u2705"
    else:
        default_present = "\u274C"
    logger.debug(
        f"Parsed mapping with {len(mapping)} different entries. Default: {default_present}."
    )
    return mapping


def parse_onnx_model_from_path(onnx_model_path):
    return onnx.load(onnx_model_path, load_external_data=False)

def add_attribute(node, name, value):
    attr = helper.make_attribute(name, value)
    if hasattr(node, "attribute"):
        node.attribute.append(attr)
    else:
        raise ValueError(f"{node} doesn't have an attribute field.")

def add_branch_attribute(graph, branch=0):
    for node in graph.node:
        add_attribute(node, BRANCH_ATTRIBUTE, branch)
        if node.op_type in ["If"]:
            g0 = node.attribute[0].g  # then or else branch
            g1 = node.attribute[1].g  # then or else branch
            add_branch_attribute(g0, branch + 1)
            add_branch_attribute(g1, branch + 1)

def unroll_branches(graph):
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
            # Add value info to originalg graph output if it's not present
            if input_name not in [vi.name for vi in graph.output]:
                graph.output.extend([value_info])
            # Add value info to subgraph input if it's not present
            if input_name not in [vi.name for vi in g0.input]:
                g0.input.extend([value_info])
            # g0 is the graph we will combine with the original one
            new_graph = compose.merge_graphs(graph, g0, io_map=[(input_name, input_name)])
            break
    return new_graph
            
def is_dynamic(model):
    return "If" in [n.op_type for n in model.graph.node]

## Modifies the given onnx model if there's dynamic behavior in terms of an 'If' operator.
# All nodes are assigned a 'branch' attribute which specifies in which branch they live.
# The branch attribute starts from 0 and increases for each seen If operator.
# The nested graphs of the 'If' operators are then unrolled into a planar onnx model.
def parse_dynamic_onnx_model(model):
    new_model = model
    if is_dynamic(model):
        add_branch_attribute(model.graph)
        pass
        graph = unroll_branches(model.graph)
        new_model = helper.make_model(graph)
    return new_model

## Retrieves the attrs[name_idx].ints from attrs.
# If attrs[name_idx] is of type INTS, attrs[name_idx].ints is returned.
# If attrs[name_idx] is of type INT, attrs[name_idx].i is returned.
# If name does not exist in attrs, the default provided by the caller is used.
# If the caller doesn't supply a default, an error is thrown.
# @param name
# @param attrs
# @param default
def get_attribute_ints_with_name(name, attrs, default=None):
    attrs_names = [attr.name for attr in attrs]
    try:
        name_idx = attrs_names.index(name)
        attr_type = attrs[name_idx].type
        if attr_type == AttributeProto.AttributeType.INT:
            return attrs[name_idx].i
        elif attr_type == AttributeProto.AttributeType.INTS:
            return attrs[name_idx].ints
        else:
            raise NotImplementedError(
                f"Attribute extraction of type {attr_type} not supported."
            )
    except ValueError:
        if default is not None:
            return default
        else:
            raise ValueError(
                f"attrs has no attribute called {name} and no default was given. Names = {attrs_names}."
            )


## Description missing
class OnnxTensorCategory(enum.Enum):
    Input = auto()
    Output = auto()
    Hidden = auto()
    Constant = auto()

    @property
    def is_output(self):
        return self == OnnxTensorCategory.Output

    @property
    def is_input(self):
        return self == OnnxTensorCategory.Input

    @property
    def is_hidden(self):
        return self == OnnxTensorCategory.Hidden

    @property
    def is_constant(self):
        return self == OnnxTensorCategory.Constant


@dataclass
## Description missing
class OnnxTensorType:
    shape: List[int]
    elem_type: int
    category: OnnxTensorCategory

    @staticmethod
    def from_tensor_type(tensor_type, category: OnnxTensorCategory):
        shape = [d.dim_value for d in tensor_type.shape.dim]
        elem_type = tensor_type.elem_type

        return OnnxTensorType(shape, elem_type, category)


def get_onnx_tensor_type(name, model):
    for input in model.graph.input:
        if input.name == name:
            return OnnxTensorType.from_tensor_type(
                input.type.tensor_type, OnnxTensorCategory.Input
            )

    for output in model.graph.output:
        if output.name == name:
            return OnnxTensorType.from_tensor_type(
                output.type.tensor_type, OnnxTensorCategory.Output
            )

    for value_info in model.graph.value_info:
        if value_info.name == name:
            return OnnxTensorType.from_tensor_type(
                value_info.type.tensor_type, OnnxTensorCategory.Hidden
            )

    for init in model.graph.initializer:
        if init.name == name:
            # initializers are represented a bit differently from other tensors
            return OnnxTensorType(
                list(init.dims), init.data_type, OnnxTensorCategory.Constant
            )

    raise KeyError(
        f""
        f"Could not find type for value {name} in model. "
        f"Make sure you are loading in an inferred model, "
        f"see https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model"
    )


def get_node_input_output_dimension_shapes(node, model):
    # assumed it is the first input, don't see a way to otherwise know
    input_name = node.input[0]
    input_shape = get_onnx_tensor_type(input_name, model).shape

    output_name = node.output[0]
    output_shape = get_onnx_tensor_type(output_name, model).shape

    return input_shape, output_shape
