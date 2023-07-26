import enum
import importlib
import logging
from dataclasses import dataclass
from enum import auto
from os import path
from typing import List

import onnx
from onnx import AttributeProto

logger = logging.getLogger(__name__)


def parse_mapping_from_path(mapping_path):
    """
    Parse the input accelerator residing in accelerator_path.
    """
    # Sanity check on mapping_path
    if mapping_path is None:
        # Update the mapping_path to the default mapping file
        if path.exists("inputs/examples/mapping/default.py"):
            mapping_path = "zigzag.inputs.examples.mapping.default"
        else:
            raise ValueError(
                "No mapping path/dict provided, and default was not found."
            )
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


def get_attribute_ints_with_name(name, attrs, default=None):
    """
    Retrieves the attrs[name_idx].ints from attrs.
    If attrs[name_idx] is of type INTS, attrs[name_idx].ints is returned.
    If attrs[name_idx] is of type INT, attrs[name_idx].i is returned.
    If name does not exist in attrs, the default provided by the caller is used.
    If the caller doesn't supply a default, an error is thrown.
    """
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
            return OnnxTensorType.from_tensor_type(input.type.tensor_type, OnnxTensorCategory.Input)

    for output in model.graph.output:
        if output.name == name:
            return OnnxTensorType.from_tensor_type(output.type.tensor_type, OnnxTensorCategory.Output)

    for value_info in model.graph.value_info:
        if value_info.name == name:
            return OnnxTensorType.from_tensor_type(value_info.type.tensor_type, OnnxTensorCategory.Hidden)

    for init in model.graph.initializer:
        if init.name == name:
            # initializers are represented a bit differently from other tensors
            return OnnxTensorType(list(init.dims), init.data_type, OnnxTensorCategory.Constant)

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
