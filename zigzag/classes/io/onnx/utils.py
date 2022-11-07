import importlib
from os import path
import onnx
from onnx import AttributeProto

import logging
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
            raise ValueError("No mapping path/dict provided, and default was not found.")
    global module
    module = importlib.import_module(mapping_path)
    mapping = module.mapping
    if "default" in mapping:
        default_present = "\u2705"
    else:
        default_present = "\u274C"
    logger.debug(f"Parsed mapping with {len(mapping)} different entries. Default: {default_present}.")
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
            raise NotImplementedError(f"Attribute extraction of type {attr_type} not supported.")
    except ValueError:
        if default is not None:
            return default
        else:
            raise ValueError(f"attrs has no attribute called {name} and no default was given. Names = {attrs_names}.")

from onnx import AttributeProto

def get_node_input_output_dimension_shapes(node, model):
        value_info = model.graph.value_info
        if not value_info:
            raise ValueError("value_info of model is empty. Make sure you are loading in an inferred model. " \
            "See https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model")
        # get tensor names of the inputs and outputs of the model
        model_input_names = [input.name for input in model.graph.input]
        model_output_names = [output.name for output in model.graph.output]
        # get tensor names of the tensors in shapes
        shapes_names = [shape.name for shape in value_info]
        # get input and output activation dimension sizes
        # get input activation name
        ia_name = node.input[0]  # assumed it is the first input, don't see a way to otherwise know
        # check if this is a global input of the model, if so, retrieve dimension shape from model inputs
        if ia_name in model_input_names:
            # Find index of this input in list of input names
            ia_index = model_input_names.index(ia_name)
            ia_dimension_shape = [dim.dim_value for dim in model.graph.input[ia_index].type.tensor_type.shape.dim]
        else:  # it should be present in the shapes variable as it's not an input or output of the model
            ia_index = shapes_names.index(ia_name)
            ia_dimension_shape = [dim.dim_value for dim in value_info[ia_index].type.tensor_type.shape.dim]

        # repeat the same for the output activation of this layer
        oa_name = node.output[0]
        if oa_name in model_output_names:
            oa_index = model_output_names.index(oa_name)
            oa_dimension_shape = [dim.dim_value for dim in model.graph.output[oa_index].type.tensor_type.shape.dim]
        else:
            oa_index = shapes_names.index(oa_name)
            oa_dimension_shape = [dim.dim_value for dim in value_info[oa_index].type.tensor_type.shape.dim]

        return ia_dimension_shape, oa_dimension_shape