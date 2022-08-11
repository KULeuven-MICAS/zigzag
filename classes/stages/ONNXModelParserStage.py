from typing import Generator
import importlib
import onnx

from classes.stages.Stage import Stage
from classes.workload.dummy_layer_node import DummyNode
from classes.workload.layer_node import LayerNode
from classes.workload.onnx_workload import ONNXWorkload

import logging
logger = logging.getLogger(__name__)


def parse_mapping_from_path(mapping_path):
    """
    Parse the input accelerator residing in accelerator_path.
    """
    global module
    module = importlib.import_module(mapping_path)
    mapping = module.mapping
    if "default" in mapping:
        logger_str = ""
    else:
        logger_str = "not "
    logger.info(f"Parsed mapping with {len(mapping)} different entries. Default is {logger_str}present.")
    return mapping

def parse_onnx_model_from_path(onnx_model_path):
    return onnx.load(onnx_model_path)


def get_attribute_ints_with_name(name, attrs, default=None):
    """
    Retrieves the attrs[name_idx].ints from attrs.
    If it does not exist, the default provided by the caller is used.
    If the caller doesn't supply a default, an error is thrown.
    """
    attrs_names = [attr.name for attr in attrs]
    try:
        name_idx = attrs_names.index(name)
        return attrs[name_idx].ints
    except ValueError:
        if default is not None:
            return default
        else:
            raise ValueError(f"attrs has no attribute called {name}. Names = {attrs_names}.")


def get_node_input_output_dimension_shapes(node, model):
        value_info = model.graph.value_info
        if not value_info:
            raise ValueError("value_info of model is empty. Make sure you are loading in an inferred model." \
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

        # For some reason there can be 0-valued elements in the list. Prune them
        ia_dimension_shape = [val for val in ia_dimension_shape if val != 0]
        oa_dimension_shape = [val for val in oa_dimension_shape if val != 0]

        return ia_dimension_shape, oa_dimension_shape


def generate_layer_node_for_qlinearconv(node_id, node, nodes_outputs, mapping, onnx_model):

    def get_input_output_weight_data_type(node, model):
        """
        Return the data type of the input, output and weight tensors of this node.
        """
        value_info = model.graph.value_info
        if not value_info:
            raise ValueError("value_info of model is empty. Make sure you are loading in an inferred model." \
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
            ia_elem_type = model.graph.input[ia_index].type.tensor_type.elem_type
        else:  # it should be present in the shapes variable as it's not an input or output of the model
            ia_index = shapes_names.index(ia_name)
            ia_elem_type = value_info[ia_index].type.tensor_type.elem_type

        # repeat the same for the output activation of this layer
        oa_name = node.output[0]
        if oa_name in model_output_names:
            oa_index = model_output_names.index(oa_name)
            oa_elem_type = model.graph.output[oa_index].type.tensor_type.elem_type
        else:
            oa_index = shapes_names.index(oa_name)
            oa_elem_type = value_info[oa_index].type.tensor_type.elem_type

        # Get the weight name for this node (for QLinearConv this is the fourth input)
        w_name = node.input[3]
        # Get the weight data type through the graph initializers
        initializer_names = [i.name for i in model.graph.initializer]
        w_data_type = model.graph.initializer[initializer_names.index(w_name)].data_type

        return ia_elem_type, oa_elem_type, w_data_type

    def get_layer_node_input_format(kernel_shape, strides, dilations, ia_shape, oa_shape, ia_data_type, oa_data_type, w_data_type, node_mapping, node_outputs):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        # convert the data types to precisions based on the onnx definition


        # Equation
        d = {}
        d["equation"] = 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]'

        # Get dimension sizes from input parameters
        assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
        B = oa_shape[0]
        K = oa_shape[1]
        OX = oa_shape[2]
        OY = oa_shape[3]
        C = ia_shape[1]
        IX = ia_shape[2]
        IY = ia_shape[3]
        FX = kernel_shape[0]
        FY = kernel_shape[1]
        d["loop_dim_size"] = {'B': B, 'K': K, "OX": OX, "OY": OY, "C": C, "FX": FX, "FY": FY}
        d["dimension_relations"] = [f'ix={strides[0]}*ox+{dilations[0]}*fx', f'iy={strides[1]}*oy+{dilations[1]}*fy']
        d["operand_precision"] =  {'O': 16, 'O_final': 8, 'W': 8, 'I': 8}
        # d["operand_source"] =  {'W': [], 'I': []}
        d["constant_operands"] =  ['W']

        d["core_allocation"] =  node_mapping["core_allocation"]
        d["spatial_mapping"] =  node_mapping["spatial_mapping"]
        d["memory_operand_links"] =  node_mapping["memory_operand_links"]

        # Find the previous layer(s) that should be this node's parent(s)
        node_inputs = node.input
        preds = []
        for node_input in node_inputs:
            for n in nodes_outputs:
                if node_input in nodes_outputs[n]:
                    preds.append(n)
        d["operand_source"] = {'I': preds}

        return d


    attrs = node.attribute
    # Find index of attrs that contains kernel shape
    # TODO: Might always be 2, should check
    kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
    # Find index of attrs that contains strides
    # TODO: Might always be , should check
    strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])
    # Find index of attrs that contain dilation rate
    # TODO: Might always be X, should check
    dilations = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])
    
    # Get the input and output activation shapes
    ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(node, onnx_model)

    # Get the input and output activation and weight data type (precision)
    ia_data_type, oa_data_type, w_data_type = get_input_output_weight_data_type(node, onnx_model)

    # Get the hw mapping of this node. 
    if node.name in mapping:
        node_mapping = mapping[node.name]
    else:
        try:
            node_mapping = mapping["default"]
        except:
            raise ValueError(f"There is no mapping provided for node {node.name}, nor a default one.")

    node_attrs = get_layer_node_input_format(kernel_shape, strides, dilations, 
                                            ia_dimension_shape, oa_dimension_shape, ia_data_type, oa_data_type, w_data_type,
                                            node_mapping, nodes_outputs)

    node_obj = LayerNode(node_id, node_attrs)
    
    logger.info(f"Parsed QLinearConv node {node.name}")

    return node_obj
    

def generate_layer_node_for_matmul(node_id, node, nodes_outputs, mapping, onnx_model):
    
    def get_layer_node_input_format(C, K, node_mapping, nodes_outputs):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        # convert the data types to precisions based on the onnx definition


        # Equation
        d = {}
        d["equation"] = 'O[k]+=B[k][c]*A[c]'

        # Get dimension sizes from input parameters
        K = K
        C = C
        d["loop_dim_size"] = {'K': K, 'C': C}
        d["dimension_relations"] = []
        d["operand_precision"] =  {'O': 16, 'O_final': 8, 'B': 8, 'A': 8}
        d["operand_source"] =  {'B': [], 'A': []}
        d["constant_operands"] =  ['B']

        d["core_allocation"] =  node_mapping["core_allocation"]
        d["spatial_mapping"] =  node_mapping["spatial_mapping"]
        d["memory_operand_links"] =  node_mapping["memory_operand_links"]

        # Find the previous layer(s) that should be this node's parent(s)
        node_inputs = node.input
        preds = []
        for node_input in node_inputs:
            for n in nodes_outputs:
                if node_input in nodes_outputs[n]:
                    preds.append(n)
        d["operand_source"] = {'A': preds}

        return d
    
    ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(node, onnx_model)

    assert len(ia_dimension_shape) == len(oa_dimension_shape) == 1  # Could also be larger for different MatMuls but assuming this for now
    C = ia_dimension_shape[0]
    K = oa_dimension_shape[0]

    # Get the hw mapping of this node. 
    if node.name in mapping:
        node_mapping = mapping[node.name]
    else:
        try:
            node_mapping = mapping["default"]
        except:
            raise ValueError(f"There is no mapping provided for node {node.name}, nor a default one.")

    node_attrs = get_layer_node_input_format(C, K, node_mapping, nodes_outputs)
    node_obj = LayerNode(node_id, node_attrs)

    return node_obj


def generate_dummy_node(node_id, node, nodes_outputs):
    preds = []
    for node_input in node.input:
        for n in nodes_outputs:
            if node_input in nodes_outputs[n]:
                preds.append(n)
    
    node_obj = DummyNode(node_id, preds)

    return node_obj

def parse_workload_from_onnx_model_and_mapping(onnx_model, mapping):
    """
    Converts an onnx model into a workload object.
    We scan the model for all convolutional layers, and setup a Layer object for each of those using the mapping.
    Then we combine the layers into a workload graph.
    """

    # If the model isn't in the format with external data, it will be slow to manipulate it, so better to work with raw models with external data
    # The line below accomplishes this.
    # onnx.save_model(model, 'model_external.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='model_external_raw_data', size_threshold=1024, convert_attribute=False)

    # In the future, assume we will have a model saved with external data, then we have to execute the code below
    # if the model isn't inferred yet
    # This approach is faster for large models because the raw model is used (w/o the external data)
    # if model is not inferred:
    #   onnx.shape_inference.infer_shapes_path('path/to/the/model.onnx')  # This will save the inferred model to the same file
    #   model = onnx.load('path/to/the/model.onnx')  # reload the inferred model

    # Saves for each node_id the inputs and outputs tensor names
    nodes_inputs = {}
    nodes_outputs = {}

    # Workload Graph
    workload = ONNXWorkload()

    for node_id, node in enumerate(onnx_model.graph.node):
        nodes_inputs[node_id] = node.input
        nodes_outputs[node_id] = node.output

        if node.op_type in ["QLinearConv"]:
            node_obj = generate_layer_node_for_qlinearconv(node_id, node, nodes_outputs, mapping, onnx_model)
        elif node.op_type in ["MatMul"]:
            node_obj = generate_layer_node_for_matmul(node_id, node, nodes_outputs, mapping, onnx_model)
        else:  # it is not a convolutional node, so create a DummyNode
            node_obj = generate_dummy_node(node_id, node, nodes_outputs)

        # Add the node_obj to the ONNXWorkload
        workload.add(node_id, node_obj)

    logger.info(
        f"Created ONNXWorkload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges.")

    return workload


class ONNXModelParserStage(Stage):
    def __init__(self, list_of_callables, *, onnx_model_path, mapping_path, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.onnx_model_path = onnx_model_path
        self.mapping_path = mapping_path
    
    def run(self) -> Generator:
        onnx_model = parse_onnx_model_from_path(self.onnx_model_path)
        mapping = parse_mapping_from_path(self.mapping_path)
        workload = parse_workload_from_onnx_model_and_mapping(onnx_model, mapping)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    # # For testing purposes
    # def is_leaf(self) -> bool:
    #     return True
