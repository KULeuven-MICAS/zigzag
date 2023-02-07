from math import ceil

from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_attribute_ints_with_name, get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.utils import pickle_deepcopy

import logging
logger = logging.getLogger(__name__)


class ConvParser(Parser):
    """Parser for ONNX Conv and QLinearConv nodes into LayerNode.
    """
    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)
    
    def run(self):
        """Run the parser and return the created LayerNode object.
        """
        layer_node = self.generate_layer_node_for_conv()
        return layer_node

    def generate_layer_node_for_conv(self):

        def get_weight_name(node):
            """Return the name of the weight input of this node depending on its operator type.
            Args:
                node (NodeProto): The node
            """
            op_type = node.op_type  # 'Conv', 'QLinearConv', ...
            if op_type == "Conv":
                return node.input[1]
            elif op_type == "QLinearConv":
                return node.input[3]
            else:
                raise NotImplementedError(f"Retrieving weight name for onnx node of type {op_type} is not supported.")


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
            w_name = get_weight_name(node)
            # w_name = node.input[3]
            # Get the weight data type through the graph initializers
            initializer_names = [i.name for i in model.graph.initializer]
            w_data_type = model.graph.initializer[initializer_names.index(w_name)].data_type

            return ia_elem_type, oa_elem_type, w_data_type


        def get_layer_node_input_format(kernel_shape, strides, dilations, groups, padding, ia_shape, oa_shape, node_mapping):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # convert the data types to precisions based on the onnx definition


            # Equation
            d = {}
            # IMPORTANT: If any of the input loops require padding, they should be defined as the rightmost dimensions in the equation
            # This is because we construct the dimensionality order and then add the padding to those last dimensions in the order
            d["equation"] = 'O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]'

            # Get dimension sizes from input parameters
            assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
            B = oa_shape[0]
            if B == 0:
                B = 1
            G = groups
            K = ceil(oa_shape[1]/G)
            OX = oa_shape[2]
            OY = oa_shape[3]
            C = ceil(ia_shape[1]/G)
            IX = ia_shape[2]
            IY = ia_shape[3]
            FX = kernel_shape[0]
            FY = kernel_shape[1]
            d["loop_dim_size"] = {'B': B, 'K': K, 'G': G, "OX": OX, "OY": OY, "C": C, "FX": FX, "FY": FY}
            d["pr_loop_dim_size"] = {'IX': IX, 'IY': IY}
            d["dimension_relations"] = [f'ix={strides[0]}*ox+{dilations[0]}*fx', f'iy={strides[1]}*oy+{dilations[1]}*fy']
            d["operand_precision"] =  {'O': 16, 'O_final': 8, 'W': 8, 'I': 8}
            # d["operand_source"] =  {'W': [], 'I': []}
            d["constant_operands"] =  ['W']

            d["core_allocation"] =  node_mapping["core_allocation"]
            d["spatial_mapping"] =  node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] =  node_mapping["memory_operand_links"]

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {'I': preds}

            # Add padding information
            
            d["padding"] = {'IY': (padding[0], padding[2]), 'IX': (padding[1], padding[3])}

            return d


        attrs = self.node.attribute
        # Find kernel shape in attrs
        kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
        # Find strides in attrs
        strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])
        # Find dilation rate in attrs
        dilations = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])
        # Find number of groups in attrs
        groups = get_attribute_ints_with_name("group", attrs, default=1)
        # Find padding in attrs
        padding = get_attribute_ints_with_name("pads", attrs, default=[0,0,0,0])
        
        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # Get the input and output activation and weight data type (precision)
        ia_data_type, oa_data_type, w_data_type = get_input_output_weight_data_type(self.node, self.onnx_model)

        # Get the hw mapping of this node. 
        if self.node.name in self.mapping:
            node_mapping = self.mapping[self.node.name]
        else:
            try:
                node_mapping = self.mapping["default"]
            except:
                raise ValueError(f"There is no mapping provided for node {self.node.name}, nor a default one.")

        # Take a deepcopy of the mapping, otherwise it will be changed for other layers if using default
        node_mapping = pickle_deepcopy(node_mapping)

        node_attrs = get_layer_node_input_format(kernel_shape, strides, dilations, groups, padding,
                                                ia_dimension_shape, oa_dimension_shape,
                                                node_mapping)

        node_obj = LayerNode(self.node_id, node_attrs, node_name=self.node.name)
        
        logger.info(f"Parsed Conv node {self.node.name}")

        return node_obj
        
