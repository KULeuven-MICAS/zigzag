from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes, get_attribute_ints_with_name
from zigzag.classes.workload.layer_node import LayerNode

import logging
logger = logging.getLogger(__name__)


class GemmParser(Parser):
    """Parses an ONNX Gemm operator into a LayerNode
    """
    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)
    
    def run(self):
        """Run the parser
        """
        layer_node = self.generate_layer_node_for_gemm()
        return layer_node

    def generate_layer_node_for_gemm(self):
    
        def get_layer_node_input_format(B, C, K, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the Node creation.
            """
            # convert the data types to precisions based on the onnx definition


            # Equation
            d = {}
            d["equation"] = 'O[b][k]+=B[k][c]*A[b][c]'

            # Get dimension sizes from input parameters
            K = K
            C = C
            B = B  # Not to be confused with operand 'B' which is the weights
            d["loop_dim_size"] = {'K': K, 'C': C, 'B': B}
            d["dimension_relations"] = []
            d["operand_precision"] =  {'O': 16, 'O_final': 8, 'B': 8, 'A': 8}
            d["operand_source"] =  {'B': [], 'A': []}
            d["constant_operands"] =  ['B']

            d["core_allocation"] =  node_mapping["core_allocation"]
            d["spatial_mapping"] =  node_mapping["spatial_mapping"]
            d["memory_operand_links"] =  {'O': 'O', 'B': 'I2', 'A': 'I1'}

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in nodes_outputs:
                    if node_input in nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {'A': preds}

            return d
        
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # The Gemm node includes flags for transpose of both of its inputs.
        # If the first input is transposed, we need to transpose its shape here.
        transA = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        if transA:
            assert len(ia_dimension_shape == 2)
            ia_dimension_shape = (ia_dimension_shape[1], ia_dimension_shape[0])

        assert len(ia_dimension_shape) == len(oa_dimension_shape) == 2  # First element is batch size, second is input/output channel
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
        node_obj = LayerNode(self.node_id, node_attrs, node_name=self.node.name)

        logger.info(f"Parsed Gemm node {self.node.name}")

        return node_obj
