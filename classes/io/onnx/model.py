from classes.io.onnx.default import DefaultNodeParser
from classes.io.onnx.gemm import GemmParser
from classes.io.onnx.matmul import MatMulParser
from classes.io.onnx.conv import ConvParser
from classes.io.onnx.utils import parse_mapping_from_path, parse_onnx_model_from_path
from classes.workload.onnx_workload import ONNXWorkload


import logging
logger = logging.getLogger(__name__)


class ONNXModelParser:
    """Parse the ONNX model into a workload.
    """
    def __init__(self, onnx_model_path, mapping_path, ) -> None:
        self.onnx_model_path = onnx_model_path
        self.mapping_path = mapping_path

        self.onnx_model = None
        self.mapping = None
        self.workload = None

    def run(self):
        """Run the parser:
        - parse the onnx_model_path into an onnx model
        - parse the mapping_path into a mapping dict
        - iterate through the onnx model and generate the workload consisting of LayerNodes and DummyNodes
        """
        onnx_model = parse_onnx_model_from_path(self.onnx_model_path)
        self.onnx_model = onnx_model

        mapping = parse_mapping_from_path(self.mapping_path)
        self.mapping = mapping

        workload = self.parse_workload_from_onnx_model_and_mapping()
        self.workload = workload

    def parse_workload_from_onnx_model_and_mapping(self):
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

        for node_id, node in enumerate(self.onnx_model.graph.node):
            nodes_inputs[node_id] = node.input
            nodes_outputs[node_id] = node.output

            if node.op_type in ["QLinearConv", "Conv"]:
                parser = ConvParser(node_id, node, nodes_outputs, self.mapping, self.onnx_model)
            elif node.op_type in ["MatMul"]:
                parser = MatMulParser(node_id, node, nodes_outputs, self.mapping, self.onnx_model)
            elif node.op_type in ["Gemm"]:
                parser = GemmParser(node_id, node, nodes_outputs, self.mapping, self.onnx_model)
            else:  # it is not a convolutional node, so create a DummyNode
                parser = DefaultNodeParser(node_id, node, nodes_outputs)
            node_obj = parser.run()
            # Add the node_obj to the ONNXWorkload
            workload.add(node_id, node_obj)

        logger.info(
            f"Created ONNXWorkload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges.")

        return workload

    def get_onnx_model(self):
        return self.onnx_model
    
    def get_mapping(self):
        return self.mapping
    
    def get_workload(self):
        return self.workload