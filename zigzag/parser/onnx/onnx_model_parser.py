import logging
from typing import Any, Type

from onnx import ModelProto, NodeProto

from zigzag.parser.onnx.conv_parser import ConvParser
from zigzag.parser.onnx.default_node_parser import DefaultNodeParser
from zigzag.parser.onnx.gemm_parser import GemmParser
from zigzag.parser.onnx.matmul_parser import MatMulParser
from zigzag.parser.onnx.onnx_operator_parser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    parse_dynamic_onnx_model,
    parse_onnx_model_from_path,
)
from zigzag.stages.parser.workload_parser import WorkloadParserStage
from zigzag.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)


class ONNXModelParser:
    """! Parses the ONNX model into a workload."""

    # Map the node's op_type to the corresponding Parser class
    PARSER_MAPPING: dict[str, Type[ONNXOperatorParser]] = {
        "QLinearConv": ConvParser,
        "Conv": ConvParser,
        "MatMul": MatMulParser,
        "Gemm": GemmParser,
    }

    def __init__(self, onnx_model: str | ModelProto, mapping_yaml_path: str) -> None:
        assert isinstance(onnx_model, (str, ModelProto)), f"Given onnx_model is of type {type(onnx_model)}."
        assert isinstance(mapping_yaml_path, str) and mapping_yaml_path.split(".")[-1] == "yaml"

        if isinstance(onnx_model, str):
            self.onnx_model: ModelProto = parse_onnx_model_from_path(onnx_model)
        else:
            self.onnx_model = onnx_model

        self.workload = None
        self.mapping_yaml_path = mapping_yaml_path

    def run(self) -> ONNXWorkload:
        """! Iterate through the onnx model and generate the workload consisting of LayerNodes and DummyNodes"""

        assert self.onnx_model is not None
        self.onnx_model = parse_dynamic_onnx_model(self.onnx_model)
        self.mapping_data = WorkloadParserStage.parse_mapping_data(self.mapping_yaml_path)

        return self.parse_workload_from_onnx_model_and_mapping()

    def get_parser_class(self, node: NodeProto):
        parser_class = ONNXModelParser.PARSER_MAPPING.get(node.op_type)
        if not parser_class:
            return DefaultNodeParser
        return parser_class

    def parse_workload_from_onnx_model_and_mapping(self):
        """! Converts an onnx model into a workload object.
        We scan the model for all convolutional layers, and setup a Layer object for each of those using the mapping.
        Then we combine the layers into a workload graph.

        If the model isn't in the format with external data, it will be slow to manipulate it, so better to work with
        raw models with external data. The line below accomplishes this.
        onnx.save_model(model, 'model_external.onnx', save_as_external_data=True, all_tensors_to_one_file=True,
        location='model_external_raw_data', size_threshold=1024, convert_attribute=False)

        In the future, assume we will have a model saved with external data, then we have to execute the code below
        if the model isn't inferred yet

        This approach is faster for large models because the raw model is used (w/o the external data)
        if model is not inferred:
        onnx.shape_inference.infer_shapes_path('path/to/the/model.onnx')  # This will save the inferred model to the
        same file
        model = onnx.load('path/to/the/model.onnx')  # reload the inferred model

        Saves for each node_id the inputs and outputs tensor names
        """
        nodes_inputs: dict[int, Any] = {}
        nodes_outputs: dict[int, Any] = {}

        # Workload Graph
        workload = ONNXWorkload()

        for node_id, node in enumerate(self.onnx_model.graph.node):  # type: ignore
            nodes_inputs[node_id] = node.input
            nodes_outputs[node_id] = node.output

            parser_class = self.get_parser_class(node)
            parser = parser_class(
                node_id=node_id,
                node=node,
                nodes_outputs=nodes_outputs,
                onnx_model=self.onnx_model,
                mapping_data=self.mapping_data,
            )

            node_obj = parser.run()
            # Add the node_obj to the ONNXWorkload
            workload.add(node_id, node_obj)

        logger.info(
            "Created ONNXWorkload graph with %i nodes and %i edges.",
            workload.number_of_nodes(),
            workload.number_of_edges(),  # type: ignore
        )

        return workload
