# Mapping

The mapping file is required for zigzag runs with the ONNX interface. Where as before the mapping information was encoded within the workload definition (see e.g. classes/inputs/workloads/resnet18.py), this mapping information is no longer present in the parsed ONNX model, and thus must be defined in a mapping file.

---

The mapping file should contain for every ONNX node that will be mapped onto a part of the accelerator some specifics:
- The accelerator core id onto which this ONNX node is mapped (the id provided when creating the core in the HW description file).
- The spatial parallelization strategy to execute the node with (this can be automated through the SpatialMappingGeneratorStage).
- The memory operand links, which link the memory operands (defined in the memory hierarchy of the core) to the layer operands (which are generated in the ONNXModelParserStage and are typically 'O', 'I', 'W' for a convolutional layer). This extra memory mapping is added to allow flexible memory allocation schemes, if you don't know what to put here, a safe bet is copying the example mapping file's memory mapping.

---

A default entry can also be defined. This is useful if you don't know the exact ONNX node names, or don't want to customize this for every mapped node.