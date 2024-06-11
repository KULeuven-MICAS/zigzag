=======
Mapping
=======

The mapping defines how the algorithmic operations are mapped onto the computational hardware resources. The ZigZag framework automates (parts of) this mapping, but some aspects need to be (at the time of writing) user-defined. The mapping input file is required for running ZigZag in combination with the onnx interface. When manually defining your algorithmic layers, the mapping information is encoded within the workload definition (e.g. `inputs/workloads/resnet18.py <https://github.com/KULeuven-MICAS/zigzag/blob/master/zigzag/inputs/workload/resnet18.py>`_).

User-defined mapping constraints
================================


The mapping file should contain following aspects for every ONNX node that will be mapped onto the accelerator:

* **core_allocation**: The accelerator core id onto which this ONNX node is mapped (the id provided when creating the core in the HW description file).
* **spatial_mapping**: The spatial parallelization strategy to execute the node with (this can be automated through the SpatialMappingGeneratorStage).
* **memory_operand_links**: The memory operand links, which link the memory operands (defined in the memory hierarchy of the core) to the layer operands (which are generated in the ONNXModelParserStage and are typically 'O', 'I', 'W' for a convolutional layer). This extra memory mapping is added to allow flexible memory allocation schemes, if you don't know what to put here, a safe bet is copying the example mapping file's memory mapping.


A default entry can also be defined. This is useful if you don't know the exact ONNX node names, or don't want to customize this for every mapped node. The default entry is automatically detected under the ``default`` key of the mapping dictionary.