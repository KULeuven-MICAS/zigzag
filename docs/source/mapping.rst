=======
Mapping
=======

The mapping defines how the algorithmic operations are mapped onto the computational hardware resources. The ZigZag framework automates (parts of) this mapping, but some aspects need to be (at the time of writing) user-defined. The mapping input file is required for running ZigZag in combination with the onnx interface. When manually defining your algorithmic layers, the mapping information is encoded within the workload definition (e.g. `inputs/workloads/resnet18.py <https://github.com/KULeuven-MICAS/zigzag/blob/master/zigzag/inputs/workload/resnet18.py>`_).

User-defined mapping constraints
================================


The mapping file should contain following aspects for every ONNX node that will be mapped onto the accelerator:

* **core_allocation**: The accelerator core id onto which this ONNX node is mapped (the id provided when creating the core in the hardware description file). Since ZigZag only supports single-core architectures, the core allocation must be set to 1.
* **spatial_mapping**: The spatial parallelization strategy to execute the node with (this can be automated through the ``SpatialMappingGeneratorStage``).
* **memory_operand_links**: The memory operand links, which link the memory operands (defined in the memory hierarchy of the core) to the layer operands (which are generated in the ONNXModelParserStage and are typically 'O', 'I', 'W' for a convolutional layer). This extra memory mapping is added to allow flexible memory allocation schemes. If left empty, a default value will be used instead.

* **core_allocation**: The accelerator core id onto which this ONNX node is mapped (the id provided when creating the core in the hardware description file). Since ZigZag only supports single-core architectures, the core allocation must be set to 1.
* **spatial_mapping**: The spatial parallelization strategy to execute the node with (this can be automated through the ``SpatialMappingGeneratorStage``).
* **memory_operand_links**: The memory operand links, which link the memory operands (defined in the memory hierarchy of the core) to the layer operands (which are generated in the ONNXModelParserStage and are typically 'O', 'I', 'W' for a convolutional layer). This extra memory mapping is added to allow flexible memory allocation schemes. If left empty, a default value will be used instead.
* **temporal_ordering**: The order in which the loop-based workload is scheduled on the accelerator. The given temporal ordering must be strictly complete. If the user-defined (or generated) spatial mapping together with the temporal ordering do not define the full workload for some layer, the temporal ordering is rejected altogether.

An example mapping is given below. The name of the mapping must correspond to an operation type (e.g. 'MatMul') or the exact name of a layer.

.. code-block:: yaml 

    - name: example_name_of_layer0
    core_allocation: [1]
    spatial_mapping:
        D1:
        - C, 32
        D2:
        - K, 32
    temporal_ordering:
        - [OX, 112] # Innermost loop
        - [OY, 112]
        - [FX, 7]
        - [FY, 7]
        - [K, 2] # Outermost loop
    memory_operand_links:
        O: O
        W: I2
        I: I1

A default entry can also be defined. This is useful if you don't know the exact ONNX node names, or don't want to customize this for every mapped node. The default entry is automatically detected under the ``default`` key of the mapping dictionary.
