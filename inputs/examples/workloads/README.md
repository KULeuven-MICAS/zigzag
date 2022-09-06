# Workload definition

This folder contains example NN workloads. 

## ONNX Model Interface

The easiest way to import a workload is to export it from your favourite tool to an inferred [ONNX](https://onnx.ai/) model. You can find more information about running shape inference for an ONNX model [here](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model). 

Running zigzag with an ONNX model can be done through the `main_onnx.py` file. This additionally requires a mapping file to be provided, which is explained in [inputs/examples/mapping](https://github.com/ZigZag-Project/zigzag/tree/master/inputs/examples/mapping).


## Manual definition

Alternatively, you can manually define your workload, as is done in e.g. [resnet18.py](https://github.com/ZigZag-Project/zigzag/blob/master/inputs/examples/workloads/resnet18.py). Here, each layer is an entry in the workload dict, with the different fields representing the encoded aspects of the layer.

Each layer should include:
- **equation**: The operational equation for this layer. The dimensions should be small letters, where as the operands are large letters. 'O' should always be used for the output operand, the input operands can be named freely.
- **dimension_relations**: The relationship between different dimensions present in the equation. This is often used in convolutional layers, where there is a relationship between the spatial input indices and the spatial output indices through the stride and with the filter indices through the dilation rate.
- **loop_dim_size**: The size of the different dimensions present in the equation. Dimensions defined (i.e. on the left hand side) in the dimension_relations are not to be provided and are inferred automatically.
- **operand_precision**: The bit precision of the different operands present in the equation. 'O' should always be used, which represents the partial output precision. 'O_final' represents the final output precision.
- **operand_source**: The layer id the input operands of this layer come from. This is important to correctly build the NN graph edges.
- **constant_operands**: The operands of this layer which are constants and do not depend on prior computations.
- **core_allocation**: The core that will execute this layer.
- **spatial_mapping**: The spatial parallelization strategy used for this layer. If none is provided, the SpatialMappingGeneratorStage should be used within zigzag's execution pipeline.
- **memory_operand_links**: The link between the virtual memory operands and the actual algorithmic operands. For more information, read the hardware readme.