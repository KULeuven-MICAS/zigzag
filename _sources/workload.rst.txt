========
Workload
========

The recommended way of defining an algorithmic workload is through an onnx model. An onnx model can contain multiple operator types, which in the context of ML are often referred to as different layers, some of which are automatically recognized and parsed by ZigZag. Alternatively, the layers can be manually defined for more customization.

Onnx models
===========

Supported onnx operators
------------------------

A complete list of all onnx operators can be found `here <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_.

The following operators are supported by ZigZag and will automatically be parsed into ``LayerNode`` objects when using your onnx model within the framework:

* `Conv <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv>`_
* `QLinearConv <https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv>`_
* `MatMul <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul>`_
* `Gemm <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_

All other operators will be parsed into a ``DummyNode`` object that is assumed to not be accelerateable, incurring zero hardware costs. If you have an onnx operator you would like to see supported, feel free to `open an issue <https://github.com/ZigZag-Project/zigzag/issues/new>`_ or manually add it yourself in the `ONNXModelParserStage <https://github.com/ZigZag-Project/zigzag/blob/8bce029a4284b720d8957357db74d629bd894dc6/classes/stages/ONNXModelParserStage.py#L314>`_ taking into account the :ref:`contributing guidelines`.

Controlling the quantization
----------------------------

To change the operand precision used in ZigZag, ONNX layers can be extended with a custom attribute to define the number of bits for weights, activations (in- and output) and intermediate output activations. The attribute name must correspondingly match ``weight_size``, ``act_size`` or ``output_size``. Attributes can be added as follows:

.. code:: python
    onnx_model = onnx.load(path)
    for node in onnx_model.graph.node:
        attr = onnx.helper.make_attribute("weight_size", 4) # 4bit weight quantization
        node.attribute.extend([attr])


Saving your onnx model with external data
-----------------------------------------

If your onnx model is rather large, and you want to avoid having it inside of your ZigZag repo, you can save the original model's weights as an external file, which can be discarded (ZigZag doesn't require the weight values but only the layers' shape). You can do so as follows:

.. code:: python

    import onnx
    # onnx_model is an in-memory ModelProto
    model = onnx.load('my_model_with_internal_data.onnx')
    onnx.save_model(model, 'path/to/save/the/model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='external_data_filename', size_threshold=1024, convert_attribute=False)
    # Then the onnx_model has converted raw data as external data and saved to a specific directory

Inferring an onnx model's shapes
--------------------------------

ZigZag requires an inferred onnx model, as it needs to know the shapes of all intermediate tensors to correctly infer the layer shapes. If you have an onnx model that is not shape inferred, you can do so by the following commands:

.. code:: python

    import onnx
    from onnx import shape_inference
    model = onnx.load("my_model.onnx")
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, "my_inferred_model.onnx")


Manual layer definition
=======================

It is also possible to manually define your own workload layers. In that case, the ``main.py`` file should be executed instead of ``main_onnx.py``. Moreover, the workload file should be provided as input together with the accelerator, and there is no onnx model loaded.

Each layer definition is represented as a YAML entry, which should have the following attributes:

* **id**: The identifier of the layer. This is important to correctly build the neural network graph edges.
* **operator_type**: The type of the layer. This can be linked to a specific mapping in the mapping file.
* **equation**: The operational equation for this layer. The dimensions should be small letters, whereas the operands are large letters. 'O' should always be used for the output operand; the input operands can be named freely.
* **dimension_relations**: The relationship between different dimensions present in the equation. This is often used in convolutional layers, where there is a relationship between the spatial input indices and the spatial output indices through the stride and with the filter indices through the dilation rate.
* **loop_dims**: A list of the different dimensions present in the equation. This should not include dimensions defined in the dimension_relations.
* **loop_sizes**: The size of the different dimensions present in the **loop_dims**.
* **operand_precision**: The bit precision of the different operands present in the equation. 'O' should always be used and represents the partial output precision. 'O_final' should always be used and represents the final output precision.
* **operand_source**: The layer id the input operands of this layer originate from. This should be set to the id of the current layer if it doesn't originate from prior layers. This information is used to correctly build the neural network graph edges.


The following loop notation is typically used to describe a layer of the workload (see loop notation in `this paper <https://ieeexplore.ieee.org/document/9360462>`_):

* **B**: Batch size
* **K**: Output channels
* **C**: Input channels
* **OY**: Output rows
* **OX**: Output columns
* **FY**: Kernel rows
* **FX**: Kernel columns

An example of this manual layer definition can be found at: `inputs/workloads/resnet18.yaml <https://github.com/KULeuven-MICAS/zigzag/blob/master/zigzag/inputs/workload/resnet18.yaml>`_. 
