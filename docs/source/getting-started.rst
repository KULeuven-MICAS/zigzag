===============
Getting Started
===============

Zigzag is a very powerful and versatile tool. It can be used to estimate and optimize the HW cost of running a DL workload on a given HW design under a multitude of constraints and settings. 

As a first step, we use it to automatically optimize the way a NN is mapped onto a HW design.

First run
=========

The NN we are going to use for this first run is AlexNet. We provide an `onnx <https://onnx.ai/>`_ model of this network in ``inputs/examples/workloads/alexnet_inferred.onnx``. The model has been shape inferred, which means that besied the input and output tensor shapes, all intermediate tensor shapes have been inferred, which is information required by zigzag. 

.. warning::
    Zigzag requires an inferred onnx model, as it needs to know the shapes of all intermediate tensors to correclty infer the layer shapes. You can find more information on how to infer an onnx model `here <https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model>`_.

The hardware the 

