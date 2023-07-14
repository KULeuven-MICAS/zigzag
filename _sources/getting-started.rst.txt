===============
Getting Started
===============

ZigZag is a versatile tool. It can be used to estimate and optimize the HW cost of running a DL workload on a given HW design under a multitude of constraints and settings. 

As a first step, we use it to automatically optimize the way a NN is mapped onto a HW design.

First run
=========

The NN we are going to use for this first run is AlexNet. We provide an `onnx <https://onnx.ai/>`_ model of this network in ``zigzag/inputs/examples/workload/alexnet.onnx``. The model has been shape inferred, which means that besied the input and output tensor shapes, all intermediate tensor shapes have been inferred, which is information required by ZigZag. 

.. warning::
    ZigZag requires an inferred onnx model, as it needs to know the shapes of all intermediate tensors to correctly infer the layer shapes. You can find more information on how to infer an onnx model `here <https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model>`_.

The hardware we will accelerate the inference of AlexNet with, is a TPU-like architecture defined in ``zigzag/inputs/examples/hardware/TPU_like.py``. 

Besides the workload and HW architecture, a mapping file must be provided which, as the name suggests, provides information on how the network's layers will be mapped onto the hardware resources. The mapping is provided in ``zigzag/inputs/examples/mapping/alexnet_on_tpu_like.py``. 

The framework is generally ran through a main file which parses the provided inputs and contains the program flow through the stages defined in the main file. 

.. note::

    You can find more information in the :doc:`stages` document.


The following command starts the execution using the provided inputs:

.. code:: sh

    python main_onnx.py --model zigzag/inputs/examples/workload/alexnet.onnx --accelerator zigzag.inputs.examples.hardware.TPU_like --mapping zigzag.inputs.examples.mapping.tpu_like
.. note::

    Note the difference in input path construction between the onnx model and the accelerator and mapping. This is because the accelerator and mapping objects are defined in their respective files and imported as python modules.

.. raw:: html

    <script id="asciicast-zdbEqPoE4odh1QsqAIVSWPi6M" src="https://asciinema.org/a/zdbEqPoE4odh1QsqAIVSWPi6M.js" async></script>

Other ZigZag runs examples:

- ZigZag can also run with user-defined workload (see section manual layer definition section in :doc:`workload`):

.. code:: sh

    python main.py --model zigzag.inputs.examples.workload.resnet18 --accelerator zigzag.inputs.examples.hardware.TPU_like --mapping zigzag.inputs.examples.mapping.tpu_like

- ZigZag can also run with `SALSA temporal mapping search engine <https://arxiv.org/pdf/2304.12931>`_ which utilizes a different scheduler than the `LOMA scheduler <https://ieeexplore.ieee.org/document/9458493>`_:

.. code:: sh

    python main_onnx_salsa.py --model zigzag/inputs/examples/workload/alexnet.onnx --accelerator zigzag.inputs.examples.hardware.TPU_like --mapping zigzag.inputs.examples.mapping.tpu_like

Analyzing results
=================

During the run, results will be saved depending on the ``dump_filename_pattern`` provided in the ``main_onnx.py`` file. In total, five result files are saved, one for each supported onnx node the ``ONNXModelParserStage`` parsed (supported meaning it can be accelerated on one of the accelerator cores). Each result file contains the optimal energy and latency of running the onnx node on the core to which it was mapped through the ``mapping`` input file. Optimality is defined through the ``MinimalLatencyStage``, for more information we refer you to :doc:`stages`.
