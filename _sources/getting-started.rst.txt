===============
Getting Started
===============

ZigZag is a versatile tool. It can be used to estimate and optimize the hardware cost of running a DL workload on a given hardware design under a multitude of constraints and settings. 

As a first step, we use it to automatically optimize the way a neural network is mapped onto a hardware design.

First run
=========

Tutorial
--------

The recommended way to get started with ZigZag is through the tutorial labs. You can find them in the `tutorial` branch of the repository. You should start with lab1 `here <https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab1>`_.

Manual run
----------

Alternatively, you can also use the provided example inputs manually. The neural network we are going to use for this first manual run is AlexNet. We provide an `onnx <https://onnx.ai/>`_ model of this network in ``zigzag/inputs/workload/alexnet.onnx``. The model has been shape inferred, which means that beside the input and output tensor shapes, all intermediate tensor shapes have been inferred, which is information required by ZigZag. 

.. warning::
    ZigZag requires an inferred onnx model, as it needs to know the shapes of all intermediate tensors to correctly infer the layer shapes. You can find more information on how to infer an onnx model `here <https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model>`_.

The hardware we will accelerate the inference of AlexNet with, is a TPU-like architecture defined in ``zigzag/inputs/hardware/tpu_like.yaml``. 

Besides the workload and hardware architecture, a mapping file must be provided which, as the name suggests, provides information on how the network's layers will be mapped onto the hardware resources. The mapping is provided in ``zigzag/inputs/mapping/tpu_like.yaml``. 

The framework is generally run through a main file which parses the provided inputs and contains the program flow through the stages defined in the main file. 

.. note::

    You can find more information in the :doc:`stages` document.


The following command starts the execution using the provided inputs:

.. code:: sh

    python main.py --model zigzag/inputs/workload/resnet18.onnx --accelerator zigzag/inputs/hardware/tpu_like.yaml --mapping zigzag/inputs/mapping/tpu_like.yaml

Below you can see the terminal outputs of this run:

.. raw:: html

    <script id="asciicast-zdbEqPoE4odh1QsqAIVSWPi6M" src="https://asciinema.org/a/zdbEqPoE4odh1QsqAIVSWPi6M.js" async></script>

Other ZigZag runs examples:

- The directory contains a directly runnable ``example.py`` file. This file directly defines the model, accelerator and mapping as Python variables, and thus doesn't require command line arguments.

- ZigZag can also run with user-defined workloads as follows (see section manual layer definition section in :doc:`workload` for more information):

.. code:: sh

    python main.py --model zigzag/inputs/workload/resnet18.yaml --accelerator zigzag/inputs/hardware/tpu_like.yaml --mapping zigzag/inputs/mapping/tpu_like.yaml

ZigZag can also run with the `SALSA <https://ieeexplore.ieee.org/document/10168625>`_ temporal mapping search engine which utilizes a different optimizer than `LOMA <https://ieeexplore.ieee.org/document/9458493>`_. This can be done by setting the ``temporal_mapping_search_engine`` to 'salsa' in the api call in ``main.py``.


Analyzing results
=================

During the run, results will be saved in the ``dump_folder`` provided in the ``main.py`` file, or as argument to the API function call. In total, five result files are saved, one for each supported onnx node the ``ONNXModelParserStage`` parsed (supported meaning it can be accelerated on one of the accelerator cores). Each result file contains the optimal energy and latency of running the onnx node on the core to which it was mapped through the ``mapping`` input file. Optimality is defined through the ``MinimalLatencyStage``, for more information we refer you to :doc:`stages`. The chosen loop ordering (spatial and temporal) and memory allocation are also saved in ``loop_ordering.txt``, as well as a graph of the hardware architecture's memory hierarchy and a bar plot of the results.
