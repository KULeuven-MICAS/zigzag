==========
ZigZag API
==========

Once ZigZag is available in your Python path, you can import the api in any Python file:

.. code-block:: python

    from zigzag import api

get_hardware_performance_zigzag()
==========================

This function takes in an workload (can be in onnx format or zigzag workload format), a hardware architecture and a mapping file and returns the hardware performance of executing the model's layers on the architecture under the given mapping constraints. We demonstrate how to use this api function in a `demo <https://github.com/ZigZag-Project/zigzag-demo/blob/main/breakdown.ipynb>`_.
