==========
ZigZag API
==========

Once ZigZag is available in your Python path, you can import the api in any Python file:

.. code-block:: python

    from zigzag import api

get_hardware_performance_zigzag()
==========================

This function takes in an workload, a hardware architecture and a mapping file and returns the hardware performance of executing the model's layers on the architecture under the given mapping constraints.

.. code:: python

    energy, latency, cme = get_hardware_performance_zigzag(
        workload,
        accelerator,
        mapping,
        opt="latency",
        dump_folder="outputs/{datetime}.json",
        pickle_filename="outputs/list_of_cmes.pickle"
    )

The inputs of the function are

* **workload**: A neural network model defined in ONNX format or ZigZag's own format.
* **accelerator**: A high-level hardware architecture description (in ZigZag's own format).
* **mapping**: A file that specifies core allocation, spatial mapping (optional), temporal ordering (optional), and memory operand link (in ZigZag's own format).
* **opt (optional)**: Optimization target. It can be 'energy', 'latency', or 'EDP' (energy-delay-product).
* **dump_folder (optional)**: The name of the result folder.
* **pickle_filename (optional)**: The name of the file which includes all the detailed metadata for analyzing and debugging.

The outputs of the function are

* **energy**: A number that indicates the overall consumed energy for running the workload on the accelerator in the user-defined optimized way.
* **latency**: A number that indicates the overall latency (cycle count) for running the workload on the accelerator in the user-defined optimized way.
* **cme**: A collection of all the detailed cost model evaluation results. "cme" stands for "cost model evaluation".
.. note::

   We demonstrate how to use this api function in multiple `demos <https://github.com/KULeuven-MICAS/zigzag-demo>`_.
