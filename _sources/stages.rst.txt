======
Stages
======

This document explains the concept of stages within the ZigZag framework. It details the different implemented stages and explains how to create your own.

Introduction
============

Stages within ZigZag are used to modularly and easily adapt the functionality of the framework. The different stages and their sequence of execution determine the goal of running the framework. The sequence of stages the framework will run through are defined in the main file. An example as follows:

.. code-block:: python

    mainstage = MainStage([  # Initializes the MainStage as entry point
        ONNXModelParserStage,  # Parses the ONNX Model into the workload
        AcceleratorParserStage,  # Parses the accelerator
        SimpleSaveStage,  # Saves all received CMEs information to a json
        WorkloadStage,  # Iterates through the different layers in the workload
        SpatialMappingConversionStage,  # Generates multiple spatial mappings (SM)
        MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
        LomaStage,  # Generates multiple temporal mappings (TM)
        CostModelStage  # Evaluates generated SM and TM through cost model
    ],
        accelerator_path=args.accelerator,  # required by AcceleratorParserStage
        onnx_model_path=args.model,  # required by ONNXModelParserStage
        mapping_path=args.mapping,  # required by ONNXModelParserStage
        dump_filename_pattern="outputs/{datetime}.json",  # output file save pattern
        loma_lpf_limit=6  # required by LomaStage
    )

    # Run the mainstage
    mainstage.run()

You can think of stages similar to those in a pipelined system. The MainStage provides an entry point for the framework to start execution from. All stages save the provided first argument as the sequence of remaining stages, of which the first one will be called when running said stage. In our example, the ``MainStage`` will automatically call the ``ONNXModelParserStage`` with the remaining stages ``[WorkloadStage, ...]`` as its first argument. The ``ONNXModelParserStage`` will then call the ``AcceleratorParserStage``, and so on. Besides the sequence of stages, the remaining arguments of the MainStage initialization are arguments required by one or more of the later stages. For example, the ``dump_filename_pattern`` is an argument required by the ``CompleteSaveStage``.

The sequence of stages in the example will parse the inputs. It then iterates through each layer in the parsed workload, and for each layer convers the user-provided spatial mapping (SM) to the framework's internal SM representation in the ``SpatialMappingConversionStage``. It generates multiple temporal mappings (TM), and each SM + TM combination is fed to the cost model for HW cost evaluation. 

So far, we have only discussed the sequential calling of stages from first to last. The reverse also holds true: when the CostModelStage finishes processing a SM + TM conbimation, it yields a CostModelEvaluation (CME) object back up the chain of stages. Some stages will simply pass this CME further up the chain, while others manipulate what is passed back up the chain. The ``MinimalLatencyStage`` for example, receives all the CMEs from the multiple cost model invocations for different TMs, but only passes the CME with the lowest latency back up the chain across all TMs. As such, the ``CompleteSaveStage`` only receives the CME with the lowest latency, which it will save to a file with the ``dump_filename_pattern`` pattern.

Implemented stages
==================



Creating your own stages
========================

Let's say you are not interested in saving the CME with minimal energy, but want to save based on another metric provided by the CME. You can easily create a custom stage. The easiest way is copying an existing stage class definition, and modifying it according to your intended behaviour. To guarantee correctness, following aspects have to be taken into account when creating a custom stage:

* aspect 1
* aspect 2

