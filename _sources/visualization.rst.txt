=============
Visualization
=============

The generated ``CostModelEvaluation`` object(s) (from e.g. the API call) can be visualized in multiple ways.

Temporal mapping
================

The temporal mapping can be visualized by a function which prints it to the terminal.
The code block demonstrates how to use it:

.. code:: python

    from zigzag.utils import pickle_load
    from zigzag.visualization.results.print_mapping import print_mapping
    cmes = pickle_load("zigzag/visualization/list_of_cmes.pickle")
    cme = cmes[0]
    print_mapping(cme)

The function will show the loops of the temporal mapping and for each operand shows at which memory level it resides.
For example:

::

    ********* Temporal Mapping - CostModelEvaluation(layer=LayerNode_0, core=1) *********
    O (O): [[('FX', 11), ('FY', 11)], [('OY', 7), ('OY', 2), ('OX', 14), ('K', 12)], []]
    W (I2): [[], [('FX', 11), ('FY', 11), ('OY', 7), ('OY', 2), ('OX', 14)], [('K', 12)]]
    I (I1): [[('FX', 11), ('FY', 11), ('OY', 7), ('OY', 2), ('OX', 14), ('K', 12)], []] 
                                                                                        
    -------------------------------------------------------------------------------------
    Temporal Loops                  O                  W                  I             
    -------------------------------------------------------------------------------------
    for K in [0:12)                 sram_2MB           dram               sram_2MB      
    -------------------------------------------------------------------------------------
    for OX in [0:14)               sram_2MB           sram_32KB          sram_2MB      
    -------------------------------------------------------------------------------------
    for OY in [0:2)               sram_2MB           sram_32KB          sram_2MB      
    -------------------------------------------------------------------------------------
        for OY in [0:7)              sram_2MB           sram_32KB          sram_2MB      
    -------------------------------------------------------------------------------------
        for FY in [0:11)            rf_2B              sram_32KB          sram_2MB      
    -------------------------------------------------------------------------------------
        for FX in [0:11)           rf_2B              sram_32KB          sram_2MB      
    -------------------------------------------------------------------------------------

The top loop is the outer-most for loop, where as the bottom loop is the inner-most. Going from bottom to top, loops are allocated to the innermost memories of the memory hierarchy for each operand.
The names of the memories match the names of the ``MemoryInstance`` object used to create the memory level using the ``add_memory()`` call in the ``MemoryHierarchy``.

Energy and latency breakdown
============================

The energy and latency breakdown of a list of ``CostModelEvaluation`` objects can be plotted using the ``bar_plot_cost_model_evaluations_breakdown`` function:

.. code:: python

    from zigzag.utils import pickle_load
    from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown

    cmes = pickle_load("zigzag/visualization/list_of_cmes.pickle")
    bar_plot_cost_model_evaluations_breakdown(cmes, "outputs/breakdown.jpg")

This will produce a bar chart, for example:

.. image:: images/visualization/breakdown.jpg
  :width: 900
