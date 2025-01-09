# Lab 6: System-Level Evaluation for IMC: Temporal Utilization Matters

## Objective
In the previous lab, you have understood a gap exists between the system-level performance and the macro-level peak performance due to the spatial under-utilization.
In this lab, we will analyze another underlying reason of the gap from another perspective: the temporal utilization.
The goal is to understand how the temporal utilization (TU) impacts on the system performance.

## Background

**The definitions of temporal utilization:**

As the spatial utilization means the percentage of the effectively working PEs compared to the total amount of PEs during the computation, the temporal utilization represents the percentage of the effective computation cycles, when PEs are consuming data, compared to the total processing cycles.

**The different reasons of the spatial and temporal under-utilization:**

The underlying reasons of the spatial under-utilization for IMC accelerators, as we saw in lab5, can be:
1. The PE array shape does not match with the layer requirement. For example, a 32×32 PE array is 75% under-utilized when processing a 32×24 tensor.
2. The mapping limits the PE processing capability. For example, a 32×32 PE array supporting (K, FX) spatial mapping has only 3/32 utilization when executing a layer with (C, K, FX, FY) = (32, 32, 3, 3).

In contrast, the reasons of the temporal under-utilization are attributed to the reasons why the PE array is in idle, which can be:
1. A memory instance does not have sufficient memory bandwidth, causing the data required by the PE array cannot be ready in time.
2. A memory instance cannot write and read simultaneously, because either the data in the memory is still under use and cannot be kicked out (such as registers), or the write and read operations share one common port (non-double-buffer memory, IMC macro).
3. A memory instance storing multiple operands has limited memory port, which causes access congestion when multiple operands require to be accessed at the same time.

## Setup
1. Ensure you have installed the requirements in `requirements.txt`.
2. Make sure you are in the base directory, as `lab6/main.py` automatically inserts this into PATH which is needed for the ZigZag imports.

## Inputs
There are three main inputs defined in the `inputs/` folder:
1. **Workload**: The second layer of ResNet18 in ONNX format. The layer name is `Conv2`. You can use [Netron](https://netron.app) to visualize the model.
2. **Hardware**: _[Same as lab5]_ A sample accelerator is encoded in `accelerator.yaml`. This accelerator includes 32x32 DIMC operational units with a hierarchy of memories attached which store different `memory operands` `I1`, `I2`, `O`.
3. **Mapping**: _[Same as lab5]_ The mapping specifies for the `Conv2` layer only the spatial mapping restriction. The `SpatialMappingGeneratorStage` automatically generate all legal spatial mappings. The `TemporalMappingGeneratorStage` automatically detects if there is any user-defined temporal loop ordering and generates multiple temporal mappings to be evaluated by the cost model.

## Running the Experiment

Run the main file:
```python
# Call this from the base folder
python lab6/main.py
```

## Outputs
Some experiment results will be printed out in the terminal (for this lab), and more detailed results will be saved in the `outputs/` folder.

The terminal will show the following parts:

- **peak performance section:** This section presents the peak performance metrics (TOP/s, TOP/s/W, and TOP/s/mm²) of the IMC array under ideal conditions: 100% array utilization, continuous switching activity on all IMC units, and zero sparsity.
- **spatial mapping:** This section presents the spatial mapping loops on the computation loops.
- **energy section:** This section presents the system-level energy and its breakdown.
- **cycle section:** This section presents the system-level cycle count and its breakdown.
- **Tclk section:** This section presents the minimum clock period (ns) and its delay breakdown.
- **area section:** This section presents the total area (mm²) and its breakdown.
- **performance comparison section:** This section presents both the macro-level and the system-level performance.

In the `outputs/` folder, following outputs are saved _[same as lab1/2/3/5]_:


- `breakdown.png` shows an energy and latency breakdown for the different layers evaluated (only one here). The energy is broken down into the operational level (MAC) and memory levels. As each memory level can store one or more operands, it is colored by operand. Moreover, it breaks down the energy cost for 4 different read/write directions of the memory. The latency is broken down into the ideal computation time (assuming perfect utilization of the operational array), the added cycles due to spatial stalls which represent the spatial underutilization (due to imperfect spatial loop unrolling), the added cycles due to temporal stalls (due to imperfect memory bandwidth), and the added on-loading and off-loading cycles (due to the very first.last iteration on/off-loading of inputs/outputs).

- `Conv2_complete.json` contains all input and output information of the cost model evaluation. 

- `overall_simple.json` aggregates the energy and latency of all layers (only one here).

- `mem_hierarchy.png` shows the constructed hierarchy of memories and for each level which operands they store and the amount of times it's replicated (more info on this in `lab3`).

- `loop_ordering.txt` shows for all evaluated layers the returned mapping. This includes both the temporal aspect, where different loops are assigned at the memory levels (which can be different for different operands due to ZigZag's uneven mapping representation). The spatial aspect shows the spatially unrolled loops.


## Questions & Answers

- What is the difference between the hardware templates in lab6 and lab5?
    > <details>
    > <summary>Answer</summary>
    >
    > There is one difference:
    > - The IMC type is different. At the end of the `accelerator.yaml` (under lab6/inputs/hardware/), we can see that the `imc_type` is `digital`, meaning the IMC type is the Digital IMC (DIMC). Checking the `bit_serial_precision` parameter, it can be seen that each cycle 1 bit is processed.
    > 
    > </details>

- What is the current temporal utilization?
    > <details>
    > <summary>Answer</summary>
    >
    > In the terminal, you can see the reports on the total cycle count (#cycles) and its breakdown. The temporal utilization (TU) can be calculated by dividing the computation cycle count with the overall cycle count.
    > 
    > TU = 903168.0 / 903226.0 = 100%
    > 
    > </details>
  
- Can the IMC cell be a 6T SRAM cell or a 8T SRAM cell?
    > <details>
    > <summary>Answer</summary>
    >
    > The difference of the 6T and 8T cells is whether the write and read ports are separate. In `accelerator.yaml`, we can see that the `cells` has one read port (`r_port`) and one write port (`w_port`), meaning the IMC behaves as 8T SRAM cells.
    >
    > 
    > </details>

- Change the IMC cell to the other type (6T or 8T). How to do it? How does the throughput change after that? Why?
    > <details>
    > <summary>Answer</summary>
    >
    > To switch to 6T SRAM cells, you need to:
    > - merge the two ports into one rw port (`rw_port`), by setting `rw_port: 1` and disabling original ports (setting `r_port: 0, w_port: 0`).
    > - update the binding of the memory directions (fh, tl) and the port name, by setting `fh: rw_port_1, tl: rw_port_1`.
    >
    > After switching the SRAM types, you can observe in the terminal that the throughput drops. The reason is the TU drops (from 100% to 89%), as now IMC cells cannot write and read simultaneously. This leads to extra memory stalling cycles and therefore lower throughput.
    > 
    > </details>

- Is the dram bandwidth realistic? Change it to 128 bit/access and rerun the simulation. How does the throughput change? Why?
    > <details>
    > <summary>Answer</summary>
    >
    > 64KB/access is not realistic. To change it to 128 bit/access, you need to set `r_bw: 128, w_bw: 128` for dram in `accelerator.yaml`.
    >
    > After rerunning the simulation, the throughput slightly drops (from 0.073 TOP/s to 0.072 TOP/s), as now the dram bandwidth is insufficient to transfer all operands in time.
    > 
    > </details>

- In the current architecture, what are the system-level bottlenecks for the throughput when designing an IMC processor?
    > <details>
    > <summary>Answer</summary>
    >
    > The two bottlenecks in terms of the Temporal Utilization (TU) are: (1) The sharing w/r port of the IMC array, (2) Insufficient dram bandwidth.
    >
    > The bottlenecks in terms of the Spatial parallelism is: the PE array shape does not match with the layer shape. Increasing the PE array shape to (64, 64) can definitely improve the throughput.
    > 
    > </details>
