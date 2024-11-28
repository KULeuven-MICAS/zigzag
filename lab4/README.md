# Lab 4: First Run of the ZigZag-IMC Extension

## Objective
The goal of this lab is to perform the first run of the ZigZag-IMC, a ZigZag extension for In-Memory-Computing (IMC). You will derive the peak performance and cost breakdown of a defined Analog IMC array.

## Setup
1. Ensure you have installed the requirements in `requirements.txt`.
2. Make sure you are in the base directory, as `lab4/main.py` automatically inserts this into PATH which is needed for the ZigZag imports.

## Inputs
There is one main input defined in the current folder:
1. **Hardware**: A sample accelerator is encoded in `imc_macro.yaml`. This accelerator features an Analog-based IMC (AIMC) array composed of 32×32 INT8 IMC units (`cells`). Each column has a 3-bit ADC, and the array processes activations in a bit-serial dataflow, handling 2 bits per cycle.

## Running the Experiment

Run the main file:
```python
# Call this from the base folder
python lab4/main.py
```

## Outputs
The results of the experiment will be printed out in the terminal, consisting of the following parts:

- **peak performance section:** This section presents the peak performance metrics (TOP/s, TOP/s/W, and TOP/s/mm²) of the IMC array under ideal conditions: 100% array utilization, continuous switching activity on all IMC units, and zero sparsity.
- **area section:** This section presents the total area (mm²) and component breakdown of the IMC array.
- **Tclk section:** This section presents the minimum clock period (ns) and its delay breakdown.
- **energy section:** This section presents the total energy consumption per clock cycle and its component breakdown under the peak performance.

The cost breakdown includes the following components:

- `cells`: memory cells, storing the constant operand for IMC.
- `dacs`: DACs (Digital-to-Analog Converters) for enabling analog computation.
- `adcs`: ADCs ((Analog-to-Digital Converters) for converting analog results to digital signals.
- `mults`: multipliers required by the MAC (Multiply-Accumulate) operations.
- `adders_regular`: regular adder trees required by the MAC operations (inputs with the same place value).
- `adders_pv`: special adder trees for summing inputs across different place values.
- `accumulators`: accumulation logics and accumulation registers for storing partial-sums.

Additionally, the following components appear only in the energy cost breakdown:

- `local_bl_precharging`: Energy consumed by precharging local bitlines within each cell group. Automatically calculated when multiple values share a cell group.
- `analog_bl_addition`: Energy consumed by analog addition on bitlines.

## Questions & Answers

- Take a look at the outputs. Which components dominate the total area?
    > <details>
    > <summary>Answer</summary>
    >
    > ADCs are the dominant component in area consumption, accounting for 54% of the total cost.
    >
    > </details>

- Which components are the top two bottlenecks for the clock speed?
    > <details>
    > <summary>Answer</summary>
    >
    > The ADCs are the primary bottleneck, consuming 53% of the clock cycle. The `adders_pv` is the secondary bottleneck, consuming 28% of the clock cycle.
    >
    > </details>

- Modify `imc_macro.yaml` by setting D1=1024 and D2=1024. Run `python lab4/main.py` again. Analyze how the peak performance changes and explain why. What is the scaling relationship between peak performance and array size?
    > <details>
    > <summary>Answer</summary>
    >
    > To set D1=1024 and D2=1024, modify the last line in `imc_macro.yaml` to `sizes: [1024, 1024]`. This change increases the array size by a factor of 1024. Then you can re-run the simulation and analyze the outputs.
    >
    > From a throughput perspective, the TOP/s increases from 0.106 to 21.6, a 204x improvement. While this gain comes from the increased parallelism, why doesn't it match the 1024-fold increase in the array size?
    >
    > The answer lies in comparing the clock periods (Tclk). The 1024×1024 IMC array operates at a clock speed 5 times slower than the 32×32 array due to the increased bitline capacitance. As a result, the TOP/s only increases by 1024/5-fold rather than the expected 1024-fold.
    >
    > From an energy perspective, the TOP/s/W increases from 6.65 to 74.1 (11x improvement). Given that TOP/s/W represents the energy efficiency per MAC operation, why will it change with the array size?
    >
    > Analysis of the energy breakdown reveals that DAC and ADC energy costs increase only 32-fold because these components are shared along rows or columns. This sharing leads to better energy amortization per MAC operation as the IMC array size increases. While components like `mults` scale linearly with array size, the overall effect results in an 11-fold improvement in TOP/s/W.
    >
    > From an area efficiency perspective, the TOP/s/mm² remains nearly constant (2.27 to 2.30). If TOP/s increases by 204-fold, why does the area efficiency (TOP/s/mm²) stay unchanged?
    >
    > The explanation lies in analyzing the area breakdown. The ADCs area increases by 32-fold. Most significantly, the multiplier area scales linearly with array size and becomes the dominant component of area consumption. These factors collectively lead to a 201x increase of the total area. While in our case the increases in area and TOP/s are coincidentally similar, determining the exact impact on TOP/s/mm² requires quantitative simulations.
    >
    > </details>

- What ADC resolution is required for [D1, D2] = [1024, 1024] to avoid any accuracy loss? How does the peak performance change when using this accuracy-preserving ADC resolution?
    > <details>
    > <summary>Answer</summary>
    >
    > Given 1024 cells per column, the required ADC resolution is log2(1024) = 10 bits. When simulated with 10-bit ADCs, all performance metrics (TOP/s, TOP/s/W, and TOP/s/mm²) decrease significantly due to ADCs becoming the dominant cost factor in area, delay, and energy consumption.
    >
    > </details>

- How do you configure an IMC array containing 16 macros, each with dimensions [D1, D2] = [1024, 1024] (ADC resolution: 3 bits)? How does the peak performance change when using this 16-bank IMC array?
    > <details>
    > <summary>Answer</summary>
    >
    > To configure a 16-macro IMC array, modify `imc_macro.yaml` by updating the following rows:
    >
    > `adc_resolution: 3`
    >
    > `dimensions: [D1, D2, D3]`
    >
    > `sizes: [1024, 1024, 16]`
    >
    > Compared to the single-macro 1024×1024 IMC, the 16-macro IMC achieves 16-fold higher TOP/s, while maintaining the same TOP/s/W and TOP/s/mm².
    >
    > </details>

- How do you configure a Digital-based IMC (DIMC) array with bit-serial processing for activation (one bit per cycle)? How do the component costs in DIMC differ from those in AIMC? How does DIMC's peak performance compare to AIMC's?
    > <details>
    > <summary>Answer</summary>
    >
    > To configure a 16-macro DIMC array, modify `imc_macro.yaml` by updating the following rows:
    >
    > `imc_type: digital`
    >
    > `bit_serial_precision: 1`
    >
    > `#  adc_resolution: 3`
    >
    > Note you need to comment out the `adc_resolution` row, as DIMC does not use ADCs.
    >
    > Comparing the 16-macro DIMC to AIMC:
    >
    > - **Component costs**: the cost of all analog components (`dacs`, `adcs`, `analog_bl_addition`) becomes zero for DIMC.
    > - **TOP/s:** DIMC achieves higher throughput due to faster clock frequency.
    > - **TOP/s/W:** DIMC shows lower efficiency due to digital addition logic overhead.
    > - **TOP/s/mm²:** Both architectures show similar efficiency but with different bottlenecks. AIMC: Limited by ADCs. DIMC: Limited by the regular adder trees (`adders_regular`).
    > 
    >
    > </details>

- Define a single-macro DIMC with [D1, D2] being [32, 32]. How does its peak performance compare to the 16-macro DIMC?
    > <details>
    > <summary>Answer</summary>
    >
    > To configure a single-macro DIMC with [D1, D2] being [32, 32], modify `imc_macro.yaml` by updating the following rows:
    >
    > `dimensions: [D1, D2]`
    >
    > `sizes: [32, 32]`
    >
    > Comparing the single-macro DIMC to the 16-macro DIMC:
    >
    > - **TOP/s:** The 16-macro DIMC achieves 16-fold higher throughput.
    > - **TOP/s/W:** The 16-macro DIMC shows slightly lower efficiency due to increased adder tree depth.
    > - **TOP/s/mm²:** Same reason as TOP/s/W. The 16-macro DIMC shows slightly lower efficiency due to increased adder tree depth.
    >
    > </details>