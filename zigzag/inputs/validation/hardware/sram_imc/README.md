## In-Memory Computing Model Extraction and Validation
This folder is where we did cost model extraction and validation for AIMC and DIMC.

To see the validation details, 
for AIMC model, you can run `python aimc_validation.py` under folder `aimc_validation/22-28nm/`.
For DIMC model, you can run `python model_extraction_28nm.py` under folder `dimc_validation/28nm/`, which will extract the best fitting value for energy/area/delay (tclk) model and the corresponding mismatch.
You can also run `python dimc_validation.py`, which will get the mismatch value and cost breakdown for each validated work.

## Cost Model Overview
Our SRAM-based In-Memory Computing model is a versatile, parameterized model designed to cater to both Analog IMC and Digital IMC.
Since hardware costs are technology-node dependent, we have performed special calibration for the 28nm technology node. The model has been validated against 7 chips from the literature. 
A summary of the hardware settings for these chips is provided in the following table.

| source                                                          | label | B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub> | macro size     | #cell_group | nb_of_macros |
|-----------------------------------------------------------------|-------|-----------------------------------------------|----------------|-------------|--------------|
| [paper](https://ieeexplore.ieee.org/abstract/document/9431575)  | AIMC1 | 7 / 2 / 7                                     | 1024&times;512 | 1           | 1            |
| [paper](https://ieeexplore.ieee.org/abstract/document/9896828)  | AIMC2 | 8 / 8 / 2                                     | 16&times;12    | 32          | 1            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067289) | AIMC3 | 8 / 8 / 1                                     | 64&times;256   | 1           | 8            |
| [paper](https://ieeexplore.ieee.org/abstract/document/9731762)  | DIMC1 | 8 / 8 / 2                                     | 32&times;6     | 1           | 64           |
| [paper](https://ieeexplore.ieee.org/abstract/document/9731545)  | DIMC2 | 8 / 8 / 1                                     | 32&times;1     | 16          | 2            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067260) | DIMC3 | 8 / 8 / 2                                     | 128&times;8    | 8           | 8            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067779) | DIMC4 | 8 / 8 / 1                                     | 128&times;8    | 2           | 4            |

B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub>: input precision/weight precision/number of bits processed per cycle per input.
#cell_group: the number of cells sharing one entry to computation logic.

The validation results are displayed in the figure below (assuming 50% input toggle rate and 50% weight sparsity are assumed). 
The gray bar represents the reported performance value, while the colored bar represents the model estimation.
The percent above the bars is the ratio between model estimation and the chip measurement results.

<p align="center">
<img src="https://github.com/KULeuven-MICAS/zigzag/blob/master/zigzag/inputs/validation/hardware/sram_imc/model_validation.png" width="100%" alt="imc model validation plot">
</p>

- AIMC1 incurs additional area costs due to repeaters/decaps.
- Sparsity information is not available for AIMC2, DIMC2, DIMC4.
- AIMC1, AIMC3 were fabricated using 22nm technology, therefore the cost estimation was scaled accordingly.

**Note:**

The current integrated IMC model has certain limitations and is applicable only under the following conditions:
- The SRAM cell is a 6T memory cell.
- The adder tree follows a RCA (Ripple Carry Adder) structure without any approximation logic.
- The operands are of integer type rather than floating point.
- The voltage used for the delay estimation is fixed at 0.9 V.
- Sparsity impact is not included in the estimated energy cost.
