# Lab 2: Automating the mapping

## Objective
The goal of this lab is to have ZigZag generate multiple temporal mappings automatically, and only return the best one it found. 

Keep in mind that each evaluated mapping is uniquely different in loop ordering and memory allocation. Using traditional simulation, this would take orders of magnitude longer to 1. encode the mappings as a different control flow and 2. use cycle-accurate simulations to obtain the performance and switching activity. The trade-off is that the analytical cost model makes simplifying assumptions on both the hardware and the mapping of the workload onto its resources.

As a second goal the user will automate the spatial mapping themselves (see Questions & Answers for more info).

## Setup
1. Ensure you have installed the requirements in `requirements.txt`.
2. Make sure you are in the base directory, as `lab2/main.py` automatically inserts this into PATH which is needed for the ZigZag imports.

## Inputs
There are three main inputs defined in the `inputs/` folder:
1. **Workload**: _[Same as lab1]_ The first layer of ResNet18 in ONNX format. The layer name is `Conv1`. You can use [Netron](https://netron.app) to visualize the model.
2. **Hardware**: _[Same as lab1]_ A sample accelerator is encoded in `accelerator1.yaml`. This accelerator includes 32x32 operational units with a hierarchy of memories attached which store different `memory operands` `I1`, `I2`, `O`.
3. **Mapping**: The mapping specifies for the `Conv1` layer only the spatial mapping. The `TemporalMappingGeneratorStage` automatically detects there is no user-defined temoral loop ordering and generates multiple temporal mappings to be evaluated by the cost model.

## Running the Experiment
Run the main file:
```python
# Call this from the base folder
python lab2/main.py
```
    
As only the spatial mapping is fixed, there will be multiple cost model evaluations. The progress is shown through a bar, where the numbers to the right indicate the evaluated and total amount of mappings that will be evaluated.

## Outputs
The results of the experiment will be saved in the `outputs/` folder.

## Questions & Answers

- What does the API call optimize for? Try changing this to a different valid criterion and analyze the impact on the performance.
    > <details>
    > <summary>Answer</summary>
    >     
    > The API call optimizes for minimal latency, defined through the `optimization_criterion` in the main file.
    > 
    > Other valid criteria are `energy` and `EDP` (energy-delay product). A custom criterion requires manual implementation of a custom `Stage` which filters cost model evlauations to only return the one that optimizes the custom criterion. 
    > 
    > **Tip:** When trying different criteria, change the `experiment_id` to automatically save the results to a different folder and easily compare them.
    >
    > </details>

- How does the `TemporalMappingGeneratorStage` detect that there is no user-defined temporal loop ordering?
    > <details>
    > <summary>Answer</summary>
    >     
    > The `WorkloadFactory` checks for each layer if there is a user-defined temporal ordering defined in the mapping file. If so, it saves it as the `temporal_ordering` attribute of the layer. The `TemporalMappingGeneratorStage` gets this attribute and passes it to the underlying `LomaEngine`, which can be seen [here](https://github.com/KULeuven-MICAS/zigzag/blob/b8a523b10215eef8f82ad4eff3be9d17446457ed/zigzag/stages/mapping/temporal_mapping_generator_stage.py#L58). The engine is responsible for generating valid temporal mappings, i.e. with allocation of the memory levels for the different loops, from the provided user-defined temporal ordering or any other constraints.
    >
    > </details>

- What is the difference in performance (latency) compared to the user-defined temporal ordering? 
    > <details>
    > <summary>Answer</summary>
    > 
    > The LOMA engine inside of the `TemporalMappingGeneratorStage` takes in the defined `temporal_ordering` and allocates the different temporal loops from inner to outer to the memories in the hierarchy. This is the extra information you see in the printed mapping: for every operand and every loop, it shows the memory level it was allocated to.
    > 
    > </details>

- How would you modify the mapping file to also automatically optimize the spatial mapping?
    > <details>
    > <summary>Answer</summary>
    > 
    > Identically to the temporal ordering, you can simply remove the defined spatial mapping in the mapping file. Then, the `SpatialMappingGeneratorStage` inside the API will automatically generate a number of spatial mappings. For each generated spatial mapping, the same flow will run as before: multiple temporal mappings are evaluated and filtered to return the best one wrt. the optimization criterion.
    >
    > The standard number of spatial mappings evaluated is 3, which are those with the highest spatial utilization. This can be increased or reduced by passing a different `nb_spatial_mappings_generated` to the API call.
    > 
    > </details>
