# Lab 3: Hardware Architecture

## Objective
In the previous two labs, you used the given `accelerator1.yaml`. This lab explains how this accelerator is constructed: an array of operational units and a memory hierarchy interconnected to these units. Secondly, the goal is to understand the interplay of thi sinterconnection with the spatial mapping of the layer.

## Setup
1. Ensure you have installed the requirements in `requirements.txt`.
2. Make sure you are in the base directory, as `lab3/main.py` automatically inserts this into PATH which is needed for the ZigZag imports.

## Inputs
There are three main inputs defined in the `inputs/` folder:
1. **Workload**: _[Same as lab1/2]_ The first layer of ResNet18 in ONNX format. The layer name is `Conv1`. You can use [Netron](https://netron.app) to visualize the model.
2. **Hardware**: Three different accelerator architectures `accelerator1.yaml`, `accelerator2.yaml` and `accelerator3.yaml`. All three have the same 32x32 operational array size with similar memories construction in varying hierarchies.
3. **Mapping**: The mapping for the three different accelerator architectures. The spatial mapping is fixed (but different) for all three accelerators.

## Understanding the hardware architecture

Before we run the experiment, let's understand the hardware architecture components of ZigZag. A hardware architecture contains two major components: the `operational_array` and the `memories`.

### Operational array
The `operational_array` specifies an array of 'operational units', where we make abstraction of what that operation is exactly. It can represent a multiplication, a multiply-accumulate, a division, etc. All that matters for ZigZag is the energy cost of an operation (in `unit_energy`) and what is the area of a unit (in `unit_area`). Then, an N-dimensional array is constructed by using the `dimensions` and the `sizes` field, where each dimension is denoted `Dx` with `x in [1, N]` and `sizes` of equal length representing the size of the dimension. The total amount of units in the array is thus the product of the `sizes` list.

The reason we allow N-dimensional arrays is that this allows us to flexibly interconnect these units to the lowest level of the memory hierarchy, which will be discussed later.

### Memories

The `memories` entry specify a number of `MemoryLevel`s which together make up a hierarchy. Each level has a name through the key of the entry, and various fields. In this lab, we focus mostly on the `served_dimensions` field to understand its link with the spatial mapping. You can find more information about the other fields [here](https://kuleuven-micas.github.io/zigzag/hardware.html#memory-instance).

It is important to keep in mind that each `MemoryLevel` can consist of one or more `MemoryInstance`s, which are unrolled with a specific replication pattern. This pattern is encoded through the `served_dimensions` attribute. It specifies which dimensions of the `operational_array` a single instance of this level will serve. 

We distinguish a couple of `served_dimensions` cases to make this more concrete:

- If the field is the empty list, this means that a single instance doesn't serve any dimensions, but rather is replicated alongside each unit. 
- If the field contains all operational array dimensions, there is a single instance connected to operational units (and/or the memory below if it's not the lowest memory level for an operand). 
- If there are some (but not all) dimensions listed, it means an instance of the level is interconnected to all units across that dimension, and other instances are replicated with the same behavior. 

Thus, the number of instances in a level always equals the product of the dimension sizes not present in `served_dimensions`.

## Example `served_dimensions` for `accelerator1.yaml`

Let's make this concrete using the `accelerator1.yaml` architecture description. First, we focus on the three lowest level memory levels that are added. These three levels store one memory operand: `I1`, `I2` and `O` each. 

**Note:** The drawings below are simplified to a 2x2 operational array for simplicity.

The `I1` lowest memory level looks as follows:
```
                      Dimension                  
                          D2                     
                 ◄─────────────────►             
┌──────────┐                                     
│          ┼──────────┬──────────┐               
│ rf_1B_I1 │          │          │               
│          │          ▼          ▼               
└──────────┘     ┌──────┐   ┌──────┐  ▲          
                 │  OP  │   │  OP  │  │          
                 └──────┘   └──────┘  │          
┌──────────┐                          │          
│          ┼──────────┬──────────┐    │ Dimension
│ rf_1B_I1 │          │          │    │    D1    
│          │          ▼          ▼    │          
└──────────┘     ┌──────┐   ┌──────┐  │          
                 │  OP  │   │  OP  │  │          
                 └──────┘   └──────┘  ▼          
```

As can be seen, each `rf_1B_I1` instance serves all operational units across dimension `D2`. There are thus 1 instances for this 2x2 example.

The `I2` lowest level:
```
      Dimension                   
          D2                      
 ◄──────────────────►             
┌────────┐  ┌────────┐            
│        │  │        │            
│rf_1B_I2│  │rf_1B_I2│            
│        │  │        │            
└──┬─────┘  └──┬─────┘            
   │           │                  
   ▼           ▼                  
  ┌──────┐   ┌──────┐  ▲          
  │  OP  │   │  OP  │  │          
  └──────┘   └──────┘  │          
┌────────┐  ┌────────┐ │          
│        │  │        │ │          
│rf_1B_I2│  │rf_1B_I2│ │          
│        │  │        │ │ Dimension
└──┬─────┘  └──┬─────┘ │    D1    
   │           │       │          
   ▼           ▼       │          
  ┌──────┐   ┌──────┐  │          
  │  OP  │   │  OP  │  │          
  └──────┘   └──────┘  ▼          
```

Each `rf_1B_I2` serves a single operational unit. There are thus 4 instances for this 2x2 example.

The `O` lowest memory level:
```
      Dimension                    
          D2                       
◄──────────────────►               
                                   
 ┌──────┐   ┌──────┐    ▲          
 │  OP  │   │  OP  │    │          
 └───┬──┘   └────┬─┘    │          
     └────┐      └───┐  │ Dimension
 ┌──────┐ │ ┌──────┐ │  │    D1    
 │  OP  │ │ │  OP  │ │  │          
 └─┬────┘ │ └─┬────┘ │  ▼          
   │ ┌────┘   │ ┌────┘             
   ▼ ▼        ▼ ▼                  
┌───────┐  ┌───────┐               
│       │  │       │               
│rf_4B_O│  │rf_4B_O│               
│       │  │       │               
└───────┘  └───────┘               
```

Each `rf_4B_O` serves the operational units across dimension `D1`. There are thus 2 instances for this 2x2 example. Note that while in reality this architecture would have an adder tree to sum up the outputs coming from different units in a column, this is abstracted out in the framework.

The higher memory levels automatically connect to these lower memory levels. These higher memory levels typically aren't unrolled (although this is possible with the representation), as such they contain more array dimensions in the `served_dimensions` attribute.

## Relationship between memory interconnection and spatial mapping

The interconnection pattern as explained above ties in closely with the potential spatial mappings. For example, the `rf_1B_I1` level, which through the mapping is linked to the `I` operand of the `Conv1` layer, has a read bandwidth of 8 bits. This means that within a clock cycle, only a single `I` element (assuming 8 bit precision) can be read out. Thus, the operational elements across `D2` should require the same input and as such only `irrelevant` dimensions of the `I` operand can be assigned to the `D2` dimension. The spatial mapping encoded in `inputs/mapping/accelerator1.yaml` unrolls the `K` dimension (output channels) across `D2`.

## Running the Experiment
Now that you fully understand the architecture definition, let's run the experiment through the main file:
```python
# Call this from the base folder
python lab3/main.py
```
    
ZigZag will optimize the temporal mapping for all three accelerator architectures with their defined spatial mappings. The resulting energy and latency is shown for the different accelerators. Note that all three are running the exact same workload.

## Outputs
The results of the experiment will be saved in the `outputs/` folder.

## Questions & Answers

- Try drawing the lowest memory levels for the `accelerator2.yaml` architecture description. Which level has the most instances?
    > <details>
    > <summary>Answer</summary>
    >     
    > The output memory operand `O` has the most instances in the lowest level. Its `served_dimensions` attribute is empty, which means that there will be 32x32 instances of the output RF. The architecture in `accelerator2` is typically referred to as an 'output-stationary' dataflow (when used in combination with temporal output reuse in these RFs).
    >
    > </details>

- How do you define a memory level with only a single instance?
    > <details>
    > <summary>Answer</summary>
    >     
    > A memory level with a single instance is defined by specifying all dimensions in the `served_dimensions` attribute.
    >
    > </details>

- Which accelerator architecture has the best latency? What causes the other ones to be worse? 
    > <details>
    > <summary>Answer</summary>
    > 
    > `accelerator2` has the best latency. In broad terms, this can be attributed mainly to the fact that the output RF `rf_4B_O` is unrolled 32x32 times. This RF has a higher capacity than the other two RFs, and a higher bandwidth. Thus, output data can be reused longer in the array, which avoids memory stalls due to insufficient bandwidths at higher memory levels. This can be checked by looking at the `mac_utilization` field of the complete output json `Conv1_complete.json`. For accelerator 3 for example, the `ideal` utilization (without taking memory stalls into account) is 67%, but when taking stalls into account, this drops to 31%. Meanwhile, the ideal utilization of accelerator2 is 87% due to better spatial mapping, and there are no memory stalls.
    > 
    > </details>

- Increase the memory size of `rf_1B_I2` of `accelerator1.yaml`. Does the latency become better than that of `accelerator2.yaml`?
    > <details>
    > <summary>Answer</summary>
    > 
    > Increasing the size of this memory allows for more data reuse. At the baseline 8 bits, the layer has a latency of `2.46e6` cycles. At 32 bits, this decreases to `1.85e6`, and at 1024 bits it further decreases to `1.23e6`. Increasing it more doesn't decrease the latency. This is due to the fundamental limitation of the spatial mapping: the `C` dimension only has a size of 3, which means the utilization can never become better than 3/32 which is roughly 9%. On the other hand, there are enough `OX` and `K` to comletely unroll across the operational array of `accelerator2.yaml`.
    > </details>

- What is the spatial utilization on `accelerator2.yaml` and why?
    > <details>
    > <summary>Answer</summary>
    > 
    > The utilization, as mentioned in previous answers, is 87.5%. The reason it's not 100% is because of a mismatch between the operational array dimension `D1` and the layer dimension unrolled: `OX`. `OX` is 112, which means the closest factor we obtain is 28, as opposed to 32. This is equivalent to a 'greedy' mapping strategy of 32, 32, 32, and remainder 16, where we would still need 4 temporal iterations.
    > </details>