# 🌀 ZigZag Tutorial Series

Welcome to the ZigZag tutorial series! This series of labs will guide you through the key concepts and functionalities of the ZigZag framework for HW Architecture-Mapping Design Space Exploration (DSE) for Deep Learning (DL) accelerators.

## 🌟 Overview

ZigZag bridges the gap between algorithmic DL decisions and their acceleration cost on specialized hardware, providing fast and accurate HW cost estimation. Through its advanced mapping engines, ZigZag automates the discovery of optimal mappings for complex DL computations on custom architectures.

## 📚 Labs Overview

### [Lab 1: First Run of the ZigZag Framework](https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab1)
- **Objective**: Perform the first run of the ZigZag framework.
- **Key Learnings**:
  - Setting up the environment and running the ZigZag framework.
  - Understanding the inputs: workload, hardware, and mapping.
  - Analyzing the outputs: energy and latency breakdown, memory hierarchy, and loop ordering.

### [Lab 2: Automating the Mapping](https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab2)
- **Objective**: Automate the generation of multiple temporal mappings and return the best one.
- **Key Learnings**:
  - Automating temporal mapping generation using the `TemporalMappingGeneratorStage`.
  - Understanding the impact of different optimization criteria.
  - Analyzing the outputs: energy and latency breakdown, memory hierarchy, and loop ordering.

### [Lab 3: Hardware Architecture](https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab3)
- **Objective**: Understand the hardware architecture components of ZigZag.
- **Key Learnings**:
  - Understanding the operational array and memory hierarchy.
  - Exploring the relationship between memory interconnection and spatial mapping.
  - Running experiments with different accelerator architectures and analyzing the results.

### [Lab 4: First Run of the ZigZag-IMC Extension](https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab4)
- **Objective**: Perform the first run of the ZigZag-IMC extension.
- **Key Learnings**:
  - Setting up and running the ZigZag-IMC extension.
  - Understanding the inputs and outputs specific to in-memory computing (IMC) macros.
  - Analyzing the macro-level results and comparing between digital and analog IMC cores.

### [Lab 5: System-Level Evaluation for IMC: Spatial Utilization Matters](https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab5)
- **Objective**: Understand the impacts of the spatial under-utilization for IMC cores.
- **Key Learnings**:
  - Understanding the differences between system-level and macro-level IMC performance. 
  - Understanding the inputs and outputs specific to in-memory computing (IMC) cores.
  - Analyzing the system-level results and comparing them with macro-level results.
  - Understanding how the spatial utilization impacts the system-level performance.

### [Lab 6: System-Level Evaluation for IMC: Temporal Utilization Matters](https://github.com/KULeuven-MICAS/zigzag/tree/tutorial/lab6)
- **Objective**: Understand the impacts of the temporal under-utilization for IMC cores.
- **Key Learnings**:
  - Understanding the inputs and outputs specific to in-memory computing (IMC) cores.
  - Understanding how the temporal utilization correlates with cell types and memory bandwidth.
  - Analyzing the results and understanding how to identify the system-level bottlenecks when designing IMC cores.

## 🚀 Getting Started

Visit the [Installation Guide](https://kuleuven-micas.github.io/zigzag/installation.html) for step-by-step instructions to set up ZigZag on your system.

## 📖 Resources

Get up to speed with ZigZag using our resources:
- Check out the [Getting Started Guide](https://kuleuven-micas.github.io/zigzag/getting-started.html).
- Explore the [Jupyter Notebook Demo](https://github.com/ZigZag-Project/zigzag-demo) to see ZigZag in action.

## 💻 Contributing

We welcome contributions! Feel free to fork the repository, submit pull requests, or open issues. Check our [Contributing Guidelines](CONTRIBUTING.md) for more details.

#### ⭐ Please consider starring this repository to stay up to date!