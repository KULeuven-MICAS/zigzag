# ZigZag
This repository presents the novel version of our tried-and-tested HW Architecture-Mapping Design Space Exploration (DSE) Framework for Deep Learning (DL) accelerators. ZigZag bridges the gap between algorithmic DL decisions and their acceleration cost on specialized accelerators through a fast and accurate HW cost estimation. 

A crucial part in this is the mapping of the algorithmic computations onto the computational HW resources and memories. In the framework, multiple engines are provided that can automatically find optimal mapping points in this search space.

## Installation

Please take a look at the [Installation](https://zigzag-project.github.io/zigzag/installation.html) page of our documentation.

## Getting Started

Please take a look at the [Getting Started](https://zigzag-project.github.io/zigzag/getting-started.html) page on how to get started using ZigZag.

## Recent changes

In this novel version, we have: 
- Added an interface with ONNX to directly parse ONNX models
- Overhauled our HW architecture definition to:
    - include multi-dimensional (>2D) MAC arrays.
    - include accurate interconnection patterns.
    - include multiple flexible accelerator cores.
- Enhanced the cost model to support complex memories with variable port structures.
- Revamped the whole project structure to be more modular.
- Written the project with OOP paradigms to facilitate user-friendly extensions and interfaces.


