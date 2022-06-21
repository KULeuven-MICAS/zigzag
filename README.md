# zigzag-v2
This repository presents the novel version of our tried-and-tested HW Architecture-Mapping Design Space Exploration (DSE) framework for Deep Learning (DL) accelerators. ZigZag bridges the gap between algorithmic DL decisions and their acceleration cost on specialized accelerators through a fast and accurate HW cost estination. 

A crucial part in this is the mapping of the algorithmic computations onto the computational HW resources and memories. In the framework, multiple engines are provided that can automatically find optimal mapping points in this search space.

## Environment

We recommend setting up an anaconda environment.

A `conda-spec-file-{platform}.txt` is provided to set up the environment for linux64 and win64.

`$ conda create --name myzigzagenv --file conda-spec-file-{platform}.txt`

Alternatively, you can also install all packages directly through pip using the pip-requirements.txt with the command:

`$ pip install -r pip-requirements.txt`

## Example

The inputs folder provides some example accelerator/workload/settings.

For example, the xception snippet running on a single core accelerator using loma:

`$ python main.py --workload inputs.workshop.workload --accelerator inputs.workshop.accelerator`
