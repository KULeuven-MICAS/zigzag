.. ZigZag documentation master file, created by
   sphinx-quickstart on Wed Sep  7 17:43:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ZigZag's documentation!
==================================

ZigZag is a hardware architecture (HW)-mapping design space exploration (DSE) framework for deep learning (DL) accelerators. ZigZag bridges the gap between algorithmic DL decisions and their HW acceleration cost on specialized accelerators through a fast and accurate analytical HW cost estimation model.

A crucial part in this is mapping the algorithmic computations onto the computational HW resources and memories. In the framework, multiple engines are provided that automatically find optimal mapping points in this search space.

Related Publications
====================

`Linyan Mei, Pouya Houshmand, Vikram Jain, Sebastian Giraldo, and Marian Verhelst, "ZigZag: Enlarging joint architecture-mapping design space exploration for DNN accelerators." IEEE Transactions on Computers 70.8 (2021): 1160-1174. <https://ieeexplore.ieee.org/abstract/document/9360462>`_

`Arne Symons, Linyan Mei, and Marian Verhelst, "Loma: Fast auto-scheduling on dnn accelerators through loop-order-based memory allocation." 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS). IEEE, 2021. <https://ieeexplore.ieee.org/abstract/document/9458493>`_

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting-started
   user-guide
   future
   contribute


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
