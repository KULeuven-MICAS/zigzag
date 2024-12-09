============
Installation
============

This guide provides step-by-step instructions for setting up ZigZag.

.. note::

   If you're already familiar with Python development environments and IDEs, skip to the **Installing ZigZag** section below.

Environment
===========

To use ZigZag effectively, we recommend setting up a Python virtual environment and optionally using an IDE like VS Code. Of course, you may use any IDE of your choice.

Basic Setup
-----------

Follow these steps to prepare your system for Python development:

Linux/Mac
~~~~~~~~~
1. **Install Python 3.11 or newer**:
   Ensure you have Python installed. You can check with:

   .. code-block:: sh

       python3 --version

   Install Python if necessary via your package manager or from the `official website <https://www.python.org/downloads/>`_.

2. **Set up a virtual environment**:

   .. code-block:: sh

       python3 -m venv zigzag_env
       source zigzag_env/bin/activate

3. **Install VS Code (Optional)**:
   Download and install VS Code from `here <https://code.visualstudio.com/>`_. Add the Python extension for an enhanced development experience.

Windows
~~~~~~~
1. **Install Python 3.11 or newer**:
   Download Python from the `official website <https://www.python.org/downloads/>`_. Ensure you check the box to "Add Python to PATH" during installation.

2. **Set up a virtual environment**:
   Open Command Prompt or PowerShell and run:
   
   .. code-block:: sh

       python -m venv zigzag_env
       zigzag_env\Scripts\activate

3. **Install VS Code (Optional)**:
   Download and install VS Code from `here <https://code.visualstudio.com/>`_. Add the Python extension for an enhanced development experience.

Installing ZigZag
=================

Installing as a Package
-----------------------

If you're an end user and don't need to modify ZigZag's internals, you can install it directly via pip:

.. code-block:: sh

    pip install zigzag-dse

After installation, you can explore the :doc:`api` documentation to understand how to use the framework.

Manual Clone
------------

For users interested in adding custom functionality or contributing to ZigZag, follow these steps:

Prerequisites
~~~~~~~~~~~~~
Ensure you have the following installed:
- ``git``: For cloning the repository.
- ``pip``: For installing the required packages.
- ``python>=3.11``: For running the framework.

Installation Steps
~~~~~~~~~~~~~~~~~~

1. **Clone the repository**:
   Use one of the following commands:
   .. code-block:: sh

       git clone git@github.com:KULeuven-MICAS/zigzag.git

   Or:
   .. code-block:: sh

       git clone https://github.com/KULeuven-MICAS/zigzag.git

2. **Install dependencies**:
   Navigate to the cloned repository folder and install the required packages:
   .. code-block:: sh

       cd zigzag
       pip install -r requirements.txt

   Alternatively, an Anaconda spec file is also provided if you prefer to use conda.

3. **Run ZigZag**:
   You can now execute ZigZag through the main file with the provided arguments. For more details, see the :doc:`getting-started` documentation.