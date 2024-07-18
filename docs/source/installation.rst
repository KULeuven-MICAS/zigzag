===================
Installing ZigZag
===================

Installing as a package
=======================

If you're interested in using the ZigZag framework as an end user, and not interested in making any modifications to its internals, you can make ZigZag available by directly installing it using pip in your ``venv`` or ``conda`` environment:

.. code-block:: sh

    pip install zigzag-dse

After installation, you can take a look at the :doc:`api` documentation on how to use the api.

Manual clone
============

If you want to add custom functionality to the framework, you can clone the repository manually:

Prerequisites
-------------

* ``git``: for cloning the repository
* ``pip``: for installing the required packages
* ``python>=3.11``: for running the framework

Installation
------------

Clone the repository

.. code-block:: sh

    git clone git@github.com:KULeuven-MICAS/zigzag.git

or 

.. code-block:: sh

    git clone https://github.com/KULeuven-MICAS/zigzag.git

Install requirements through pip. Alternatively, anaconda spec file is also provided.

.. code-block:: sh

    pip install -r requirements.txt

Once the dependencies are installed, ZigZag can be run through the main file with provided arguments. More details are provided in :doc:`getting-started`.
