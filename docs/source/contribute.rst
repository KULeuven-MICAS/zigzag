===========================
Contribute
===========================

.. _contributing guidelines:

Contributing guidelines
=======================

When contributing to the framework, please consider the following guidelines:

* Use Google's `Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_
* Use Google docstrings to document your classes, functions, methods, .... Examples can be found throughout the code and `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
* Update the documentation accordingly

Upgrading the project version (for ZigZag developers)
=====================================================

In order to upgrade the project version, we use Python packages called bumpver, build and twine. These can be installed as follows:

.. code-block:: sh

    pip install bumpver
    pip install build
    pip install twine
    pip install pytest
    pip install typeguard
    pip install ruff

First, pull to make sure you have all the remote changes. Merge any conflicts with your new changes, and commit.

Check linting and typing:

.. code-block:: sh
 
    python -m ruff check --select=E9,F63,F7,F82 --target-version=py37 .

Test whether functionalities have been broken by the changes:

.. code-block:: sh

    python -m pytest --typeguard-packages=zigzag tests/main/test_origin/
    python -m pytest --typeguard-packages=zigzag tests/main/test_imc/

Finally, execute the following commands to build and publish the package :

.. code-block:: sh

    bumpver update --patch
    python -m build
    twine upload dist/zigzag_dse-x.y.z-<...>.whl dist/zigzag-dse-x.y.z.tar.gz

Documentation
=============

The ZigZag project provides several different ways of documentation:

1. There are many :doc:`publications` available which are related to the project.
2. The `general documentation <https://kuleuven-micas.github.io/zigzag/index.html>`_ on these pages allows everyone to get familiar with the framework.
3. There is a :doc:`code-documentation` which provides more details about the implementation of this framework.

Writing new parts for the general documentation
-----------------------------------------------

When adding new functionality, it's mandatory to document what this does, how it achieves this, and how to use the newly added functionality.
Explicit documentation resides in the `docs/` folder, using the `reStructuredText <https://docutils.sourceforge.io/rst.html>`_ format (.rst).

When writing new documentation, decide if it would best fit in an existing document, or in a new one. If you decide to create a new file, create this file under docs/source/ and use lower-case letters with a hyphen (-) in between words. After writing the new file, you need to add it to the `toctree` in `docs/source/index.rst`.

Building the general documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The general documentation (same as on this webpage) is build using Sphinx. You should install both sphinx and sphinx-press-theme, which is easy through the requirements.txt file provided in `docs/`.


.. code-block:: sh

    cd docs/
    pip install -r requirements.txt

After, you can simply use the following commands to build the general documentation locally:

.. code-block:: sh

    sphinx-build -b html source build

Use the ``index.html`` file in the ``docs/build/`` folder as the entry point to the general documentation.

Writing code which supports the code documentation with Doxygen
----------------------------------------------------------------

Please follow the `general Doxygen guidlines <https://www.doxygen.nl/manual/docblocks.html#pythonblocks:~:text=Here%20is%20the%20same%20example%20again%20but%20now%20using%20doxygen%20style%20comments%3A>`_ to document new code added to the ZigZag project.

The the following parts of your code should be documented with comments in the Doxygen format:

1. Classes (including the parameter of the constructor)
2. Functions (including the parameter of it)

Building the code documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `code documentation <doxygen/html/index.html>`_ of the ZigZag project can be build locally using Doxygen. You have to download and install Doxygen as described on `this page. <https://www.doxygen.nl/download.html>`_

After successfully installing Doxygen, you can use the provided `configuration file <https://github.com/KULeuven-MICAS/zigzag/blob/master/docs/doxygen-conf>`_ to generate the code documentation locally. This can be done either through importing the configuration file into the `GUI of Doxygen <https://www.doxygen.nl/manual/doxywizard_usage.html>`_ or through running

.. code-block:: sh

    cd docs
    doxygen doxygen-conf

Use the ``index.html`` file in the ``docs/html/`` folder as the entry point to the code documentation.
