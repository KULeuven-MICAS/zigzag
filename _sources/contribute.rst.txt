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

 First, pull to make sure you have all the remote cahnges. Merge any conflicts with your new changes, and commit. Then, execute the following commands:

.. code-block:: sh

    bumpver update --patch
    python -m build
    twine upload dist/zigzag_dse-x.y.z-<...>.whl dist/zigzag-dse-x.y.z.tar.gz

Documentation
=============

When adding new functionality, it's mandatory to document what this does, how it achieves this, and how to use the newly added functionality.
Explicit documentation resides in the `docs/` folder, using the `reStructuredText <https://docutils.sourceforge.io/rst.html>`_ format (.rst).

Writing new documentation
-------------------------

When writing new documentation, decide if it would best fit in an existing document, or in a new one. If you decide to create a new file, create this file under docs/source/ and use lower-case letters with a hyphen (-) in between words. After writing the new file, you need to add it to the `toctree` in `docs/source/index.rst`.

Building the documentation
--------------------------

The documentation is build using Sphinx. You should install both sphinx and sphinx-press-theme, which is easy through the requirements.txt file provided in `docs/`.


.. code-block:: sh

    cd docs/
    pip install -r requirements.txt

After, you can build the documentation using the provide Makefile (Linux). For Windows, you can run the ``make.bat`` file.

.. code-block:: sh

    make html

In the future, this will be automated through Github Actions.
