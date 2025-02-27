# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "ZigZag"
copyright = "2022, Arne Symons"
author = "Arne Symons"

# The full version, including alpha/beta/rc tags
release = "2.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc"]

# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.napoleon',
#     'sphinx.ext.viewcode',
#     'autoapi.extension'
# ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_material"

# Logo
html_logo = "zigzag_logo_white_32x32.svg"

# Favicon
html_favicon = "zigzag_logo_white_32x32.svg"

# Material theme options
html_theme_options = {
    "base_url": "http://kuleuven-micas.github.io/zigzag/",
    "repo_url": "https://github.com/kuleuven-micas/zigzag",
    "repo_name": "ZigZag Framework",
    "google_analytics_account": "UA-XXXXX",
    "html_minify": True,
    "css_minify": True,
    "nav_title": "ZigZag Framework",
    # 'logo_icon': '&#x1D56B',  # '&#x2124',
    "globaltoc_depth": 2,
    "color_primary": "blue-grey",
    "color_accent": "grey",
}

html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
