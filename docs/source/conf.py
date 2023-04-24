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

# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "ibicus"
copyright = "2022, ECMWF"
author = "ECMWF"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    # "matplotlib.sphinxext.plot_directive",
    "numpydoc",  # "sphinx.ext.napoleon"
]

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
html_theme = "sphinx_rtd_theme"  # "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Some new settings ------------------------------------------------------------------
# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "none"  # description
autodoc_typehints_format = "fully-qualified"

# Styling of autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# Don't show class signature with the class' name.
autodoc_class_signature = "mixed"
autodoc_member_order = "groupwise"

# Do not preserve default arguments
autodoc_preserve_defaults = False

# Google Search Console authentication as extra file to copy into the build directory
html_extra_path = ["google0b2594d760cfdda3.html"]

# Sort by order specified and not alphabetically
autodoc_member_order = "bysource"

"""numpydoc_attributes_as_param_list = True
numpydoc_show_class_members = True
"""
