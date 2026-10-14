# Configuration file for the Sphinx documentation builder.
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib.metadata

project = "hanoi"
author = "Jun Ishigohoka"
copyright = "2025, Jun Ishigohoka"
release = importlib.metadata.version("hanoi")  # read from pyproject.toml via the installed package
version = release

extensions = [
    "myst_parser",             # lets Sphinx read Markdown (the README)
    "sphinx.ext.autodoc",      # pulls documentation from docstrings
    "sphinx.ext.autosummary",  # summary tables for the API page
    "sphinx.ext.napoleon",     # understands numpy-style docstrings
    "sphinx.ext.viewcode",     # links to highlighted source code
    "sphinx.ext.intersphinx",  # links to numpy/scipy/python docs
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

myst_enable_extensions = ["dollarmath"]  # lets Markdown use $...$ math

html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build"]
