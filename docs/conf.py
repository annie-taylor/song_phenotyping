"""Sphinx configuration for song_phenotyping documentation."""

import os
import sys

# Make the package importable from the repo root
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project info
# ---------------------------------------------------------------------------
project = "song_phenotyping"
copyright = "2024, Annie Taylor"
author = "Annie Taylor"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from source
    "sphinx.ext.autosummary",   # generate summary tables
    "sphinx.ext.napoleon",      # NumPy / Google style docstrings
    "sphinx.ext.viewcode",      # [source] links in docs
    "sphinx.ext.intersphinx",   # cross-links to NumPy, SciPy docs
]

# Napoleon settings (scipy / NumPy style)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

# intersphinx: link to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
