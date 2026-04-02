Installation
============

Requirements
------------

- Python 3.9 or later
- A working `conda <https://docs.conda.io>`_ environment (recommended)

Install from source
-------------------

Clone the repository and install in editable mode::

    git clone https://github.com/annie-taylor/song_phenotyping.git
    cd song_phenotyping
    pip install -e .

This installs the ``song_phenotyping`` package and all runtime dependencies
(NumPy, SciPy, PyTables, UMAP-learn, HDBSCAN, pandas, scikit-learn, PyYAML,
tqdm, matplotlib, seaborn).

Optional extras
---------------

To build the documentation locally::

    pip install -e ".[docs]"
    cd docs && make html

To install development tools (pytest)::

    pip install -e ".[dev]"

Configuration
-------------

Copy ``config.yaml.example`` to ``config.yaml`` in the project root and edit
the paths for your machine::

    cp config.yaml.example config.yaml

The two required fields are:

.. code-block:: yaml

    paths:
      local_cache: /Volumes/Extreme SSD   # root of your local data cache
      run_registry: db.sqlite3            # path to the SQLite run registry

``macaw_root`` is optional and can be left unset — it will be auto-detected if
the Macaw server is mounted.
