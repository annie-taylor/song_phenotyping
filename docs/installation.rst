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
    conda env create -f environment.yml
    conda activate song_phenotyping
    pip install -e .

This installs the ``song_phenotyping`` package and all runtime dependencies
(NumPy, SciPy, PyTables, UMAP-learn, HDBSCAN, pandas, scikit-learn, PyYAML,
tqdm, matplotlib).

Optional extras
---------------

To build the documentation locally::

    pip install -e ".[docs]"
    cd docs && make html

To install development tools (pytest)::

    pip install -e ".[dev]"

----

Configuration
-------------

The pipeline reads all settings from ``config.yaml`` in the project root.
This file is **gitignored** — each machine keeps its own copy. A fully
annotated template is provided at ``config.yaml.example``.

Step 1 — create your local config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    cp config.yaml.example config.yaml

Then open ``config.yaml`` and edit the values for your machine. You only
**need** to set the two required path fields; everything else has a sensible
default.

Step 2 — set required paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    paths:
      local_cache: /Volumes/Extreme SSD   # macOS example
      # local_cache: E:/                  # Windows example

      run_registry: db.sqlite3            # SQLite run log; relative = project root

    pipeline:
      save_path: /Volumes/Extreme SSD/pipeline_runs
      evsong_source: /Volumes/Extreme SSD/birds/evsong

``local_cache``
    Root of your local data cache (an external SSD or a fast local directory).
    Bird data and pipeline outputs that need to be kept close to the machine
    should live under here.

``run_registry``
    Path to the SQLite database that records every pipeline run. A relative
    path is resolved from the project root.  The default (``db.sqlite3``)
    places it in the project root — fine for most setups.

``save_path``
    Root output directory.  One sub-directory per bird is created here::

        <save_path>/
        └── or18or24/
            ├── stages/       ← per-stage HDF5 artefacts
            └── results/      ← human-facing CSVs, plots, HTML catalogs

``evsong_source``
    Parent directory that contains evsonganaly bird folders.  Each bird is
    expected at ``<evsong_source>/<bird_id>/source/``.  Set to ``null`` to
    skip evsonganaly processing.

Step 3 — optional pipeline settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All keys below have defaults; omit any key to accept the default.

.. code-block:: yaml

    pipeline:
      wseg_metadata: null      # WhisperSeg metadata dir; null = skip wseg
      birds: null              # null = all; or [or18or24, bu78bu77]
      songs_per_bird: 30       # null = all songs; small number for quick tests
      songs_seed: 42           # null = random; int = reproducible subset
      run_name: null           # null = auto-hash; str = fixed name (e.g. "baseline")
      generate_catalog: true   # write HTML catalogs after Stage E

``wseg_metadata``
    Directory containing WhisperSeg-format metadata.  Set only if you have
    wseg annotations; ``null`` skips that ingestion path entirely.

``birds``
    ``null`` auto-discovers every bird found under ``evsong_source`` /
    ``wseg_metadata``.  To process a subset, provide an explicit list::

        birds: [or18or24, bu78bu77]

``songs_per_bird``
    Limits the number of songs loaded per bird in Stage A.  Useful for
    quick smoke tests (set to 5–10) without touching the rest of the config.
    ``null`` processes all available songs.

``songs_seed``
    When ``songs_per_bird`` is less than the total available songs, Stage A
    selects a random subset.  Setting a seed makes the selection reproducible
    across machines and re-runs.  ``null`` gives a fresh random draw each time.

``run_name``
    By default (``null``) the pipeline computes a short hash from all
    computational parameters and uses it as the run directory name.  Changing
    any parameter automatically produces a new directory, so old results are
    never silently overwritten.  Set a fixed string (e.g. ``"baseline"``) if
    you want to pin a name across runs regardless of parameters.

``generate_catalog``
    Whether to build HTML syllable catalogs after Stage E completes.
    The catalogs are written to ``<save_path>/<bird>/results/catalog/`` and
    can be opened in any browser without a server.

Spectrogram parameters (Stage A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    pipeline:
      spectrograms:
        nfft: 512
        hop: 1
        target_shape: [513, 300]   # [freq_bins, time_bins]
        min_freq: 400.0
        max_freq: 10000.0
        max_dur: 0.080             # seconds
        fs: 32000.0
        save_inst_freq: false
        save_group_delay: false
        duration_feature_weight: 0.0
        use_warping: false

``nfft``
    FFT window size in samples.  Larger values give finer frequency resolution
    at the cost of time resolution.

``target_shape``
    ``[freq_bins, time_bins]`` — the fixed size each syllable spectrogram is
    resampled to.  All downstream stages expect this shape; changing it
    invalidates any existing Stage A outputs.

``min_freq`` / ``max_freq``
    Frequency band (Hz) retained in the spectrogram.  Set to match the
    typical range of the species you are analysing.

``max_dur``
    Syllables longer than this duration (seconds) are truncated or split.

``save_inst_freq`` / ``save_group_delay``
    Append an instantaneous-frequency or group-delay channel alongside the
    magnitude spectrogram.  Disabled by default; may improve clustering for
    some species.

``duration_feature_weight``
    ``0.0`` (default) disables duration as a feature.  Setting to ``~1.0``
    appends a synthetic "duration" time-bin column, giving UMAP a hint about
    syllable length.

``overwrite_existing`` *(Stage A)*
    ``false`` (default) skips any bird/song whose spectrogram HDF5 file is
    already present on disk, making re-runs fast.  Set to ``true`` to force
    recomputation — useful when the source audio has changed or you have
    altered spectrogram parameters that affect the output (e.g. ``nfft``,
    ``target_shape``).

    .. note::
       Changing computational parameters (``nfft``, ``target_shape``, etc.)
       automatically produces a new run directory via the config hash, so you
       rarely need to set ``overwrite_existing: true`` unless you want to
       rebuild within the *same* run.

Embedding parameters (Stage C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    pipeline:
      embedding:
        n_neighbors: 20
        min_dist: 0.5
        metric: euclidean
        n_components: 2
        subsample_seed: 42
        n_neighbors_grid: [5, 10, 20, 50, 100]
        min_dist_grid: [0.01, 0.05, 0.1, 0.3, 0.5]
        max_workers: null
        overwrite: false

``n_neighbors`` / ``min_dist`` / ``metric``
    Default UMAP hyperparameters, used as starting values for the grid search
    and as the single-run default.

``n_neighbors_grid`` / ``min_dist_grid``
    All combinations are tried during the grid search.  The best is chosen by
    the Stage D clustering metrics.  Narrowing these lists speeds up Stage C
    significantly.

``subsample_seed``
    When the dataset is too large for UMAP, a random subset of syllables is
    drawn.  This seed makes that draw reproducible.

``max_workers``
    ``null`` lets the pipeline decide how many parallel workers to use based
    on available memory.  Set an integer to cap usage on shared machines.

``overwrite`` *(Stage C)*
    ``false`` (default) skips any UMAP parameter combination whose ``.h5``
    embedding file already exists on disk and is compatible with the current
    data (verified by sample count and hash overlap).  This makes iterative
    runs much faster — only new grid combinations are computed.

    Set to ``true`` to force recomputation of all embeddings.  This is needed
    when the flattened features have changed (e.g. after re-running Stage A or
    B with different parameters within the same run directory).

    .. note::
       Like ``overwrite_existing``, this flag is excluded from the run-name
       hash — toggling it does not create a new run directory.

Clustering parameters (Stage D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    pipeline:
      labelling:
        metrics: [silhouette, dbi]
        # metric_weights:
        #   silhouette: 2.0
        #   dbi: 1.0
        replace_labels: false
        hdbscan_grid: null
        generate_cluster_pdf: false
        max_workers: null

``metrics``
    Clustering quality metrics used to rank HDBSCAN results.  Supported values:
    ``silhouette``, ``dbi``, ``ch``, ``dunn``, ``nmi``.  ``ch``
    (Calinski-Harabasz) is excluded by default as it tends to favour large
    clusters.

``metric_weights``
    Optional per-metric weights for the composite ranking score.  Comment out
    to use equal weights.

``replace_labels`` *(Stage D)*
    ``false`` (default) skips re-clustering if label files already exist for a
    parameter combination.  Set to ``true`` to delete existing label files and
    force a full re-run — equivalent to the ``overwrite`` flags above but
    scoped to Stage D.

``hdbscan_grid``
    ``null`` uses the built-in default grid
    (``min_cluster_size ∈ [5, 20, 60]``, ``min_samples ∈ [5, 15]``).
    Override with a custom grid::

        hdbscan_grid:
          min_cluster_size: [10, 30]
          min_samples: [5, 10]

Phenotyping parameters (Stage E)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    pipeline:
      phenotyping:
        min_syllable_proportion: 0.02
        repeat_significance_threshold: 0.4
        dyad_threshold: 0.95
        use_top_n_clusterings: 3
        generate_plots: true

``min_syllable_proportion``
    Syllable types that appear in fewer than this fraction of songs are
    excluded from phenotype calculations.

``repeat_significance_threshold`` / ``dyad_threshold``
    Thresholds governing repeat and dyad detection in the transition matrix
    analysis.

``use_top_n_clusterings``
    How many top-ranked clusterings (from Stage D) are averaged to produce
    the final phenotype.

``generate_plots``
    Write phenotype summary figures to ``<save_path>/<bird>/results/plots/``.

----

Macaw server configuration
---------------------------

The pipeline can read source audio directly from the Macaw network server.
By default it attempts to **auto-detect** the mount point based on your
operating system:

- **macOS**: ``/Volumes/users/``
- **Linux**: ``/run/user/1005/gvfs/smb-share:server=macaw.local,share=users/``
- **Windows**: ``Z:\\``

These defaults are lab-specific and **may not match your setup** — in
particular the Linux path encodes a user ID (``1005``) that differs between
accounts, and the Windows drive letter varies by machine.

To set your Macaw mount point explicitly, add it to ``config.yaml``::

    paths:
      macaw_root: /Volumes/users        # macOS (standard mount)
      # macaw_root: /run/user/1234/gvfs/smb-share:server=macaw.local,share=users/
      # macaw_root: Z:/                 # Windows

If Macaw is not used in your workflow, set it to ``null`` (or leave it
absent) and the pipeline will simply skip any server-path resolution::

    paths:
      macaw_root: null

You can verify the detected root at runtime::

    from song_phenotyping.tools.project_config import ProjectConfig
    cfg = ProjectConfig.load()
    print(cfg.macaw_root)   # None if not mounted or not configured
