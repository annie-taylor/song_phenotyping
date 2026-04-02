Usage guide
===========

The pipeline runs in five sequential stages (A → E).  Each stage reads the
output of the previous one, so they must be run in order.  A convenience
script ``run_pipeline.py`` at the project root chains all five stages for
one or more birds.

.. contents:: On this page
   :local:
   :depth: 2

----

Quick start
-----------

Edit the paths at the top of ``run_pipeline.py``, then run::

    python run_pipeline.py

Outputs land under ``SAVE_PATH/<bird>/`` on your local drive.

----

Stage A — Spectrogram ingestion
--------------------------------

Stage A reads raw audio recordings and their segmentation metadata, computes
a mel spectrogram for each detected syllable, and saves the results to
``<bird>/syllable_data/specs/syllables_<song_id>.h5``.

Two annotation formats are supported:

**evsonganaly** (``batch.txt.labeled`` metadata files)::

    from song_phenotyping.ingestion import (
        filepaths_from_evsonganaly,
        save_specs_for_evsonganaly_birds,
    )

    meta, audio = filepaths_from_evsonganaly(
        wav_directory="/data/raw/evsonganaly",
        bird_subset=["or18or24"],          # omit to process all birds
    )
    save_specs_for_evsonganaly_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path="/data/pipeline_runs",
        songs_per_bird=50,                 # None = all songs
    )

**wseg** (``<bird>/song/*.wav.not.mat`` segmentation files)::

    from song_phenotyping.ingestion import (
        filepaths_from_wseg,
        save_specs_for_wseg_birds,
    )

    meta, audio = filepaths_from_wseg(
        seg_directory="/data/raw/wseg",
        bird_subset=["bu78bu77"],
    )
    save_specs_for_wseg_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path="/data/pipeline_runs",
        songs_per_bird=50,
    )

Each output HDF5 file contains the arrays ``spectrograms``, ``manual``,
``position_idxs``, and ``hashes``.

Stage A1 — Fixed-length slicing (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer equal-duration windows instead of boundary-segmented syllables,
use :mod:`song_phenotyping.slicing` as a drop-in replacement for Stage A::

    from song_phenotyping.slicing import slice_syllable_files_from_evsonganaly

    slice_syllable_files_from_evsonganaly(
        wav_directory="/data/raw/evsonganaly",
        save_path="/data/pipeline_runs",
        slice_length=50,   # ms
    )

----

Stage B — Flattening
---------------------

Stage B reshapes each ``(n_syllables, n_freq, n_time)`` spectrogram stack
into a ``(n_features, n_syllables)`` matrix ready for UMAP::

    from song_phenotyping.flattening import flatten_bird_spectrograms

    flatten_bird_spectrograms(
        directory="/data/pipeline_runs",
        bird="or18or24",
    )

Output: ``<bird>/syllable_data/flattened/flattened_<song_id>.h5``

----

Stage C — UMAP embedding
-------------------------

Stage C projects the high-dimensional feature vectors into 2-D using UMAP,
exploring a grid of hyperparameters::

    from song_phenotyping.embedding import explore_embedding_parameters_robust

    explore_embedding_parameters_robust(
        save_path="/data/pipeline_runs",
        bird="or18or24",
        n_neighbors_list=[10, 30, 50],
        min_dists=[0.1, 0.3],
        metrics=["euclidean", "cosine"],
    )

One ``.h5`` file per parameter combination is written to
``<bird>/syllable_data/embeddings/``.  Each file contains arrays
``embeddings``, ``hashes``, and ``labels``.

----

Stage D — Clustering and labelling
------------------------------------

Stage D runs HDBSCAN over every embedding, evaluates each clustering with a
set of internal metrics, and writes a ranked ``master_summary.csv``::

    from song_phenotyping.labelling import label_bird, DEFAULT_HDBSCAN_GRID

    label_bird(
        save_path="/data/pipeline_runs",
        bird="or18or24",
        metrics=["silhouette", "dbi", "ch"],
        hdbscan_params=[p.to_dict() for p in DEFAULT_HDBSCAN_GRID],
    )

Cluster label files are written to
``<bird>/syllable_data/labelling/<umap_id>/``.

Custom HDBSCAN grid
~~~~~~~~~~~~~~~~~~~

::

    from song_phenotyping.labelling import HDBSCANParams

    my_grid = [
        HDBSCANParams(min_cluster_size=5,  min_samples=2),
        HDBSCANParams(min_cluster_size=10, min_samples=3),
        HDBSCANParams(min_cluster_size=20, min_samples=5),
    ]
    label_bird(..., hdbscan_params=[p.to_dict() for p in my_grid])

----

Stage E — Phenotyping
----------------------

Stage E reads the top-ranked clusterings and computes phenotypic measures
(vocabulary, entropy, transition matrices, repeat patterns)::

    from song_phenotyping.phenotyping import phenotype_bird, PhenotypingConfig

    cfg = PhenotypingConfig(
        use_top_n_clusterings=5,
        generate_plots=True,
    )
    phenotype_bird(
        bird_path="/data/pipeline_runs/or18or24",
        config=cfg,
    )

Outputs:

- ``phenotype_results.csv`` — one row per clustering rank with all phenotype
  columns.
- ``syllable_data/phenotype_detailed/automated_phenotype_data_rank*.pkl`` —
  full data structures for downstream PDF generation.

----

Visual inspection — HTML catalogs
----------------------------------

After Stage E you can generate interactive HTML catalogs for visual inspection
of the clustering results::

    from song_phenotyping.catalog import generate_all_catalogs

    results = generate_all_catalogs(
        bird_path="/data/pipeline_runs/or18or24",
        rank=0,   # use the best-ranked clustering
    )
    print(results["song_catalog"])         # continuous song view
    print(results["syllable_types_auto"])  # per-label spectrogram grid

HTML files are written to ``<bird>/syllable_data/html/``.  Open them in any
browser — no server required.

----

Configuration
-------------

Path configuration is loaded from ``config.yaml`` via
:class:`~song_phenotyping.tools.project_config.ProjectConfig`::

    from song_phenotyping.tools.project_config import ProjectConfig

    cfg = ProjectConfig.load()

    # Resolve a bird directory on the local cache drive
    bird_path = cfg.bird_dir("or18or24", experiment="evsong test")

    # Access the Macaw server root (None if not mounted)
    print(cfg.macaw_root)

Spectrogram parameters are controlled by
:class:`~song_phenotyping.tools.spectrogram_configs.SpectrogramParams`.
The defaults match the settings used for Bengalese finch recordings but can
be overridden::

    from song_phenotyping.tools.spectrogram_configs import SpectrogramParams

    params = SpectrogramParams(
        min_freq=500,
        max_freq=8000,
        song_gap=200,    # ms between songs
    )
