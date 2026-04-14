"""Central path constants for the song phenotyping pipeline output tree.

All pipeline modules import from here so that directory names can be
changed in one place.  Use :func:`stage_path` to build absolute paths.

Output tree layout
------------------
::

    <save_path>/<bird>/
        <run_name>/              ← e.g. "d5dfde49" (SHA256[:8] of config)
            stages/
                01_specs/        ← Stage A  spectrogram HDF5 files
                02_features/     ← Stage B  flattened feature HDF5 files
                03_embeddings/   ← Stage C  UMAP HDF5 + pkl models
                04_labels/       ← Stage D  cluster label HDF5 files
                syllable_database/  ← syllable_features.{csv,h5} + feature_params.json
                05_phenotype/    ← Stage E  detailed phenotype pkl files
            results/
                master_summary.csv
                phenotype_results.csv
                run_config.json
                catalog/         ← HTML song catalogs
                plots/           ← PDFs / images

Each unique combination of computational parameters produces a different
``<run_name>`` directory, isolating run artifacts completely.  The
``stages/`` subtree contains internal computation artifacts; ``results/``
contains outputs intended for human inspection.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Top-level subdirectory names
# ---------------------------------------------------------------------------

STAGES_DIR  = "stages"
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Stage subdirectories (relative to run root)
# ---------------------------------------------------------------------------

SPECS_DIR       = "stages/01_specs"
FEATURES_DIR    = "stages/02_features"
EMBEDDINGS_DIR  = "stages/03_embeddings"
LABELS_DIR      = "stages/04_labels"
SYLLABLE_DB_DIR = "stages/syllable_database"
PHENOTYPE_DIR   = "stages/05_phenotype"

# ---------------------------------------------------------------------------
# Results subdirectories (relative to run root)
# ---------------------------------------------------------------------------

CATALOG_DIR = "results/catalog"
PLOTS_DIR   = "results/plots"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def stage_path(bird_root: str | Path, subdir: str) -> Path:
    """Return ``bird_root / subdir`` as an absolute :class:`~pathlib.Path`.

    Parameters
    ----------
    bird_root:
        Root directory for a single bird, e.g. ``E:/pipeline_runs/or18or24``.
    subdir:
        One of the ``*_DIR`` constants defined in this module, or any
        relative path string.

    Example
    -------
    >>> from song_phenotyping.tools.pipeline_paths import stage_path, SPECS_DIR
    >>> stage_path("/data/pipeline_runs/or18or24", SPECS_DIR)
    PosixPath('/data/pipeline_runs/or18or24/stages/01_specs')
    """
    return Path(bird_root) / subdir


# ---------------------------------------------------------------------------
# Run-scoped helpers
# ---------------------------------------------------------------------------


def run_root(bird_root, run_name: str) -> Path:
    """Return ``<bird_root>/<run_name>`` as a :class:`~pathlib.Path`.

    Parameters
    ----------
    bird_root:
        Root directory for a single bird, e.g. ``/data/pipeline_runs/or18or24``.
    run_name:
        Human-readable or hash-derived run identifier, e.g. ``"baseline"``
        or ``"a1b2c3d4"``.

    Example
    -------
    >>> run_root("/data/or18or24", "baseline")
    PosixPath('/data/or18or24/baseline')
    """
    return Path(bird_root) / run_name


def run_stage_path(bird_root, run_name: str, subdir: str) -> Path:
    """Return ``<bird_root>/<run_name>/<subdir>`` as a :class:`~pathlib.Path`.

    Parameters
    ----------
    bird_root:
        Root directory for a single bird.
    run_name:
        Run identifier (see :func:`run_root`).
    subdir:
        One of the stage ``*_DIR`` constants, e.g. :data:`SPECS_DIR`.

    Example
    -------
    >>> run_stage_path("/data/or18or24", "baseline", SPECS_DIR)
    PosixPath('/data/or18or24/baseline/stages/01_specs')
    """
    return run_root(bird_root, run_name) / subdir
