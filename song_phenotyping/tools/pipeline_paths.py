"""Central path constants for the song phenotyping pipeline output tree.

All pipeline modules import from here so that directory names can be
changed in one place.  Use :func:`stage_path` to build absolute paths.

Output tree layout
------------------
::

    <save_path>/<bird>/
        stages/
            01_specs/        ← Stage A  spectrogram HDF5 files
            02_features/     ← Stage B  flattened feature HDF5 files
            03_embeddings/   ← Stage C  UMAP HDF5 + pkl models
            04_labels/       ← Stage D  cluster label HDF5 files
            05_phenotype/    ← Stage E  detailed phenotype pkl files
        results/
            master_summary.csv
            phenotype_results.csv
            catalog/         ← HTML song catalogs
            plots/           ← PDFs / images

The ``stages/`` subtree contains internal computation artifacts that are
not typically opened by hand.  The ``results/`` subtree contains outputs
intended for human inspection.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Top-level subdirectory names
# ---------------------------------------------------------------------------

STAGES_DIR  = "stages"
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Stage subdirectories (relative to bird root)
# ---------------------------------------------------------------------------

SPECS_DIR       = "stages/01_specs"
FEATURES_DIR    = "stages/02_features"
EMBEDDINGS_DIR  = "stages/03_embeddings"
LABELS_DIR      = "stages/04_labels"
PHENOTYPE_DIR   = "stages/05_phenotype"

# ---------------------------------------------------------------------------
# Results subdirectories (relative to bird root)
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
