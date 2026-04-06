"""Flatten 2-D syllable spectrograms into 1-D feature vectors (Stage B).

Each syllable spectrogram saved by Stage A is an ``(n_freq, n_time)`` array.
This module reshapes the full set of spectrograms for one song into a 2-D
matrix of shape ``(n_features, n_syllables)`` — where
``n_features = n_freq × n_time`` — and writes the result to a paired HDF5
file under ``<bird>/syllable_data/flattened/``.

Public API
----------
- :func:`flatten_bird_spectrograms` — run Stage B for a single bird
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import tables
from tqdm import tqdm

from song_phenotyping.tools.system_utils import optimize_pytables_for_network


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def extract_song_id(filepath: str) -> str:
    """Return the song ID embedded in a syllable HDF5 filename.

    Parameters
    ----------
    filepath : str
        Path to a file named ``syllables_<song_id>.h5``.

    Returns
    -------
    str
        The ``<song_id>`` portion of the filename.

    Raises
    ------
    ValueError
        If ``'syllables_'`` is not found in the filename stem.
    """
    stem = Path(filepath).stem
    if "syllables_" not in stem:
        raise ValueError(f"Expected 'syllables_' in filename: {filepath}")
    return stem.replace("syllables_", "")


def create_flattened_output_path(bird_root: str, song_id: str, run_name: str = "default") -> str:
    """Build the output path for a flattened HDF5 file, creating the directory.

    Parameters
    ----------
    bird_root : str
        Bird root directory (e.g. ``<save_path>/<bird>``).
    song_id : str
        Song identifier (from :func:`extract_song_id`).
    run_name : str, optional
        Run identifier; output goes under ``runs/<run_name>/``.

    Returns
    -------
    str
        Full path ``<bird_root>/runs/<run_name>/stages/02_features/flattened_<song_id>.h5``.
    """
    from song_phenotyping.tools.pipeline_paths import run_stage_path, FEATURES_DIR
    flattened_dir = str(run_stage_path(bird_root, run_name, FEATURES_DIR))
    os.makedirs(flattened_dir, exist_ok=True)
    return os.path.join(flattened_dir, f"flattened_{song_id}.h5")


def load_syllable_data(filepath: str) -> tuple:
    """Load spectrograms and metadata from a Stage A HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to ``syllables_<song_id>.h5`` as written by Stage A.

    Returns
    -------
    specs : numpy.ndarray, shape (n_syllables, n_freq, n_time)
        Raw spectrogram array.
    labels : numpy.ndarray
        Syllable label for each entry.
    position_idxs : numpy.ndarray
        Position indices within the original recording.
    hashes : numpy.ndarray
        Unique hash per syllable for cross-stage tracking.
    durations : numpy.ndarray or None
        Syllable durations in seconds; ``None`` if the node is absent.
    inst_freq : numpy.ndarray or None
        Instantaneous-frequency array, shape ``(n_syllables, n_freq, n_time-1)``;
        ``None`` if the node is absent.
    group_delay : numpy.ndarray or None
        Group-delay array, shape ``(n_syllables, n_freq-1, n_time)``;
        ``None`` if the node is absent.

    Raises
    ------
    ValueError
        If required HDF5 nodes are missing or array lengths are inconsistent.
    OSError
        If the file cannot be opened.
    """
    try:
        with tables.open_file(filepath, mode="r") as f:
            specs = f.root.spectrograms.read()
            if hasattr(f.root, 'manual'):
                labels = f.root.manual[:]
            else:
                labels = np.array([''] * len(specs), dtype='U1')
            position_idxs = f.root.position_idxs[:]
            hashes = f.root.hashes[:]
            durations = f.root.durations[:] if hasattr(f.root, "durations") else None
            inst_freq = f.root.inst_freq.read() if hasattr(f.root, "inst_freq") else None
            group_delay = f.root.group_delay.read() if hasattr(f.root, "group_delay") else None
    except (tables.NoSuchNodeError, AttributeError) as e:
        logging.error(f"Missing required data in {filepath}: {e}")
        raise ValueError(f"Invalid HDF5 structure in {filepath}")
    except (OSError, IOError) as e:
        logging.error(f"File access error for {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load syllable data from {filepath}: {e}")
        raise

    if not (len(specs) == len(labels) == len(position_idxs) == len(hashes)):
        raise ValueError(f"Inconsistent data lengths in {filepath}")

    return specs, labels, position_idxs, hashes, durations, inst_freq, group_delay


def flatten_spectrograms(
    specs: np.ndarray,
    inst_freq: Optional[np.ndarray] = None,
    group_delay: Optional[np.ndarray] = None,
    durations: Optional[np.ndarray] = None,
    duration_feature_weight: float = 0.0,
) -> np.ndarray:
    """Reshape a stack of 2-D spectrograms into a feature matrix.

    Optionally concatenates instantaneous-frequency and group-delay channels,
    and a duration token, before transposing to column-per-syllable form.

    Parameters
    ----------
    specs : numpy.ndarray, shape (n_syllables, n_freq, n_time)
        Stack of spectrogram arrays.
    inst_freq : numpy.ndarray or None, shape (n_syllables, n_freq, n_time-1)
        Instantaneous-frequency channel; appended when provided.
    group_delay : numpy.ndarray or None, shape (n_syllables, n_freq-1, n_time)
        Group-delay channel; appended when provided.
    durations : numpy.ndarray or None, shape (n_syllables,)
        Normalised syllable durations in [0, 1].  Only used when
        *duration_feature_weight* is non-zero.
    duration_feature_weight : float
        Scale factor applied to the duration block before concatenation.
        Zero (default) disables the duration feature entirely.

    Returns
    -------
    numpy.ndarray, shape (n_features, n_syllables), dtype float32
        Column-per-syllable feature matrix suitable for UMAP input.
        ``n_features`` equals ``n_freq × n_time`` plus any optional channels.

    Raises
    ------
    ValueError
        If *specs* is empty or not 3-D.
    """
    if specs.size == 0:
        raise ValueError("Cannot flatten empty spectrogram array")
    if specs.ndim != 3:
        raise ValueError(f"Expected 3D spectrogram array, got shape {specs.shape}")
    n_specs, height, width = specs.shape
    flat = specs.reshape(n_specs, -1)
    if inst_freq is not None:
        flat = np.concatenate([flat, inst_freq.reshape(n_specs, -1)], axis=1)
    if group_delay is not None:
        flat = np.concatenate([flat, group_delay.reshape(n_specs, -1)], axis=1)
    if duration_feature_weight and durations is not None:
        dur_block = np.outer(durations * duration_feature_weight, np.ones(height))
        flat = np.concatenate([flat, dur_block], axis=1)
    return flat.T.astype(np.float32)


def save_flattened_data(
    output_path: str,
    flattened_specs: np.ndarray,
    labels: np.ndarray,
    position_idxs: np.ndarray,
    hashes: np.ndarray,
    durations: Optional[np.ndarray] = None,
) -> None:
    """Write flattened spectrograms and metadata to an HDF5 file.

    Parameters
    ----------
    output_path : str
        Destination file path (``flattened_<song_id>.h5``).
    flattened_specs : numpy.ndarray, shape (n_features, n_syllables)
        Column-per-syllable feature matrix.
    labels : numpy.ndarray
        Syllable labels (same length as number of columns).
    position_idxs : numpy.ndarray
        Position indices within the original recording.
    hashes : numpy.ndarray
        Per-syllable hash values.
    durations : numpy.ndarray or None
        Syllable durations in seconds; written when provided.

    Raises
    ------
    Exception
        Re-raises any PyTables error after logging it.
    """
    try:
        with tables.open_file(output_path, mode="w") as f:
            f.create_array(f.root, "flattened_specs", flattened_specs)
            f.create_array(f.root, "labels", labels)
            f.create_array(f.root, "position_idxs", position_idxs)
            f.create_array(f.root, "hashes", hashes)
            if durations is not None:
                f.create_array(f.root, "durations", durations)
    except Exception as e:
        logging.error(f"Failed to save flattened data to {output_path}: {e}")
        raise


def process_single_syllable_file(
    filepath: str,
    bird_root: str,
    duration_feature_weight: float = 0.0,
    run_name: str = "default",
) -> bool:
    """Flatten one Stage A HDF5 file and write the result.

    Skips files whose output already exists.

    Parameters
    ----------
    filepath : str
        Path to ``syllables_<song_id>.h5``.
    bird_root : str
        Bird root directory (e.g. ``<save_path>/<bird>``).
    duration_feature_weight : float
        Forwarded to :func:`flatten_spectrograms`.  Zero disables the
        duration feature block (default).

    Returns
    -------
    bool
        ``True`` on success or if the output already existed;
        ``False`` if an error occurred.
    """
    try:
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        logging.debug(f"Processing {filepath} ({file_size:.1f} MB)")

        song_id = extract_song_id(filepath)
        output_path = create_flattened_output_path(bird_root, song_id, run_name=run_name)

        if os.path.exists(output_path):
            logging.debug(f"Flattened file already exists, skipping: {output_path}")
            return True

        specs, labels, position_idxs, hashes, durations, inst_freq, group_delay = load_syllable_data(filepath)
        flattened_specs = flatten_spectrograms(
            specs,
            inst_freq=inst_freq,
            group_delay=group_delay,
            durations=durations,
            duration_feature_weight=duration_feature_weight,
        )
        save_flattened_data(output_path, flattened_specs, labels, position_idxs, hashes, durations=durations)

        logging.info(f"Successfully flattened {len(specs)} syllables from {filepath}")
        return True

    except Exception as e:
        logging.error(f"Failed to process syllable file {filepath}: {e}")
        return False


def find_syllable_files(syllables_dir: str) -> List[str]:
    """Return all Stage A HDF5 files in *syllables_dir*.

    Parameters
    ----------
    syllables_dir : str
        Directory to search (typically ``<bird>/syllable_data/specs/``).

    Returns
    -------
    list of str
        Absolute paths to ``syllables_*.h5`` files; empty list if the
        directory does not exist.
    """
    if not os.path.exists(syllables_dir):
        return []
    return [
        os.path.join(syllables_dir, f)
        for f in os.listdir(syllables_dir)
        if f.endswith(".h5") and "syllables" in f
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flatten_bird_spectrograms(directory: str, bird: str, params=None, run_name: str = "default") -> bool:
    """Flatten all Stage A spectrograms for one bird (Stage B entry point).

    Reads every ``syllables_*.h5`` file from
    ``<directory>/<bird>/syllable_data/specs/``, flattens each spectrogram
    stack from ``(n_syllables, n_freq, n_time)`` to
    ``(n_freq × n_time, n_syllables)``, and writes
    ``flattened_<song_id>.h5`` files to
    ``<directory>/<bird>/syllable_data/flattened/``.

    Already-flattened files are skipped, so re-running is safe.

    Parameters
    ----------
    directory : str
        Project root directory containing bird subdirectories.
    bird : str
        Bird identifier (e.g. ``'or18or24'``).
    params : SpectrogramParams or None
        Pipeline parameters.  When provided,
        ``params.duration_feature_weight`` controls whether a duration
        block is appended to the feature vector.  ``None`` uses defaults
        (duration feature disabled).

    Returns
    -------
    bool
        ``True`` if at least one file was processed successfully (or if
        there were no files to process); ``False`` if all files failed.

    Examples
    --------
    >>> from song_phenotyping.flattening import flatten_bird_spectrograms
    >>> flatten_bird_spectrograms("/Volumes/Extreme SSD/pipeline_runs", "or18or24")
    True

    See Also
    --------
    song_phenotyping.ingestion.save_specs_for_evsonganaly_birds : Stage A (produces input).
    song_phenotyping.embedding.explore_embedding_parameters_robust : Stage C (consumes output).
    """
    duration_feature_weight = getattr(params, "duration_feature_weight", 0.0) or 0.0

    try:
        from song_phenotyping.tools.pipeline_paths import run_stage_path, SPECS_DIR, FEATURES_DIR
        bird_folder = os.path.join(directory, bird)
        syllables_path = str(run_stage_path(bird_folder, run_name, SPECS_DIR))
        features_path = str(run_stage_path(bird_folder, run_name, FEATURES_DIR))
        os.makedirs(features_path, exist_ok=True)

        syllable_files = find_syllable_files(syllables_path)
        if not syllable_files:
            logging.warning(
                f"No syllable files found for bird {bird} in {syllables_path}"
            )
            return True

        logging.info(f"Found {len(syllable_files)} syllable files for bird {bird}")

        success_count = sum(
            process_single_syllable_file(fp, bird_folder, duration_feature_weight=duration_feature_weight, run_name=run_name)
            for fp in tqdm(syllable_files, desc=f"Flattening {bird}")
        )

        logging.info(
            f"Bird {bird}: {success_count}/{len(syllable_files)} files processed"
        )
        return success_count > 0

    except Exception as e:
        logging.error(f"Error processing bird {bird}: {e}")
        return False
