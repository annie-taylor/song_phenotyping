"""Compute song phenotypes from clustering results (Stage E).

For each bird, this module ingests the top-ranked HDBSCAN clusterings
produced by Stage D, reconstructs syllable sequences from raw spec files,
and computes a standardised set of phenotypic measures:

* **Vocabulary** — repertoire size and syllable proportions.
* **Entropy** — song-level and transition-level information content.
* **Transitions** — first-order Markov transition matrices.
* **Repeats** — detection and characterisation of stereotyped syllable
  repetitions (dyads, longer motifs).

Results are written to ``phenotype_results.csv`` in the bird directory,
and detailed data structures (for PDF generation) are pickled to
``syllable_data/phenotype_detailed/``.

Public API
----------
- :func:`phenotype_bird` — run Stage E for a single bird.
- :class:`PhenotypingConfig` — configurable analysis parameters.
"""

import os
from pathlib import Path
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import tables
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from song_phenotyping.tools.label_handler import LabelType, LabelHandler, has_manual_labels


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class PhenotypingConfig:
    """Configurable parameters for Stage E phenotyping analysis.

    Parameters
    ----------
    min_syllable_proportion : float, optional
        Minimum fraction of all syllables a type must represent to be
        counted toward repertoire size.  Default is ``0.02``.
    repeat_significance_threshold : float, optional
        Minimum fraction of a syllable's instances that must participate in
        a repeat for that repeat to be considered significant.
        Default is ``0.2``.
    repeat_candidate_range : range or None, optional
        Repeat lengths (number of consecutive syllables) to search.
        Defaults to ``range(2, 20)`` when ``None``.
    dyad_threshold : float, optional
        Fraction of length-2 repeats above which a syllable is classified
        as a dyad.  Default is ``0.7``.
    adaptive_repeat_factor : float, optional
        Scales the per-syllable minimum prevalence used to filter spurious
        repeats.  Default is ``0.25``.
    use_top_n_clusterings : int, optional
        Number of top-ranked clustering results (from ``master_summary.csv``)
        to process.  Default is ``5``.
    generate_plots : bool, optional
        Whether to save scatter plots and heatmaps.  Default is ``True``.
    figure_dpi : int, optional
        Resolution of saved figures in dots per inch.  Default is ``300``.
    heatmap_annotation_size : int, optional
        Font size for transition-matrix heatmap annotations.  Default is ``8``.

    Examples
    --------
    >>> cfg = PhenotypingConfig(min_syllable_proportion=0.05, generate_plots=False)
    >>> cfg.repeat_candidate_range
    range(2, 20)
    """
    # Vocabulary parameters
    min_syllable_proportion: float = 0.02  # min proportion for syllable to count toward repertoire size

    # Repeat analysis parameters
    repeat_significance_threshold: float = 0.2  # min proportion of syllable instances that must be repeats
    repeat_candidate_range: range = None         # repeat lengths to search (default: range(2, 20))
    dyad_threshold: float = 0.7                  # proportion of length-2 repeats for syllable to be a dyad
    adaptive_repeat_factor: float = 0.25         # scales per-syllable min prevalence for repeat filtering

    # Processing options
    use_top_n_clusterings: int = 5  # how many top clustering results to process
    generate_plots: bool = True

    # Visualization
    figure_dpi: int = 300
    heatmap_annotation_size: int = 8

    def __post_init__(self):
        if self.repeat_candidate_range is None:
            self.repeat_candidate_range = range(2, 20)


# LabelType, LabelHandler, has_manual_labels — imported from tools.label_handler above.



# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_bird_syllable_data(bird_path: str) -> Dict[str, Any]:
    """Load manual syllable labels from a bird's Stage A spec files.

    Parameters
    ----------
    bird_path : str
        Absolute path to the bird's root directory (contains
        ``syllable_data/specs/``).

    Returns
    -------
    dict
        Keys: ``'manual_syllables'`` (list), ``'song_paths'`` (list of
        filenames), ``'bird_name'`` (str), ``'syllables_dir'`` (str).
        ``'manual_syllables'`` is empty when no manual labels are found.
    """
    bird_name = os.path.basename(bird_path)

    # Look for spec files in syllable_data/specs directory
    specs_dir = os.path.join(bird_path, 'syllable_data', 'specs')

    if not os.path.exists(specs_dir):
        logging.warning(f"No specs directory found at {specs_dir} for {bird_name}")
        return {
            'manual_syllables': [],
            'song_paths': [],
            'bird_name': bird_name,
            'syllables_dir': specs_dir  # Keep for consistency with other functions
        }

    # Look for syllable spec files (should start with 'syllables_')
    spec_files = [f for f in os.listdir(specs_dir) if f.endswith('.h5') and f.startswith('syllables_')]

    if not spec_files:
        logging.warning(f"No syllable spec files found in {specs_dir} for {bird_name}")
        return {
            'manual_syllables': [],
            'song_paths': [],
            'bird_name': bird_name,
            'syllables_dir': specs_dir
        }

    logging.info(f"Found {len(spec_files)} syllable spec files for {bird_name}")

    # Initialize data containers
    all_manual_syllables = []
    song_paths = []

    # Load data from each spec file
    for filename in spec_files:
        file_path = os.path.join(specs_dir, filename)
        song_paths.append(filename)

        try:
            with tables.open_file(file_path, 'r') as f:
                available_nodes = [node._v_name for node in f.list_nodes(f.root)]
                logging.debug(f"Available nodes in {filename}: {available_nodes}")

                if 'manual' in available_nodes:
                    raw_labels = f.root._f_get_child('manual').read()
                    logging.debug(f"Found {len(raw_labels)} manual labels in {filename}")

                    if len(raw_labels) > 0:
                        # Create handler for manual labels
                        handler = LabelHandler(LabelType.MANUAL)

                        # Normalize and add tokens
                        normalized_labels = handler.normalize_labels(raw_labels)
                        song_with_tokens = handler.add_sequence_tokens(normalized_labels)

                        # Add to collection
                        all_manual_syllables.extend(song_with_tokens)
                else:
                    logging.debug(f"No 'manual' node found in {filename}")

        except Exception as e:
            logging.error(f'Error processing {file_path}: {e}')
            continue

    if all_manual_syllables:
        logging.info(f"Loaded {len(all_manual_syllables)} manual syllable labels for {bird_name}")
    else:
        logging.info(f"No manual labels found in spec files for {bird_name}")

    return {
        'manual_syllables': all_manual_syllables,
        'song_paths': song_paths,
        'bird_name': bird_name,
        'syllables_dir': specs_dir  # Point to specs directory instead
    }


def load_clustering_results(bird_path: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Load the top-*top_n* clustering results from ``master_summary.csv``.

    Parameters
    ----------
    bird_path : str
        Bird root directory containing ``master_summary.csv``.
    top_n : int, optional
        Number of rows to return (rows are already ranked by composite
        score in the CSV).  Default is ``5``.

    Returns
    -------
    list of dict
        One dict per row with keys: ``'rank'``, ``'label_path'``,
        ``'composite_score'``, ``'nmi'``, ``'silhouette'``, ``'dbi'``,
        ``'n_clusters'``, ``'clustering_method'``, ``'n_neighbors'``,
        ``'min_dist'``, ``'metric'``, ``'min_cluster_size'``,
        ``'min_samples'``.  Returns an empty list if the CSV is absent or
        cannot be read.
    """
    master_summary_path = os.path.join(bird_path, 'master_summary.csv')

    if not os.path.exists(master_summary_path):
        logging.warning(f"No master summary found: {master_summary_path}")
        return []

    try:
        master_summary = pd.read_csv(master_summary_path)

        # Take top N results (already sorted by performance in clustering module)
        top_results = master_summary.head(top_n)

        clustering_results = []
        for idx, row in top_results.iterrows():
            # Extract clustering metadata
            result = {
                'rank': idx,
                'label_path': row['label_path'],
                'composite_score': row.get('composite_score', np.nan),
                'nmi': row.get('nmi', np.nan),
                'silhouette': row.get('silhouette', np.nan),
                'dbi': row.get('dbi', np.nan),
                'n_clusters': row.get('n_syls', np.nan),
                'clustering_method': 'hdbscan',  # CHANGED: Only hdbscan now

                # UMAP parameters
                'n_neighbors': row.get('n_neighbors', np.nan),
                'min_dist': row.get('min_dist', np.nan),
                'metric': row.get('metric', 'euclidean'),

                # HDBSCAN parameters
                'min_cluster_size': row.get('min_cluster_size', np.nan),
                'min_samples': row.get('min_samples', np.nan)
            }
            clustering_results.append(result)

        return clustering_results

    except Exception as e:
        logging.error(f"Error loading clustering results from {master_summary_path}: {e}")
        return []


def load_clustering_labels_for_syllables(clustering_result: Dict[str, Any], syllable_data: Dict[str, Any]) -> List[
    Union[str, int]]:
    """
    Load clustering labels and map them to syllable sequences.
    Updated to work with spec files in syllable_data/specs/.
    """
    try:
        # Load clustering labels from HDF5 file
        label_path = clustering_result['label_path']
        logging.info(f"Loading clustering labels from: {label_path}")
        resolved_path = _resolve_file_path(label_path)
        logging.info(f"Resolved path: {resolved_path}")

        if not os.path.exists(resolved_path):
            logging.error(f"Clustering label file does not exist: {resolved_path}")
            return []

        with tables.open_file(resolved_path, mode='r') as f:
            cluster_labels = f.root.labels.read()
            cluster_hashes = f.root.hashes.read()
            logging.info(f"Loaded {len(cluster_labels)} cluster labels and {len(cluster_hashes)} hashes")

        # Convert hashes to strings (fix for numpy bytes)
        cluster_hashes = [
            h.decode('utf-8') if isinstance(h, (bytes, np.bytes_)) else str(h)
            for h in cluster_hashes
        ]

        # Create hash to label mapping
        hash_to_label = dict(zip(cluster_hashes, cluster_labels))
        logging.info(f"Created hash-to-label mapping with {len(hash_to_label)} entries")

        # Map labels to syllable sequences
        mapped_labels = []
        song_paths = syllable_data['song_paths']
        specs_dir = syllable_data['syllables_dir']  # This now points to specs directory

        logging.info(f"Processing {len(song_paths)} song files from {specs_dir}")

        # Load syllable hashes from spec files and map to clustering labels
        total_syllables_processed = 0
        for song_path in song_paths:
            full_song_path = os.path.join(specs_dir, song_path)
            try:
                with tables.open_file(full_song_path, mode='r') as f:
                    # Check if hashes exist in the spec file
                    available_nodes = [node._v_name for node in f.list_nodes(f.root)]

                    if 'hashes' in available_nodes:
                        song_hashes = f.root.hashes.read()
                        song_hashes = [
                            h.decode('utf-8') if isinstance(h, (bytes, np.bytes_)) else str(h)
                            for h in song_hashes
                        ]

                        logging.debug(f"Found {len(song_hashes)} hashes in {song_path}")
                        total_syllables_processed += len(song_hashes)

                        # Map each syllable hash to its cluster label
                        for hash_id in song_hashes:
                            mapped_labels.append(hash_to_label.get(hash_id, -1))  # -1 for missing
                    else:
                        logging.warning(f"No 'hashes' node found in {full_song_path}")
                        logging.warning(f"Available nodes: {available_nodes}")

            except Exception as e:
                logging.error(f"Error loading hashes from {full_song_path}: {e}")
                continue

        logging.info(f"Processed {total_syllables_processed} syllables, mapped {len(mapped_labels)} labels")

        # Count successful mappings
        successful_mappings = sum(1 for label in mapped_labels if label != -1)
        logging.info(f"Successfully mapped {successful_mappings}/{len(mapped_labels)} syllables to cluster labels")

        if mapped_labels:
            # Add sequence tokens
            handler = LabelHandler(LabelType.AUTO)
            result = handler.add_sequence_tokens(mapped_labels)
            logging.info(f"Final sequence length with tokens: {len(result)}")
            return result
        else:
            logging.warning("No clustering labels could be mapped to syllables")
            return []

    except Exception as e:
        logging.error(f"Error loading clustering labels: {e}")
        logging.error(traceback.format_exc())
        return []

def _resolve_file_path(file_path: str) -> str:
    """
    Resolve file path, handling cross-platform and network path issues.

    Args:
        file_path: Original file path that may need resolution

    Returns:
        str: Resolved file path
    """
    # If file exists as-is, return it
    if os.path.exists(file_path):
        return file_path

    try:
        # Handle network path resolution
        from song_phenotyping.tools.system_utils import check_sys_for_macaw_root
        path_to_macaw = check_sys_for_macaw_root()

        path_parts = file_path.replace('\\', '/').split('/')
        # Try different relative path lengths to handle various structures
        for path_length in [9, 7, 5]:  # Try different path component counts
            if len(path_parts) >= path_length:
                relative_path = '/'.join(path_parts[-path_length:])
                resolved_path = os.path.join(path_to_macaw, relative_path)
                if os.path.exists(resolved_path):
                    return resolved_path

        # If all else fails, return original path
        return file_path

    except Exception as e:
        logging.warning(f"Error resolving path {file_path}: {e}")
        return file_path


def calculate_phenotypes_for_label_type(
        syllables: List[Union[str, int]],
        label_type: str,
        bird_name: str,
        config: PhenotypingConfig,
        clustering_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Add metadata from clustering results
    """
    Calculate all phenotype metrics for one label type.

    Args:
        syllables: List of syllable labels including start/end tokens
        label_type: 'manual' or 'hdbscan'
        bird_name: Bird identifier
        config: Configuration object

    Returns:
        Dictionary of phenotype metrics
    """
    if not syllables:
        return _create_empty_phenotype_results()

    try:
        # Create label handler
        enum_type = LabelType.MANUAL if label_type == 'manual' else LabelType.AUTO
        handler = LabelHandler(enum_type)

        # Calculate vocabulary and basic stats
        vocab_results = analyze_vocabulary_and_entropy(syllables, handler, config)

        # Calculate transition patterns
        transition_results = analyze_transitions(syllables, handler)

        # Calculate repeat patterns
        repeat_results = analyze_repeats(syllables, handler, config)

        # Combine all results
        phenotype_results = {
            **vocab_results,
            **transition_results,
            **repeat_results,
            'bird_name': bird_name,
            'label_type': label_type,
            'n_songs': _count_songs_in_sequence(syllables, handler),
            'n_syllables_total': _count_total_syllables(syllables, handler)
        }

        # Include clustering metadata if available
        if clustering_metadata:
            phenotype_results.update(clustering_metadata)

        return phenotype_results

    except Exception as e:
        logging.error(f"Error calculating phenotypes for {label_type} labels in {bird_name}: {e}")
        return _create_empty_phenotype_results()


def _create_empty_phenotype_results() -> Dict[str, Any]:
    """Create empty phenotype results dictionary with NaN values."""
    return {
        'repertoire_size': np.nan,
        'entropy': np.nan,
        'entropy_scaled': np.nan,
        'repeat_bool': False,
        'dyad_bool': False,
        'num_dyad': 0,
        'num_longer_reps': 0,
        'mean_repeat_syls': np.nan,
        'median_repeat_syls': np.nan,
        'var_repeat_syls': np.nan,
        'skew_repeat_syls': np.nan,
        'kurt_repeat_syls': np.nan,
        'n_songs': 0,
        'n_syllables_total': 0
    }


def analyze_vocabulary_and_entropy(syllables: List[Union[str, int]], handler: LabelHandler,
                                   config: PhenotypingConfig) -> Dict[str, Any]:
    """
    Analyze vocabulary size, syllable proportions, and sequence entropy.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type
        config: Configuration object

    Returns:
        Dictionary with vocabulary and entropy metrics
    """
    try:
        # Generate vocabulary statistics
        vocabulary, vocab_size, syl_counts, n_syls_total, syl_proportions = _generate_vocabulary(
            syllables, handler, config
        )

        # Calculate transition matrix for entropy computation
        transition_counts_df, t_mat_df, _, _ = _calculate_transition_counts(syllables, handler)

        # Calculate entropy measures
        entropy, entropy_scaled = _calculate_entropy(t_mat_df, syl_proportions, handler)

        return {
            'repertoire_size': vocab_size,
            'vocabulary': vocabulary,
            'syllable_counts': syl_counts,
            'syllable_proportions': syl_proportions,
            'entropy': entropy,
            'entropy_scaled': entropy_scaled
        }

    except Exception as e:
        logging.error(f"Error in vocabulary analysis: {e}")
        return {
            'repertoire_size': np.nan,
            'vocabulary': [],
            'syllable_counts': {},
            'syllable_proportions': np.array([]),
            'entropy': np.nan,
            'entropy_scaled': np.nan
        }


def analyze_transitions(syllables: List[Union[str, int]], handler: LabelHandler) -> Dict[str, Any]:
    """
    Analyze syllable transition patterns and matrices.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type

    Returns:
        Dictionary with transition analysis results
    """
    try:
        # Calculate transition matrices
        transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df = _calculate_transition_counts(syllables, handler)

        return {
            'transition_counts': transition_counts_df,
            'transition_matrix': t_mat_df,
            'transition_matrix_2nd': t2_mat_df,
            'transition_matrix_3rd': t3_mat_df
        }

    except Exception as e:
        logging.error(f"Error in transition analysis: {e}")
        return {
            'transition_counts': pd.DataFrame(),
            'transition_matrix': pd.DataFrame(),
            'transition_matrix_2nd': pd.DataFrame(),
            'transition_matrix_3rd': pd.DataFrame()
        }


def analyze_repeats(syllables: List[Union[str, int]], handler: LabelHandler, config: PhenotypingConfig) -> Dict[
    str, Any]:
    """
    Analyze syllable repeat patterns and statistics.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type
        config: Configuration object with repeat analysis parameters

    Returns:
        Dictionary with repeat analysis results
    """
    try:
        # Count repeats
        repeat_counts_df = _count_repeats_optimized(syllables, handler, config)

        # Calculate repeat significance and filter
        syl_counts = _get_syllable_counts(syllables, handler)
        repeat_counts_df = _remove_insignificant_repeats(repeat_counts_df, syl_counts, config, handler)

        # Calculate repeat phenotype statistics
        repeat_stats, _ = _repeat_phenotypes(repeat_counts_df, config)

        return {
            'repeat_counts': repeat_counts_df,
            **repeat_stats
        }

    except Exception as e:
        logging.error(f"Error in repeat analysis: {e}")
        return {
            'repeat_counts': pd.DataFrame(),
            'repeat_bool': False,
            'dyad_bool': False,
            'num_dyad': 0,
            'num_longer_reps': 0,
            'mean_repeat_syls': np.nan,
            'median_repeat_syls': np.nan,
            'var_repeat_syls': np.nan,
            'skew_repeat_syls': np.nan,
            'kurt_repeat_syls': np.nan
        }


def _generate_vocabulary(syllables: List[Union[str, int]], handler: LabelHandler, config: PhenotypingConfig) -> Tuple[
    List[Union[str, int]], int, Dict[Union[str, int], int], int, np.ndarray]:
    """
    Generate vocabulary and statistics from syllable sequences.

    Args:
        syllables: Complete sequence of syllable labels including start/stop tokens
        handler: LabelHandler for the specific label type being processed
        config: Configuration object with filtering parameters

    Returns:
        Tuple of (vocabulary, vocab_size, syl_counts, n_syls_total, syl_proportions)
    """
    vocabulary = []
    syl_counts = {}
    non_syl_tokens = handler.non_syl_tokens

    # Get unique syllables excluding tokens
    unique_syls = np.unique(syllables)

    # Build initial vocabulary and counts
    for syl in unique_syls:
        if syl not in non_syl_tokens:
            vocabulary.append(syl)
            syl_counts[syl] = 0

    # Count syllable occurrences
    for syl in syllables:
        if syl not in non_syl_tokens:
            syl_counts[syl] += 1

    # Calculate total syllables and proportions
    n_syls_total = sum(syl_counts.values())
    if n_syls_total == 0:
        logging.warning("No syllables found after filtering tokens")
        return [], 0, {}, 0, np.array([])

    syl_proportions = np.array(list(syl_counts.values())) / n_syls_total

    # Calculate vocabulary size excluding rare syllables
    vocab_size = sum(1 for prop in syl_proportions if prop > config.min_syllable_proportion)

    return vocabulary, vocab_size, syl_counts, n_syls_total, syl_proportions


def _calculate_transition_counts(syllables: List[Union[str, int]], handler: LabelHandler) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate syllable transition probabilities and higher-order matrices.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type being processed

    Returns:
        Tuple of (transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df)
    """
    if len(syllables) < 2:
        logging.warning("Sequence too short for transition analysis")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df

    # Create syllable-to-integer mapping
    unique_syls = np.unique(syllables)
    syl_to_int_map = {syl: idx for idx, syl in enumerate(unique_syls)}

    # Handle potential negative indices by remapping
    if any(isinstance(syl, int) and syl < 0 for syl in unique_syls):
        max_positive = max((syl for syl in unique_syls if isinstance(syl, int) and syl >= 0), default=-1)
        remap_start = max_positive + 1

        for syl in unique_syls:
            if isinstance(syl, int) and syl < 0:
                syl_to_int_map[syl] = remap_start
                remap_start += 1

    # Remove carriage return if present
    if '\r' in syl_to_int_map:
        del syl_to_int_map['\r']
        unique_syls = [s for s in unique_syls if s != '\r']

    # Initialize transition count matrix
    n_syls = len(syl_to_int_map)
    syl_transition_counts = np.zeros((n_syls, n_syls), dtype=np.int64)

    # Calculate transition counts
    start_token = handler.start_token
    end_token = handler.end_token

    for i in range(len(syllables) - 1):
        current_syl = syllables[i]
        next_syl = syllables[i + 1]

        # Skip transitions from end token to start token (between songs)
        if current_syl == end_token and next_syl == start_token:
            continue

        if current_syl in syl_to_int_map and next_syl in syl_to_int_map:
            current_idx = syl_to_int_map[current_syl]
            next_idx = syl_to_int_map[next_syl]
            syl_transition_counts[next_idx, current_idx] += 1

    # Order syllables with start/end tokens at beginning/end
    ordered_syls = [s for s in unique_syls if s not in [start_token, end_token]]
    ordered_syls = [start_token] + sorted(ordered_syls) + [end_token]

    # Create reordered matrices
    n_ordered = len(ordered_syls)
    transition_mat = np.zeros((n_ordered, n_ordered), dtype=float)
    reordered_transition_counts = np.zeros((n_ordered, n_ordered), dtype=int)

    for i, syl_from in enumerate(ordered_syls):
        for j, syl_to in enumerate(ordered_syls):
            if syl_from in syl_to_int_map and syl_to in syl_to_int_map:
                from_idx = syl_to_int_map[syl_from]
                to_idx = syl_to_int_map[syl_to]

                count = syl_transition_counts[to_idx, from_idx]
                reordered_transition_counts[i, j] = count

                # Calculate probability (row sum for normalization)
                row_sum = np.sum(syl_transition_counts[from_idx, :])
                if row_sum > 0:
                    transition_mat[i, j] = count / row_sum

    # Calculate higher-order transition matrices TODO this is not helpful
    t2_mat = np.linalg.matrix_power(transition_mat, 2)
    t3_mat = np.linalg.matrix_power(transition_mat, 3)

    # Create DataFrames with proper indexing
    transition_counts_df = pd.DataFrame(reordered_transition_counts, index=ordered_syls, columns=ordered_syls)
    t_mat_df = pd.DataFrame(transition_mat, index=ordered_syls, columns=ordered_syls)
    t2_mat_df = pd.DataFrame(t2_mat, index=ordered_syls, columns=ordered_syls)
    t3_mat_df = pd.DataFrame(t3_mat, index=ordered_syls, columns=ordered_syls)

    return transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df


def _calculate_entropy(t_mat_df: pd.DataFrame, syl_proportions: np.ndarray, handler: LabelHandler) -> Tuple[
    float, float]:
    """
    Calculate the entropy of the transition matrix.

    Args:
        t_mat_df: Transition probability matrix
        syl_proportions: Proportion of each syllable type in the sequence
        handler: LabelHandler for the specific label type

    Returns:
        Tuple of (entropy, entropy_scaled) - raw and syllable-proportion-weighted entropy
    """
    # Exclude start/stop tokens (first and last rows/columns)
    if len(t_mat_df) < 3:  # Need at least start, one syllable, stop
        logging.warning("Transition matrix too small for entropy calculation")
        return 0.0, 0.0

    # Get core transition matrix (excluding start/stop tokens)
    core_matrix = t_mat_df.iloc[1:-1, 1:-1].to_numpy()

    if core_matrix.size == 0:
        return 0.0, 0.0

    # Calculate raw entropy
    nonzero_probs = core_matrix[core_matrix > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))

    # Calculate scaled entropy using syllable proportions
    n_core_syls = core_matrix.shape[0]

    if len(syl_proportions) >= n_core_syls:
        # Use the first n_core_syls proportions (excluding start/stop tokens)
        core_syl_proportions = syl_proportions[:n_core_syls]

        if n_core_syls > 0 and len(core_syl_proportions) == n_core_syls:
            syl_prop_mat = np.tile(core_syl_proportions.reshape(-1, 1), (1, n_core_syls))
            weighted_probs = core_matrix * syl_prop_mat
            nonzero_weighted = weighted_probs[weighted_probs > 0]
            if len(nonzero_weighted) > 0:
                entropy_scaled = -np.sum(nonzero_weighted * np.log(core_matrix[weighted_probs > 0]))
            else:
                entropy_scaled = entropy
        else:
            entropy_scaled = entropy
    else:
        logging.warning(f"Syllable proportions ({len(syl_proportions)}) don't match core matrix size ({n_core_syls})")
        entropy_scaled = entropy

    return entropy, entropy_scaled


def _count_repeats_optimized(syllables: List[Union[str, int]], handler: LabelHandler,
                             config: PhenotypingConfig) -> pd.DataFrame:
    """
    Count repeated sequences of syllables using optimized algorithm.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type
        config: Configuration object with repeat analysis parameters

    Returns:
        DataFrame with repeat counts, indexed by repeat length and columned by syllable
    """
    if len(syllables) < 4:  # Need at least start + 2 syllables + stop for a repeat
        return pd.DataFrame()

    # Convert to numpy array for faster operations
    syls_array = np.array(syllables)
    non_syl_tokens = set(handler.non_syl_tokens)

    # Get unique syllables excluding tokens
    unique_syls = [syl for syl in np.unique(syls_array) if syl not in non_syl_tokens]

    if not unique_syls:
        return pd.DataFrame()

    repeat_counts = {}
    candidate_lengths = list(config.repeat_candidate_range)

    # For each syllable type, find all its positions
    for syl in unique_syls:
        positions = np.where(syls_array == syl)[0]

        # For each position, check for consecutive repeats
        for pos in positions:
            max_run_length = 1

            # Find maximum consecutive run starting at this position
            while (
                    pos + max_run_length < len(syls_array)
                    and syls_array[pos + max_run_length] == syl
                    and max_run_length < max(candidate_lengths)
            ):
                max_run_length += 1

            # Record all repeat lengths for this run (2, 3, 4, ... up to max_run_length)
            for run_len in range(2, max_run_length + 1):
                if run_len in candidate_lengths:
                    repeat_counts[(syl, run_len)] = repeat_counts.get((syl, run_len), 0) + 1

    # Convert to DataFrame format
    if not repeat_counts:
        return pd.DataFrame()

    # Create DataFrame with repeat lengths as index and syllables as columns
    repeat_df_data = {}
    for syl in unique_syls:
        repeat_df_data[syl] = [
            repeat_counts.get((syl, length), 0) for length in candidate_lengths
        ]

    repeat_df = pd.DataFrame(repeat_df_data, index=candidate_lengths)

    # Remove overlapping counts (subtract shorter repeats contained in longer ones)
    repeat_df = _remove_overlapping_repeats(repeat_df)

    # Remove empty rows and columns
    repeat_df = repeat_df.loc[(repeat_df != 0).any(axis=1), (repeat_df != 0).any(axis=0)]

    return repeat_df


def _remove_overlapping_repeats(repeat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove shorter repeats that are contained within longer repeats.

    Args:
        repeat_df: DataFrame with repeat counts before overlap removal

    Returns:
        DataFrame with overlapping repeats removed
    """
    if repeat_df.empty:
        return repeat_df

    # Work from longest to shortest repeats
    lengths = sorted(repeat_df.index, reverse=True)

    for i, longer_len in enumerate(lengths[:-1]):
        for shorter_len in lengths[i + 1:]:
            # For each syllable, subtract overlaps
            for syl in repeat_df.columns:
                n_longer = repeat_df.loc[longer_len, syl]
                if n_longer > 0:
                    # Each longer repeat contains (longer_len - shorter_len + 1) shorter repeats
                    overlaps = n_longer * (longer_len - shorter_len + 1)
                    repeat_df.loc[shorter_len, syl] = max(
                        0, repeat_df.loc[shorter_len, syl] - overlaps
                    )

    return repeat_df


def _get_syllable_counts(syllables: List[Union[str, int]], handler: LabelHandler) -> Dict[Union[str, int], int]:
    """
    Get count of each syllable type excluding non-syllable tokens.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type

    Returns:
        Dictionary mapping syllable to count
    """
    syl_counts = {}
    non_syl_tokens = handler.non_syl_tokens

    for syl in syllables:
        if syl not in non_syl_tokens:
            syl_counts[syl] = syl_counts.get(syl, 0) + 1

    return syl_counts


def _remove_insignificant_repeats(repeat_counts: pd.DataFrame, syl_counts: Dict[Union[str, int], int],
                                  config: PhenotypingConfig, handler: LabelHandler) -> pd.DataFrame:
    """
    Remove syllables with insignificant repeat patterns from analysis.

    Args:
        repeat_counts: DataFrame with repeat counts by syllable and repeat length
        syl_counts: Total count of each syllable type in the sequence
        config: Configuration object with filtering parameters
        handler: LabelHandler for the specific label type

    Returns:
        Filtered DataFrame with only significant repeats
    """
    if repeat_counts.empty:
        return repeat_counts

    # Calculate overall syllable proportions for adaptive threshold
    total_syls = sum(syl_counts.values())
    overall_syl_prop = {syl: count / total_syls for syl, count in syl_counts.items()}

    # Remove syllables with very low overall occurrence
    adaptive_threshold = config.adaptive_repeat_factor * (1 / len(syl_counts))
    columns_to_remove = []

    for syl in repeat_counts.columns:
        if syl in overall_syl_prop:
            if overall_syl_prop[syl] < adaptive_threshold:
                columns_to_remove.append(syl)

    # Remove low-occurrence syllables
    repeat_counts = repeat_counts.drop(columns=columns_to_remove, errors='ignore')

    # Remove syllables without significant repeat proportions
    significance = _repeat_significance(repeat_counts, syl_counts, config, handler)
    insignificant_syls = [syl for syl, (is_sig, _) in significance.items() if not is_sig]
    repeat_counts = repeat_counts.drop(columns=insignificant_syls, errors='ignore')

    return repeat_counts


def _repeat_significance(repeat_counts_df: pd.DataFrame, syl_counts: Dict[Union[str, int], int],
                         config: PhenotypingConfig, handler: LabelHandler) -> Dict[Union[str, int], Tuple[bool, float]]:
    """
    Check for significant proportion of repeats from syllable instances.

    Args:
        repeat_counts_df: DataFrame with repeat counts by syllable and repeat length
        syl_counts: Total count of each syllable type in the sequence
        config: Configuration object with significance threshold
        handler: LabelHandler for the specific label type

    Returns:
        Dictionary mapping syllable to (is_significant, repeat_proportion)
    """
    repeated_syls = repeat_counts_df.columns if not repeat_counts_df.empty else []
    significance = {}
    non_syl_tokens = set(handler.non_syl_tokens)

    for syl in repeated_syls:
        if syl not in non_syl_tokens and syl in syl_counts:
            n_total_syls = syl_counts[syl]

            # Calculate total syllables involved in repeats
            # Each repeat of length N involves N syllables
            n_repeated_syls = sum(
                repeat_len * repeat_count
                for repeat_len, repeat_count in repeat_counts_df[syl].items()
                if repeat_count > 0
            )

            repeat_prop = n_repeated_syls / n_total_syls if n_total_syls > 0 else 0
            is_significant = repeat_prop >= config.repeat_significance_threshold

            significance[syl] = (is_significant, repeat_prop)
        else:
            significance[syl] = (False, 0.0)

    return significance


def _repeat_phenotypes(repeat_counts: pd.DataFrame, config: PhenotypingConfig) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """
    Calculate phenotypic statistics from repeat count data.

    Args:
        repeat_counts: DataFrame with repeat counts by syllable and repeat length
        config: Configuration object with analysis parameters

    Returns:
        Tuple of (repeat_stats, repeat_stats_all) - summary statistics and optional individual distributions
    """
    if repeat_counts.empty or len(repeat_counts.columns) == 0:
        return {
            'repeat_bool': False,
            'dyad_bool': False,
            'num_dyad': 0,
            'num_longer_reps': 0,
            'mean_repeat_syls': 0.0,
            'median_repeat_syls': 0.0,
            'var_repeat_syls': 0.0,
            'skew_repeat_syls': 0.0,
            'kurt_repeat_syls': 0.0
        }, None

    num_unique_repeaty_syls = len(repeat_counts.columns)
    dyad_bool = False
    num_dyads = 0
    combined_rep_list = []

    # Process each syllable's repeat distribution
    for syl in repeat_counts.columns:
        # Create list of repeat lengths weighted by their occurrence
        rep_list = []
        for repeat_len in repeat_counts.index:
            n_instances = repeat_counts.loc[repeat_len, syl]
            rep_list.extend([repeat_len] * n_instances)

        if not rep_list:
            continue

        # Check for dyad dominance (repeats of length 2)
        dyad_count = sum(1 for length in rep_list if length == 2)
        dyad_prop = dyad_count / len(rep_list) if rep_list else 0

        if dyad_prop > config.dyad_threshold:
            dyad_bool = True
            num_dyads += 1
            # Remove dyads from longer repeat analysis
            rep_list = [length for length in rep_list if length != 2]

        # Add to combined distribution for overall statistics
        combined_rep_list.extend(rep_list)

    # Calculate overall repeat statistics
    if combined_rep_list:
        mean_repeat_syls = np.mean(combined_rep_list)
        median_repeat_syls = np.median(combined_rep_list)
        var_repeat_syls = np.var(combined_rep_list)
        skew_repeat_syls = skew(combined_rep_list)
        kurt_repeat_syls = kurtosis(combined_rep_list)
    else:
        mean_repeat_syls = median_repeat_syls = var_repeat_syls = 0.0
        skew_repeat_syls = kurt_repeat_syls = 0.0

    repeat_stats = {
        'repeat_bool': True,
        'dyad_bool': dyad_bool,
        'num_dyad': num_dyads,
        'num_longer_reps': num_unique_repeaty_syls - num_dyads,
        'mean_repeat_syls': mean_repeat_syls,
        'median_repeat_syls': median_repeat_syls,
        'var_repeat_syls': var_repeat_syls,
        'skew_repeat_syls': skew_repeat_syls,
        'kurt_repeat_syls': kurt_repeat_syls
    }

    return repeat_stats, None


def _count_songs_in_sequence(syllables: List[Union[str, int]], handler: LabelHandler) -> int:
    """
    Count the number of songs in a syllable sequence by counting start tokens.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type

    Returns:
        Number of songs (start tokens) in the sequence
    """
    return syllables.count(handler.start_token)


def _count_total_syllables(syllables: List[Union[str, int]], handler: LabelHandler) -> int:
    """
    Count the total number of syllables excluding non-syllable tokens.

    Args:
        syllables: Complete syllable sequence including start/stop tokens
        handler: LabelHandler for the specific label type

    Returns:
        Total number of syllables (excluding tokens)
    """
    non_syl_tokens = set(handler.non_syl_tokens)
    return sum(1 for syl in syllables if syl not in non_syl_tokens)


def create_unified_phenotype_row(
        bird_name: str,
        manual_results: Dict[str, Any],
        auto_results: List[Dict[str, Any]],
        clustering_results: List[Dict[str, Any]],
        config: PhenotypingConfig
) -> pd.DataFrame:
    """
    Create unified phenotype DataFrame with manual labels as first row (if available),
    followed by ranked automated results.

    Args:
        bird_name: Bird identifier
        manual_results: Dictionary with manual phenotype results
        auto_results: List of dictionaries with automated phenotype results
        clustering_results: List of dictionaries with clustering metadata
        config: Configuration object

    Returns:
        DataFrame with phenotype data (manual first, then ranked automated)
    """
    rows = []

    # Add manual row first if manual labels exist
    if manual_results.get('repertoire_size') is not None and not np.isnan(manual_results.get('repertoire_size', np.nan)):
        manual_row = {
            'bird_name': bird_name,
            'rank': 'manual',

            # Phenotype metrics (no prefixes)
            'repertoire_size': manual_results.get('repertoire_size', np.nan),
            'entropy': manual_results.get('entropy', np.nan),
            'entropy_scaled': manual_results.get('entropy_scaled', np.nan),
            'repeat_bool': manual_results.get('repeat_bool', False),
            'dyad_bool': manual_results.get('dyad_bool', False),
            'num_dyad': manual_results.get('num_dyad', 0),
            'num_longer_reps': manual_results.get('num_longer_reps', 0),
            'mean_repeat_syls': manual_results.get('mean_repeat_syls', np.nan),
            'median_repeat_syls': manual_results.get('median_repeat_syls', np.nan),
            'var_repeat_syls': manual_results.get('var_repeat_syls', np.nan),
            'skew_repeat_syls': manual_results.get('skew_repeat_syls', np.nan),
            'kurt_repeat_syls': manual_results.get('kurt_repeat_syls', np.nan),
            'n_songs': manual_results.get('n_songs', 0),
            'n_syllables_total': manual_results.get('n_syllables_total', 0),

            # Clustering metadata (empty for manual)
            'clustering_method': np.nan,
            'composite_score': np.nan,
            'nmi': np.nan,
            'silhouette': np.nan,
            'dbi': np.nan,
            'n_clusters': np.nan,
            'n_neighbors': np.nan,
            'min_dist': np.nan,
            'metric': np.nan,
            'min_cluster_size': np.nan,
            'min_samples': np.nan
        }
        rows.append(manual_row)

    # Add automated rows for each clustering result
    for i, (auto_result, cluster_result) in enumerate(zip(auto_results, clustering_results)):
        auto_row = {
            'bird_name': bird_name,
            'rank': i,  # 0, 1, 2, etc. for automated rankings

            # Phenotype metrics (no prefixes)
            'repertoire_size': auto_result.get('repertoire_size', np.nan),
            'entropy': auto_result.get('entropy', np.nan),
            'entropy_scaled': auto_result.get('entropy_scaled', np.nan),
            'repeat_bool': auto_result.get('repeat_bool', False),
            'dyad_bool': auto_result.get('dyad_bool', False),
            'num_dyad': auto_result.get('num_dyad', 0),
            'num_longer_reps': auto_result.get('num_longer_reps', 0),
            'mean_repeat_syls': auto_result.get('mean_repeat_syls', np.nan),
            'median_repeat_syls': auto_result.get('median_repeat_syls', np.nan),
            'var_repeat_syls': auto_result.get('var_repeat_syls', np.nan),
            'skew_repeat_syls': auto_result.get('skew_repeat_syls', np.nan),
            'kurt_repeat_syls': auto_result.get('kurt_repeat_syls', np.nan),
            'n_songs': auto_result.get('n_songs', 0),
            'n_syllables_total': auto_result.get('n_syllables_total', 0),

            # Clustering metadata
            'clustering_method': cluster_result.get('clustering_method', 'hdbscan'),
            'composite_score': cluster_result.get('composite_score', np.nan),
            'nmi': cluster_result.get('nmi', np.nan),
            'silhouette': cluster_result.get('silhouette', np.nan),
            'dbi': cluster_result.get('dbi', np.nan),
            'n_clusters': cluster_result.get('n_clusters', np.nan),
            'n_neighbors': cluster_result.get('n_neighbors', np.nan),
            'min_dist': cluster_result.get('min_dist', np.nan),
            'metric': cluster_result.get('metric', 'euclidean'),
            'min_cluster_size': cluster_result.get('min_cluster_size', np.nan),
            'min_samples': cluster_result.get('min_samples', np.nan)
        }
        rows.append(auto_row)

    # Handle edge case: no manual labels and no clustering results
    if not rows:
        empty_row = {
            'bird_name': bird_name,
            'rank': 0,
            'repertoire_size': np.nan,
            'entropy': np.nan,
            'entropy_scaled': np.nan,
            'repeat_bool': False,
            'dyad_bool': False,
            'num_dyad': 0,
            'num_longer_reps': 0,
            'mean_repeat_syls': np.nan,
            'median_repeat_syls': np.nan,
            'var_repeat_syls': np.nan,
            'skew_repeat_syls': np.nan,
            'kurt_repeat_syls': np.nan,
            'n_songs': 0,
            'n_syllables_total': 0,
            'clustering_method': np.nan,
            'composite_score': np.nan,
            'nmi': np.nan,
            'silhouette': np.nan,
            'dbi': np.nan,
            'n_clusters': np.nan,
            'n_neighbors': np.nan,
            'min_dist': np.nan,
            'metric': np.nan,
            'min_cluster_size': np.nan,
            'min_samples': np.nan
        }
        rows.append(empty_row)

    return pd.DataFrame(rows)


def phenotype_bird(bird_path: str, config: PhenotypingConfig = None) -> bool:
    """Run the complete Stage E phenotyping pipeline for one bird.

    Processes manual labels (when available) and the top automated
    clustering results from Stage D.  Writes:

    * ``<bird_path>/phenotype_results.csv`` — one row per clustering rank.
    * ``<bird_path>/syllable_data/phenotype_detailed/automated_phenotype_data_rank*.pkl``
      — detailed data structures for PDF generation.

    Parameters
    ----------
    bird_path : str
        Absolute path to the bird's root directory.
    config : PhenotypingConfig or None, optional
        Analysis parameters.  Uses default :class:`PhenotypingConfig` when
        ``None``.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if a fatal error occurred.

    Examples
    --------
    >>> from song_phenotyping.phenotyping import phenotype_bird, PhenotypingConfig
    >>> cfg = PhenotypingConfig(generate_plots=False)
    >>> phenotype_bird("/Volumes/Extreme SSD/pipeline_runs/or18or24", cfg)
    True

    See Also
    --------
    song_phenotyping.labelling.label_bird : Stage D (produces input).
    """
    if config is None:
        config = PhenotypingConfig()

    bird_name = os.path.basename(bird_path)
    logging.info(f"Starting phenotyping for bird: {bird_name}")

    try:
        # Load syllable data
        syllable_data = load_bird_syllable_data(bird_path)

        # Check if manual labels are available
        has_manual = has_manual_labels(syllable_data)
        logging.info(f"Manual labels available for {bird_name}: {has_manual}")

        # Load clustering results
        clustering_results = load_clustering_results(bird_path, config.use_top_n_clusterings)
        has_clustering = len(clustering_results) > 0
        logging.info(
            f"Clustering results available for {bird_name}: {has_clustering} ({len(clustering_results)} results)")

        # Process manual labels if available
        manual_results = {}
        if has_manual:
            logging.info(f"Processing manual labels for {bird_name}")
            manual_syllables = syllable_data['manual_syllables']
            logging.info(f"Manual syllables sequence length: {len(manual_syllables)}")
            manual_results = calculate_phenotypes_for_label_type(
                manual_syllables, 'manual', bird_name, config
            )
            logging.info(f"Manual results: repertoire_size={manual_results.get('repertoire_size', 'N/A')}")
        else:
            logging.info(f"No manual labels found for {bird_name}")
            manual_results = _create_empty_phenotype_results()

        # Process automated labels for each clustering result
        auto_results = []
        for i, cluster_result in enumerate(clustering_results):
            logging.info(f"Processing automated labels for {bird_name}, rank {i}")
            logging.info(f"Cluster result metadata: {cluster_result}")

            # Load clustering labels mapped to syllables
            try:
                auto_syllables = load_clustering_labels_for_syllables(cluster_result, syllable_data)
                logging.info(f"Auto syllables sequence length for rank {i}: {len(auto_syllables)}")

                if auto_syllables:
                    auto_result = calculate_phenotypes_for_label_type(
                        auto_syllables, 'hdbscan', bird_name, config
                    )
                    logging.info(f"Auto results rank {i}: repertoire_size={auto_result.get('repertoire_size', 'N/A')}")
                else:
                    logging.warning(f"No auto syllables loaded for rank {i}")
                    auto_result = _create_empty_phenotype_results()
            except Exception as e:
                logging.error(f"Error processing clustering rank {i} for {bird_name}: {e}")
                logging.error(traceback.format_exc())
                auto_result = _create_empty_phenotype_results()

            auto_results.append(auto_result)

        # If no clustering results, create empty auto results
        if not auto_results:
            logging.info("No clustering results found, creating empty auto results")
            auto_results = [_create_empty_phenotype_results()]
            clustering_results = [{
                'rank': 0,
                'clustering_method': 'hdbscan',
                'composite_score': np.nan,
                'nmi': np.nan,
                'silhouette': np.nan,
                'dbi': np.nan,
                'n_clusters': np.nan,
                'n_neighbors': np.nan,
                'min_dist': np.nan,
                'metric': 'euclidean',
                'min_cluster_size': np.nan,
                'min_samples': np.nan
            }]

        # Save detailed phenotype data for PDF generation
        save_detailed_phenotype_data(bird_path, manual_results, auto_results, clustering_results)

        # Create unified results DataFrame
        results_df = create_unified_phenotype_row(
            bird_name, manual_results, auto_results, clustering_results, config
        )

        # Save results
        output_path = os.path.join(bird_path, 'phenotype_results.csv')
        results_df.to_csv(output_path, index=False)
        logging.info(f"Saved phenotype results to: {output_path}")

        # Log summary of results
        logging.info(f"Results summary for {bird_name}:")
        for idx, row in results_df.iterrows():
            rank = row['rank']
            rep_size = row['repertoire_size']
            entropy = row['entropy']
            logging.info(f"  Rank {rank}: repertoire_size={rep_size}, entropy={entropy}")

        # Generate plots if requested
        if config.generate_plots:
            _generate_phenotype_plots(bird_path, syllable_data, manual_results, auto_results, clustering_results,
                                      config)

        # Generate PDFs if requested
        if config.generate_plots:  # Use same flag for now
            try:
                from phenotype_pdfs import integrate_with_phenotyping_pipeline
                pdf_results = integrate_with_phenotyping_pipeline(bird_path, config)
                if pdf_results:
                    logging.info(f"Generated phenotype PDFs for {bird_name}: {list(pdf_results.keys())}")
            except ImportError:
                logging.warning("phenotype_pdfs module not available, skipping PDF generation")

        logging.info(f"Successfully completed phenotyping for bird: {bird_name}")
        return True

    except Exception as e:
        logging.error(f"Error in phenotyping pipeline for {bird_name}: {e}")
        logging.error(traceback.format_exc())
        return False

def save_detailed_phenotype_data(bird_path: str,
                                 manual_results: Dict[str, Any],
                                 auto_results: List[Dict[str, Any]],
                                 clustering_results: List[Dict[str, Any]]) -> bool:
    """Pickle detailed phenotype data structures for downstream PDF generation.

    Writes one ``.pkl`` file per clustering rank to
    ``<bird_path>/syllable_data/phenotype_detailed/``.

    Parameters
    ----------
    bird_path : str
        Bird root directory.
    manual_results : dict
        Phenotype results computed from manual labels.
    auto_results : list of dict
        Phenotype results for each automated clustering rank.
    clustering_results : list of dict
        Clustering metadata dicts (as returned by
        :func:`load_clustering_results`).

    Returns:
        bool: Success status
    """
    try:
        # Create detailed data directory
        detailed_data_dir = os.path.join(bird_path, 'syllable_data', 'phenotype_detailed')
        os.makedirs(detailed_data_dir, exist_ok=True)

        # Save manual results if available
        if manual_results.get('repertoire_size') is not None and not np.isnan(
                manual_results.get('repertoire_size', np.nan)):
            manual_path = os.path.join(detailed_data_dir, 'manual_phenotype_data.pkl')
            with open(manual_path, 'wb') as f:
                pkl.dump(manual_results, f)
            logging.info(f"Saved detailed manual phenotype data to: {manual_path}")

        # Save automated results for each rank
        for i, (auto_result, cluster_result) in enumerate(zip(auto_results, clustering_results)):
            auto_data = {
                'phenotype_results': auto_result,
                'clustering_metadata': cluster_result
            }
            auto_path = os.path.join(detailed_data_dir, f'automated_phenotype_data_rank{i}.pkl')
            with open(auto_path, 'wb') as f:
                pkl.dump(auto_data, f)
            logging.info(f"Saved detailed automated phenotype data rank {i} to: {auto_path}")

        return True

    except Exception as e:
        logging.error(f"Error saving detailed phenotype data: {e}")
        return False


def _generate_phenotype_plots(
        bird_path: str,
        syllable_data: Dict[str, Any],
        manual_results: Dict[str, Any],
        auto_results: List[Dict[str, Any]],
        clustering_results: List[Dict[str, Any]],
        config: PhenotypingConfig
) -> None:
    """
    Generate all phenotype visualization plots for a bird.
    ENHANCED: Now generates plots for both manual and automated results.
    """
    try:
        # Create plots directory
        plots_dir = os.path.join(bird_path, 'figures', 'phenotyping')
        os.makedirs(plots_dir, exist_ok=True)

        bird_name = os.path.basename(bird_path)

        # Plot manual results if available
        if manual_results.get('transition_matrix') is not None and not manual_results['transition_matrix'].empty:
            plot_transition_matrices(
                plots_dir, manual_results, "manual", bird_name, config
            )

        if manual_results.get('repeat_counts') is not None and not manual_results['repeat_counts'].empty:
            plot_repeat_patterns(
                plots_dir, manual_results['repeat_counts'], "manual", bird_name, config
            )

        # Plot transition matrices for each automated rank
        for i, auto_result in enumerate(auto_results):
            if 'transition_matrix' in auto_result and not auto_result['transition_matrix'].empty:
                plot_transition_matrices(
                    plots_dir, auto_result, f"rank{i}", bird_name, config
                )

        # Plot repeat patterns for each automated rank
        for i, auto_result in enumerate(auto_results):
            if 'repeat_counts' in auto_result and not auto_result['repeat_counts'].empty:
                plot_repeat_patterns(
                    plots_dir, auto_result['repeat_counts'], f"rank{i}", bird_name, config
                )

        # Plot vocabulary comparison (manual vs auto)
        if manual_results.get('vocabulary') and any(auto_result.get('vocabulary') for auto_result in auto_results):
            plot_vocabulary_comparison(
                plots_dir, manual_results, auto_results, bird_name, config
            )

        # Generate manual UMAP plot if manual labels exist and clustering data available
        if (manual_results.get('vocabulary') and
            clustering_results and
            syllable_data.get('manual_syllables')):
            generate_manual_umap_plot(bird_path, syllable_data, manual_results, clustering_results[0], config)

        logging.info(f"Generated phenotype plots for {bird_name} in {plots_dir}")

    except Exception as e:
        logging.error(f"Error generating plots for {bird_name}: {e}")


def generate_manual_umap_plot(
        bird_path: str,
        syllable_data: Dict[str, Any],
        manual_results: Dict[str, Any],
        clustering_result: Dict[str, Any],
        config: PhenotypingConfig
) -> str:
    """
    Generate UMAP plot colored by manual labels to match automated clustering plots.
    """
    try:
        import umap
        from sklearn.preprocessing import StandardScaler

        # Load syllable features for UMAP
        syllable_db_path = os.path.join(bird_path, 'syllable_data', 'syllable_database', 'syllable_features.csv')
        if not os.path.exists(syllable_db_path):
            logging.warning(f"No syllable database found for manual UMAP plot: {syllable_db_path}")
            return ""

        df = pd.read_csv(syllable_db_path)

        # Get feature columns (exclude metadata columns)
        feature_cols = [col for col in df.columns if
                        not col.startswith(('manual_label', 'cluster_', 'song_file', 'hash'))]

        if not feature_cols:
            logging.warning("No feature columns found for UMAP")
            return ""

        # Prepare features and labels
        features = df[feature_cols].values
        manual_labels = df.get('manual_label', pd.Series(['unknown'] * len(df)))

        # Filter out samples without manual labels
        valid_mask = manual_labels.notna() & (manual_labels != '') & (manual_labels != 'unknown')
        if not valid_mask.any():
            logging.warning("No valid manual labels found for UMAP")
            return ""

        features_clean = features[valid_mask]
        labels_clean = manual_labels[valid_mask]

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)

        # Use same UMAP parameters as automated clustering
        umap_params = {
            'n_neighbors': clustering_result.get('n_neighbors', 15),
            'min_dist': clustering_result.get('min_dist', 0.1),
            'metric': clustering_result.get('metric', 'euclidean'),
            'random_state': 42
        }

        # Fit UMAP
        reducer = umap.UMAP(**umap_params)
        embedding = reducer.fit_transform(features_scaled)

        # Create plot
        plt.figure(figsize=(10, 8))

        # Get unique labels and assign colors
        unique_labels = sorted(labels_clean.unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels_clean == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                        c=[colors[i]], label=str(label), alpha=0.7, s=20)

        plt.title(f'Manual Labels UMAP - {os.path.basename(bird_path)}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save plot
        clusters_dir = os.path.join(bird_path, 'figures', 'clusters')
        os.makedirs(clusters_dir, exist_ok=True)

        plot_path = os.path.join(clusters_dir, 'manual_labels_umap.jpg')
        plt.savefig(plot_path, dpi=config.figure_dpi, bbox_inches='tight')
        plt.close()

        logging.info(f"Generated manual UMAP plot: {plot_path}")
        return plot_path

    except Exception as e:
        logging.error(f"Error generating manual UMAP plot: {e}")
        plt.close()
        return ""


def plot_transition_matrices(
        plots_dir: str,
        auto_result: Dict[str, Any],
        rank_str: str,
        bird_name: str,
        config: PhenotypingConfig
) -> List[str]:
    """
    Generate and save transition matrix heatmaps.

    Args:
        plots_dir: Directory to save plots
        auto_result: Automated phenotype results containing transition matrices
        rank_str: Rank identifier (e.g., "rank0")
        bird_name: Bird identifier
        config: Configuration object

    Returns:
        List of paths to generated image files
    """
    saved_plots = []

    try:
        matrices = [
            (auto_result.get('transition_counts', pd.DataFrame()), "counts", "Transition Counts"),
            (auto_result.get('transition_matrix', pd.DataFrame()), "1st", "1st Order Transitions"),
            (auto_result.get('transition_matrix_2nd', pd.DataFrame()), "2nd", "2nd Order Transitions"),
            (auto_result.get('transition_matrix_3rd', pd.DataFrame()), "3rd", "3rd Order Transitions")
        ]

        for matrix, order_str, title in matrices:
            if matrix.empty:
                continue

            plt.figure(figsize=(10, 10))

            # Format annotations based on matrix type
            if order_str == "counts":
                annot = True
                fmt = "d"
            else:
                annot = _format_transition_annotations(matrix)
                fmt = ""

            # Create heatmap
            sns.heatmap(
                matrix,
                annot=annot,
                fmt=fmt,
                annot_kws={'size': config.heatmap_annotation_size},
                linewidths=0.2,
                square=True,
                cbar_kws={"shrink": 0.7}
            )

            plt.title(f"{title} - {bird_name} ({rank_str})")
            plt.xlabel('Syllable n-1')
            plt.ylabel('Syllable n')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(plots_dir, f"{rank_str}_transition_{order_str}.png")
            plt.savefig(plot_path, dpi=config.figure_dpi, bbox_inches='tight')
            plt.close()

            saved_plots.append(plot_path)

    except Exception as e:
        logging.error(f"Error creating transition plots: {e}")
        plt.close('all')  # Clean up any open figures

    return saved_plots


def plot_repeat_patterns(
        plots_dir: str,
        repeat_counts: pd.DataFrame,
        rank_str: str,
        bird_name: str,
        config: PhenotypingConfig
) -> str:
    """
    Generate and save repeat pattern heatmap.

    Args:
        plots_dir: Directory to save plot
        repeat_counts: DataFrame with repeat counts by syllable and repeat length
        rank_str: Rank identifier (e.g., "rank0")
        bird_name: Bird identifier
        config: Configuration object

    Returns:
        Path to generated image file
    """
    try:
        plt.figure(figsize=(10, 8))

        if repeat_counts.empty:
            # Create placeholder for empty data
            plt.figure(figsize=(5, 5))
            sns.heatmap(
                np.zeros((1, 1), dtype=int),
                annot=True,
                annot_kws={'size': config.heatmap_annotation_size},
                fmt="d",
                linewidths=0.2,
                square=True,
                cbar_kws={"shrink": 0.7}
            )
            plt.title(f'No Repeated Syllables - {bird_name} ({rank_str})')
        else:
            # Create heatmap of repeat counts
            sns.heatmap(
                repeat_counts.astype(int),
                annot=True,
                annot_kws={'size': config.heatmap_annotation_size},
                fmt="d",
                linewidths=0.2,
                square=True,
                cbar_kws={"shrink": 0.7}
            )
            plt.title(f'Repeat Counts - {bird_name} ({rank_str})')
            plt.xlabel('Syllable')
            plt.ylabel('Number of Repeats')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(plots_dir, f"{rank_str}_repeats.png")
        plt.savefig(plot_path, dpi=config.figure_dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    except Exception as e:
        logging.error(f"Error creating repeat plot: {e}")
        plt.close()
        return ""


def plot_vocabulary_comparison(
        plots_dir: str,
        manual_results: Dict[str, Any],
        auto_results: List[Dict[str, Any]],
        bird_name: str,
        config: PhenotypingConfig
) -> str:
    """
    Generate and save vocabulary size comparison plot between manual and automated labels.

    Args:
        plots_dir: Directory to save plot
        manual_results: Manual phenotype results
        auto_results: List of automated phenotype results
        bird_name: Bird identifier
        config: Configuration object

    Returns:
        Path to generated image file
    """
    try:
        plt.figure(figsize=(12, 6))

        # Prepare data for plotting
        manual_vocab_size = manual_results.get('repertoire_size', 0)
        auto_vocab_sizes = [result.get('repertoire_size', 0) for result in auto_results]

        # Create comparison data
        comparison_data = []

        # Add manual data
        if not np.isnan(manual_vocab_size):
            comparison_data.append({
                'Label Type': 'Manual',
                'Rank': 'Manual',
                'Vocabulary Size': manual_vocab_size
            })

        # Add automated data for each rank
        for i, vocab_size in enumerate(auto_vocab_sizes):
            if not np.isnan(vocab_size):
                comparison_data.append({
                    'Label Type': 'Automated',
                    'Rank': f'Rank {i}',
                    'Vocabulary Size': vocab_size
                })

        if not comparison_data:
            logging.warning(f"No vocabulary data available for comparison plot for {bird_name}")
            return ""

        # Create DataFrame and plot
        df = pd.DataFrame(comparison_data)

        # Create bar plot
        ax = plt.subplot(1, 2, 1)
        manual_data = df[df['Label Type'] == 'Manual']
        auto_data = df[df['Label Type'] == 'Automated']

        if not manual_data.empty:
            plt.bar('Manual', manual_data['Vocabulary Size'].iloc[0],
                    color='skyblue', alpha=0.7, label='Manual')

        if not auto_data.empty:
            plt.bar(auto_data['Rank'], auto_data['Vocabulary Size'],
                    color='orange', alpha=0.7, label='Automated')

        plt.title(f'Vocabulary Size Comparison - {bird_name}')
        plt.ylabel('Vocabulary Size')
        plt.xlabel('Label Method')
        plt.legend()
        plt.xticks(rotation=45)

        # Create entropy comparison if available
        ax = plt.subplot(1, 2, 2)
        manual_entropy = manual_results.get('entropy', np.nan)
        auto_entropies = [result.get('entropy', np.nan) for result in auto_results]

        entropy_data = []
        if not np.isnan(manual_entropy):
            entropy_data.append(manual_entropy)
            labels = ['Manual']
        else:
            labels = []

        for i, entropy in enumerate(auto_entropies):
            if not np.isnan(entropy):
                entropy_data.append(entropy)
                labels.append(f'Rank {i}')

        if entropy_data:
            colors = ['skyblue'] + ['orange'] * (len(entropy_data) - 1)
            plt.bar(labels, entropy_data, color=colors, alpha=0.7)
            plt.title(f'Sequence Entropy Comparison - {bird_name}')
            plt.ylabel('Entropy')
            plt.xlabel('Label Method')
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(plots_dir, "vocabulary_comparison.png")
        plt.savefig(plot_path, dpi=config.figure_dpi, bbox_inches='tight')
        plt.close()

        return plot_path

    except Exception as e:
        logging.error(f"Error creating vocabulary comparison plot: {e}")
        plt.close()
        return ""


def _format_transition_annotations(matrix: pd.DataFrame) -> np.ndarray:
    """
    Format numerical values for transition matrix annotations with appropriate precision.

    Args:
        matrix: Transition probability matrix

    Returns:
        Array of formatted string representations
    """
    try:
        array = matrix.to_numpy()
        r, c = array.shape
        annot_array = np.empty((r, c), dtype=object)

        # Format each value
        for i in range(r):
            for j in range(c):
                value = array[i, j]

                # Handle special cases
                if np.isnan(value):
                    annot_array[i, j] = ''
                elif np.isclose(value, 0, atol=1e-3):
                    annot_array[i, j] = '0'
                elif np.isclose(value, 1, atol=1e-3):
                    annot_array[i, j] = '1'
                elif abs(value) < 0.01 or abs(value) >= 100:
                    annot_array[i, j] = f'{value:.2e}'
                else:
                    formatted_value = f'{value:.3f}'
                    # Switch to scientific notation if formatted string is too long
                    if len(formatted_value) > 5:
                        annot_array[i, j] = f'{value:.2e}'
                    else:
                        annot_array[i, j] = formatted_value

        return annot_array

    except Exception as e:
        logging.error(f"Error formatting transition annotations: {e}")
        return np.array([[''] * matrix.shape[1]] * matrix.shape[0])


def _get_available_birds(save_path: str) -> List[str]:
    """
    Get list of available birds for processing.
    Updated to work with the actual directory structure (syllable_data/specs/ and master_summary.csv).
    """
    try:
        birds = []
        if not os.path.exists(save_path):
            logging.error(f"Save path does not exist: {save_path}")
            return birds

        logging.info(f"Scanning directory: {save_path}")

        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)

            # Skip files and hidden directories
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue

            logging.info(f"Checking potential bird: {item}")

            # Check if this looks like a bird directory
            # Look for either:
            # 1. syllable_data/specs/ with .h5 files (for manual labels)
            # 2. master_summary.csv (for clustering results)

            is_valid_bird = False

            # Method 1: Check for syllable spec files
            specs_dir = os.path.join(item_path, 'syllable_data', 'specs')
            if os.path.exists(specs_dir):
                spec_files = [f for f in os.listdir(specs_dir) if f.endswith('.h5') and f.startswith('syllables_')]
                if spec_files:
                    logging.info(f"  Found {len(spec_files)} syllable spec files for {item}")
                    is_valid_bird = True

            # Method 2: Check for master_summary.csv (clustering results)
            master_summary_path = os.path.join(item_path, 'master_summary.csv')
            if os.path.exists(master_summary_path):
                logging.info(f"  Found master_summary.csv for {item}")
                is_valid_bird = True

            if is_valid_bird:
                birds.append(item)
                logging.info(f"  ✅ Added {item} as valid bird")
            else:
                logging.info(f"  ❌ Skipping {item}: no syllable specs or master_summary.csv found")

        logging.info(f"Found {len(birds)} valid birds: {birds}")
        return sorted(birds)

    except Exception as e:
        logging.error(f"Error getting available birds from {save_path}: {e}")
        logging.error(traceback.format_exc())
        return []


def main(save_path: str) -> None:
    """
    Main function to run the unified phenotyping pipeline.

    Args:
        save_path: Root directory containing bird data
    """
    try:
        # Get available birds
        birds = _get_available_birds(save_path)
        if not birds:
            logging.error("No birds found for processing")
            return

        logging.info(f"Found {len(birds)} birds for processing: {birds}")

        # Create configuration
        config = PhenotypingConfig()

        # Process each bird
        successful_birds = []
        failed_birds = []

        for bird in tqdm(birds, desc='Processing birds'):
            bird_path = os.path.join(save_path, bird)
            success = phenotype_bird(bird_path, config)

            if success:
                successful_birds.append(bird)
            else:
                failed_birds.append(bird)

        # Report results
        logging.info(f"Processing complete. Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
        if failed_birds:
            logging.warning(f"Failed birds: {failed_birds}")

        logging.info("Unified phenotyping pipeline execution complete!")

    except Exception as e:
        logging.error(f"Error in main phenotyping pipeline: {e}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    # Setup logging
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'unified_phenotyping.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting unified phenotyping pipeline")

    # Process test datasets
    test_paths = [
        #os.path.join('E:', 'ssharma_RNA_seq'),  # Updated to your actual path
        # Add other paths as needed
        # os.path.join('/Volumes', 'Extreme SSD', 'wseg test'),
        # os.path.join('/Volumes', 'Extreme SSD', 'evsong test'),
        #os.path.join('E:', 'ssharma_RNA_seq')
        Path('E:/') / 'xfosters'
    ]

    for save_path in test_paths:
        if os.path.exists(save_path):
            dataset_name = os.path.basename(save_path)
            logging.info(f"Processing {dataset_name} dataset...")
            main(save_path=save_path)
        else:
            logging.warning(f"Path does not exist: {save_path}")

    logging.info("All datasets processed!")