"""HDBSCAN clustering and cluster quality evaluation (Stage D).

Takes UMAP embeddings produced by Stage C and clusters them using HDBSCAN
over a parameter grid. For each ``(embedding, HDBSCAN params)`` combination,
cluster labels and quality metrics are saved to HDF5 files. A PDF summary
report ranks parameter combinations by a composite score.

Supported quality metrics
-------------------------
``'silhouette'``
    Silhouette coefficient (higher is better; range −1 to 1).
``'dbi'``
    Davies–Bouldin index (lower is better).
``'ch'``
    Calinski–Harabasz index (higher is better).
``'dunn'``
    Dunn index (higher is better).
``'nmi'``
    Normalised mutual information against manual labels, when available.
``'aic'``, ``'bic'``, ``'log-likelihood'``
    Gaussian mixture information criteria.

Public API
----------
- :class:`HDBSCANParams` — HDBSCAN hyperparameter dataclass
- :data:`DEFAULT_HDBSCAN_GRID` — default parameter grid used by :func:`label_bird`
- :func:`label_bird` — Stage D entry point
- :func:`compute_scores` — evaluate cluster quality metrics
- :func:`cluster_embeddings` — cluster a single embedding array
"""

import warnings
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, \
    normalized_mutual_info_score
from scipy.spatial.distance import euclidean, cdist
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import shutil
from pathlib import Path
import tables
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from song_phenotyping.tools.system_utils import check_sys_for_macaw_root
from song_phenotyping.tools.logging_utils import setup_logger

logger = setup_logger(__name__, 'labeling.log')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class HDBSCANParams:
    """Hyperparameters for a single HDBSCAN clustering run.

    Parameters
    ----------
    min_cluster_size : int, optional
        Minimum number of points required to form a cluster. Larger values
        produce fewer, broader clusters. Default is 20.
    min_samples : int, optional
        Number of samples in the neighbourhood for a point to be
        considered a core point. Higher values make the algorithm more
        conservative (more points classified as noise). Default is 5.

    Examples
    --------
    >>> p = HDBSCANParams(min_cluster_size=10, min_samples=3)
    >>> p.to_dict()
    {'min_cluster_size': 10, 'min_samples': 3}
    """

    min_cluster_size: int = 20
    min_samples: int = 5

    def to_dict(self) -> dict:
        """Return parameters as a plain dictionary.

        Returns
        -------
        dict
            Keys: ``min_cluster_size``, ``min_samples``.
        """
        return {'min_cluster_size': self.min_cluster_size, 'min_samples': self.min_samples}

    @classmethod
    def from_dict(cls, data: dict) -> 'HDBSCANParams':
        """Construct a :class:`HDBSCANParams` from a dictionary.

        Parameters
        ----------
        data : dict
            Must contain ``min_cluster_size`` and ``min_samples``.

        Returns
        -------
        HDBSCANParams
        """
        return cls(min_cluster_size=data['min_cluster_size'], min_samples=data['min_samples'])


#: Default HDBSCAN parameter grid explored by :func:`label_bird`.
#: Combines ``min_cluster_size`` in ``[5, 20, 60]`` with
#: ``min_samples`` in ``[5, 15]`` (6 combinations total).
DEFAULT_HDBSCAN_GRID: List[HDBSCANParams] = [
    HDBSCANParams(min_cluster_size=n, min_samples=m)
    for n in [5, 20, 60]
    for m in [5, 15]
]


# ============================================================================
# CORE CLUSTERING FUNCTIONS
# ============================================================================

def compute_scores(embeddings, labels: list, metrics: list, true_labels: list or None = None):
    """Compute clustering evaluation metrics for given embeddings and labels.

    Parameters
    ----------
    embeddings : numpy.ndarray
        UMAP embedding coordinates, shape ``(n_samples, n_components)``.
    labels : list
        Cluster label assigned to each sample.
    metrics : list of str
        Names of metrics to compute.  Supported values: ``'silhouette'``,
        ``'dbi'``, ``'ch'``, ``'dunn'``, ``'nmi'``, ``'aic'``, ``'bic'``,
        ``'log-likelihood'``.
    true_labels : list or None, optional
        Ground-truth syllable labels used for NMI calculation.  When *None*,
        ``'nmi'`` is set to ``nan``.

    Returns
    -------
    dict
        Mapping of metric name → score.  Metrics that cannot be computed
        (e.g. only one cluster present) are set to ``numpy.nan``.
    """
    scores = {}
    n_unique_labels = len(np.unique(labels))

    # Early return if only one cluster (most metrics undefined)
    if n_unique_labels <= 1:
        for metric in metrics:
            scores[metric] = np.nan
        return scores

    # Compute each requested metric
    if 'silhouette' in metrics:
        scores['silhouette'] = silhouette_score(embeddings, labels)

    if 'dbi' in metrics:
        scores['dbi'] = davies_bouldin_score(embeddings, labels)

    if 'ch' in metrics:
        scores['ch'] = calinski_harabasz_score(embeddings, labels)

    if 'dunn' in metrics:
        scores['dunn'] = dunn_index(embeddings, labels)

    # Handle NMI with ground truth labels
    if 'nmi' in metrics and true_labels is not None:
        scores['nmi'] = _compute_nmi(true_labels, labels)
    elif 'nmi' in metrics:
        scores['nmi'] = np.nan

    # Handle information criteria
    info_metrics = [m for m in metrics if m in ['aic', 'bic', 'log-likelihood']]
    if info_metrics:
        ll, aic, bic = information_criterion(embeddings, labels, type=info_metrics)
        if 'log-likelihood' in metrics:
            scores['log-likelihood'] = ll
        if 'aic' in metrics:
            scores['aic'] = aic
        if 'bic' in metrics:
            scores['bic'] = bic

    return scores


def _compute_nmi(true_labels, predicted_labels):
    """Helper function to compute NMI, handling string labels and missing data."""
    try:
        # Handle string/byte labels
        if isinstance(true_labels[0], (bytes, str, np.bytes_)):
            # Find labeled indices (exclude '-' or similar missing indicators)
            labeled_mask = np.array(true_labels) != '-'
            if not np.any(labeled_mask):
                return np.nan

            # Convert to integer labels
            _, true_int_labels = np.unique(true_labels, return_inverse=True)

            # Use only labeled samples
            true_subset = true_int_labels[labeled_mask]
            pred_subset = np.array(predicted_labels)[labeled_mask]

            return normalized_mutual_info_score(true_subset, pred_subset)
        else:
            return normalized_mutual_info_score(true_labels, predicted_labels)
    except Exception as e:
        logger.error(f"Error computing NMI: {e}")
        return np.nan


def cluster_embeddings(embeddings, hashes, method: str = 'hdbscan', cluster_params: dict = {},
                       return_scores: bool = False, path_to_clusters: str = None,
                       path_to_imgs: str = None, umap_id: str = None,
                       true_labels: np.ndarray = None, metrics: list = None):
    """Cluster embeddings using a specified algorithm and hyperparameters.

    Parameters
    ----------
    embeddings : numpy.ndarray
        UMAP embedding coordinates, shape ``(n_samples, n_components)``.
    hashes : array-like
        Per-sample hash identifiers used to cross-reference syllables across
        pipeline stages.
    method : {'hdbscan', 'kmeans'}, optional
        Clustering algorithm.  Default is ``'hdbscan'``.
    cluster_params : dict, optional
        Keyword arguments forwarded directly to the chosen clustering
        algorithm (e.g. ``{'min_cluster_size': 10}`` for HDBSCAN).
    return_scores : bool, optional
        When ``True``, evaluate the resulting partition with *metrics* and
        include the scores in the return value.  Default is ``False``.
    path_to_clusters : str or None, optional
        Directory where cluster label HDF5 files are saved.  No files are
        written when ``None``.
    path_to_imgs : str or None, optional
        Directory where cluster scatter plots are saved.  No images are
        written when ``None``.
    umap_id : str or None, optional
        Short identifier for the UMAP run (embedded in output filenames).
    true_labels : numpy.ndarray or None, optional
        Ground-truth syllable labels used for NMI evaluation.
    metrics : list of str or None, optional
        Metric names to compute when *return_scores* is ``True``.  See
        :func:`compute_scores` for supported values.

    Returns
    -------
    labels : numpy.ndarray
        Cluster label per sample (``-1`` = noise for HDBSCAN).
    hashes : numpy.ndarray
        Input *hashes* unchanged.
    scores : dict or None
        Metric scores if *return_scores* is ``True``, else ``None``.
    label_path : str or None
        Path of the saved label HDF5 file, or ``None`` if not saved.
    plot_path : str or None
        Path of the saved scatter plot, or ``None`` if not saved.
    """
    # Create directories
    if path_to_clusters:
        os.makedirs(path_to_clusters, exist_ok=True)
    if path_to_imgs:
        os.makedirs(path_to_imgs, exist_ok=True)

    # Normalize method name and validate
    normalized_method = method.strip().lower()
    if normalized_method not in ['kmeans', 'hdbscan']:
        raise ValueError(f"Invalid method '{method}'. Choose either 'kmeans' or 'hdbscan'.")

    # Generate file paths
    label_path = None
    plot_path = None
    if path_to_clusters:
        cluster_id = '_'.join([f'{key}{value}' for key, value in cluster_params.items()])
        label_path = os.path.join(path_to_clusters, f'{normalized_method}_{cluster_id}_labels.h5')
        if path_to_imgs and umap_id:
            plot_path = os.path.join(path_to_imgs, f'{umap_id}_{cluster_id}_clusters.jpg')

    # Check if labels already exist (no overwrite for now)
    if label_path and os.path.exists(label_path):
        labels, hashes, scores = load_labels(label_path)
        return labels, hashes, scores, label_path, plot_path

    # Perform clustering
    labels = _perform_clustering(embeddings, normalized_method, cluster_params)

    # Compute scores if requested
    scores = {}
    if return_scores and metrics:
        scores = compute_scores(embeddings, labels, metrics, true_labels)

    # Save results
    if label_path:
        save_labels(label_path, labels, hashes, scores)

    if plot_path:
        plot_umap(embeddings, labels, save=plot_path)

    return labels, hashes, scores, label_path, plot_path


def _perform_clustering(embeddings, method, cluster_params):
    """Helper function to perform the actual clustering."""
    if method == 'kmeans':
        kmeans = KMeans(**cluster_params)
        return kmeans.fit_predict(embeddings)
    elif method == 'hdbscan':
        # Filter parameters to only valid HDBSCAN parameters
        valid_params = {k: v for k, v in cluster_params.items()
                        if k in hdbscan.HDBSCAN().get_params()}
        clusterer = hdbscan.HDBSCAN(**valid_params)
        return clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def search_cluster_params(embeddings, hashes, algorithm, umap_id, directory_path, figure_path,
                          candidate_params: list = None, true_labels: np.ndarray = None,
                          metrics: list = None, sample_size: int = None, use_parallel: bool = True,
                          max_workers: int = None):
    """
    Search through candidate clustering parameters and evaluate performance.

    Args:
        embeddings: UMAP embeddings array
        hashes: Sample hash identifiers
        algorithm: Clustering algorithm name
        umap_id: UMAP parameter identifier
        directory_path: Base directory for saving results
        figure_path: Directory for saving plots
        candidate_params: List of parameter dictionaries to test
        true_labels: Ground truth labels for evaluation
        metrics: List of evaluation metrics to compute
        sample_size: Total number of samples (for efficiency, pass if known)
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of parallel workers (default: CPU count)

    Returns:
        pd.DataFrame: Summary of results for all parameter combinations
    """
    if candidate_params is None:
        candidate_params = []

    if metrics is None:
        metrics = ['silhouette', 'dbi', 'ch']

    # Calculate sample size if not provided
    if sample_size is None:
        sample_size = len(embeddings)

    # Create algorithm-specific directory
    algorithm_path = os.path.join(directory_path, algorithm)
    os.makedirs(algorithm_path, exist_ok=True)

    # Choose parallel or sequential processing
    if use_parallel and len(candidate_params) > 1:
        try:
            return _search_cluster_params_parallel(
                embeddings, hashes, algorithm, umap_id, algorithm_path, figure_path,
                candidate_params, true_labels, metrics, sample_size, max_workers
            )
        except Exception as e:
            logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
            return _search_cluster_params_sequential(
                embeddings, hashes, algorithm, umap_id, algorithm_path, figure_path,
                candidate_params, true_labels, metrics, sample_size
            )
    else:
        return _search_cluster_params_sequential(
            embeddings, hashes, algorithm, umap_id, algorithm_path, figure_path,
            candidate_params, true_labels, metrics, sample_size
        )


def _search_cluster_params_parallel(embeddings, hashes, algorithm, umap_id, algorithm_path,
                                    figure_path, candidate_params, true_labels, metrics,
                                    sample_size, max_workers):
    """Parallel implementation of parameter search."""

    # Control worker count
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(candidate_params))
    else:
        max_workers = min(max_workers, len(candidate_params))

    logger.info(f"Using {max_workers} parallel workers for parameter search")

    # Prepare arguments for parallel processing
    args_list = []
    for params in candidate_params:
        args_list.append((
            embeddings, hashes, algorithm, params, algorithm_path,
            figure_path, umap_id, true_labels, metrics, sample_size
        ))

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(_cluster_single_params, args): args for args in args_list}

        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_params):
            result = future.result()
            completed += 1

            if result is not None:
                results.append(result)
                # Log success/failure
                if not pd.isna(result.get('n_syls', np.nan)):
                    logger.debug(f"✅ {completed}/{len(args_list)}: Completed parameter combination")
                else:
                    logger.debug(f"❌ {completed}/{len(args_list)}: Failed parameter combination")

    # Clear embeddings from memory
    del embeddings

    return pd.DataFrame(results)


def _search_cluster_params_sequential(embeddings, hashes, algorithm, umap_id, algorithm_path,
                                      figure_path, candidate_params, true_labels, metrics, sample_size):
    """Sequential implementation of parameter search (original logic)."""

    logger.info("Using sequential processing for parameter search")

    results = []

    # Process each parameter combination sequentially
    for params in tqdm(candidate_params, desc="Searching cluster parameters..."):
        try:
            labels, hashes_returned, scores, label_path, fig_path = cluster_embeddings(
                embeddings=embeddings,
                hashes=hashes,
                method=algorithm,
                cluster_params=params,
                return_scores=True,
                path_to_clusters=algorithm_path,
                path_to_imgs=figure_path,
                true_labels=true_labels,
                umap_id=umap_id,
                metrics=metrics
            )

            # Build result row
            result_row = {
                'sample_size': sample_size,
                'n_syls': len(np.unique(labels)),
                'png_path': Path(fig_path) if fig_path else None,
                'label_path': Path(label_path) if label_path else None,
            }

            # Add parameter values
            result_row.update(params)

            # Add metric scores (raw values for cross-bird comparison)
            result_row.update(scores)

            results.append(result_row)

        except Exception as e:
            logger.error(f"Error processing parameters {params}: {e}")
            # Add failed result with NaN scores
            result_row = {
                'sample_size': sample_size,
                'n_syls': np.nan,
                'png_path': None,
                'label_path': None,
            }
            result_row.update(params)
            result_row.update({metric: np.nan for metric in metrics})
            results.append(result_row)

    # Clear embeddings from memory since we're done with clustering
    del embeddings

    # Create summary DataFrame
    return pd.DataFrame(results)


def _cluster_single_params(args):
    """Worker function for parallel parameter search."""
    (embeddings, hashes, algorithm, params, algorithm_path,
     figure_path, umap_id, true_labels, metrics, sample_size) = args

    try:
        labels, hashes_returned, scores, label_path, fig_path = cluster_embeddings(
            embeddings=embeddings,
            hashes=hashes,
            method=algorithm,
            cluster_params=params,
            return_scores=True,
            path_to_clusters=algorithm_path,
            path_to_imgs=figure_path,
            true_labels=true_labels,
            umap_id=umap_id,
            metrics=metrics
        )

        # Build result row
        result_row = {
            'sample_size': sample_size,
            'n_syls': len(np.unique(labels)),
            'png_path': Path(fig_path) if fig_path else None,
            'label_path': Path(label_path) if label_path else None,
        }

        # Add parameter values
        result_row.update(params)

        # Add metric scores
        result_row.update(scores)

        return result_row

    except Exception as e:
        logger.error(f"Error processing parameters {params}: {e}")
        # Return failed result with NaN scores
        result_row = {
            'sample_size': sample_size,
            'n_syls': np.nan,
            'png_path': None,
            'label_path': None,
        }
        result_row.update(params)
        result_row.update({metric: np.nan for metric in metrics})
        return result_row


def information_criterion(embeddings, labels, type=['aic', 'bic', 'log-likelihood']):
    """
    Compute information criteria (AIC, BIC) and log-likelihood for clustering results.

    Args:
        embeddings: UMAP embeddings array
        labels: Cluster labels
        type: List of criteria to compute

    Returns:
        tuple: (log_likelihood, aic, bic) - None for uncomputed metrics
    """
    try:
        n_samples = len(labels)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise points
        k = len(unique_labels)

        # Handle edge cases
        if k == 0:  # All noise
            return np.nan, np.nan, np.nan
        if k == 1:  # Single cluster
            return np.nan, np.nan, np.nan

        # Compute Within-Cluster Sum of Squares (WCSS)
        wcss = 0
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum(cdist(cluster_points, [centroid], 'euclidean') ** 2)

        # Compute log-likelihood (simplified Gaussian assumption)
        log_likelihood = -wcss / (2 * n_samples)

        # Compute information criteria
        aic = 2 * k - 2 * log_likelihood if 'aic' in type else None
        bic = np.log(n_samples) * k - 2 * log_likelihood if 'bic' in type else None
        log_likelihood = log_likelihood if 'log-likelihood' in type else None

        return log_likelihood, aic, bic

    except Exception as e:
        logger.error(f"Error computing information criteria: {e}")
        return np.nan, np.nan, np.nan


def dunn_index(embeddings, labels):
    """
    Compute Dunn index for clustering evaluation.
    Higher values indicate better clustering (compact clusters, well-separated).

    Args:
        embeddings: UMAP embeddings array
        labels: Cluster labels

    Returns:
        float: Dunn index value
    """
    try:
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise

        if len(unique_labels) < 2:
            return np.nan

        # Compute maximum intra-cluster distances
        max_intra_distances = []
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) < 2:
                max_intra_distances.append(0)
                continue

            # Compute all pairwise distances within cluster
            intra_distances = cdist(cluster_points, cluster_points, 'euclidean')
            # Get upper triangle (avoid diagonal and duplicates)
            upper_triangle = intra_distances[np.triu_indices_from(intra_distances, k=1)]
            max_intra_distances.append(np.max(upper_triangle) if len(upper_triangle) > 0 else 0)

        # Compute minimum inter-cluster distances
        min_inter_distances = []
        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i + 1:]:
                cluster_i_points = embeddings[labels == label_i]
                cluster_j_points = embeddings[labels == label_j]
                inter_distances = cdist(cluster_i_points, cluster_j_points, 'euclidean')
                min_inter_distances.append(np.min(inter_distances))

        if not min_inter_distances or max(max_intra_distances) == 0:
            return np.nan

        return min(min_inter_distances) / max(max_intra_distances)

    except Exception as e:
        logger.error(f"Error calculating Dunn index: {e}")
        return np.nan


# ============================================================================
# DATA MANAGEMENT & I/O
# ============================================================================

def load_umap_embeddings(embedding_path: str):
    """
    Load embeddings, hashes, and labels from HDF5 file using PyTables.

    Args:
        embedding_path: Path to HDF5 embedding file

    Returns:
        tuple: (embeddings, hashes, labels) or (None, None, None) on error
    """
    try:
        with tables.open_file(embedding_path, mode='r') as f:
            embeddings = f.root.embeddings.read()
            hashes = [hash_id.decode('utf-8') for hash_id in f.root.hashes.read()]
            labels = [label.decode('utf-8') for label in f.root.labels.read()]
        return embeddings, hashes, labels
    except Exception as e:
        logger.error(f"Error loading embeddings from {embedding_path}: {e}")
        return None, None, None


def save_labels(label_save_path: str, labels, hashes, scores):
    """
    Save cluster labels, hashes, and evaluation scores to HDF5 file.

    Args:
        label_save_path: Path where to save the HDF5 file
        labels: Cluster labels array
        hashes: Sample hash identifiers
        scores: Dictionary of evaluation scores

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with tables.open_file(label_save_path, mode='w') as f:
            # Save labels
            f.create_array('/', 'labels', obj=np.array(labels))

            # Save hashes as Unicode strings
            f.create_array('/', 'hashes', obj=np.array(hashes, dtype='U'))

            # Save each score as separate array
            for key, value in scores.items():
                if value is not None and not np.isnan(value):
                    f.create_array('/', key, obj=np.array(value))

        return True

    except Exception as e:
        logger.error(f"Error saving labels to {label_save_path}: {e}")
        return False


def load_labels(label_save_path: str):
    """
    Load cluster labels, hashes, and evaluation scores from HDF5 file.

    Args:
        label_save_path: Path to HDF5 file

    Returns:
        tuple: (labels, hashes, scores) or (None, None, None) on error
    """
    try:
        # Handle path resolution (simplified from original)
        resolved_path = _resolve_file_path(label_save_path)

        with tables.open_file(resolved_path, mode='r') as f:
            # Load labels
            labels = f.root.labels.read()

            # Load hashes
            hashes_raw = f.root.hashes.read()
            hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h) for h in hashes_raw]

            # Load scores (all arrays except labels and hashes)
            scores = {}
            for node in f.list_nodes(f.root, classname='Array'):
                if node._v_name not in ['labels', 'hashes']:
                    score_value = node.read()
                    # Handle scalar vs array values
                    scores[node._v_name] = float(score_value) if score_value.ndim == 0 else score_value

        return labels, hashes, scores

    except Exception as e:
        logger.error(f"Error loading labels from {label_save_path}: {e}")
        return None, None, None


def _resolve_file_path(file_path: str) -> str:
    """
    Resolve file path, handling cross-platform and network path issues.

    Args:
        file_path: Original file path

    Returns:
        str: Resolved file path
    """
    # If file exists as-is, return it
    if os.path.exists(file_path):
        return file_path

    try:
        # Handle network path resolution (simplified from original complex logic)
        path_to_macaw = check_sys_for_macaw_root()

        # Extract relative path (last 9 components as in original)
        path_parts = file_path.replace('\\', '/').split('/')
        if len(path_parts) >= 9:
            relative_path = '/'.join(path_parts[-9:])
            resolved_path = os.path.join(path_to_macaw, relative_path)
            if os.path.exists(resolved_path):
                return resolved_path

        # If all else fails, return original path
        return file_path

    except Exception as e:
        logger.warning(f"Error resolving path {file_path}: {e}")
        return file_path


def parse_embedding_filename(filename: str):
    """
    Parse UMAP parameters from embedding filename.
    Updated to handle new filename format with sample info.

    Args:
        filename: Embedding filename (e.g., 'euclidean_10neighbors_0.1dist_full3301.h5')

    Returns:
        tuple: (metric, n_neighbors, min_dist, umap_id) or None on error
    """
    try:
        # Remove file extension
        base_name = filename.replace('.h5', '')

        # Handle both old and new filename formats
        if '_full' in base_name or '_subsample' in base_name:
            # New format: euclidean_10neighbors_0.1dist_full3301
            # Split at the sample info part
            if '_full' in base_name:
                main_part = base_name.split('_full')[0]
            else:  # _subsample
                main_part = base_name.split('_subsample')[0]
        else:
            # Old format: euclidean_10neighbors_0.1dist
            main_part = base_name

        # Split into components
        parts = main_part.split('_')
        if len(parts) != 3:
            raise ValueError(f"Unexpected filename format: {filename}")

        metric = parts[0]
        n_neighbors = int(parts[1].replace('neighbors', ''))
        min_dist = float(parts[2].replace('dist', ''))

        return metric, n_neighbors, min_dist, main_part

    except Exception as e:
        logger.error(f"Error parsing embedding filename {filename}: {e}")
        return None

# ============================================================================
# EVALUATION & RANKING
# ============================================================================

def compute_composite_score(summary_df, metrics, n_syls: list = None, weights=None,
                            target_clusters=20, penalty_decay=0.1, use_cluster_penalty=False):
    """
    Compute composite scores with normalization and optional cluster count penalty.

    Args:
        summary_df: DataFrame with raw metric scores
        metrics: List of metrics to include in composite score
        n_syls: List of cluster counts (extracted from summary_df if None)
        weights: Weights for each metric (equal weights if None)
        target_clusters: Target number of clusters for penalty function
        penalty_decay: Decay rate for cluster penalty
        use_cluster_penalty: Whether to apply cluster count penalty (default: False)

    Returns:
        pd.DataFrame: DataFrame with added normalized scores and composite score
    """
    df = summary_df.copy()

    # Get cluster counts
    if n_syls is None:
        n_syls = df['n_syls'].tolist()

    # Set equal weights if not provided
    if weights is None:
        weights = [1.0] * len(metrics)

    if len(weights) != len(metrics):
        raise ValueError("Number of weights must match number of metrics")

    # Normalize each metric
    normalized_scores = []
    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in DataFrame, skipping")
            continue

        metric_values = df[metric]

        # Skip if all values are NaN
        if metric_values.isna().all():
            df[f'normalized_{metric}'] = np.nan
            normalized_scores.append(pd.Series([0] * len(df)))
            continue

        # Normalize based on metric type (higher vs lower is better)
        if metric in ['nmi', 'silhouette', 'ch', 'log-likelihood', 'dunn']:
            # Higher is better
            norm_metric = _normalize_higher_better(metric_values)
        elif metric in ['dbi', 'aic', 'bic']:
            # Lower is better
            norm_metric = _normalize_lower_better(metric_values)
        else:
            logger.warning(f"Unknown metric {metric}, treating as higher-is-better")
            norm_metric = _normalize_higher_better(metric_values)

        df[f'normalized_{metric}'] = norm_metric
        normalized_scores.append(norm_metric)

    # Compute weighted composite score
    if normalized_scores:
        weighted_scores = [w * norm.fillna(0) for w, norm in zip(weights, normalized_scores)]
        composite_base = sum(weighted_scores) / sum(weights)  # Weighted average

        # Apply cluster penalty only if requested
        if use_cluster_penalty:
            cluster_penalties = [score_cluster_penalty(n, target_clusters, penalty_decay) for n in n_syls]
            df['composite_score'] = composite_base * cluster_penalties
            logger.info("Applied cluster count penalty to composite scores")
        else:
            df['composite_score'] = composite_base

    else:
        df['composite_score'] = np.nan

    return df


def _normalize_higher_better(values):
    """Normalize values where higher is better (0-1 scale)."""
    min_val, max_val = values.min(), values.max()
    if min_val == max_val:
        return pd.Series([1.0] * len(values))
    return (values - min_val) / (max_val - min_val)


def _normalize_lower_better(values):
    """Normalize values where lower is better (0-1 scale, inverted)."""
    min_val, max_val = values.min(), values.max()
    if min_val == max_val:
        return pd.Series([1.0] * len(values))
    return (max_val - values) / (max_val - min_val)


def score_cluster_penalty(n, target=20, decay_rate=0.1):
    """
    Compute penalty for cluster count deviation from target.

    Args:
        n: Number of clusters
        target: Target number of clusters
        decay_rate: Exponential decay rate for penalty

    Returns:
        float: Penalty multiplier (1.0 = no penalty, <1.0 = penalty applied)
    """
    if n <= target:
        return 1.0
    else:
        # Exponential decay penalty for too many clusters
        return 0.8 * np.exp(-(n - target) / decay_rate)


def select_best_params(summary_df, selection_method='composite', top_n=1):
    """
    Select best parameter combinations based on specified method.

    Args:
        summary_df: DataFrame with scores and composite scores
        selection_method: Method for selection ('composite' or specific metric name)
        top_n: Number of top parameter combinations to return

    Returns:
        pd.DataFrame: Top N parameter combinations sorted by selection criteria
    """
    if summary_df.empty:
        return pd.DataFrame()

    # Select based on method
    if selection_method == 'composite':
        if 'composite_score' not in summary_df.columns:
            raise ValueError("Composite score not found. Run compute_composite_score first.")
        sorted_df = summary_df.sort_values('composite_score', ascending=False)
    else:
        if selection_method not in summary_df.columns:
            raise ValueError(f"Selection method '{selection_method}' not found in DataFrame")

        # Determine sort order based on metric type
        if selection_method in ['nmi', 'silhouette', 'ch', 'log-likelihood', 'dunn']:
            ascending = False  # Higher is better
        else:
            ascending = True  # Lower is better

        sorted_df = summary_df.sort_values(selection_method, ascending=ascending)

    return sorted_df.head(top_n).reset_index(drop=True)


def compute_metric_ranking(summary_df, metrics):
    """
    Compute individual metric rankings and aggregate rank.

    Args:
        summary_df: DataFrame with metric scores
        metrics: List of metrics to rank

    Returns:
        pd.DataFrame: DataFrame with added ranking columns
    """
    df = summary_df.copy()

    rank_columns = []
    for metric in metrics:
        if metric not in df.columns:
            continue

        rank_col = f'{metric}_rank'

        # Rank based on metric type
        if metric in ['nmi', 'silhouette', 'ch', 'log-likelihood', 'dunn']:
            df[rank_col] = df[metric].rank(ascending=False, na_option='bottom')
        else:
            df[rank_col] = df[metric].rank(ascending=True, na_option='bottom')

        rank_columns.append(rank_col)

    # Compute aggregate rank (sum of individual ranks)
    if rank_columns:
        df['aggregate_rank'] = df[rank_columns].sum(axis=1)

    return df

# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def plot_umap(embeddings, labels, save: str = None, figsize=(12, 10), title=None):
    """
    Create scatter plot of UMAP embeddings colored by cluster labels.

    Args:
        embeddings: UMAP embeddings array (N x 2)
        labels: Cluster labels
        save: Path to save plot (optional)
        figsize: Figure size tuple
        title: Plot title (optional)

    Returns:
        None
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)

        # Handle string/byte labels
        plot_labels = labels
        if isinstance(labels[0], (str, bytes)):
            # Convert to integer labels for coloring
            unique_labels, plot_labels = np.unique(labels, return_inverse=True)

        # Create scatter plot
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                             c=plot_labels, cmap='tab10', alpha=0.7, s=20)

        # Customize plot
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=14)

        # Add legend for cluster labels
        n_clusters = len(np.unique(plot_labels))
        if n_clusters <= 20:  # Only show legend if reasonable number of clusters
            legend = ax.legend(*scatter.legend_elements(),
                               title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.add_artist(legend)

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error creating UMAP plot: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_summary_matrix(summary_df, save_path: str, metrics: list, figsize=(12, 8)):
    """
    Create heatmap visualization of normalized metric scores.

    Args:
        summary_df: DataFrame with normalized scores
        save_path: Path to save the plot
        metrics: List of metrics to include
        figsize: Figure size tuple

    Returns:
        None
    """
    try:
        # Select normalized metrics and composite score
        plot_columns = [f'normalized_{metric}' for metric in metrics if f'normalized_{metric}' in summary_df.columns]
        if 'composite_score' in summary_df.columns:
            plot_columns.append('composite_score')

        if not plot_columns:
            logger.warning("No normalized metrics found for plotting")
            return

        plot_data = summary_df[plot_columns]

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(plot_data.T, annot=True, fmt=".3f", cmap="viridis",
                    cbar_kws={'label': 'Normalized Score'})

        plt.title('Clustering Performance Summary (Normalized Scores)', fontsize=14)
        plt.xlabel('Parameter Combination')
        plt.ylabel('Metric')
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error creating summary matrix plot: {e}")


def create_cluster_summary_pdf(master_summary_df, bird: str, save_path: str, top_n: int = None):
    """
    Create comprehensive PDF report with cluster visualizations and scores.

    Args:
        master_summary_df: DataFrame with all clustering results
        bird: Bird identifier
        save_path: Directory to save PDF
        top_n: Number of top results to include (all if None)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        pdf_path = os.path.join(save_path, f'{bird}_cluster_summary.pdf')

        # Select subset if requested
        if top_n is not None:
            plot_df = master_summary_df.head(top_n)
        else:
            plot_df = master_summary_df

        with PdfPages(pdf_path) as pdf:
            for index, row in plot_df.iterrows():
                # Resolve PNG path
                png_path = _resolve_plot_path(row['png_path'])

                if not os.path.exists(png_path):
                    logger.warning(f"PNG file not found: {png_path}")
                    continue

                try:
                    # Create figure with plot and metadata
                    fig = _create_summary_page(row, index, len(plot_df), png_path)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

                except Exception as e:
                    logger.error(f"Error processing page {index}: {e}")
                    continue

        logger.info(f"Created PDF summary: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating PDF summary: {e}")
        return False


def _resolve_plot_path(png_path):
    """Resolve PNG path, handling different path formats."""
    if isinstance(png_path, Path):
        png_path = str(png_path)

    # Handle Windows network paths
    if 'Y:' in str(png_path):
        png_path = str(png_path).replace(f'Y:{os.sep}', check_sys_for_macaw_root())

    return png_path


def _create_summary_page(row, index, total_rows, png_path):
    """Create individual summary page for PDF report."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Top subplot: cluster visualization
    img = plt.imread(png_path)
    ax1.imshow(img)
    ax1.axis('off')

    # Create title with parameters and scores
    title_lines = [
        f"UMAP: n_neighbors={int(row['n_neighbors'])}, min_dist={row['min_dist']:.3f}, metric={row['metric']}",
        f"Clustering: min_cluster_size={int(row['min_cluster_size'])}, min_samples={int(row['min_samples'])}",
        f"Clusters: {int(row['n_syls'])} | Composite Score: {row['composite_score']:.3f} | Rank: {index + 1}/{total_rows}"
    ]
    ax1.set_title('\n'.join(title_lines), fontsize=12, pad=20)

    # Bottom subplot: metrics table
    _create_metrics_table(ax2, row)

    plt.tight_layout()
    return fig


def _create_metrics_table(ax, row):
    """Create metrics table for summary page."""
    # Extract metric scores (exclude non-metric columns)
    exclude_cols = ['sample_size', 'n_syls', 'png_path', 'label_path',
                    'n_neighbors', 'min_dist', 'metric', 'min_cluster_size',
                    'min_samples', 'composite_score']

    metric_data = []
    for col in row.index:
        if col not in exclude_cols and not col.startswith('normalized_'):
            value = row[col]
            if pd.notna(value):
                metric_data.append([col.upper(), f"{value:.4f}"])

    if metric_data:
        # Create table
        table = ax.table(cellText=metric_data,
                         colLabels=['Metric', 'Score'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(metric_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')

    ax.axis('off')


def reorder_columns(master_summary_df: pd.DataFrame, metrics: list):
    """
    Reorder DataFrame columns for better readability and analysis.

    Args:
        master_summary_df: DataFrame to reorder
        metrics: List of metrics used

    Returns:
        pd.DataFrame: Reordered DataFrame
    """
    # Define column order priority
    priority_cols = []

    # Add NMI first if present (special handling for ground truth comparison)
    if 'nmi' in metrics and 'nmi' in master_summary_df.columns:
        priority_cols.append('nmi')

    # Core results columns
    core_cols = ['composite_score', 'sample_size', 'n_syls']
    priority_cols.extend([col for col in core_cols if col in master_summary_df.columns])

    # Parameter columns
    param_cols = ['n_neighbors', 'min_dist', 'metric', 'min_cluster_size', 'min_samples']
    priority_cols.extend([col for col in param_cols if col in master_summary_df.columns])

    # Raw metric scores (excluding NMI if already added)
    metric_cols = [m for m in metrics if m in master_summary_df.columns and m not in priority_cols]
    priority_cols.extend(metric_cols)

    # Normalized scores
    normalized_cols = [f'normalized_{m}' for m in metrics
                       if f'normalized_{m}' in master_summary_df.columns]
    priority_cols.extend(normalized_cols)

    # Path columns last
    path_cols = ['png_path', 'label_path']
    priority_cols.extend([col for col in path_cols if col in master_summary_df.columns])

    # Add any remaining columns
    remaining_cols = [col for col in master_summary_df.columns if col not in priority_cols]
    final_columns = priority_cols + remaining_cols

    return master_summary_df[final_columns]


def save_master_summary(master_summary_df, bird_path: str):
    """
    Save master summary DataFrame to CSV under the results/ directory.

    Args:
        master_summary_df: DataFrame to save
        bird_path: Bird root directory

    Returns:
        bool: True if successful, False otherwise
    """
    from song_phenotyping.tools.pipeline_paths import RESULTS_DIR
    try:
        results_dir = os.path.join(bird_path, RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        master_summary_path = os.path.join(results_dir, 'master_summary.csv')
        master_summary_df.to_csv(master_summary_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Error saving master summary: {e}")
        return False


def load_master_summary(bird_path: str):
    """
    Load master summary DataFrame from CSV.

    Args:
        bird_path: Bird root directory

    Returns:
        pd.DataFrame: Loaded DataFrame or empty DataFrame if not found
    """
    from song_phenotyping.tools.pipeline_paths import RESULTS_DIR
    try:
        master_summary_path = os.path.join(bird_path, RESULTS_DIR, 'master_summary.csv')
        if os.path.exists(master_summary_path):
            return pd.read_csv(master_summary_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading master summary: {e}")
        return pd.DataFrame()


# ============================================================================
# CROSS-BIRD ANALYSIS
# ============================================================================

def aggregate_raw_scores_across_birds(save_path: str, birds: list = None, top_n: int = 10):
    """
    Aggregate raw clustering scores across multiple birds for cross-bird comparison.

    Args:
        save_path: Root directory containing bird subdirectories
        birds: List of bird names to include (all birds if None)
        top_n: Number of top parameter combinations to include per bird

    Returns:
        pd.DataFrame: Aggregated raw scores with bird identifiers
    """
    if birds is None:
        birds = _get_available_birds(save_path)

    aggregated_results = []

    for bird in tqdm(birds, desc="Aggregating scores across birds"):
        try:
            bird_path = os.path.join(save_path, bird)

            # Load bird's master summary
            master_summary = load_master_summary(bird_path)
            if master_summary.empty:
                logger.warning(f"No master summary found for bird {bird}")
                continue

            # Select top N parameter combinations for this bird
            top_results = select_best_params(master_summary, selection_method='composite', top_n=top_n)

            # Add bird identifier
            top_results['bird'] = bird

            # Ensure sample_size is included (calculate if missing)
            if 'sample_size' not in top_results.columns:
                top_results['sample_size'] = _estimate_sample_size_from_paths(top_results)

            aggregated_results.append(top_results)

        except Exception as e:
            logger.error(f"Error processing bird {bird}: {e}")
            continue

    if not aggregated_results:
        logger.warning("No valid results found for cross-bird analysis")
        return pd.DataFrame()

    # Combine all results
    combined_df = pd.concat(aggregated_results, ignore_index=True)

    # Reorder columns for cross-bird analysis
    combined_df = _reorder_for_cross_bird_analysis(combined_df)

    return combined_df


def analyze_parameter_performance_by_sample_size(aggregated_df, sample_size_bins=None,
                                                 metrics=None):
    """
    Analyze how parameter combinations perform across different sample size ranges.

    Args:
        aggregated_df: DataFrame from aggregate_raw_scores_across_birds
        sample_size_bins: List of bin edges for sample size categorization
        metrics: List of metrics to analyze

    Returns:
        pd.DataFrame: Analysis results by sample size bins
    """
    if aggregated_df.empty:
        return pd.DataFrame()

    if sample_size_bins is None:
        sample_size_bins = [0, 1000, 5000, 10000, 20000, np.inf]

    if metrics is None:
        metrics = ['silhouette', 'dbi', 'ch', 'aic']

    # Create sample size categories
    aggregated_df = aggregated_df.copy()
    aggregated_df['sample_size_bin'] = pd.cut(aggregated_df['sample_size'],
                                              bins=sample_size_bins,
                                              labels=[
                                                  f"{int(sample_size_bins[i])}-{int(sample_size_bins[i + 1]) if sample_size_bins[i + 1] != np.inf else 'inf'}"
                                                  for i in range(len(sample_size_bins) - 1)])

    # Group by sample size bin and parameter combinations
    param_cols = ['n_neighbors', 'min_dist', 'min_cluster_size', 'min_samples']
    available_param_cols = [col for col in param_cols if col in aggregated_df.columns]

    if not available_param_cols:
        logger.error("No parameter columns found for analysis")
        return pd.DataFrame()

    # Analyze performance by sample size bin
    analysis_results = []

    for bin_name, bin_group in aggregated_df.groupby('sample_size_bin'):
        bin_analysis = {
            'sample_size_bin': bin_name,
            'n_birds': bin_group['bird'].nunique(),
            'n_parameter_combinations': len(bin_group),
            'avg_sample_size': bin_group['sample_size'].mean(),
            'avg_n_clusters': bin_group['n_syls'].mean(),
        }

        # Add metric statistics
        for metric in metrics:
            if metric in bin_group.columns:
                bin_analysis[f'{metric}_mean'] = bin_group[metric].mean()
                bin_analysis[f'{metric}_std'] = bin_group[metric].std()

        # Find most common parameter values
        for param in available_param_cols:
            if param in bin_group.columns:
                mode_value = bin_group[param].mode()
                bin_analysis[f'{param}_mode'] = mode_value.iloc[0] if len(mode_value) > 0 else np.nan

        analysis_results.append(bin_analysis)

    return pd.DataFrame(analysis_results)


def compute_cross_bird_composite_scores(aggregated_df, metrics, weights=None):
    """
    Compute new composite scores based on raw scores across all birds.

    Args:
        aggregated_df: DataFrame with raw scores from multiple birds
        metrics: List of metrics to include in composite score
        weights: Metric weights (equal if None)

    Returns:
        pd.DataFrame: DataFrame with cross-bird composite scores
    """
    if aggregated_df.empty:
        return aggregated_df

    # Use the existing composite score function but on the full dataset
    df_with_composite = compute_composite_score(
        aggregated_df,
        metrics=metrics,
        weights=weights,
        n_syls=aggregated_df['n_syls'].tolist(),
        use_cluster_penalty=False  # Add this parameter
    )

    # Add cross-bird ranking
    df_with_composite['cross_bird_rank'] = df_with_composite['composite_score'].rank(ascending=False)

    return df_with_composite


def _get_available_birds(save_path: str):
    """Get list of available bird directories."""
    try:
        all_items = os.listdir(save_path)
        birds = []

        for item in all_items:
            item_path = os.path.join(save_path, item)
            # Check if it's a directory and has expected pipeline structure
            if os.path.isdir(item_path) and not item.startswith('.'):
                from song_phenotyping.tools.pipeline_paths import RESULTS_DIR, STAGES_DIR
                if (os.path.exists(os.path.join(item_path, RESULTS_DIR, 'master_summary.csv')) or
                        os.path.exists(os.path.join(item_path, STAGES_DIR))):
                    birds.append(item)

        return sorted(birds)

    except Exception as e:
        logger.error(f"Error getting available birds: {e}")
        return []


def _estimate_sample_size_from_paths(results_df):
    """Estimate sample size from embedding files if not directly available."""
    sample_sizes = []

    for _, row in results_df.iterrows():
        try:
            # Try to get from existing column first
            if 'sample_size' in row and pd.notna(row['sample_size']):
                sample_sizes.append(int(row['sample_size']))
                continue

            # Otherwise estimate from label path by loading the file
            if 'label_path' in row and pd.notna(row['label_path']):
                try:
                    labels, _, _ = load_labels(str(row['label_path']))
                    if labels is not None:
                        sample_sizes.append(len(labels))
                        continue
                except Exception as e:
                    logger.debug(f"Could not load labels from {row['label_path']}: {e}")

            # Fallback: use a default estimate or NaN
            logger.warning(f"Could not determine sample size for row, using NaN")
            sample_sizes.append(np.nan)

        except Exception as e:
            logger.error(f"Error estimating sample size: {e}")
            sample_sizes.append(np.nan)

    return sample_sizes


def _reorder_for_cross_bird_analysis(combined_df):
    """Reorder columns for cross-bird analysis readability."""
    priority_cols = ['bird', 'sample_size', 'composite_score', 'n_syls']

    # Add parameter columns
    param_cols = ['n_neighbors', 'min_dist', 'metric', 'min_cluster_size', 'min_samples']
    priority_cols.extend([col for col in param_cols if col in combined_df.columns])

    # Add metric columns (raw scores)
    metric_cols = [col for col in combined_df.columns
                   if col not in priority_cols + ['png_path', 'label_path']
                   and not col.startswith('normalized_')]
    priority_cols.extend(metric_cols)

    # Add path columns last
    path_cols = ['png_path', 'label_path']
    priority_cols.extend([col for col in path_cols if col in combined_df.columns])

    # Ensure all columns are included
    remaining_cols = [col for col in combined_df.columns if col not in priority_cols]
    final_columns = priority_cols + remaining_cols

    return combined_df[final_columns]


def save_cross_bird_analysis(aggregated_df, analysis_by_size, save_path: str, filename_prefix: str = 'cross_bird'):
    """
    Save cross-bird analysis results to CSV files.

    Args:
        aggregated_df: Aggregated raw scores DataFrame
        analysis_by_size: Sample size analysis DataFrame
        save_path: Directory to save files
        filename_prefix: Prefix for output filenames

    Returns:
        dict: Paths to saved files
    """
    saved_files = {}

    try:
        # Save aggregated raw scores
        aggregated_path = os.path.join(save_path, f'{filename_prefix}_aggregated_scores.csv')
        aggregated_df.to_csv(aggregated_path, index=False)
        saved_files['aggregated_scores'] = aggregated_path

        # Save sample size analysis
        if not analysis_by_size.empty:
            analysis_path = os.path.join(save_path, f'{filename_prefix}_sample_size_analysis.csv')
            analysis_by_size.to_csv(analysis_path, index=False)
            saved_files['sample_size_analysis'] = analysis_path

        logger.info(f"Saved cross-bird analysis files: {list(saved_files.keys())}")
        return saved_files

    except Exception as e:
        logger.error(f"Error saving cross-bird analysis: {e}")
        return {}


def identify_optimal_parameters_by_sample_size(analysis_by_size_df):
    """
    Identify which parameter combinations work best for different sample size ranges.

    Args:
        analysis_by_size_df: DataFrame from analyze_parameter_performance_by_sample_size

    Returns:
        dict: Recommendations by sample size bin
    """
    recommendations = {}

    for _, row in analysis_by_size_df.iterrows():
        bin_name = row['sample_size_bin']

        # Extract parameter recommendations
        param_recommendations = {}
        param_cols = [col for col in row.index if col.endswith('_mode')]

        for col in param_cols:
            param_name = col.replace('_mode', '')
            if pd.notna(row[col]):
                param_recommendations[param_name] = row[col]

        # Extract performance metrics
        performance_summary = {}
        metric_cols = [col for col in row.index if col.endswith('_mean')]

        for col in metric_cols:
            metric_name = col.replace('_mean', '')
            if pd.notna(row[col]):
                performance_summary[metric_name] = {
                    'mean': row[col],
                    'std': row.get(f'{metric_name}_std', np.nan)
                }

        recommendations[bin_name] = {
            'sample_info': {
                'n_birds': row['n_birds'],
                'n_parameter_combinations': row['n_parameter_combinations'],
                'avg_sample_size': row['avg_sample_size'],
                'avg_n_clusters': row['avg_n_clusters']
            },
            'recommended_parameters': param_recommendations,
            'expected_performance': performance_summary
        }

    return recommendations


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def remove_directory(path: str):
    """
    Safely remove directory and all contents.

    Args:
        path: Directory path to remove

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info(f"Successfully removed {path}")
            return True
        else:
            logger.info(f"Directory {path} does not exist")
            return True
    except Exception as e:
        logger.error(f"Error removing directory {path}: {e}")
        return False


def label_bird(save_path: str, bird: str, metrics: list, replace_labels: bool = False,
               hdbscan_params: list = None, top_n_for_pdf: int = 20,
               generate_cluster_pdf: bool = False, metric_weights: dict = None,
               max_workers: int = None, run_name: str = "default"):
    """Run the complete Stage D labelling pipeline for one bird.

    Iterates over all UMAP embedding files produced by Stage C, searches a
    grid of HDBSCAN hyperparameters for each embedding, evaluates the
    resulting clusterings with *metrics*, and writes a master CSV summary
    together with a PDF report of the top-*top_n_for_pdf* parameter
    combinations.

    Parameters
    ----------
    save_path : str
        Project root directory containing the bird subdirectory.
    bird : str
        Bird identifier (e.g. ``'or18or24'``).
    metrics : list of str
        Evaluation metrics to compute for each clustering.  Supported values:
        ``'silhouette'``, ``'dbi'``, ``'ch'``, ``'dunn'``, ``'nmi'``,
        ``'aic'``, ``'bic'``, ``'log-likelihood'``.
    replace_labels : bool, optional
        When ``True``, delete any existing ``labelling/`` directory and
        re-cluster from scratch.  Default is ``False``.
    hdbscan_params : list of dict or None, optional
        Custom HDBSCAN parameter grid.  Each element is a dict accepted by
        :func:`cluster_embeddings`.  Uses :data:`DEFAULT_HDBSCAN_GRID` when
        ``None``.
    top_n_for_pdf : int, optional
        Number of highest-scoring parameter combinations to include in the
        PDF report.  Default is ``20``.
    generate_cluster_pdf : bool, optional
        When ``True``, write a PDF summary of the top clusterings to
        ``results/plots/``.  Default is ``False`` (opt-in, avoids
        matplotlib PDF overhead on every run).
    metric_weights : dict or None, optional
        Per-metric weights for the composite score, e.g.
        ``{'silhouette': 2.0, 'dbi': 1.0}``.  ``None`` uses equal weights.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if a fatal error occurred.

    See Also
    --------
    song_phenotyping.embedding.explore_embedding_parameters_robust : Stage C (produces input).
    song_phenotyping.phenotyping.phenotype_bird : Stage E (consumes output).
    """
    try:
        logger.info(f"Starting labeling pipeline for bird {bird}")

        # Setup paths
        from song_phenotyping.tools.pipeline_paths import (
            EMBEDDINGS_DIR, LABELS_DIR, RESULTS_DIR, PLOTS_DIR, run_stage_path, run_root
        )
        bird_path = os.path.join(save_path, bird)
        run_path       = str(run_root(bird_path, run_name))
        labelling_path = str(run_stage_path(bird_path, run_name, LABELS_DIR))
        embedding_path = str(run_stage_path(bird_path, run_name, EMBEDDINGS_DIR))
        figure_path    = os.path.join(run_path, PLOTS_DIR, 'clusters')

        # Create directories
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(labelling_path, exist_ok=True)

        # Handle replacement of existing labels
        if replace_labels and os.path.exists(labelling_path):
            if not remove_directory(labelling_path):
                logger.error(f"Failed to remove existing labelling directory for {bird}")
                return False
            os.makedirs(labelling_path, exist_ok=True)

            # Remove existing master summary
            master_summary_path = os.path.join(run_path, RESULTS_DIR, 'master_summary.csv')
            if os.path.exists(master_summary_path):
                os.remove(master_summary_path)

        # Check for embeddings
        if not os.path.exists(embedding_path):
            logger.error(f"No embeddings directory found for bird {bird}")
            return False

        embeddings_files = [f for f in os.listdir(embedding_path) if f.endswith('.h5')]
        if not embeddings_files:
            logger.error(f"No embedding files found for bird {bird}")
            return False

        # Default HDBSCAN parameters if not provided
        if hdbscan_params is None:
            hdbscan_params = [p.to_dict() for p in DEFAULT_HDBSCAN_GRID]

        # Process each embedding file
        all_summaries = []

        for embedding_file in tqdm(embeddings_files, desc=f'Processing embeddings for {bird}'):
            try:
                # Parse UMAP parameters from filename - UPDATED for new format
                parsed_params = parse_embedding_filename(embedding_file)
                if parsed_params is None:
                    logger.warning(f"Could not parse filename {embedding_file}, skipping")
                    continue

                metric, n_neighbors, min_dist, umap_id = parsed_params

                # Load embeddings
                embeddings, hashes, labels = load_umap_embeddings(
                    os.path.join(embedding_path, embedding_file)
                )

                if embeddings is None:
                    logger.warning(f"Could not load embeddings from {embedding_file}, skipping")
                    continue

                # Calculate sample size
                sample_size = len(embeddings)

                # Setup directory for this embedding
                embedding_labelling_path = os.path.join(labelling_path, umap_id)

                # Search clustering parameters
                summary_df = search_cluster_params(
                    embeddings=embeddings,
                    hashes=hashes,
                    algorithm='hdbscan',
                    umap_id=umap_id,
                    directory_path=embedding_labelling_path,
                    figure_path=figure_path,
                    candidate_params=hdbscan_params,
                    true_labels=labels,
                    metrics=metrics,
                    sample_size=sample_size,
                    use_parallel=True,
                    max_workers=max_workers,
                )

                # Add UMAP parameters to summary
                summary_df['n_neighbors'] = n_neighbors
                summary_df['min_dist'] = min_dist
                summary_df['metric'] = metric

                all_summaries.append(summary_df)

                # Clear embeddings from memory
                del embeddings

            except Exception as e:
                logger.error(f"Error processing {embedding_file} for bird {bird}: {e}")
                continue

        if not all_summaries:
            logger.error(f"No valid embeddings processed for bird {bird}")
            return False

        # Combine all summaries
        master_summary_df = pd.concat(all_summaries, ignore_index=True)

        # Compute composite scores across all parameter combinations
        master_summary_df = compute_composite_score(
            master_summary_df,
            metrics=metrics,
            n_syls=master_summary_df['n_syls'].tolist(),
            weights=metric_weights,
            use_cluster_penalty=False,
        )

        # Reorder columns and sort by performance
        master_summary_df = reorder_columns(master_summary_df, metrics)

        # Sort by primary metric (NMI if available, otherwise composite score)
        if 'nmi' in metrics and 'nmi' in master_summary_df.columns:
            master_summary_df = master_summary_df.sort_values('nmi', ascending=False)
        else:
            master_summary_df = master_summary_df.sort_values('composite_score', ascending=False)

        master_summary_df = master_summary_df.reset_index(drop=True)

        # Save master summary
        if not save_master_summary(master_summary_df, run_path):
            logger.error(f"Failed to save master summary for bird {bird}")
            return False

        # Create PDF report (opt-in: set generate_cluster_pdf=True to enable)
        if generate_cluster_pdf:
            pdf_success = create_cluster_summary_pdf(
                master_summary_df,
                bird=bird,
                save_path=os.path.join(run_path, PLOTS_DIR),
                top_n=top_n_for_pdf
            )
            if not pdf_success:
                logger.warning(f"PDF creation failed for bird {bird}, but pipeline completed")

        logger.info(f"Successfully completed labeling pipeline for bird {bird}")
        return True

    except Exception as e:
        logger.error(f"Error in labeling pipeline for bird {bird}: {e}")
    return False

def main(save_path: str) -> None:
    """Main function to run the clustering pipeline."""
    try:
        # Define evaluation metrics
        metrics = ['silhouette', 'dbi']  # remember not to use nmi for wseg files!

        # Get available birds
        birds = _get_available_birds(save_path)
        if not birds:
            logger.error("No birds found for processing")
            return

        logger.info(f"Found {len(birds)} birds for processing: {birds}")

        # Process each bird
        successful_birds = []
        failed_birds = []

        for bird in tqdm(birds, desc='Processing birds'):
            success = label_bird(
                save_path=save_path,
                bird=bird,
                metrics=metrics,
                replace_labels=True
            )

            if success:
                successful_birds.append(bird)
            else:
                failed_birds.append(bird)

        # Report results
        logger.info(f"Processing complete. Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
        if failed_birds:
            logger.warning(f"Failed birds: {failed_birds}")

        # Perform cross-bird analysis if we have successful results
        if len(successful_birds) >= 2:
            logger.info("Starting cross-bird analysis...")

            try:
                # Aggregate raw scores across birds
                aggregated_df = aggregate_raw_scores_across_birds(
                    save_path=save_path,
                    birds=successful_birds,
                    top_n=10
                )

                if not aggregated_df.empty:
                    # Analyze performance by sample size
                    analysis_by_size = analyze_parameter_performance_by_sample_size(
                        aggregated_df=aggregated_df,
                        metrics=metrics
                    )

                    # Compute cross-bird composite scores
                    aggregated_df = compute_cross_bird_composite_scores(
                        aggregated_df=aggregated_df,
                        metrics=metrics
                    )

                    # Save cross-bird analysis results
                    saved_files = save_cross_bird_analysis(
                        aggregated_df=aggregated_df,
                        analysis_by_size=analysis_by_size,
                        save_path=save_path,
                        filename_prefix='cross_bird'
                    )

                    # Generate parameter recommendations
                    if not analysis_by_size.empty:
                        recommendations = identify_optimal_parameters_by_sample_size(analysis_by_size)

                        # Log key recommendations
                        logger.info("Parameter recommendations by sample size:")
                        for bin_name, rec in recommendations.items():
                            logger.info(f"  {bin_name}: {rec['recommended_parameters']}")

                    logger.info(f"Cross-bird analysis complete. Files saved: {list(saved_files.keys())}")
                else:
                    logger.warning("No data available for cross-bird analysis")

            except Exception as e:
                logger.error(f"Error in cross-bird analysis: {e}")

        else:
            logger.info("Insufficient successful birds for cross-bird analysis (need at least 2)")

        logger.info("Pipeline execution complete!")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

def clear_clustering_outputs(save_path: str, bird: str = None, confirm: bool = True):
    """
    Clear clustering label files and images to avoid clutter.
    Updated to work with new directory structure (syllable_data instead of data).

    Args:
        save_path: Root directory containing bird data
        bird: Specific bird to clear (all birds if None)
        confirm: Whether to ask for confirmation before deleting

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get birds to process
        if bird is not None:
            birds_to_clear = [bird]
        else:
            birds_to_clear = _get_available_birds(save_path)

        if not birds_to_clear:
            logger.info("No birds found to clear")
            return True

        # Calculate what will be removed
        total_items = 0
        paths_to_remove = []

        for bird_name in birds_to_clear:
            bird_path = os.path.join(save_path, bird_name)

            from song_phenotyping.tools.pipeline_paths import LABELS_DIR, RESULTS_DIR, PLOTS_DIR
            # Labelling directory (contains all cluster labels)
            labelling_path = os.path.join(bird_path, LABELS_DIR)
            if os.path.exists(labelling_path):
                paths_to_remove.append(('labelling', labelling_path))
                for root, dirs, files in os.walk(labelling_path):
                    total_items += len(files)
            # Cluster figures directory (check both new and legacy locations)
            for cluster_figures_path in [
                os.path.join(bird_path, PLOTS_DIR, 'clusters'),
                os.path.join(bird_path, 'figures', 'clusters'),
            ]:
                if os.path.exists(cluster_figures_path):
                    paths_to_remove.append(('cluster_figures', cluster_figures_path))
                    for root, dirs, files in os.walk(cluster_figures_path):
                        total_items += len(files)

            # Master summary CSV
            master_summary_path = os.path.join(bird_path, RESULTS_DIR, 'master_summary.csv')
            if os.path.exists(master_summary_path):
                paths_to_remove.append(('master_summary', master_summary_path))
                total_items += 1

            # PDF report (check both new and legacy locations)
            for pdf_path in [
                os.path.join(bird_path, PLOTS_DIR, f'{bird_name}_cluster_summary.pdf'),
                os.path.join(bird_path, f'{bird_name}_cluster_summary.pdf'),
            ]:
                if os.path.exists(pdf_path):
                    paths_to_remove.append(('pdf_report', pdf_path))
                    total_items += 1

        if not paths_to_remove:
            logger.info("No clustering outputs found to clear")
            return True

        # Show what will be removed
        logger.info(f"Found {total_items} items to remove across {len(birds_to_clear)} bird(s):")
        for bird_name in birds_to_clear:
            bird_items = [path for path_type, path in paths_to_remove if bird_name in path]
            if bird_items:
                logger.info(f"  {bird_name}: {len(bird_items)} directories/files")

        # Confirmation
        if confirm:
            response = input(f"\nAre you sure you want to delete {total_items} clustering output items? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Deletion cancelled")
                return False

        # Remove items
        removed_count = 0
        failed_count = 0

        for path_type, path in paths_to_remove:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.debug(f"Removed directory: {path}")
                else:
                    os.remove(path)
                    logger.debug(f"Removed file: {path}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {path}: {e}")
                failed_count += 1

        # Report results
        if failed_count == 0:
            logger.info(f"✅ Successfully removed all {removed_count} items")
        else:
            logger.warning(f"⚠️ Removed {removed_count} items, failed to remove {failed_count} items")

        return failed_count == 0

    except Exception as e:
        logger.error(f"Error clearing clustering outputs: {e}")
        return False


if __name__ == '__main__':
    # Setup paths and parameters
    # Updated to use new directory structure
    save_path = Path('E:/') / 'xfosters'

    # Optional: Clear existing clustering outputs
    clear_clustering_outputs(save_path=save_path)

    # Run main pipeline
    main(save_path=save_path)
