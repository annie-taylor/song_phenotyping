import os
from datetime import datetime
import logging
import tables
import numpy as np
from typing import Tuple, Dict, List, Optional
import psutil
import traceback
import umap
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import pickle as pkl
import matplotlib.pyplot as plt


from tools.system_utils import optimize_pytables_for_network

@dataclass
class UMAPParams:
    n_neighbors: int = 20
    min_dist: float = 0.5
    metric: str = 'euclidean'
    n_components: int = 2
    n_epochs: int = 10  # not optimized

    def __post_init__(self):
        self.validate_params()

    def validate_params(self):
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if self.min_dist < 0:
            raise ValueError("min_dist must be non-negative")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be positive")

    def to_dict(self) -> Dict:
        return {
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'n_components': self.n_components,
            'n_epochs': self.n_epochs,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        instance = cls(
            n_neighbors=data['n_neighbors'],
            min_dist=data['min_dist'],
            metric=data['metric'],
            n_components=data['n_components'],
            n_epochs=data['n_epochs']
        )
        return instance


def complex_spectrogram_distance(spec1, spec2):
    # spec1, spec2 are complex spectrograms (magnitude + phase)
    return np.linalg.norm(spec1 - spec2)


def phase_aware_spectrogram_distance(spec1, spec2):
    # Magnitude component (cosine-like)
    mag1, mag2 = np.abs(spec1), np.abs(spec2)
    mag_sim = np.dot(mag1.flatten(), mag2.flatten()) / (
            np.linalg.norm(mag1) * np.linalg.norm(mag2))

    # Phase coherence component
    phase_diff = np.angle(spec1 * np.conj(spec2))
    phase_coherence = np.mean(np.cos(phase_diff))

    # Combine (higher weight on magnitude typically)
    return 1 - (0.8 * mag_sim + 0.2 * phase_coherence)

#
# def polar_distance(spec1, spec2):
#     mag1, phase1 = np.abs(spec1), np.angle(spec1)
#     mag2, phase2 = np.abs(spec2), np.angle(spec2)
#
#     # Weighted combination
#     mag_dist = np.linalg.norm(mag1 - mag2)
#     phase_dist = np.mean(np.abs(np.angle(np.exp(1j * (phase1 - phase2)))))
#
#     return alpha * mag_dist + beta * phase_dist

def group_delay_distance(spec1, spec2):
    # Group delay = -d(phase)/d(frequency)
    gd1 = -np.diff(np.unwrap(np.angle(spec1)), axis=0)
    gd2 = -np.diff(np.unwrap(np.angle(spec2)), axis=0)

    return np.linalg.norm(gd1 - gd2)

def instantaneous_freq_distance(spec1, spec2):
    # Compute instantaneous frequency from phase derivatives
    if1 = np.diff(np.unwrap(np.angle(spec1)), axis=1)
    if2 = np.diff(np.unwrap(np.angle(spec2)), axis=1)

    return np.linalg.norm(if1 - if2)


def generate_embedding_paths(paths: dict, params: UMAPParams, n_samples: int,
                             was_subsampled: bool, subsample_seed: int = None) -> Tuple[str, str]:
    """Generate paths that include sample size information"""

    # Base filename
    base_name = f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist'

    # Add sample info
    if was_subsampled:
        base_name += f'_subsample{n_samples}_seed{subsample_seed}'
    else:
        base_name += f'_full{n_samples}'

    model_path = os.path.join(paths['model'], f'{base_name}.pkl')
    embedding_path = os.path.join(paths['embeddings'], f'{base_name}.h5')

    return model_path, embedding_path


def check_embedding_compatibility(embedding_path: str, current_n_samples: int,
                                  current_hashes: list, overwrite: bool = False) -> bool:
    """
    Check if existing embedding is compatible with current data.

    Returns:
        True if compatible or should proceed, False if should skip
    """
    if not os.path.exists(embedding_path):
        return True  # No existing file, proceed

    if overwrite:
        logging.info(f"🔄 Overwrite=True, will replace: {os.path.basename(embedding_path)}")
        return True

    try:
        with tables.open_file(embedding_path, mode='r') as f:
            existing_hashes = [h.decode('utf-8') for h in f.root.hashes[:]]

            # Check if processing metadata exists
            if hasattr(f.root, 'processing_metadata'):
                metadata_str = f.root.processing_metadata[0].decode('utf-8')
                metadata = eval(metadata_str)  # Convert string back to dict
                logging.info(f"📋 Existing embedding metadata: {metadata}")

            # Compare sample sizes
            if len(existing_hashes) != current_n_samples:
                logging.warning(
                    f"⚠️ Sample size mismatch: existing={len(existing_hashes)}, current={current_n_samples}")
                logging.warning(f"   Use overwrite=True to replace with current data")
                return False

            # Check hash overlap (sample)
            overlap = len(set(existing_hashes[:100]) & set(current_hashes[:100]))
            if overlap < 50:  # Less than 50% overlap in first 100 samples
                logging.warning(f"⚠️ Low data overlap detected: {overlap}/100 hashes match")
                logging.warning(f"   This suggests different underlying data")
                return False

        logging.info(f"✅ Compatible embedding found: {os.path.basename(embedding_path)}")
        return False  # Compatible but exists, so skip

    except Exception as e:
        logging.warning(f"⚠️ Could not verify compatibility for {embedding_path}: {e}")
        return overwrite  # If can't check, depend on overwrite flag


def inspect_existing_embeddings(bird_path: str) -> None:
    """Utility to inspect what embeddings already exist and their metadata"""

    embeddings_path = os.path.join(bird_path, 'syllable_data', 'embeddings')

    if not os.path.exists(embeddings_path):
        logging.info(f"📁 No embeddings directory found at {embeddings_path}")
        return

    embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('.h5')]

    if not embedding_files:
        logging.info(f"📁 No embedding files found in {embeddings_path}")
        return

    logging.info(f"🔍 Found {len(embedding_files)} existing embeddings:")

    for filename in embedding_files:
        filepath = os.path.join(embeddings_path, filename)
        try:
            with tables.open_file(filepath, mode='r') as f:
                n_samples = len(f.root.hashes[:])

                metadata_info = "No metadata"
                if hasattr(f.root, 'processing_metadata'):
                    metadata_str = f.root.processing_metadata[0].decode('utf-8')
                    metadata = eval(metadata_str)
                    metadata_info = f"Original: {metadata.get('original_samples', 'unknown')}, " \
                                    f"Final: {metadata.get('final_samples', 'unknown')}, " \
                                    f"Subsampled: {metadata.get('was_subsampled', 'unknown')}"

                logging.info(f"  📄 {filename}: {n_samples} samples, {metadata_info}")

        except Exception as e:
            logging.warning(f"  ❌ Could not read {filename}: {e}")

def load_embedding_from_file(embedding_path: str) -> Optional[np.ndarray]:
    """Load embeddings from HDF5 file"""
    try:
        with tables.open_file(embedding_path, mode='r') as f:
            return f.root.embeddings[:]
    except Exception as e:
        logging.warning(f"Could not load embedding from {embedding_path}: {e}")
        return None

def load_flattened_specs(paths_to_specs: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all flattened spectrogram files from a directory.

    Args:
        paths_to_specs: Path to directory containing flattened HDF5 files

    Returns:
        Tuple of (flattened_specs, labels, position_idxs, hashes)
    """
    flattened_syl_filenames = [f for f in os.listdir(paths_to_specs)
                               if f.endswith('.h5') and 'flattened' in f]

    if not flattened_syl_filenames:
        raise ValueError(f"No flattened spectrogram files found in {paths_to_specs}")

    # Determine total number of syllables to preallocate arrays
    total_syllables = 0
    for filename in flattened_syl_filenames:
        file_path = os.path.join(paths_to_specs, filename)
        try:
            with tables.open_file(file_path, mode='r') as f:
                total_syllables += f.root.flattened_specs.shape[1]
        except Exception as e:
            logging.warning(f"Could not read {file_path} for size calculation: {e}")
            continue

    if total_syllables == 0:
        raise ValueError("No valid syllables found in any flattened files")

    # Get dimensions from first valid file
    first_file_path = os.path.join(paths_to_specs, flattened_syl_filenames[0])
    with tables.open_file(first_file_path, mode='r') as f:
        spec_dim0 = f.root.flattened_specs.shape[0]

    # Preallocate arrays
    flattened_specs = np.empty((spec_dim0, total_syllables), dtype=np.float32)
    labels = np.empty(total_syllables, dtype=object)
    position_idxs = np.empty(total_syllables, dtype=np.int32)
    hashes = np.empty(total_syllables, dtype=object)

    # Load data from all files
    current_idx = 0
    for filename in flattened_syl_filenames:
        file_path = os.path.join(paths_to_specs, filename)
        try:
            with tables.open_file(file_path, mode='r') as f:
                n_syllables = f.root.flattened_specs.shape[1]
                end_idx = current_idx + n_syllables

                flattened_specs[:, current_idx:end_idx] = f.root.flattened_specs[:]
                labels[current_idx:end_idx] = [label.decode('utf-8') for label in f.root.labels[:]]
                position_idxs[current_idx:end_idx] = f.root.position_idxs[:]
                hashes[current_idx:end_idx] = [hash_id.decode('utf-8') for hash_id in f.root.hashes[:]]
                current_idx = end_idx

        except Exception as e:
            logging.warning(f"Failed to load flattened spec data from {filename}: {e}")
            continue

    # Trim arrays if some files failed to load
    if current_idx < total_syllables:
        flattened_specs = flattened_specs[:, :current_idx]
        labels = labels[:current_idx]
        position_idxs = position_idxs[:current_idx]
        hashes = hashes[:current_idx]

    return flattened_specs, labels, position_idxs, hashes


def save_umap_embeddings(embedding_path: str, embeddings: np.ndarray, hashes: list,
                         labels: Optional[list] = None, metadata: Optional[dict] = None) -> bool:
    """"""
    try:
        with tables.open_file(embedding_path, mode='w') as f:
            # Save embeddings and existing data
            f.create_array(f.root, 'embeddings', obj=np.array(embeddings))

            hash_atom = tables.StringAtom(itemsize=max(len(h) for h in hashes))
            hash_array = f.create_earray(f.root, 'hashes', atom=hash_atom, shape=(0,))
            hash_array.append(np.array([str(h) for h in hashes], dtype='S'))

            if labels is not None:
                label_atom = tables.StringAtom(itemsize=1)
                label_array = f.create_earray(f.root, 'labels', atom=label_atom, shape=(0,))
                label_array.append(np.array([str(l) for l in labels], dtype='S'))

            if metadata is not None:
                f.create_array(f.root, 'processing_metadata', obj=np.array([str(metadata)], dtype='S'))

        return True
    except Exception as e:
        logging.error(f"Error saving embeddings to {embedding_path}: {e}")
        return False


def save_umap_model(model_path: str, umap_model: umap.UMAP, params: UMAPParams) -> bool:
    try:
        data = {'model': umap_model, 'params': params}
        with open(model_path, 'wb') as f:
            pkl.dump(data, f)
        return True
    except Exception as e:
        logging.error(f"Error saving UMAP model to {model_path}: {e}")
        logging.error(traceback.format_exc())
        return False

def subsample_data(specs: np.ndarray, labels: np.ndarray, position_idxs: np.ndarray,
                   hashes: np.ndarray, max_samples: int, random_seed: int = 42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly subsample the data if it exceeds max_samples.

    Args:
        specs, labels, position_idxs, hashes: Original data arrays
        max_samples: Maximum number of samples to keep
        random_seed: Random seed for reproducibility

    Returns:
        Subsampled versions of input arrays
    """
    if len(labels) <= max_samples:
        return specs, labels, position_idxs, hashes

    np.random.seed(random_seed)
    indices = np.random.choice(len(labels), size=max_samples, replace=False)
    indices.sort()  # Keep chronological order

    logging.info(f"🎯 Subsampling from {len(labels)} to {max_samples} samples")

    return (specs[:, indices], labels[indices],
            position_idxs[indices], hashes[indices])


def estimate_umap_memory_usage(n_samples: int, n_features: int, n_neighbors: int) -> float:
    """
    More accurate UMAP memory estimation based on algorithm internals.

    Returns estimated peak memory usage in GB.
    """
    # Base data storage (float32)
    input_data_gb = (n_samples * n_features * 4) / (1024 ** 3)

    # k-NN graph storage (typically sparse, but can be dense for large k)
    # UMAP builds a k-neighbor graph: n_samples * n_neighbors * (index + distance)
    knn_graph_gb = (n_samples * n_neighbors * 8) / (1024 ** 3)  # 4 bytes each for index + distance

    # Distance matrix (worst case for small datasets or large n_neighbors)
    # For large n_neighbors, UMAP may compute full pairwise distances
    if n_neighbors > n_samples * 0.1:  # Heuristic: if k > 10% of samples
        distance_matrix_gb = (n_samples * n_samples * 4) / (1024 ** 3)
    else:
        distance_matrix_gb = knn_graph_gb * 2  # Approximate

    # Embedding space (small)
    embedding_gb = (n_samples * 2 * 4) / (1024 ** 3)  # 2D embedding, float32

    # Optimization overhead (gradients, temporary arrays)
    overhead_gb = max(input_data_gb * 0.5, 0.5)  # At least 500MB overhead

    # Peak usage (not all components exist simultaneously, but be conservative)
    peak_memory_gb = input_data_gb + max(distance_matrix_gb, knn_graph_gb * 3) + embedding_gb + overhead_gb

    return peak_memory_gb


def calculate_safe_batch_size(available_memory_gb: float, n_features: int,
                              max_n_neighbors: int, safety_factor: float = 0.7) -> int:
    """Calculate maximum safe batch size for UMAP given memory constraints."""

    # Binary search for largest feasible batch size
    min_size, max_size = 1000, 100000
    safe_size = min_size

    while min_size <= max_size:
        mid_size = (min_size + max_size) // 2
        estimated_memory = estimate_umap_memory_usage(mid_size, n_features, max_n_neighbors)

        if estimated_memory <= available_memory_gb * safety_factor:
            safe_size = mid_size
            min_size = mid_size + 1
        else:
            max_size = mid_size - 1

    return safe_size


def calculate_adaptive_workers_improved(n_samples: int, n_features: int,
                                        max_n_neighbors: int,
                                        memory_per_worker_gb: float = None,
                                        max_workers: Optional[int] = None) -> int:
    """
    Improved worker calculation with better memory estimation.
    """
    if memory_per_worker_gb is None:
        # Calculate optimal memory per worker inline
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024 ** 3)

        if max_workers is None:
            max_workers = mp.cpu_count()

        usable_memory_gb = available_gb * 0.7  # 70% safety factor
        memory_per_worker_gb = max(0.5, min(8.0, usable_memory_gb / max(2, max_workers)))

        logging.info(f"[MEMORY] Auto-calculated memory per worker: {memory_per_worker_gb:.1f}GB")

    # Estimate memory for worst-case UMAP (highest n_neighbors)
    estimated_memory_per_job = estimate_umap_memory_usage(n_samples, n_features, max_n_neighbors)

    # Available memory for all workers
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    usable_memory_gb = available_memory_gb * 0.6  # Conservative safety factor

    # Calculate workers based on memory constraint
    memory_limited_workers = max(1, int(usable_memory_gb / estimated_memory_per_job))

    # CPU-based limit
    cpu_workers = max_workers if max_workers else mp.cpu_count()

    # For very large datasets, be more conservative
    if n_samples > 20000:
        conservative_factor = 0.5
        memory_limited_workers = max(1, int(memory_limited_workers * conservative_factor))

    recommended_workers = min(memory_limited_workers, cpu_workers)

    logging.info(f"[ANALYSIS] Estimated UMAP memory per job: {estimated_memory_per_job:.2f}GB")
    logging.info(f"[WORKERS] Memory-limited workers: {memory_limited_workers}")
    logging.info(f"[FINAL] Final worker count: {recommended_workers}")

    return recommended_workers

def monitor_memory_usage():
    """Return current memory usage info."""
    memory = psutil.virtual_memory()
    return {
        'available_gb': memory.available / (1024**3),
        'percent_used': memory.percent,
        'free_gb': memory.free / (1024**3)
    }

def compute_single_umap_worker_safe(args):
    """Enhanced UMAP worker with memory monitoring and fallback."""
    samples, labels, hashes, n_neighbors, min_dist, paths, overwrite, processing_metadata = args

    try:
        # Check memory before starting
        memory_info = monitor_memory_usage()
        if memory_info['percent_used'] > 85:
            logging.warning(f"High memory usage ({memory_info['percent_used']:.1f}%) before UMAP computation")
            # Could implement waiting or skipping logic here

        params = UMAPParams(n_neighbors=n_neighbors, metric='euclidean', min_dist=min_dist)

        # Generate paths with sample info
        model_path, embedding_path = generate_embedding_paths(
            paths, params, len(labels),
            processing_metadata['was_subsampled'],
            processing_metadata.get('subsample_seed')
        )

        # Enhanced UMAP computation with memory-aware settings
        embeddings, model, success = compute_and_save_umap_memory_aware(
            samples=samples,
            labels=labels,
            hashes=hashes,
            params=params,
            model_path=model_path,
            embedding_path=embedding_path,
            save_model=False,
            overwrite=overwrite,
            processing_metadata=processing_metadata
        )

        return (n_neighbors, min_dist, success)

    except MemoryError as e:
        logging.error(f"Memory error in UMAP n={n_neighbors}, dist={min_dist}: {e}")
        return (n_neighbors, min_dist, False)
    except Exception as e:
        logging.error(f"Failed UMAP n={n_neighbors}, dist={min_dist}: {e}")
        return (n_neighbors, min_dist, False)

def compute_and_save_umap_memory_aware(samples: np.ndarray, labels, hashes, params: UMAPParams,
                                       model_path: str, embedding_path: str, save_model: bool = False,
                                       overwrite: bool = False, processing_metadata: dict = None) -> Tuple[
    Optional[np.ndarray], Optional[umap.UMAP], bool]:
    """
    Memory-aware UMAP computation with fallback strategies.
    """
    # Check compatibility first
    if not check_embedding_compatibility(embedding_path, len(labels), hashes, overwrite):
        return None, None, True  # Skip but report success

    try:
        # Pre-flight memory check
        memory_info = monitor_memory_usage()
        estimated_memory = estimate_umap_memory_usage(
            len(labels), samples.shape[1], params.n_neighbors
        )

        if estimated_memory > memory_info['available_gb'] * 0.8:
            logging.warning(
                f"⚠️ Estimated memory ({estimated_memory:.2f}GB) exceeds available ({memory_info['available_gb']:.2f}GB)")
            logging.warning(f"   Attempting with reduced precision and optimized settings")

            # Fallback strategy 1: Use float32 and optimize UMAP settings
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            # More memory-efficient UMAP settings
            umap_model = umap.UMAP(
                n_components=params.n_components,
                metric=params.metric,
                min_dist=params.min_dist,
                n_neighbors=min(params.n_neighbors, len(labels) // 10),  # Cap n_neighbors
                n_epochs=min(params.n_epochs, 200),  # Cap epochs to save memory
                low_memory=True,  # Enable UMAP's low memory mode
                verbose=False,
                random_state=42
            )
        else:
            # Standard UMAP settings
            umap_model = umap.UMAP(
                n_components=params.n_components,
                metric=params.metric,
                min_dist=params.min_dist,
                n_neighbors=params.n_neighbors,
                n_epochs=params.n_epochs,
                verbose=False,
                random_state=42
            )

        # Monitor memory during fit_transform
        logging.info(f"🧠 Pre-UMAP memory: {monitor_memory_usage()['percent_used']:.1f}% used")

        # Fit and transform with error handling
        embeddings = umap_model.fit_transform(samples)

        logging.info(f"🧠 Post-UMAP memory: {monitor_memory_usage()['percent_used']:.1f}% used")

        # Create output directories
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

        # Save results with metadata
        if save_model:
            save_umap_model(model_path, umap_model, params)
        save_umap_embeddings(embedding_path, embeddings, hashes, labels, processing_metadata)

        logging.info(f"✅ Computed new embedding: {os.path.basename(embedding_path)}")
        return embeddings, umap_model, True

    except MemoryError as e:
        logging.error(f"💥 Memory error during UMAP computation: {e}")
        logging.error(f"   Consider reducing max_samples or using fewer workers")
        return None, None, False
    except Exception as e:
        logging.error(f"❌ Error creating UMAP embeddings: {e}")
        return None, None, False

def explore_embedding_parameters_robust(save_path: str, bird: str,
                                        min_dists: List[float] = None,
                                        n_neighbors_list: List[int] = None,
                                        use_parallel: bool = True,
                                        overwrite: bool = False,
                                        max_samples: Optional[int] = None,
                                        memory_per_worker_gb: Optional[float] = None,
                                        auto_memory_management: bool = True,
                                        subsample_seed: int = 42) -> bool:
    """
    Robust version with dynamic memory management and fallback strategies.
    """
    try:
        # Default parameter ranges
        if min_dists is None:
            min_dists = [0.01, 0.05, 0.1, 0.3, 0.5]
        if n_neighbors_list is None:
            n_neighbors_list = [5, 10, 20, 50, 100]

        # Setup paths
        bird_path = os.path.join(save_path, bird)
        data_path = os.path.join(bird_path, 'syllable_data')

        paths = {
            'specs': os.path.join(data_path, 'flattened'),
            'model': os.path.join(data_path, 'models'),
            'embeddings': os.path.join(data_path, 'embeddings'),
            'figures': os.path.join(bird_path, 'figures')
        }

        # Load data
        specs, labels, position_idxs, hashes = load_flattened_specs(paths_to_specs=paths['specs'])
        logging.info(f"🐦 Loaded {len(labels)} samples for bird {bird}")

        # Dynamic memory management
        if auto_memory_management:
            # Calculate safe batch size based on largest n_neighbors
            max_n_neighbors = max(n_neighbors_list)
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

            safe_batch_size = calculate_safe_batch_size(
                available_memory_gb, specs.shape[0], max_n_neighbors
            )

            if max_samples is None:
                max_samples = safe_batch_size
            else:
                max_samples = min(max_samples, safe_batch_size)

            logging.info(f"🎯 Safe batch size: {safe_batch_size}, using max_samples: {max_samples}")

        # Apply subsampling if needed
        was_subsampled = False
        original_n_samples = len(labels)

        if max_samples is not None and len(labels) > max_samples:
            specs, labels, position_idxs, hashes = subsample_data(
                specs, labels, position_idxs, hashes, max_samples, subsample_seed
            )
            was_subsampled = True
            logging.info(f"📉 Subsampled from {original_n_samples} to {len(labels)} samples")

        # Create processing metadata
        processing_metadata = {
            'original_samples': original_n_samples,
            'final_samples': len(labels),
            'was_subsampled': was_subsampled,
            'subsample_seed': subsample_seed if was_subsampled else None,
            'max_samples_limit': max_samples,
            'memory_per_worker_gb': memory_per_worker_gb,
            'processing_date': datetime.now().isoformat(),
            'bird_id': bird
        }

        # Calculate adaptive worker count with improved estimation
        if use_parallel:
            adaptive_workers = calculate_adaptive_workers_improved(
                n_samples=len(labels),
                n_features=specs.shape[0],
                max_n_neighbors=max(n_neighbors_list),
                memory_per_worker_gb=memory_per_worker_gb
            )

            # Additional safety check: if we have very few workers, consider sequential processing
            if adaptive_workers < 2:
                logging.warning(
                    f"[WARNING] Only {adaptive_workers} workers recommended, switching to sequential processing")
                use_parallel = False
                adaptive_workers = 1
        else:
            adaptive_workers = 1

        # Always use the robust parallel function, but with 1 worker for sequential
        successful_params = compute_embedding_grid_parallel_robust(
            samples=specs.T,
            labels=labels,
            hashes=hashes,
            min_dists=min_dists,
            n_neighbors=n_neighbors_list,
            paths=paths,
            plot=True,
            bird=bird,
            max_workers=adaptive_workers,  # This will be 1 for sequential
            overwrite=overwrite,
            processing_metadata=processing_metadata
        )

        logging.info(f"Successfully explored UMAP parameters for bird {bird}. "
                     f"Computed {len(successful_params)} parameter combinations.")
        return True

    except Exception as e:
        logging.error(f"Failed to explore UMAP parameters for bird {bird}: {e}")
        logging.error(traceback.format_exc())
        return False


def compute_embedding_grid_parallel_robust(samples, labels, hashes, min_dists, n_neighbors, paths,
                                           plot: bool = True, bird: str = '', max_workers: int = None,
                                           overwrite: bool = False, processing_metadata: dict = None):
    """
    Robust parallel version with memory monitoring and graceful degradation.
    """
    # Prepare arguments for parallel processing
    args_list = []
    for n in n_neighbors:
        for dist in min_dists:
            args_list.append((samples, labels, hashes, n, dist, paths, overwrite, processing_metadata))

    print(f"    🚀 Computing {len(args_list)} UMAPs in parallel...")

    # Control worker count
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(args_list))
    else:
        max_workers = min(max_workers, len(args_list))

    print(f"    Using {max_workers} parallel workers")

    successful_params = []
    failed_params = []

    # Monitor system memory before starting
    initial_memory = monitor_memory_usage()
    logging.info(f"🧠 Starting parallel processing with {initial_memory['percent_used']:.1f}% memory used")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(compute_single_umap_worker_safe, args): args for args in args_list}

        # Collect results with progress tracking and memory monitoring
        completed = 0
        for future in as_completed(future_to_params):
            n_neighbors_val, min_dist_val, success = future.result()
            completed += 1

            # Monitor memory usage periodically
            if completed % 5 == 0:
                current_memory = monitor_memory_usage()
                logging.info(f"🧠 Progress: {completed}/{len(args_list)}, Memory: {current_memory['percent_used']:.1f}%")

                # If memory usage is getting high, warn but continue
                if current_memory['percent_used'] > 90:
                    logging.warning(f"⚠️ High memory usage detected: {current_memory['percent_used']:.1f}%")

            if success:
                successful_params.append((n_neighbors_val, min_dist_val))
                print(f"    ✅ {completed}/{len(args_list)}: n_neighbors={n_neighbors_val}, min_dist={min_dist_val}")
            else:
                failed_params.append((n_neighbors_val, min_dist_val))
                print(
                    f"    ❌ {completed}/{len(args_list)}: FAILED n_neighbors={n_neighbors_val}, min_dist={min_dist_val}")

    # Final memory check
    final_memory = monitor_memory_usage()
    logging.info(f"🧠 Completed parallel processing. Final memory usage: {final_memory['percent_used']:.1f}%")

    # Report results
    if failed_params:
        logging.warning(f"⚠️ {len(failed_params)} parameter combinations failed:")
        for n, d in failed_params:
            logging.warning(f"   - n_neighbors={n}, min_dist={d}")

    # Create plot if requested
    if plot:
        fig_savepath = paths['figures']
        if not os.path.isdir(fig_savepath):
            os.makedirs(fig_savepath)
        compare_umap_embeddings_plot(successful_params, min_dists, n_neighbors, paths, fig_savepath, bird,
                                     processing_metadata)

    return successful_params
#
# def compute_embedding_grid(samples, labels, hashes, min_dists, n_neighbors, paths,
#                            plot: bool = True, bird: str = '', overwrite: bool = False,
#                            processing_metadata: dict = None):  # ADD this parameter
#     """
#     Non-parallel version of embedding grid computation.
#     Updated to include processing_metadata parameter.
#
#     Returns:
#         List of successful (n_neighbors, min_dist) tuples
#     """
#     print(f"    🔄 Computing {len(n_neighbors) * len(min_dists)} UMAPs sequentially...")
#
#     successful_params = []
#     completed = 0
#     total_tasks = len(n_neighbors) * len(min_dists)
#
#     for n in n_neighbors:
#         for dist in min_dists:
#             try:
#                 params = UMAPParams(n_neighbors=n, metric='euclidean', min_dist=dist)
#
#                 # Generate paths with sample info (if metadata available)
#                 if processing_metadata:
#                     model_path, embedding_path = generate_embedding_paths(
#                         paths, params, len(labels),
#                         processing_metadata['was_subsampled'],
#                         processing_metadata.get('subsample_seed')
#                     )
#                 else:
#                     # Fallback to old naming convention
#                     model_path = os.path.join(paths['model'],
#                                               f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.pkl')
#                     embedding_path = os.path.join(paths['embeddings'],
#                                                   f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.h5')
#
#                 embeddings, model, success = compute_and_save_umap(
#                     samples=samples,
#                     labels=labels,
#                     hashes=hashes,
#                     params=params,
#                     model_path=model_path,
#                     embedding_path=embedding_path,
#                     save_model=False,
#                     overwrite=overwrite,
#                     processing_metadata=processing_metadata  # Pass metadata
#                 )
#
#                 if success:
#                     successful_params.append((n, dist))
#                     completed += 1
#                     if embeddings is not None:
#                         print(f"    ✅ {completed}/{total_tasks}: n_neighbors={n}, min_dist={dist}")
#                     else:
#                         print(f"    ⏭️ {completed}/{total_tasks}: SKIPPED n_neighbors={n}, min_dist={dist}")
#                 else:
#                     completed += 1
#                     print(f"    ❌ {completed}/{total_tasks}: FAILED n_neighbors={n}, min_dist={dist}")
#
#             except Exception as e:
#                 print(f"    ❌ {completed + 1}/{total_tasks}: FAILED n_neighbors={n}, min_dist={dist}: {e}")
#                 completed += 1
#                 continue
#
#     # Create plot if requested
#     if plot:
#         fig_savepath = paths['figures']
#         if not os.path.isdir(fig_savepath):
#             os.makedirs(fig_savepath)
#         compare_umap_embeddings_plot(successful_params, min_dists, n_neighbors, paths, fig_savepath, bird, processing_metadata)  # Pass metadata
#
#     return successful_params

def compare_umap_embeddings_plot(successful_params: List[Tuple[int, float]], min_dists, n_neighbors,
                                 paths: dict, save_path: str, bird: str = '', processing_metadata: dict = None):
    """
    Create comparison plot by loading embeddings on-demand.
    Updated to handle new filename structure with sample info.
    """

    fig, axs = plt.subplots(len(n_neighbors), len(min_dists), figsize=(20, 20))

    # Handle different subplot configurations
    if len(n_neighbors) == 1 and len(min_dists) == 1:
        axs = [[axs]]
    elif len(n_neighbors) == 1:
        axs = [axs]
    elif len(min_dists) == 1:
        axs = [[ax] for ax in axs]

    # Convert to set for faster lookup
    successful_set = set(successful_params)

    for i, n in enumerate(n_neighbors):
        for j, dist in enumerate(min_dists):
            ax = axs[i][j]

            if (n, dist) in successful_set:
                # Generate the embedding path using the new naming convention
                if processing_metadata:
                    n_samples = processing_metadata['final_samples']
                    was_subsampled = processing_metadata['was_subsampled']
                    subsample_seed = processing_metadata.get('subsample_seed')

                    # Create params object for path generation
                    params = UMAPParams(n_neighbors=n, metric='euclidean', min_dist=dist)
                    _, embedding_path = generate_embedding_paths(
                        paths, params, n_samples, was_subsampled, subsample_seed
                    )
                else:
                    # Fallback to old naming convention
                    embedding_path = os.path.join(paths['embeddings'],
                                                  f'euclidean_{n}neighbors_{dist}dist.h5')

                embeddings = load_embedding_from_file(embedding_path)

                if embeddings is not None:
                    ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5, s=1)
                else:
                    ax.text(0.5, 0.5, 'Missing\nFile', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(f"min_dist = {dist}", size=15)
            if j == 0:
                ax.set_ylabel(f"n_neighbors = {n}", size=15)

    fig.suptitle(f"UMAP embedding with grid of parameters - {bird}", y=0.92, size=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if save_path:
        # Create descriptive filename with all parameter values
        n_str = "_".join(map(str, n_neighbors))
        d_str = "_".join(map(str, min_dists))
        filename = f"umap_grid_{bird}_n{n_str}_d{d_str}.png"

        plt.savefig(os.path.join(save_path, filename))
        plt.close(fig)


def main():
    logging.info("Optimizing PyTables for network access")
    optimize_pytables_for_network()

    # EVSong processing
    evsong_test_directory = os.path.join('E:', 'ssharma_RNA_seq')
    logging.info(f"Processing EVSong directory: {evsong_test_directory}")

    if os.path.exists(evsong_test_directory):
        birds = [b for b in os.listdir(evsong_test_directory) if b != 'copied_data' and
                 os.path.isdir(os.path.join(evsong_test_directory, b))]
        logging.info(f"Found {len(birds)} birds in EVSong directory: {birds}")

        for bird in birds:
            logging.info(f"Processing EVSong bird: {bird}")

            # Inspect existing embeddings first
            bird_path = os.path.join(evsong_test_directory, bird)
            inspect_existing_embeddings(bird_path)

            success = explore_embedding_parameters_robust(  # Use robust version
                save_path=evsong_test_directory,
                bird=bird,
                min_dists=[0.01, 0.1, 0.2, 0.5],
                n_neighbors_list=[5, 10, 25, 50, 100],
                use_parallel=True,
                overwrite=False,
                max_samples=None,  # Let auto-management decide
                memory_per_worker_gb=None,  # Auto-detect based on system
                auto_memory_management=True,
                subsample_seed=42
            )

            if success:
                logging.info(f"✅ Successfully processed EVSong bird: {bird}")
            else:
                logging.error(f"❌ Failed to process EVSong bird: {bird}")
    else:
        logging.warning(f"EVSong directory not found: {evsong_test_directory}")

    logging.info("UMAP embeddings pipeline completed")

    # # WSeg processing
    # wseg_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'wseg test new')
    # logging.info(f"Processing WSeg directory: {wseg_test_directory}")
    #
    # if os.path.exists(wseg_test_directory):
    #     birds = [b for b in os.listdir(wseg_test_directory) if b != 'copied_data' and
    #              os.path.isdir(os.path.join(wseg_test_directory, b))]
    #     logging.info(f"Found {len(birds)} birds in WSeg directory: {birds}")
    #
    #     for bird in birds:
    #         logging.info(f"Processing WSeg bird: {bird}")
    #
    #         # Inspect existing embeddings first
    #         bird_path = os.path.join(wseg_test_directory, bird)
    #         inspect_existing_embeddings(bird_path)
    #
    #         success = explore_embedding_parameters(
    #             save_path=wseg_test_directory,
    #             bird=bird,
    #             min_dists=[0.01, 0.1, 0.5],
    #             n_neighbors_list=[5, 10, 50, 100],
    #             use_parallel=True,
    #             overwrite=False,  # Set to True if you want to regenerate all
    #             max_samples=30000,  # Smaller limit for WSeg data (often more samples)
    #             memory_per_worker_gb=None,  # Auto-detect based on system
    #             auto_memory_management=True,
    #             subsample_seed=42  # Fixed seed for reproducibility
    #         )
    #         if success:
    #             logging.info(f"✅ Successfully processed WSeg bird: {bird}")
    #         else:
    #             logging.error(f"❌ Failed to process WSeg bird: {bird}")
    # else:
    #     logging.warning(f"WSeg directory not found: {wseg_test_directory}")
    #
    #logging.info("UMAP embeddings pipeline completed")


if __name__ == "__main__":
    # Create logs directory
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'umap_embeddings.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting UMAP embeddings pipeline")
    main()
