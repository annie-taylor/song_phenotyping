import os
import logging
import tables
import numpy as np
from typing import Tuple, Dict, List, Optional
import traceback
import umap
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import pickle as pkl


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
    first_file_path = os.path.join(paths_to_specs, flattened_syl_filenames[0])  # FIXED: was [1]
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


def save_umap_embeddings(embedding_path: str, embeddings: np.ndarray, hashes: list, labels: Optional[list] = None) -> bool:
    try:
        # Open the file in write mode
        with tables.open_file(embedding_path, mode='w') as f:
            # Save embeddings
            f.create_array(f.root, 'embeddings', obj=np.array(embeddings))
            # Save hashes
            hash_atom = tables.StringAtom(itemsize=max(len(h) for h in hashes))  # Define max string length
            hash_array = f.create_earray(f.root, 'hashes', atom=hash_atom, shape=(0,))
            hash_array.append(np.array([str(h) for h in hashes], dtype='S'))  # Convert to byte strings
            # Save labels if provided
            if labels is not None:
                label_atom = tables.StringAtom(itemsize=1)  # Define max string length
                label_array = f.create_earray(f.root, 'labels', atom=label_atom, shape=(0,))
                label_array.append(np.array([str(l) for l in labels], dtype='S'))  # Convert to byte strings
        return True
    except Exception as e:
        logging.error(f"Error saving embeddings to {embedding_path}: {e}")
        logging.error(traceback.format_exc())
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


def create_embeddings(samples: np.ndarray, labels, hashes, params: UMAPParams, paths: dict,
                      save_model: bool = False) -> Tuple[np.ndarray, umap.UMAP]:
    """
    Create UMAP embeddings from samples and save results.
    """
    try:
        umap_model = umap.UMAP(
            n_components=params.n_components,
            metric=params.metric,
            min_dist=params.min_dist,
            n_neighbors=params.n_neighbors,
            n_epochs=params.n_epochs
        )

        embeddings = umap_model.fit_transform(samples)

        # Create output directories
        os.makedirs(paths['model'], exist_ok=True)
        os.makedirs(paths['embeddings'], exist_ok=True)

        # Create filenames based on parameters - FIXED: removed repeat_cluster logic
        model_path = os.path.join(paths['model'],
                                  f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.pkl')
        embedding_path = os.path.join(paths['embeddings'],
                                      f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.h5')

        # Save results
        if save_model:
            save_umap_model(model_path, umap_model, params)
        save_umap_embeddings(embedding_path, embeddings, hashes, labels)

        return embeddings, umap_model

    except Exception as e:
        logging.error(f"Error creating UMAP embeddings: {e}")
        logging.error(traceback.format_exc())
        raise


def generate_embedding_for_bird(save_path: str, bird: str) -> Tuple[
    Optional[np.ndarray], Optional[umap.UMAP], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate UMAP embedding for a single bird using default parameters.

    Args:
        save_path: Root project directory
        bird: Bird identifier

    Returns:
        Tuple of (embeddings, umap_model, specs, labels) or (None, None, None, None) if failed
    """
    try:
        params = UMAPParams(n_neighbors=10, min_dist=0.2, metric='euclidean', n_components=2)

        # Fixed path structure to match previous modules
        bird_path = os.path.join(save_path, bird)
        data_path = os.path.join(bird_path, 'data')

        paths = {
            'specs': os.path.join(data_path, 'flattened'),
            'model': os.path.join(data_path, 'models'),
            'embeddings': os.path.join(data_path, 'embeddings')
        }

        # Load spectrograms
        specs, labels, position_idxs, hashes = load_flattened_specs(paths_to_specs=paths['specs'])

        # Create embeddings (transpose specs to get samples as rows)
        embeddings, umap_model = create_embeddings(
            samples=specs.T,
            labels=labels,
            hashes=hashes,
            params=params,
            paths=paths,
            save_model=True  # Save model for single embedding
        )

        logging.info(f"Successfully processed UMAP for bird {bird}")
        return embeddings, umap_model, specs, labels

    except Exception as e:
        logging.error(f"Failed to process UMAP for bird {bird}: {e}")
        logging.error(traceback.format_exc())
        return None, None, None, None


def explore_embedding_parameters(save_path: str, bird: str,
                                 min_dists: List[float] = None,
                                 n_neighbors_list: List[int] = None,
                                 use_parallel: bool = True) -> bool:
    """
    Explore different UMAP parameters for a bird and create comparison plots.

    Args:
        save_path: Root project directory
        bird: Bird identifier
        min_dists: List of min_dist values to test
        n_neighbors_list: List of n_neighbors values to test
        use_parallel: Whether to use parallel processing

    Returns:
        True if successful, False otherwise
    """
    try:
        # Default parameter ranges
        if min_dists is None:
            min_dists = [0.01, 0.05, 0.1, 0.3, 0.5]
        if n_neighbors_list is None:
            n_neighbors_list = [5, 10, 20, 50, 100]

        # Setup paths
        bird_path = os.path.join(save_path, bird)
        data_path = os.path.join(bird_path, 'data')

        paths = {
            'specs': os.path.join(data_path, 'flattened'),
            'model': os.path.join(data_path, 'models'),
            'embeddings': os.path.join(data_path, 'embeddings'),
            'figures': os.path.join(bird_path, 'figures')
        }

        # Load data
        specs, labels, position_idxs, hashes = load_flattened_specs(paths_to_specs=paths['specs'])

        # Compute parameter grid
        if use_parallel:
            compute_embedding_grid_parallel(
                samples=specs.T,
                labels=labels,
                hashes=hashes,
                min_dists=min_dists,
                n_neighbors=n_neighbors_list,
                paths=paths,
                plot=True,
                bird=bird
            )
        else:
            compute_embedding_grid(
                samples=specs.T,
                labels=labels,
                hashes=hashes,
                min_dists=min_dists,
                n_neighbors=n_neighbors_list,
                paths=paths,
                plot=True,
                bird=bird
            )

        logging.info(f"Successfully explored UMAP parameters for bird {bird}")
        return True

    except Exception as e:
        logging.error(f"Failed to explore UMAP parameters for bird {bird}: {e}")
        logging.error(traceback.format_exc())
        return False


def compare_umap_embeddings_plot(embeddings, min_dists, n_neighbors, save: None or str = None, bird: str = ''):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(n_neighbors), len(min_dists), figsize=(20, 20))
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            ax.scatter(embeddings[i, j, :, 0], embeddings[i, j, :, 1], alpha=0.5, s=1, )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"min_dist = {min_dists[j]}", size=15)
            if j == 0:
                ax.set_ylabel(f"n_neighbors = {n_neighbors[i]}", size=15)
    fig.suptitle("UMAP embedding with grid of parameters", y=0.92, size=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if save:
        plt.savefig(os.path.join(save, "umap_embedding_grid.png"))
        plt.close(fig)


def compute_single_umap(args):
    """Single UMAP computation for parallel execution"""
    samples, labels, hashes, n_neighbors, min_dist, paths = args

    try:
        params = UMAPParams(n_neighbors=n_neighbors, metric='euclidean', min_dist=min_dist)
        embedding, _ = create_embeddings(samples=samples, labels=labels, hashes=hashes,
                                         params=params, paths=paths, save_model=False)
        return (n_neighbors, min_dist, embedding)
    except Exception as e:
        print(f"Failed UMAP n={n_neighbors}, dist={min_dist}: {e}")
        return (n_neighbors, min_dist, None)


def compute_embedding_grid_parallel(samples, labels, hashes, min_dists, n_neighbors, paths,
                                    plot: bool = True, bird: str = '', max_workers: int = None):
    """
    Parallel version of compute_embedding_grid function with worker control.

    Args:
        max_workers: Maximum number of parallel workers. If None, uses min(cpu_count(), num_tasks)
    """
    # Prepare arguments for parallel processing
    args_list = []
    for n in n_neighbors:
        for dist in min_dists:
            args_list.append((samples, labels, hashes, n, dist, paths))

    print(f"    🚀 Computing {len(args_list)} UMAPs in parallel...")

    # Control worker count
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(args_list))
    else:
        max_workers = min(max_workers, len(args_list))

    print(f"    Using {max_workers} parallel workers")

    all_embeddings = np.empty((len(n_neighbors), len(min_dists), len(samples), 2))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(compute_single_umap, args): args for args in args_list}

        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_params):
            n_neighbors_val, min_dist_val, embedding = future.result()
            completed += 1

            if embedding is not None:
                # Find indices for this parameter combination
                i = n_neighbors.index(n_neighbors_val)
                j = min_dists.index(min_dist_val)
                all_embeddings[i, j] = embedding
                print(f"    ✅ {completed}/{len(args_list)}: n_neighbors={n_neighbors_val}, min_dist={min_dist_val}")
            else:
                print(
                    f"    ❌ {completed}/{len(args_list)}: FAILED n_neighbors={n_neighbors_val}, min_dist={min_dist_val}")

    # Create plot if requested
    if plot:
        fig_savepath = paths['figures']
        if not os.path.isdir(fig_savepath):
            os.makedirs(fig_savepath)
        compare_umap_embeddings_plot(all_embeddings, min_dists, n_neighbors, fig_savepath, bird)

    return all_embeddings


def compute_embedding_grid(samples, labels, hashes, min_dists, n_neighbors, paths,
                           plot: bool = True, bird: str = ''):
    """
    Non-parallel version of embedding grid computation.
    """
    print(f"    🔄 Computing {len(n_neighbors) * len(min_dists)} UMAPs sequentially...")

    all_embeddings = np.empty((len(n_neighbors), len(min_dists), len(samples), 2))

    completed = 0
    total_tasks = len(n_neighbors) * len(min_dists)

    for i, n in enumerate(n_neighbors):
        for j, dist in enumerate(min_dists):
            try:
                params = UMAPParams(n_neighbors=n, metric='euclidean', min_dist=dist)
                embedding, _ = create_embeddings(
                    samples=samples,
                    labels=labels,
                    hashes=hashes,
                    params=params,
                    paths=paths,
                    save_model=False
                )
                all_embeddings[i, j] = embedding
                completed += 1
                print(f"    ✅ {completed}/{total_tasks}: n_neighbors={n}, min_dist={dist}")

            except Exception as e:
                print(f"    ❌ {completed + 1}/{total_tasks}: FAILED n_neighbors={n}, min_dist={dist}: {e}")
                completed += 1
                continue

    # Create plot if requested
    if plot:
        fig_savepath = paths['figures']
        if not os.path.isdir(fig_savepath):
            os.makedirs(fig_savepath)
        compare_umap_embeddings_plot(all_embeddings, min_dists, n_neighbors, fig_savepath, bird)

    return all_embeddings
