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


def compute_and_save_umap(samples: np.ndarray, labels, hashes, params: UMAPParams,
                          model_path: str, embedding_path: str, save_model: bool = False,
                          overwrite: bool = False) -> Tuple[Optional[np.ndarray], Optional[umap.UMAP], bool]:
    """
    Core function: Create UMAP embeddings, save results, and return objects

    Args:
        samples: Input data for UMAP
        labels: Sample labels
        hashes: Sample hashes
        params: UMAP parameters
        model_path: Path to save UMAP model
        embedding_path: Path to save embeddings
        save_model: Whether to save the UMAP model
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (embeddings, umap_model, success) where success is True if file exists or was created
    """
    # Check if files already exist
    if not overwrite and os.path.exists(embedding_path):
        logging.debug(f"⏭️ Skipped existing embedding: {os.path.basename(embedding_path)}")
        return None, None, True  # File exists = success

    if not overwrite and save_model and os.path.exists(model_path):
        logging.debug(f"⏭️ Skipped existing model: {os.path.basename(model_path)}")
        return None, None, True  # File exists = success

    try:
        # Create UMAP model
        umap_model = umap.UMAP(
            n_components=params.n_components,
            metric=params.metric,
            min_dist=params.min_dist,
            n_neighbors=params.n_neighbors,
            n_epochs=params.n_epochs
        )

        # Fit and transform
        embeddings = umap_model.fit_transform(samples)

        # Create output directories
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

        # Save results
        if save_model:
            save_umap_model(model_path, umap_model, params)
        save_umap_embeddings(embedding_path, embeddings, hashes, labels)

        logging.info(f"✅ Computed new embedding: n_neighbors={params.n_neighbors}, min_dist={params.min_dist}")
        return embeddings, umap_model, True  # Successfully created

    except Exception as e:
        logging.error(f"Error creating UMAP embeddings: {e}")
        logging.error(traceback.format_exc())
        return None, None, False  # Failed to create


def compute_single_umap_worker(args):
    """Single UMAP computation for parallel execution"""
    samples, labels, hashes, n_neighbors, min_dist, paths, overwrite = args

    try:
        params = UMAPParams(n_neighbors=n_neighbors, metric='euclidean', min_dist=min_dist)

        # Generate file paths
        model_path = os.path.join(paths['model'],
                                  f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.pkl')
        embedding_path = os.path.join(paths['embeddings'],
                                      f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.h5')

        embeddings, model, success = compute_and_save_umap(  # Added the third value here
            samples=samples,
            labels=labels,
            hashes=hashes,
            params=params,
            model_path=model_path,
            embedding_path=embedding_path,
            save_model=False,
            overwrite=overwrite
        )

        return (n_neighbors, min_dist, success)

    except Exception as e:
        logging.error(f"Failed UMAP n={n_neighbors}, dist={min_dist}: {e}")
        return (n_neighbors, min_dist, False)


def compute_embedding_grid_parallel(samples, labels, hashes, min_dists, n_neighbors, paths,
                                    plot: bool = True, bird: str = '', max_workers: int = None,
                                    overwrite: bool = False):
    """
    Parallel version of compute_embedding_grid function with worker control.

    Args:
        max_workers: Maximum number of parallel workers. If None, uses min(cpu_count(), num_tasks)
        overwrite: Whether to overwrite existing embedding files

    Returns:
        List of successful (n_neighbors, min_dist) tuples
    """
    # Prepare arguments for parallel processing
    args_list = []
    for n in n_neighbors:
        for dist in min_dists:
            args_list.append((samples, labels, hashes, n, dist, paths, overwrite))

    print(f"    🚀 Computing {len(args_list)} UMAPs in parallel...")

    # Control worker count
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(args_list))
    else:
        max_workers = min(max_workers, len(args_list))

    print(f"    Using {max_workers} parallel workers")

    successful_params = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(compute_single_umap_worker, args): args for args in args_list}

        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_params):
            n_neighbors_val, min_dist_val, success = future.result()
            completed += 1

            if success:
                successful_params.append((n_neighbors_val, min_dist_val))
                print(f"    ✅ {completed}/{len(args_list)}: n_neighbors={n_neighbors_val}, min_dist={min_dist_val}")
            else:
                print(
                    f"    ❌ {completed}/{len(args_list)}: FAILED n_neighbors={n_neighbors_val}, min_dist={min_dist_val}")

    # Create plot if requested
    if plot:
        fig_savepath = paths['figures']
        if not os.path.isdir(fig_savepath):
            os.makedirs(fig_savepath)
        compare_umap_embeddings_plot(successful_params, min_dists, n_neighbors, paths, fig_savepath, bird)

    return successful_params


def compute_embedding_grid(samples, labels, hashes, min_dists, n_neighbors, paths,
                           plot: bool = True, bird: str = '', overwrite: bool = False):
    """
    Non-parallel version of embedding grid computation.

    Returns:
        List of successful (n_neighbors, min_dist) tuples
    """
    print(f"    🔄 Computing {len(n_neighbors) * len(min_dists)} UMAPs sequentially...")

    successful_params = []
    completed = 0
    total_tasks = len(n_neighbors) * len(min_dists)

    for n in n_neighbors:
        for dist in min_dists:
            try:
                params = UMAPParams(n_neighbors=n, metric='euclidean', min_dist=dist)

                # Generate file paths
                model_path = os.path.join(paths['model'],
                                          f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.pkl')
                embedding_path = os.path.join(paths['embeddings'],
                                              f'{params.metric}_{params.n_neighbors}neighbors_{params.min_dist}dist.h5')

                embeddings, model, success = compute_and_save_umap(
                    samples=samples,
                    labels=labels,
                    hashes=hashes,
                    params=params,
                    model_path=model_path,
                    embedding_path=embedding_path,
                    save_model=False,
                    overwrite=overwrite
                )

                if success:
                    successful_params.append((n, dist))
                    completed += 1
                    if embeddings is not None:
                        print(f"    ✅ {completed}/{total_tasks}: n_neighbors={n}, min_dist={dist}")
                    else:
                        print(f"    ⏭️ {completed}/{total_tasks}: SKIPPED n_neighbors={n}, min_dist={dist}")
                else:
                    completed += 1
                    print(f"    ❌ {completed}/{total_tasks}: FAILED n_neighbors={n}, min_dist={dist}")

            except Exception as e:
                print(f"    ❌ {completed + 1}/{total_tasks}: FAILED n_neighbors={n}, min_dist={dist}: {e}")
                completed += 1
                continue

    # Create plot if requested
    if plot:
        fig_savepath = paths['figures']
        if not os.path.isdir(fig_savepath):
            os.makedirs(fig_savepath)
        compare_umap_embeddings_plot(successful_params, min_dists, n_neighbors, paths, fig_savepath, bird)

    return successful_params


def explore_embedding_parameters(save_path: str, bird: str,
                                 min_dists: List[float] = None,
                                 n_neighbors_list: List[int] = None,
                                 use_parallel: bool = True,
                                 overwrite: bool = False) -> bool:
    """
    Explore different UMAP parameters for a bird and create comparison plots.

    Args:
        save_path: Root project directory
        bird: Bird identifier
        min_dists: List of min_dist values to test
        n_neighbors_list: List of n_neighbors values to test
        use_parallel: Whether to use parallel processing
        overwrite: Whether to overwrite existing embedding files

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
            successful_params = compute_embedding_grid_parallel(
                samples=specs.T,
                labels=labels,
                hashes=hashes,
                min_dists=min_dists,
                n_neighbors=n_neighbors_list,
                paths=paths,
                plot=True,
                bird=bird,
                overwrite=overwrite
            )
        else:
            successful_params = compute_embedding_grid(
                samples=specs.T,
                labels=labels,
                hashes=hashes,
                min_dists=min_dists,
                n_neighbors=n_neighbors_list,
                paths=paths,
                plot=True,
                bird=bird,
                overwrite=overwrite
            )

        logging.info(f"Successfully explored UMAP parameters for bird {bird}. "
                     f"Computed {len(successful_params)} parameter combinations.")
        return True

    except Exception as e:
        logging.error(f"Failed to explore UMAP parameters for bird {bird}: {e}")
        logging.error(traceback.format_exc())
        return False


def load_embedding_from_file(embedding_path: str) -> Optional[np.ndarray]:
    """Load embeddings from HDF5 file"""
    try:
        with tables.open_file(embedding_path, mode='r') as f:
            return f.root.embeddings[:]
    except Exception as e:
        logging.warning(f"Could not load embedding from {embedding_path}: {e}")
        return None


def compare_umap_embeddings_plot(successful_params: List[Tuple[int, float]], min_dists, n_neighbors,
                                 paths: dict, save_path: str, bird: str = ''):
    """
    Create comparison plot by loading embeddings on-demand
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
    evsong_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
    logging.info(f"Processing EVSong directory: {evsong_test_directory}")

    birds = os.listdir(evsong_test_directory)
    birds.remove('copied_data')
    logging.info(f"Found {len(birds)} birds in EVSong directory: {birds}")

    for bird in birds:
        logging.info(f"Processing EVSong bird: {bird}")
        success = explore_embedding_parameters(
            save_path=evsong_test_directory,
            bird=bird,
            min_dists=[0.01, 0.1, 0.5],
            n_neighbors_list=[10, 20, 50],
            use_parallel=True,
            overwrite=True
        )
        if success:
            logging.info(f"✅ Successfully processed EVSong bird: {bird}")
        else:
            logging.error(f"❌ Failed to process EVSong bird: {bird}")

    # WSeg processing
    wseg_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'wseg test')
    logging.info(f"Processing WSeg directory: {wseg_test_directory}")

    birds = [b for b in os.listdir(evsong_test_directory) if b != 'copied_data' and
             os.path.isdir(os.path.join(evsong_test_directory, b))]
    logging.info(f"Found {len(birds)} birds in WSeg directory: {birds}")

    for bird in birds:
        logging.info(f"Processing WSeg bird: {bird}")
        success = explore_embedding_parameters(
            save_path=wseg_test_directory,
            bird=bird,
            min_dists=[0.01, 0.1, 0.5],
            n_neighbors_list=[5, 100],
            use_parallel=True,
            overwrite=True
        )
        if success:
            logging.info(f"✅ Successfully processed WSeg bird: {bird}")
        else:
            logging.error(f"❌ Failed to process WSeg bird: {bird}")

    logging.info("UMAP embeddings pipeline completed")


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
