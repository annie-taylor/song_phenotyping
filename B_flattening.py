import os
import logging
import tables
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm


from tools.system_utils import optimize_pytables_for_network


def extract_song_id(filepath: str) -> str:
    """Extract song ID from syllables filepath."""
    stem = Path(filepath).stem
    if 'syllables_' not in stem:
        raise ValueError(f"Expected 'syllables_' in filename: {filepath}")
    return stem.replace('syllables_', '')


def create_flattened_output_path(data_folder: str, song_id: str) -> str:
    """Create output path for flattened spectrograms and ensure directory exists."""
    flattened_dir = os.path.join(data_folder, 'flattened')
    os.makedirs(flattened_dir, exist_ok=True)
    return os.path.join(flattened_dir, f'flattened_{song_id}.h5')


def load_syllable_data(filepath: str) -> tuple:
    """Load spectrograms and metadata from HDF5 file."""
    try:
        with tables.open_file(filepath, mode='r') as f:
            specs = f.root.spectrograms.read()
            labels = f.root.manual[:]
            position_idxs = f.root.position_idxs[:]
            hashes = f.root.hashes[:]
    except (tables.NoSuchNodeError, AttributeError) as e:
        logging.error(f"Missing required data in {filepath}: {e}")
        raise ValueError(f"Invalid HDF5 structure in {filepath}")
    except (OSError, IOError) as e:
        logging.error(f"File access error for {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load syllable data from {filepath}: {e}")
        raise

    # Validate data consistency
    if not (len(specs) == len(labels) == len(position_idxs) == len(hashes)):
        raise ValueError(f"Inconsistent data lengths in {filepath}")

    return specs, labels, position_idxs, hashes


def flatten_spectrograms(specs: np.ndarray) -> np.ndarray:
    """Convert 2D spectrograms to 1D vectors efficiently."""
    if specs.size == 0:
        raise ValueError("Cannot flatten empty spectrogram array")

    if len(specs.shape) != 3:
        raise ValueError(f"Expected 3D spectrogram array, got shape {specs.shape}")

    try:
        n_specs, height, width = specs.shape
        flattened = specs.reshape(n_specs, -1).T.astype(np.float32)
        return flattened
    except Exception as e:
        logging.error(f"Failed to flatten spectrograms with shape {specs.shape}: {e}")
        raise


def save_flattened_data(output_path: str, flattened_specs: np.ndarray,
                        labels: np.ndarray, position_idxs: np.ndarray,
                        hashes: np.ndarray) -> None:
    """Save flattened data and metadata to HDF5 file."""
    try:
        with tables.open_file(output_path, mode='w') as f:
            f.create_array(f.root, 'flattened_specs', flattened_specs)
            f.create_array(f.root, 'labels', labels)
            f.create_array(f.root, 'position_idxs', position_idxs)
            f.create_array(f.root, 'hashes', hashes)

    except Exception as e:
        logging.error(f"Failed to save flattened data to {output_path}: {e}")
        raise


def process_single_syllable_file(filepath: str, data_folder: str) -> bool:
    """Process a single syllable file and create flattened version."""
    try:
        # Log file size for debugging
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        logging.debug(f"Processing {filepath} ({file_size:.1f} MB)")

        # Extract song ID
        song_id = extract_song_id(filepath)

        # Create output path
        output_path = create_flattened_output_path(data_folder, song_id)

        # Skip if already exists
        if os.path.exists(output_path):
            logging.debug(f"Flattened file already exists, skipping: {output_path}")
            return True

        # Load data
        specs, labels, position_idxs, hashes = load_syllable_data(filepath)

        # Flatten spectrograms
        flattened_specs = flatten_spectrograms(specs)

        # Save results
        save_flattened_data(output_path, flattened_specs, labels, position_idxs, hashes)

        logging.info(f"Successfully flattened {len(specs)} syllables from {filepath}")
        return True

    except Exception as e:
        logging.error(f"Failed to process syllable file {filepath}: {e}")
        return False


def find_syllable_files(syllables_dir: str) -> List[str]:
    """Find all HDF5 syllable files in directory."""
    if not os.path.exists(syllables_dir):
        return []

    return [
        os.path.join(syllables_dir, f)
        for f in os.listdir(syllables_dir)
        if f.endswith('.h5') and 'syllables' in f
    ]


def flatten_bird_spectrograms(directory: str, bird: str) -> bool:
    """Process all syllable files for a bird."""
    try:
        # Setup paths TODO make this work for slices and syllables OR consider making seperate projects!!
                            # the latter might be best b/c then we could define a project by spec params, etc.
        bird_folder = os.path.join(directory, bird)
        data_path = os.path.join(bird_folder, 'data')
        syllables_path = os.path.join(data_path, 'syllables')

        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)

        # Find syllable files
        syllable_files = find_syllable_files(syllables_path)
        if not syllable_files:
            logging.warning(f"No syllable files found for bird {bird} in {syllables_path}")
            return True  # Not an error, just nothing to process

        logging.info(f"Found {len(syllable_files)} syllable files for bird {bird}")

        # Process each file
        success_count = 0
        for filepath in tqdm(syllable_files, desc=f"Flattening {bird}"):
            if process_single_syllable_file(filepath, data_path):
                success_count += 1

        logging.info(f"Bird {bird}: {success_count}/{len(syllable_files)} files processed successfully")
        return success_count > 0

    except Exception as e:
        logging.error(f"Error processing bird {bird}: {e}")
        return False


def main():
    logging.info("Optimizing PyTables for network access")
    optimize_pytables_for_network()

    # EVSong processing
    evsong_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
    logging.info(f"Processing EVSong directory: {evsong_test_directory}")

    if not os.path.exists(evsong_test_directory):
        logging.error(f"EVSong directory does not exist: {evsong_test_directory}")
    else:
        birds = [b for b in os.listdir(evsong_test_directory)
                 if b != 'copied_data' and os.path.isdir(os.path.join(evsong_test_directory, b))]
        logging.info(f"Found {len(birds)} birds in EVSong directory: {birds}")

        for bird in birds:
            logging.info(f"Processing EVSong bird: {bird}")
            success = flatten_bird_spectrograms(evsong_test_directory, bird)
            if success:
                logging.info(f"✅ Successfully processed EVSong bird: {bird}")
            else:
                logging.error(f"❌ Failed to process EVSong bird: {bird}")

    # WSeg processing
    wseg_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'wseg test')
    logging.info(f"Processing WSeg directory: {wseg_test_directory}")

    if not os.path.exists(wseg_test_directory):
        logging.error(f"WSeg directory does not exist: {wseg_test_directory}")
    else:
        birds = [b for b in os.listdir(wseg_test_directory)
                 if b != 'copied_data' and os.path.isdir(os.path.join(wseg_test_directory, b))]
        logging.info(f"Found {len(birds)} birds in WSeg directory: {birds}")

        for bird in birds:
            logging.info(f"Processing WSeg bird: {bird}")
            success = flatten_bird_spectrograms(wseg_test_directory, bird)
            if success:
                logging.info(f"✅ Successfully processed WSeg bird: {bird}")
            else:
                logging.error(f"❌ Failed to process WSeg bird: {bird}")

    logging.info("Spectrogram flattening pipeline completed")


if __name__ == '__main__':
    # Create logs directory
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'flatten_spectrograms.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting spectrogram flattening pipeline")
    main()