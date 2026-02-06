import re
from sys import platform
from scipy.io import loadmat
import numpy as np
import hashlib
from typing import Any, Dict, List
from random import sample
import gc
import shutil
from tqdm import tqdm

from tools.song_io import save_segmented_audio_data, get_song_specs
from tools.system_utils import check_sys_for_macaw_root, optimize_pytables_for_network
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_path_management import *


def filepaths_from_wseg(seg_directory: str, save_path: str = None,
                        song_or_call: str = 'song',
                        file_ext: str = '.wav.not.mat',
                        bird_subset: None | list = None,
                        copy_locally: bool = False) -> Dict[str, List[str]]:
    """
    Extract WhisperSeg metadata file paths organized by bird and optionally copy audio files.

    Parameters:
    - copy_locally (bool): If True, copy audio files to local copied_data directory
    """
    metadata_file_paths = {}

    for root, dirs, files in os.walk(seg_directory):
        if song_or_call.lower() not in root.lower():
            continue

        for file in files:
            if file.endswith(file_ext):
                file_path = os.path.join(root, file)
                path_parts = root.split(os.sep)
                bird = path_parts[-2]

                if bird not in metadata_file_paths:
                    metadata_file_paths[bird] = []
                metadata_file_paths[bird].append(file_path)

                # Handle copying and mapping for wseg files
                if save_path and copy_locally:
                    if (bird_subset is None) or (bird in bird_subset):
                        bird_folder = os.path.join(save_path, bird)
                        os.makedirs(bird_folder, exist_ok=True)

                        # For wseg, we need to load metadata to get audio path
                        try:
                            metadata_matfile = loadmat(file_path, squeeze_me=True)
                            audio_path, _ = resolve_audio_file_path(
                                file_path, metadata_matfile, read_songpath_from_metadata=True
                            )

                            if audio_path and os.path.exists(audio_path):
                                filename = os.path.basename(audio_path)

                                # Copy file locally
                                copied_data_dir = os.path.join(save_path, 'copied_data', bird)
                                os.makedirs(copied_data_dir, exist_ok=True)
                                local_path = os.path.join(copied_data_dir, filename)

                                if not os.path.exists(local_path):
                                    try:
                                        shutil.copy2(audio_path, local_path)
                                        logging.debug(f"Copied {audio_path} to {local_path}")
                                    except Exception as e:
                                        logging.error(f"Failed to copy {audio_path}: {e}")
                                        local_path = None

                                # Update mapping with both paths
                                update_audio_paths_file(bird_folder, filename,
                                                       local_path=local_path,
                                                       server_path=audio_path)
                            else:
                                logging.warning(f"Could not resolve audio path for {file_path}")

                        except Exception as e:
                            logging.error(f"Failed to process wseg metadata {file_path}: {e}")

                elif save_path:
                    # Just create bird folders without copying
                    if (bird_subset is None) or (bird in bird_subset):
                        bird_folder = os.path.join(save_path, bird)
                        os.makedirs(bird_folder, exist_ok=True)

    if bird_subset is not None:
        metadata_file_paths = {bird: paths for bird, paths in metadata_file_paths.items()
                               if bird in bird_subset}

    return metadata_file_paths


def filepaths_from_evsonganaly(wav_directory: str = None, save_path: str = None,
                               batch_file_naming: str = 'batch.txt.keep',
                               bird_subset: None | list = None,
                               copy_locally: bool = False) -> tuple[
    dict[str, list[str]], dict[str, list[str]]]:
    """
    Extract file paths from evsonganaly batch files and optionally copy files locally.

    Parameters:
    - wav_directory (str): The root directory to search for audio files containing batch files.
    - save_path (str): The directory where bird folders will be created and audio_paths.txt files saved.
    - batch_file_naming (str): Specifies the name of the batch file we're referencing
    - bird_subset (None | list): If not processing all birds in directory, list of birds to be processed
    - copy_locally (bool): If True, copy audio files to local copied_data directory

    Returns:
    - tuple: (metadata_file_paths, wav_file_paths) where:
        - metadata_file_paths (dict[str, list[str]]): Dictionary mapping bird names to lists of metadata file paths
        - wav_file_paths (dict[str, list[str]]): Dictionary mapping bird names to lists of wav file paths
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    birds = []
    metadata_file_paths = {}
    wav_file_paths = {}

    for root, dirs, files in os.walk(wav_directory, topdown=False):
        for file in files:
            if batch_file_naming in file:
                song_metadata = []
                song_audio = []

                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        song_name = line.replace('\n', '.not.mat')
                        song_metadata_path = os.path.join(root, song_name)

                        if os.path.isfile(song_metadata_path):
                            audio_path = os.path.join(root, line.replace('\n', ''))
                            song_audio.append(audio_path)
                            song_metadata.append(song_metadata_path)

                            # Extract bird name
                            try:
                                [bird, _, _] = line.split('_')[0:3]
                            except ValueError:
                                logging.error(f"Trouble reading birdname from {line}: {ValueError}")
                                try:
                                    [bird, _] = line.split('_')[0:2]
                                except ValueError:
                                    logging.error(f"Trouble reading birdname from {line}: {ValueError}")
                                    [bird, _] = line.split('.')[0:2]

                            # Create bird folder and handle copying/mapping
                            if save_path:
                                bird_folder = os.path.join(save_path, bird)
                                os.makedirs(bird_folder, exist_ok=True)

                                filename = os.path.basename(audio_path)

                                if copy_locally:
                                    # Copy file locally
                                    copied_data_dir = os.path.join(save_path, 'copied_data', bird)
                                    os.makedirs(copied_data_dir, exist_ok=True)
                                    local_path = os.path.join(copied_data_dir, filename)

                                    if not os.path.exists(local_path):
                                        try:
                                            shutil.copy2(audio_path, local_path)
                                            logging.debug(f"Copied {audio_path} to {local_path}")
                                        except Exception as e:
                                            logging.error(f"Failed to copy {audio_path}: {e}")
                                            local_path = None

                                    # Update mapping with both paths
                                    update_audio_paths_file(bird_folder, filename,
                                                            local_path=local_path,
                                                            server_path=audio_path)
                                else:
                                    # Just update with server path
                                    update_audio_paths_file(bird_folder, filename,
                                                            server_path=audio_path)

                            # Collect paths by bird
                            if bird not in birds:
                                birds.append(bird)
                                metadata_file_paths[bird] = song_metadata
                                wav_file_paths[bird] = song_audio
                            else:
                                metadata_file_paths[bird].extend(song_metadata)
                                wav_file_paths[bird].extend(song_audio)

    # Apply bird subset filter
    if bird_subset is not None:
        metadata_file_paths = {bird: paths for bird, paths in metadata_file_paths.items()
                               if bird in bird_subset}
        wav_file_paths = {bird: paths for bird, paths in wav_file_paths.items()
                          if bird in bird_subset}

    return metadata_file_paths, wav_file_paths


def select_new_files(available_files: list[str], already_saved_files: list[str], needed_count: int) -> list[str]:
    """
    Select files that haven't been processed yet.

    Parameters:
    - available_files: All possible files for this bird
    - already_saved_files: Files already processed
    - needed_count: How many new files we need

    Returns:
    - List of file paths to process
    """
    if needed_count <= 0:
        return []

    # Get base names of already processed files
    processed_bases = set()
    for saved_file in already_saved_files:
        base_name = extract_base_name(saved_file)
        if base_name:
            processed_bases.add(base_name)

    # Find unprocessed files
    unprocessed_files = []
    for file_path in available_files:
        base_name = extract_base_name(file_path)
        if base_name and base_name not in processed_bases:
            unprocessed_files.append(file_path)
        elif base_name is None:
            logging.warning(f"Skipping file with unparseable name: {file_path}")

    # Return up to needed_count files
    if len(unprocessed_files) <= needed_count:
        return unprocessed_files
    else:
        return sample(unprocessed_files, needed_count)


def standardize_bird_band(band_string):
    """
    Convert bird band strings from various formats to standardized 'co#co#' format.

    Args:
        band_string (str): Bird band identifier in various formats

    Returns:
        str: Standardized band string in 'co#co#' format, or None if invalid
    """
    if not band_string or not isinstance(band_string, str):
        return None

    # Remove whitespace and convert to lowercase
    band_string = band_string.strip().lower()

    # Color abbreviation mapping (single char -> two char)
    color_map = {
        'b': 'bk',  # black (not blue)
        'w': 'wh',  # white
        'g': 'gr',  # green
        'y': 'ye',  # yellow
        'r': 'rd',  # red
        'o': 'or',  # orange
        # purple, pink, and blue have two char instances and full name instances, but not in original dbquery function
    }

    # Full color name to two-char abbreviation
    full_color_map = {
        'black': 'bk',
        'white': 'wh',
        'green': 'gr',
        'yellow': 'ye',
        'red': 'rd',
        'pink': 'pk',
        'orange': 'or',
        'purple': 'pu',
        'brown': 'br',
        'blue': 'bl',
        'noband': 'nb'
    }

    def normalize_color(color_str):
        """Convert color string to standardized two-character format"""
        if not color_str:
            return ''

        color_str = color_str.lower().strip()

        # If it's already a two-character code, check if it's valid
        if len(color_str) == 2:
            # Check if it's already in the correct format
            valid_two_char = set(color_map.values()) | set(full_color_map.values())
            if color_str in valid_two_char:
                return color_str

        # If it's a single character, map it
        if len(color_str) == 1 and color_str in color_map:
            return color_map[color_str]

        # If it's a full color name, map it
        if color_str in full_color_map:
            return full_color_map[color_str]

        # If it's an abbreviation we don't recognize, try to guess
        # This handles cases like "pk" for pink, "rd" for red, etc.
        for full_name, abbrev in full_color_map.items():
            if color_str == abbrev or full_name.startswith(color_str):
                return abbrev

        # Return as-is if we can't map it (might need manual review)
        return color_str

    # Parse the band string using regex to separate colors and numbers
    # This pattern captures sequences of letters and sequences of digits
    parts = re.findall(r'([a-zA-Z]+)|(\d+)', band_string)

    if not parts:
        return None

    # Extract components
    colors = []
    numbers = []

    for letter_part, digit_part in parts:
        if letter_part:
            colors.append(normalize_color(letter_part))
        if digit_part:
            numbers.append(digit_part)

    # Build standardized string based on what we found
    if len(colors) == 1 and len(numbers) == 1:
        # Single band: co#
        return f"{colors[0]}{numbers[0]}"
    elif len(colors) == 2 and len(numbers) == 2:
        # Two bands: co#co#
        return f"{colors[0]}{numbers[0]}{colors[1]}{numbers[1]}"
    elif len(colors) == 2 and len(numbers) == 1:
        # Two colors, one number - might be co#co format
        return f"{colors[0]}{numbers[0]}{colors[1]}"
    elif len(colors) == 1 and len(numbers) == 2:
        # One color, two numbers - unusual but handle it
        return f"{colors[0]}{numbers[0]}{numbers[1]}"
    else:
        # Return None for unrecognized patterns
        return None


def parse_audio_filename(file_path: str) -> dict:
    """
    Parse audio filename to extract bird, date, and time components.

    Returns:
        dict with keys: 'bird', 'day', 'time', 'success'
    """
    try:
        base_name = os.path.basename(file_path).split('.')[0]

        if base_name is None:
            return {'bird': None, 'day': None, 'time': None, 'success': False}

        # Split the standardized base name
        parts = base_name.split('_')

        if len(parts) >= 2:
            bird = parts[0]  # Already standardized by parse_bird_filename_simple
            day = parts[1]  # Date component
            time = parts[2] if len(parts) > 2 else None  # Time component (optional)

            return {
                'bird': bird,
                'day': day,
                'time': time,
                'success': True
            }
        else:
            logging.warning(f"Insufficient filename components: {base_name}")
            return {'bird': None, 'day': None, 'time': None, 'success': False}

    except Exception as e:
        logging.error(f"Failed to parse filename {file_path}: {e}")
        return {'bird': None, 'day': None, 'time': None, 'success': False}


def resolve_audio_file_path(metadata_file_path: str, metadata_matfile: dict,
                            read_songpath_from_metadata: bool,
                            bird_folder: str = None, prefer_local: bool = True) -> tuple[str, float]:
    """
    Resolve the path to the audio file and return offset.

    Args:
        bird_folder: Path to bird folder for audio path mapping (optional)
        prefer_local: If True and bird_folder provided, prefer local files

    Returns:
        tuple: (audio_file_path, wseg_offset) or (None, offset) if file not found
    """
    if not read_songpath_from_metadata:
        # Simple case: same path as metadata but with .wav extension
        audio_file_path = metadata_file_path.replace('.wav.not.mat', '.wav')
        wseg_offset = 0.0

        # Try to use path mapping if available
        if bird_folder:
            try:
                audio_file_path = get_audio_path(bird_folder, audio_file_path, prefer_local)
            except FileNotFoundError:
                # Fall back to original logic
                pass
    else:
        # Complex case: read path from metadata and reconstruct with server root
        try:
            fs = metadata_matfile.get('Fs', 32000.0)
            wseg_offset = (256 * 1000) / fs

            # Try to get filename from metadata (handle both field names)
            fname = None
            for key in ['fname', 'fnamecell']:
                if key in metadata_matfile:
                    fname = metadata_matfile[key]
                    break

            if fname is None:
                raise KeyError("No filename found in metadata (tried 'fname' and 'fnamecell')")

            # Try path mapping first if available
            if bird_folder:
                try:
                    audio_file_path = get_audio_path(bird_folder, fname, prefer_local)
                except FileNotFoundError:
                    # Fall back to server reconstruction
                    audio_file_path = reconstruct_server_path(fname)
            else:
                # Reconstruct path using your server root function
                audio_file_path = reconstruct_server_path(fname)

        except KeyError as e:
            logging.error(f"Could not extract filename from metadata {metadata_file_path}: {e}")
            return None, 0.0

    # Verify file exists
    if not os.path.exists(audio_file_path):
        logging.warning(f"Audio file not found: {audio_file_path}")
        return None, wseg_offset

    return audio_file_path, wseg_offset


def reconstruct_server_path(stored_path: str) -> str:
    """
    Reconstruct full server path from stored relative path using current platform.
    """
    # Get the appropriate server root for current platform
    server_root = check_sys_for_macaw_root()

    if not server_root:
        raise RuntimeError("Could not determine server root for current platform")

    # Clean up the stored path and extract relative part
    # Handle mixed separators in stored path
    clean_path = stored_path.replace('\\', '/').replace('//', '/')

    # Remove any existing server root prefixes and get relative path
    path_parts = [part for part in clean_path.split('/') if part]
    if len(path_parts) > 0:
        # Skip the first part which is usually the old server root
        relative_path = '/'.join(path_parts[1:])

        # Combine with current platform's server root
        if platform == "win32":
            return os.path.join(server_root, relative_path.replace('/', '\\'))
        else:
            return os.path.join(server_root, relative_path)

    return stored_path  # Fallback if parsing fails


def load_and_validate_metadata(metadata_file_path: str, wseg_offset: float = 0.0) -> dict:
    """
    Load metadata from .mat file and validate required fields.

    Returns:
        dict with keys: 'fs', 'onsets', 'offsets', 'labels', 'is_valid_song', 'error'
    """
    try:
        metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)

        # Extract sampling rate with fallback (i.e. defaults to 32000 if fs unavailable)
        fs = metadata_matfile.get('Fs') or metadata_matfile.get('fs') or 32000.0

        # Extract required arrays
        raw_onsets = metadata_matfile.get('onsets')
        raw_offsets = metadata_matfile.get('offsets')
        labels = np.array(list(metadata_matfile.get('labels')))

        # Check for missing required fields
        if raw_onsets is None or raw_offsets is None or labels is None:
            missing = []
            if raw_onsets is None: missing.append('onsets')
            if raw_offsets is None: missing.append('offsets')
            if labels is None: missing.append('labels')
            return {
                'fs': fs, 'onsets': None, 'offsets': None, 'labels': None,
                'is_valid_song': False, 'error': f"Missing required fields: {missing}"
            }

        # Apply offset and ensure arrays
        onsets = np.atleast_1d(raw_onsets) + wseg_offset
        offsets = np.atleast_1d(raw_offsets) + wseg_offset
        labels = np.atleast_1d(labels)

        # Validate array lengths match
        if not (len(onsets) == len(offsets) == len(labels)):
            return {
                'fs': fs, 'onsets': onsets, 'offsets': offsets, 'labels': labels,
                'is_valid_song': False,
                'error': f"Array length mismatch: onsets={len(onsets)}, offsets={len(offsets)}, labels={len(labels)}"
            }

        # Validate onset/offset logic
        if np.any(offsets <= onsets):
            return {
                'fs': fs, 'onsets': onsets, 'offsets': offsets, 'labels': labels,
                'is_valid_song': False, 'error': "Invalid onset/offset pairs (offset <= onset)"
            }

        # Determine if this is a valid song (more than one syllable)
        is_valid_song = len(onsets) > 1

        return {
            'fs': fs, 'onsets': onsets, 'offsets': offsets, 'labels': labels,
            'is_valid_song': is_valid_song, 'error': None
        }

    except Exception as e:
        return {
            'fs': 32000.0, 'onsets': None, 'offsets': None, 'labels': None,
            'is_valid_song': False, 'error': f"Failed to load metadata: {str(e)}"
        }


def split_long_syllables(onsets: np.ndarray, offsets: np.ndarray, labels: np.ndarray,
                         max_duration_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split syllables longer than max_duration into smaller segments.

    Args:
        onsets: Array of syllable onset times (ms)
        offsets: Array of syllable offset times (ms)
        labels: Array of syllable labels
        max_duration_ms: Maximum allowed syllable duration (ms)

    Returns:
        tuple: (new_onsets, new_offsets, new_labels)
    """
    if len(onsets) == 0:
        return onsets.copy(), offsets.copy(), labels.copy()

    new_onsets = []
    new_offsets = []
    new_labels = []

    # TODO probably can just subtract all at once and then only process long syllables rather than looping through each syl instance
    for i, (onset, offset, label) in enumerate(zip(onsets, offsets, labels)):
        duration = offset - onset

        if duration <= max_duration_ms:
            # Short syllable - keep as is
            new_onsets.append(onset)
            new_offsets.append(offset)
            new_labels.append(label)
        else:
            # Long syllable - split into segments
            segments = split_single_syllable(onset, offset, label, max_duration_ms)
            for seg_onset, seg_offset, seg_label in segments:
                new_onsets.append(seg_onset)
                new_offsets.append(seg_offset)
                new_labels.append(seg_label)

    return np.array(new_onsets), np.array(new_offsets), np.array(new_labels)


def split_single_syllable(onset: float, offset: float, label: str,
                          max_duration_ms: float) -> list[tuple[float, float, str]]:
    """
    Split a single long syllable into segments of max_duration_ms.

    Returns:
        list of (onset, offset, label) tuples
    """
    duration = offset - onset
    n_full_segments = int(duration // max_duration_ms)
    remainder = duration % max_duration_ms

    segments = []

    # Add full-length segments
    for i in range(n_full_segments):
        seg_onset = onset + (i * max_duration_ms)
        seg_offset = onset + ((i + 1) * max_duration_ms)
        segments.append((seg_onset, seg_offset, label))

    # Add final segment if there's a remainder
    if remainder > 0:
        final_onset = onset + (n_full_segments * max_duration_ms)
        final_offset = offset  # Use original offset
        segments.append((final_onset, final_offset, label))

    return segments


def pad_waveforms_to_same_length(waveforms: list[np.ndarray],
                                 time_arrays: list[np.ndarray] = None,
                                 pad_value: float = np.nan) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Pad all waveforms (and optionally time arrays) to the same length.

    Args:
        waveforms: List of 1D numpy arrays of different lengths
        time_arrays: Optional list of time arrays to pad alongside waveforms
        pad_value: Value to use for padding (default: NaN)

    Returns:
        tuple: (padded_waveforms, padded_time_arrays or None)
    """
    if not waveforms:
        return [], [] if time_arrays is not None else []

    # Validate inputs
    if time_arrays is not None and len(waveforms) != len(time_arrays):
        raise ValueError(f"Waveforms and time arrays must have same length: {len(waveforms)} vs {len(time_arrays)}")

    # Find maximum length
    max_length = max(len(wav) for wav in waveforms)

    # Pad waveforms
    padded_waveforms = []
    for wav in waveforms:
        if len(wav) == max_length:
            padded_waveforms.append(wav.copy())  # Copy to avoid modifying original
        else:
            padded_wav = np.full(max_length, pad_value, dtype=wav.dtype)
            padded_wav[:len(wav)] = wav
            padded_waveforms.append(padded_wav)

    # Pad time arrays if provided
    padded_time_arrays = None
    if time_arrays is not None:
        padded_time_arrays = []
        for ts in time_arrays:
            if len(ts) == max_length:
                padded_time_arrays.append(ts.copy())
            else:
                padded_ts = np.full(max_length, pad_value, dtype=ts.dtype)
                padded_ts[:len(ts)] = ts
                padded_time_arrays.append(padded_ts)

    return padded_waveforms, padded_time_arrays


def pad_segmented_audio_data(segmented_data: dict) -> dict:
    """
    Pad waveforms and time arrays in segmented audio data to same length.

    Args:
        segmented_data: Dictionary containing 'waveforms' and 'spec_t' keys

    Returns:
        Updated dictionary with padded arrays
    """
    waveforms = segmented_data.get('waveforms', [])
    time_arrays = segmented_data.get('spec_t', [])

    if not waveforms:
        return segmented_data

    # Pad the arrays
    padded_wavs, padded_ts = pad_waveforms_to_same_length(waveforms, time_arrays)

    # Update the dictionary
    updated_data = segmented_data.copy()
    updated_data['waveforms'] = padded_wavs
    if padded_ts is not None:
        updated_data['spec_t'] = padded_ts

    return updated_data


def create_segmented_audio_data(specs: List[np.ndarray], wavs: List[np.ndarray],
                                ts: List[np.ndarray], onsets: np.ndarray,
                                offsets: np.ndarray, labels: np.ndarray, mean_top_3: np.float64, low_f_mean: np.float64,
                                mean_all: np.float64,
                                valid_indices: List[int], file_identifier: str) -> Dict[str, Any]:
    """
    Create organized segmented audio data structure from processing results.

    Args:
        specs: List of spectrogram arrays
        wavs: List of waveform arrays
        ts: List of time reference arrays
        onsets: Array of onset times
        offsets: Array of offset times
        labels: Array of syllable labels
        mean_top_3: Mean of the three highest area frequency peaks (representing tempo)
        low_f_mean: Mean of frequency peaks below cutoff of 3Hz (possibly rep. motif structure)
        mean_all: Mean of all frequency peaks, scaled by area
        valid_indices: Indices of successfully processed syllables
        file_identifier: Base string for generating unique hashes

    Returns:
        Dictionary with organized segmented audio data
    """
    if not valid_indices:
        return create_empty_segmented_data()

    # Validate input lengths
    expected_length = len(valid_indices)
    if not (len(specs) == len(wavs) == len(ts) == expected_length):
        raise ValueError(
            f"Inconsistent array lengths: specs={len(specs)}, wavs={len(wavs)}, ts={len(ts)}, valid_indices={expected_length}")

    # Filter arrays to valid indices only
    valid_onsets = [onsets[i] for i in valid_indices]
    valid_offsets = [offsets[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    # Generate unique hashes for each syllable
    hashes = generate_syllable_hashes(file_identifier, valid_indices)

    # Pad waveforms and time arrays to same length
    padded_wavs, padded_ts = pad_waveforms_to_same_length(wavs, ts)

    # Build the data structure
    segmented_data = {
        'spectrograms': specs,
        'waveforms': padded_wavs,
        'spec_t': padded_ts,
        'manual': valid_labels,
        'onsets': valid_onsets,
        'offsets': valid_offsets,
        'mean_top_3': mean_top_3,
        'low_f_mean': low_f_mean,
        'mean_all': mean_all,
        'position_idxs': valid_indices,
        'hashes': hashes
    }

    return segmented_data


def generate_syllable_hashes(base_identifier: str, indices: List[int]) -> List[str]:
    """
    Generate unique hash IDs for each syllable.

    Args:
        base_identifier: Base string (usually file path)
        indices: List of syllable indices

    Returns:
        List of unique hash strings
    """
    hashes = []
    for idx in indices:
        # Create unique string for each syllable
        unique_str = f"{base_identifier}_{idx}"
        # Generate hash
        hash_obj = hashlib.sha256()
        hash_obj.update(unique_str.encode('utf-8'))
        hashes.append(hash_obj.hexdigest())

    return hashes


def create_empty_segmented_data() -> Dict[str, Any]:
    """Create empty segmented data structure."""
    return {
        'spectrograms': [],
        'waveforms': [],
        'spec_t': [],
        'manual': [],
        'onsets': [],
        'offsets': [],
        'position_idxs': [],
        'hashes': []
    }


def create_output_path(save_path: str, filename_info: Dict[str, str]) -> str:
    """Create output file path and ensure directory structure exists."""
    bird = filename_info['bird']
    day = filename_info['day']
    time = filename_info['time'] or 'unknown'

    # Create directory structure
    bird_dir = os.path.join(save_path, bird)
    data_dir = os.path.join(bird_dir, 'data')
    syllables_dir = os.path.join(data_dir, 'syllables')

    os.makedirs(syllables_dir, exist_ok=True)

    # Create output filename
    output_filename = f'syllables_{bird}_{day}_{time}.h5'
    return os.path.join(syllables_dir, output_filename)


def process_and_save_audio(audio_file_path: str, output_path: str, metadata: Dict[str, Any],
                           params: SpectrogramParams, verbose: bool) -> bool:
    """
    Process audio file and save segmented data.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Split long syllables if needed
        if hasattr(params, 'max_dur') and params.max_dur:
            max_dur_ms = params.max_dur * 1000
            syl_onsets, syl_offsets, labels = split_long_syllables(
                metadata['onsets'], metadata['offsets'], metadata['labels'], max_dur_ms
            )
        else:
            syl_onsets = metadata['onsets']
            syl_offsets = metadata['offsets']
            labels = metadata['labels']

        # Generate spectrograms
        specs, wavs, ts, valid_inds, tempos = get_song_specs(
            audio_file_path, syl_onsets, syl_offsets, params=params
        )

        if not valid_inds:
            logging.warning(f"No valid spectrograms generated for {audio_file_path}")
            return False
        (mean_top_3, low_f_mean, mean_all) = tempos

        # Create organized data structure
        segmented_audio_data = create_segmented_audio_data(
            specs=specs,
            wavs=wavs,
            ts=ts,
            onsets=syl_onsets,
            offsets=syl_offsets,
            labels=labels,
            mean_top_3=mean_top_3,
            low_f_mean=low_f_mean,
            mean_all=mean_all,
            valid_indices=valid_inds,
            file_identifier=output_path
        )

        # Save to HDF5 file
        save_segmented_audio_data(output_path, audio_file_path, segmented_audio_data)
        del segmented_audio_data  # Clear the large data structure
        gc.collect()  # Force garbage collection

        if verbose:
            logging.info(f"Saved {len(valid_inds)} syllables to {output_path}")

        return True

    except Exception as e:
        logging.error(f"Error processing audio {audio_file_path}: {e}")
        return False


def process_single_metadata_file(metadata_file_path: str, save_path: str, params: SpectrogramParams,
                                 read_songpath_from_metadata: bool, verbose: bool,
                                 prefer_local: bool = True) -> Dict[str, str]:
    """
    Process a single metadata file and save spectrograms if conditions are met.

    Args:
        prefer_local: If True, prefer local audio files over server files
    """
    # Check if metadata file exists
    if not os.path.exists(metadata_file_path):
        return {'status': 'failed', 'reason': 'Metadata file not found'}

    # Load and validate metadata
    metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)

    # Determine bird folder for path mapping
    filename_info = parse_audio_filename(metadata_file_path)
    bird_folder = None
    if filename_info['success']:
        bird_folder = os.path.join(save_path, filename_info['bird'])

    # Resolve audio file path
    audio_file_path, wseg_offset = resolve_audio_file_path(
        metadata_file_path, metadata_matfile, read_songpath_from_metadata,
        bird_folder, prefer_local
    )

    if audio_file_path is None:
        return {'status': 'failed', 'reason': 'Audio file not found'}

    # Update audio paths mapping if we have a bird folder and server path
    if bird_folder and not prefer_local:
        try:
            filename = os.path.basename(audio_file_path)
            update_audio_paths_file(bird_folder, filename, server_path=audio_file_path)
        except Exception as e:
            logging.debug(f"Could not update audio paths file: {e}")

    # Load and validate metadata arrays
    metadata = load_and_validate_metadata(metadata_file_path, wseg_offset)
    if metadata['error']:
        return {'status': 'failed', 'reason': metadata['error']}

    if not metadata['is_valid_song']:
        return {'status': 'skipped', 'reason': 'Single syllable file'}

    # Parse filename for output path
    filename_info = parse_audio_filename(audio_file_path)
    if not filename_info['success']:
        return {'status': 'failed', 'reason': 'Could not parse filename'}

    # Check song duration meets minimum threshold
    total_duration_ms = metadata['offsets'][-1] - metadata['onsets'][0]
    if total_duration_ms <= 2000:  # 2 seconds in milliseconds
        return {'status': 'skipped', 'reason': 'Song too short (< 2 seconds)'}

    # Create output path and check if already exists
    output_path = create_output_path(save_path, filename_info)
    if os.path.exists(output_path):
        return {'status': 'skipped', 'reason': 'Output file already exists'}

    # Process the audio
    try:
        success = process_and_save_audio(
            audio_file_path, output_path, metadata, params, verbose
        )

        if success:
            return {'status': 'processed', 'reason': 'Successfully saved'}
        else:
            return {'status': 'failed', 'reason': 'Audio processing failed'}

    except Exception as e:
        logging.error(f"Error processing audio for {metadata_file_path}: {e}")
        return {'status': 'failed', 'reason': f'Processing error: {str(e)}'}


def save_data_specs(metadata_file_paths: List[str], save_path: str, params: SpectrogramParams,
                    verbose: bool = False, read_songpath_from_metadata: bool = True,
                    prefer_local: bool = True) -> Dict[str, List[str]]:
    """
    Process metadata files and save spectrograms to HDF5 files.

    Args:
        prefer_local: If True, prefer local audio files over server files
    """
    results = {
        'processed': [],
        'skipped': [],
        'failed': []
    }

    for metadata_file_path in tqdm(metadata_file_paths, desc="Processing audio files"):
        try:
            result = process_single_metadata_file(
                metadata_file_path, save_path, params,
                read_songpath_from_metadata, verbose, prefer_local
            )
            results[result['status']].append(metadata_file_path)

        except Exception as e:
            logging.error(f"Unexpected error processing {metadata_file_path}: {e}")
            results['failed'].append(metadata_file_path)

    # Log summary
    logging.info(f"Processing complete: {len(results['processed'])} processed, "
                 f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")

    return results

def save_specs_for_evsonganaly_birds(metadata_file_paths: dict, save_path: str = None,
                                     songs_per_bird: int = 5, params: 'SpectrogramParams' = None,
                                     verbose: bool = False, prefer_local: bool = True):
    if params is None:
        params = SpectrogramParams()

    if save_path is None:
        raise ValueError("save_path cannot be None")

    birds = list(metadata_file_paths.keys())

    for bird in tqdm(birds, desc="Processing birds"):
        syllables_dir = os.path.join(save_path, bird, 'data', 'syllables')

        if os.path.isdir(syllables_dir):
            already_saved_files = os.listdir(syllables_dir)
        else:
            already_saved_files = []

        needed_count = songs_per_bird - len(already_saved_files)

        if needed_count <= 0:
            print(f'{bird} already processed, skipping...')
            continue

        candidate_file_paths = select_new_files(
            metadata_file_paths[bird],
            already_saved_files,
            needed_count
        )

        if candidate_file_paths:
            save_data_specs(
                metadata_file_paths=candidate_file_paths,
                save_path=save_path,
                params=params,
                verbose=verbose,
                read_songpath_from_metadata=False,
                prefer_local=prefer_local
            )

        else:
            logging.warning(f"No new files found for {bird}")


def save_specs_for_wseg_birds(metadata_file_paths: Dict[str, List[str]],
                              save_path: str,
                              songs_per_bird: int = 20,
                              params: SpectrogramParams = None,
                              verbose: bool = False, prefer_local: bool = True):
    """
    Process WhisperSeg files for multiple birds using the new modular pipeline.
    """
    if params is None:
        params = SpectrogramParams()

    birds = list(metadata_file_paths.keys())

    for bird in tqdm(birds, desc="Processing birds"):
        syllables_dir = os.path.join(save_path, bird, 'data', 'syllables')

        if os.path.isdir(syllables_dir):
            already_saved_files = os.listdir(syllables_dir)
        else:
            already_saved_files = []

        needed_count = songs_per_bird - len(already_saved_files)

        if needed_count <= 0:
            print(f'{bird} already processed, skipping...')
            continue

        candidate_file_paths = select_new_files(
            metadata_file_paths[bird],
            already_saved_files,
            needed_count
        )

        if candidate_file_paths:
            save_data_specs(
                metadata_file_paths=candidate_file_paths,
                save_path=save_path,
                params=params,
                verbose=verbose,
                read_songpath_from_metadata=True,
                prefer_local=prefer_local
            )
        else:
            logging.warning(f"No new files found for {bird}")


def main():
    path_to_macaw = check_sys_for_macaw_root()
    optimize_pytables_for_network()

    evsong_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
    evsong_directory = os.path.join(path_to_macaw, 'ssharma', 'RNA_seq', 'family_analysis_labeled', 'or-or')

    # Get file paths and copy locally in one step
    metadata_file_paths, wav_file_paths = filepaths_from_evsonganaly(
        wav_directory=evsong_directory,
        save_path=evsong_test_directory,
        batch_file_naming='batch.txt.labeled',
        copy_locally=True,  # Copy files as we discover them
        bird_subset=['or16or22', 'or18or24']
    )

    # Process spectrograms using local files
    save_specs_for_evsonganaly_birds(
        metadata_file_paths=metadata_file_paths,
        save_path=evsong_test_directory,
        songs_per_bird=15,
        prefer_local=True  # Use local copies
    )

    wseg_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'wseg test')
    wseg_directory = os.path.join(path_to_macaw, 'annietaylor', 'bubu-rdyw', 'metadata')

    # Get wseg paths and copy locally
    wseg_metadata_paths = filepaths_from_wseg(
        seg_directory=wseg_directory,
        save_path=wseg_test_directory,
        song_or_call='song',
        bird_subset=['bu68bu81', 'bu85bu97'],
        copy_locally=True
    )

    # Process spectrograms using local files
    save_specs_for_wseg_birds(
        metadata_file_paths=wseg_metadata_paths,
        save_path=wseg_test_directory,
        songs_per_bird=15,
        prefer_local=True
    )

if __name__ == '__main__':
    main()