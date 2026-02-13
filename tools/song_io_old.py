# import h5py

import matplotlib.pyplot as plt
from numpy import floating

plt.switch_backend('agg')
import os
import warnings
import numpy as np
import scipy
import hashlib
import gc
import tables
import scipy.signal as signal
import logging

from sys import platform
from tqdm import tqdm
from scipy.io import loadmat
from scipy.signal import get_window, ShortTimeFFT
from scipy.ndimage import zoom
from typing import List, Optional, Tuple, Dict, Any

from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file
from tools.signal_utils import smooth, butter_bandpass_filter_sos
from tools.system_utils import fix_mixture_of_separators
# from Z_add_syllable_database import fix_mixture_of_separators

import pyfftw.interfaces.scipy_fft
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union


# ----------------------------------------------------------------------
# SETUP LOGGING
# ----------------------------------------------------------------------

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup centralized logging for song processing."""
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('song_processing')
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    log_file = os.path.join(log_dir, f"song_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger for this module
logger = setup_logging()


@dataclass
class ProcessingResult:
    """Standardized return type for processing operations."""
    specs: List[np.ndarray]
    waveforms: List[np.ndarray]
    spec_times: List[np.ndarray]
    valid_indices: List[int]
    tempos: Optional[Tuple[float, float, float]] = None


@dataclass
class ProcessingError:
    """Information about processing failures."""
    file_path: str
    error_type: str
    error_message: str
    timestamp: datetime


# Set globally when module is imported
scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------

EPSILON = 1e-9  # small constant to avoid log(0)
# Tempo estimation parameters
PEAK_HEIGHT_THRESHOLD = 5e33
LOW_FREQ_THRESHOLD = 3.0
FREQ_RANGE_MIN = 0.0
FREQ_RANGE_MAX = 30.0


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def parse_filename_info(filepath: str) -> Dict[str, str]:
    """Extract bird, day, and time information from filepath."""
    try:
        if platform == 'win32':
            filename = filepath.split('\\')[-1]
        else:
            filename = filepath.split('/')[-1]

        split_filename = filename.split('.')
        if len(split_filename) > 4:
            bird = split_filename[0]
            daytime = split_filename[-1]
            logger.warning(f'Unusual filename format detected: {filename}')

        split_filename = filename.split('_')
        if len(split_filename) == 3:
            bird, day, time = split_filename[0:3]
            time = time.split('.')[0]
        elif len(split_filename) == 2:
            bird, daytime = filename.split('_')[0:2]
            day = daytime.split('.')[0][0:8]
            time = daytime.split('.')[0][8:]
        elif len(split_filename) == 4:
            bird, _, day, time = split_filename[0:4]
            time = time.split('.')[0]
        else:
            raise ValueError(f'Unrecognized filename format: {filename}')

        return {'bird': bird, 'day': day, 'time': time, 'filename': filename}

    except Exception as e:
        logger.error(f'Failed to parse filename {filepath}: {e}')
        raise


def create_output_paths(save_path: str, bird: str) -> Dict[str, str]:
    """Create and return standardized output directory paths."""
    paths = {
        'bird_dir': os.path.join(save_path, bird),
        'data_dir': os.path.join(save_path, bird, 'data'),
        'syllables_dir': os.path.join(save_path, bird, 'data', 'syllables')
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def generate_syllable_hashes(h5file_save_path: str, valid_indices: List[int]) -> List[str]:
    """Generate consistent hash IDs for syllables."""
    syl_strs = [f'{h5file_save_path}_{i}' for i in valid_indices]
    hash_obj = hashlib.sha256()
    hashes = []
    for syl_str in syl_strs:
        hash_obj.update(syl_str.encode('utf-8'))
        hash_id = hash_obj.hexdigest()
        hashes.append(hash_id)
    return hashes


def should_process_file(h5file_path: str, params: SpectrogramParams) -> bool:
    """Determine if a file should be processed based on existence and overwrite settings."""
    if not os.path.exists(h5file_path):
        return True
    if params.overwrite_existing:
        logger.info(f"Overwriting existing file: {h5file_path}")
        return True

    logger.info(f"Skipping existing file: {h5file_path}")
    return False


# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------


def process_single_file(metadata_file_path: str, save_path: str, params: SpectrogramParams,
                        read_songpath_from_metadata: bool = True,
                        slice_mode: bool = False) -> Optional[ProcessingError]:
    """
    Process a single metadata file and save the results.

    Returns ProcessingError if processing fails, None if successful.
    """
    try:
        # Read metadata
        if not os.path.exists(metadata_file_path):
            error_msg = f"Metadata file not found: {metadata_file_path}"
            logger.error(error_msg)
            return ProcessingError(metadata_file_path, "FileNotFound", error_msg, datetime.now())

        song_file_path, fs, syl_onsets, syl_offsets, labels = read_metadata(
            metadata_file_path, read_songpath_from_metadata
        )

        # Validate song file exists
        if not os.path.exists(song_file_path):
            error_msg = f"Audio file not found: {song_file_path}"
            logger.error(error_msg)
            return ProcessingError(song_file_path, "AudioFileNotFound", error_msg, datetime.now())

        # Parse filename information
        file_info = parse_filename_info(song_file_path)

        # Create output paths
        paths = create_output_paths(save_path, file_info['bird'])
        h5file_save_path = os.path.join(
            paths['syllables_dir'],
            f"syllables_{file_info['bird']}_{file_info['day']}_{file_info['time']}.h5"
        )

        # Check if we should process this file
        if not should_process_file(h5file_save_path, params):
            return None

        # Validate song data
        if not _validate_song_data(syl_onsets, syl_offsets, labels):
            logger.warning(f"Invalid or insufficient song data in {metadata_file_path}")
            return None

        # Process based on mode
        if slice_mode:
            return _process_slice_mode(song_file_path, syl_onsets, syl_offsets, labels,
                                       h5file_save_path, params)
        else:
            return _process_syllable_mode(song_file_path, syl_onsets, syl_offsets, labels,
                                          h5file_save_path, params)

    except Exception as e:
        error_msg = f"Unexpected error processing {metadata_file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ProcessingError(metadata_file_path, "ProcessingError", error_msg, datetime.now())


def _validate_song_data(syl_onsets, syl_offsets, labels) -> bool:
    """Validate that song data is suitable for processing."""
    try:
        assert len(labels) == len(syl_offsets)
        if len(syl_onsets) == 1:
            return False
        if syl_offsets[-1] - syl_onsets[0] <= 2:  # Less than 2 seconds
            return False
        return True
    except (TypeError, AssertionError):
        return False


def _process_syllable_mode(song_file_path: str, syl_onsets: np.ndarray, syl_offsets: np.ndarray,
                           labels: np.ndarray, h5file_save_path: str,
                           params: SpectrogramParams) -> Optional[ProcessingError]:
    """Process file in syllable detection mode."""
    try:
        # Handle long syllables by splitting them
        syl_lengths = syl_offsets - syl_onsets
        long_mask = syl_lengths > params.max_dur * 1000

        if np.any(long_mask):
            syl_onsets, syl_offsets, labels = _split_long_syllables(
                syl_onsets, syl_offsets, labels, long_mask, params
            )

        # Extract spectrograms
        result = get_song_specs(song_file_path, syl_onsets, syl_offsets, params=params)
        specs, wavs, ts, valid_inds = result[:4]  # Handle variable return length

        if len(valid_inds) == 0:
            logger.warning(f"No valid syllables found in {song_file_path}")
            return None

        # Prepare data for saving
        segmented_data = _prepare_syllable_data(specs, wavs, ts, valid_inds,
                                                syl_onsets, syl_offsets, labels, h5file_save_path)

        # Save data
        save_segmented_audio_data(h5file_save_path, song_file_path, segmented_data)
        logger.info(f"Successfully processed syllable file: {song_file_path}")

        return None  # Success

    except Exception as e:
        error_msg = f"Error in syllable mode processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ProcessingError(song_file_path, "SyllableProcessingError", error_msg, datetime.now())


def _process_slice_mode(song_file_path: str, syl_onsets: np.ndarray, syl_offsets: np.ndarray,
                        labels: np.ndarray, h5file_save_path: str,
                        params: SpectrogramParams) -> Optional[ProcessingError]:
    """Process file in slice mode."""
    try:
        if params.slice_length is None:
            raise ValueError("slice_length must be specified for slice mode")

        # Create slices and labels
        slice_onsets, slice_offsets, slice_labels, mapping = label_slices(
            onsets=syl_onsets, offsets=syl_offsets, labels=labels,
            time_bin_size=params.slice_length
        )

        # Extract spectrograms for slices
        result = get_song_specs(song_file_path, slice_onsets, slice_offsets, params=params)
        specs, audio_segs, spec_t, valid_inds = result[:4]

        if len(valid_inds) == 0:
            logger.warning(f"No valid slices found in {song_file_path}")
            return None

        # Generate hashes
        hashes = generate_syllable_hashes(h5file_save_path, valid_inds)

        # Save slice data
        _save_slice_data(h5file_save_path, song_file_path, specs, audio_segs, spec_t,
                         slice_onsets, slice_offsets, slice_labels, syl_onsets, syl_offsets,
                         valid_inds, hashes)

        logger.info(f"Successfully processed slice file: {song_file_path}")
        return None  # Success

    except Exception as e:
        error_msg = f"Error in slice mode processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ProcessingError(song_file_path, "SliceProcessingError", error_msg, datetime.now())


def _split_long_syllables(syl_onsets: np.ndarray, syl_offsets: np.ndarray, labels: np.ndarray,
                          long_mask: np.ndarray, params: SpectrogramParams) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Split syllables that exceed max_dur into smaller segments."""
    syl_lengths = syl_offsets - syl_onsets

    # Calculate new array sizes
    n_sub_syls = (syl_lengths[long_mask] // (params.max_dur * 1000)).astype(int)
    n_specs = len(syl_onsets) + np.sum(n_sub_syls)

    # Initialize new arrays
    new_onsets = np.zeros(n_specs)
    new_offsets = np.zeros(n_specs)
    new_labels = np.empty(n_specs, dtype=str)

    # Track where we're inserting new values
    current_idx = 0

    # Process all syllables in order
    for i in range(len(syl_onsets)):
        if not long_mask[i]:
            # Copy short syllables directly
            new_onsets[current_idx] = syl_onsets[i]
            new_offsets[current_idx] = syl_offsets[i]
            new_labels[current_idx] = labels[i]
            current_idx += 1
        else:
            # Split long syllables
            n_splits = n_sub_syls[np.sum(long_mask[:i])]
            syl_dur = syl_lengths[i]
            end_dur = syl_dur - (params.max_dur * 1000 * n_splits)

            # Add full-length segments
            for n in range(n_splits):
                new_onsets[current_idx] = syl_onsets[i] + params.max_dur * 1000 * n
                new_offsets[current_idx] = syl_onsets[i] + params.max_dur * 1000 * (n + 1)
                new_labels[current_idx] = labels[i]
                current_idx += 1

            # Add final segment
            new_onsets[current_idx] = syl_offsets[i] - end_dur
            new_offsets[current_idx] = syl_offsets[i]
            new_labels[current_idx] = labels[i]
            current_idx += 1

    return new_onsets, new_offsets, new_labels


def _prepare_syllable_data(specs: List[np.ndarray], wavs: List[np.ndarray], ts: List[np.ndarray],
                           valid_inds: List[int], syl_onsets: np.ndarray, syl_offsets: np.ndarray,
                           labels: np.ndarray, h5file_save_path: str) -> Dict[str, Any]:
    """Prepare syllable data for saving to HDF5."""
    # Extract valid data
    onsets = [syl_onsets[i] for i in valid_inds]
    offsets = [syl_offsets[i] for i in valid_inds]
    valid_labels = [labels[i] for i in valid_inds]

    # Generate hashes
    hashes = generate_syllable_hashes(h5file_save_path, valid_inds)

    # Pad waveforms to same length
    padded_waveforms, padded_ts = _pad_waveforms(wavs, ts)

    return {
        'spectrograms': specs,
        'waveforms': padded_waveforms,
        'spec_t': padded_ts,
        'manual': valid_labels,
        'onsets': onsets,
        'offsets': offsets,
        'position_idxs': valid_inds,
        'hashes': hashes
    }


def _pad_waveforms(wavs: List[np.ndarray], ts: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Pad waveforms and time arrays to same length with NaNs."""
    if not wavs:
        return [], []

    max_l = max(len(wav) for wav in wavs)
    padded_waveforms = []
    padded_ts = []

    for wav, t in zip(wavs, ts):
        if len(wav) != max_l:
            padded_wav = np.full(max_l, np.nan)
            padded_t = np.full(max_l, np.nan)
            padded_wav[:len(wav)] = wav
            padded_t[:len(t)] = t
        else:
            padded_wav = wav
            padded_t = t

        padded_waveforms.append(padded_wav)
        padded_ts.append(padded_t)

    return padded_waveforms, padded_ts


def _save_slice_data(h5file_save_path: str, song_file_path: str, specs: List[np.ndarray],
                     audio_segs: List[np.ndarray], spec_t: List[np.ndarray],
                     slice_onsets: np.ndarray, slice_offsets: np.ndarray, slice_labels: np.ndarray,
                     syl_onsets: np.ndarray, syl_offsets: np.ndarray,
                     valid_inds: List[int], hashes: List[str]) -> None:
    """Save slice data to HDF5 file."""
    try:
        with tables.open_file(h5file_save_path, "w") as f:
            f.create_array(f.root, 'audio_filename', np.array([song_file_path]).astype('S'))
            f.create_array(f.root, 'spectrograms', np.array(specs))
            f.create_array(f.root, 'offsets', np.array(slice_offsets))
            f.create_array(f.root, 'onsets', np.array(slice_onsets))
            f.create_array(f.root, 'manual', np.array(slice_labels[valid_inds]))
            f.create_array(f.root, 'hashes', np.array(hashes).astype('S'))
            f.create_array(f.root, 'syl_offsets', np.array(syl_offsets))
            f.create_array(f.root, 'syl_onsets', np.array(syl_onsets))
            f.create_array(f.root, 'waveforms', np.array(audio_segs))
            f.create_array(f.root, 'spec_t', np.array(spec_t))
            f.create_array(f.root, 'position_idxs', np.array(valid_inds))
            f.flush()
        logger.info(f"Successfully saved slice data to {h5file_save_path}")
    except Exception as e:
        logger.error(f"Failed to save slice data to {h5file_save_path}: {e}")
        raise


def save_data_specs(metadata_file_paths: List[str], save_path: str, params: SpectrogramParams,
                    verbose: bool = False, read_songpath_from_metadata: bool = True) -> None:
    """
    Process syllable-based spectrograms from metadata files.

    Maintains backward compatibility with existing interface.
    """
    logger.info(f"Starting syllable processing for {len(metadata_file_paths)} files")

    problem_files = []

    for metadata_file_path in tqdm(metadata_file_paths, "Processing syllable spectrograms..."):
        error = process_single_file(
            metadata_file_path=metadata_file_path,
            save_path=save_path,
            params=params,
            read_songpath_from_metadata=read_songpath_from_metadata,
            slice_mode=False
        )

        if error:
            problem_files.append(error)
            if verbose:
                print(f"Problem with {error.file_path}: {error.error_message}")

    # Log summary
    logger.info(f"Completed syllable processing. Success: {len(metadata_file_paths) - len(problem_files)}, "
                f"Failures: {len(problem_files)}")

    if problem_files:
        logger.warning(f"Failed files: {[err.file_path for err in problem_files]}")
        if verbose:
            for error in problem_files:
                print(f"Error in {error.file_path} ({error.error_type}): {error.error_message}")


def save_spec_slices(metadata_file_paths: List[str], save_path: str, params: SpectrogramParams,
                     slice_length: Optional[float] = None, read_songpath_from_metadata: bool = True,
                     verbose: bool = False) -> None:
    """
    Process fixed-length slice spectrograms from metadata files.

    Maintains backward compatibility with existing interface.
    """
    # Set slice_length in params if provided
    if slice_length is not None:
        params.slice_length = slice_length

    if params.slice_length is None:
        raise ValueError("slice_length must be specified either in params or as argument")

    logger.info(f"Starting slice processing for {len(metadata_file_paths)} files "
                f"with slice_length={params.slice_length}ms")

    problem_files = []

    for metadata_file_path in tqdm(metadata_file_paths, 'Processing slice spectrograms...'):
        error = process_single_file(
            metadata_file_path=metadata_file_path,
            save_path=save_path,
            params=params,
            read_songpath_from_metadata=read_songpath_from_metadata,
            slice_mode=True
        )

        if error:
            problem_files.append(error)
            if verbose:
                print(f"Problem with {error.file_path}: {error.error_message}")

        # Memory cleanup after each file
        gc.collect()

    # Log summary
    logger.info(f"Completed slice processing. Success: {len(metadata_file_paths) - len(problem_files)}, "
                f"Failures: {len(problem_files)}")

    if problem_files:
        logger.warning(f"Failed files: {[err.file_path for err in problem_files]}")
        if verbose:
            for error in problem_files:
                print(f"Error in {error.file_path} ({error.error_type}): {error.error_message}")


def verify_save(filepath):
    with tables.open_file(filepath, mode='r') as f:
        specs = f.root.spectrograms.read()
        print(f"Verification - Contains NaN: {np.isnan(specs).any()}")
        print(f"Verification - Data shape: {specs.shape}")
        return specs


def save_segmented_audio_data(h5file_save_path, song_file_path, segmented_audio_data):
    spec_data = np.array(segmented_audio_data['spectrograms'], dtype=np.float64)
    print(f"Min value: {np.min(spec_data)}")
    print(f"Max value: {np.max(spec_data)}")
    print(f"Mean value: {np.mean(spec_data)}")
    print(f"Data shape: {spec_data.shape}")
    if np.isnan(spec_data).any():
        print("Warning: NaN values detected before saving")
    atom = tables.Float64Atom()
    with tables.open_file(h5file_save_path, mode='w') as h5file:
        h5file.create_array(h5file.root, 'audio_filename',
                            obj=np.array([song_file_path], dtype=np.str_))  # Save as string array
        h5file.create_array(h5file.root, 'waveforms', obj=np.array(segmented_audio_data['waveforms']))
        h5file.create_array(h5file.root, 'spectrograms', obj=np.array(segmented_audio_data['spectrograms'],
                                                                      dtype=np.float64), atom=atom)
        h5file.create_array(h5file.root, 'spec_t', obj=np.array(segmented_audio_data['spec_t']))
        h5file.create_array(h5file.root, 'manual',
                            obj=np.array(segmented_audio_data['manual'], dtype=np.str_))  # Save as string array
        h5file.create_array(h5file.root, 'onsets', obj=np.array(segmented_audio_data['onsets']))
        h5file.create_array(h5file.root, 'offsets', obj=np.array(segmented_audio_data['offsets']))
        h5file.create_array(h5file.root, 'position_idxs', obj=np.array(segmented_audio_data['position_idxs']))
        h5file.create_array(h5file.root, 'hashes', obj=np.array(segmented_audio_data['hashes'], dtype=np.str_))
        saved_data = h5file.root.spectrograms.read()
        if np.isnan(saved_data).any():
            print("Warning: NaN values detected after saving")
            print(f"Original data contains NaN: {np.isnan(spec_data).any()}")
        h5file.flush()
    spectrograms = verify_save(h5file_save_path)
    assert np.array_equal(spectrograms, segmented_audio_data['spectrograms'])
    return


def rms_norm(array):
    """
    RMS normalization with safe dtype handling
    """
    # Convert to float64 first
    array_f64 = array.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(array_f64)))
    return array_f64 / rms


def tempo_estimates(audio_norm: np.ndarray, fs: int) -> Tuple[float, float, float]:
    """
    Estimate tempo characteristics from normalized audio signal.

    Cleaned up version with better variable names and structure.
    """
    try:
        # Rectify and smooth the audio signal
        rectified_audio = smooth(audio_norm ** 2)
        autocorr = np.correlate(rectified_audio, rectified_audio, 'same')

        # Compute power spectrum using FFT
        power_spectrum = np.abs(np.fft.rfft(autocorr)) ** 2
        frequencies = np.fft.rfftfreq(autocorr.size, 1 / fs)

        # Filter to relevant frequency range
        freq_mask = (frequencies >= FREQ_RANGE_MIN) & (frequencies <= FREQ_RANGE_MAX)
        filtered_power = power_spectrum[freq_mask]
        filtered_freqs = frequencies[freq_mask]

        # Find peaks in the power spectrum
        peak_indices = signal.find_peaks(filtered_power, height=PEAK_HEIGHT_THRESHOLD)[0]

        if len(peak_indices) == 0:
            logger.warning("No tempo peaks found in audio signal")
            return np.nan, np.nan, np.nan

        # Calculate peak characteristics
        peak_powers = filtered_power[peak_indices]
        peak_freqs = filtered_freqs[peak_indices]

        # Relative heights for top 3 peaks
        relative_heights = peak_powers / np.sum(peak_powers)
        top_3_indices = np.argsort(relative_heights)[-3:]

        # Low frequency peaks
        low_freq_mask = peak_freqs < LOW_FREQ_THRESHOLD
        low_freq_peaks = peak_freqs[low_freq_mask] if low_freq_mask.any() else np.array([])

        # Calculate tempo estimates
        mean_top_3 = np.mean(peak_freqs[top_3_indices]) if len(top_3_indices) > 0 else np.nan
        low_freq_mean = np.mean(low_freq_peaks) if len(low_freq_peaks) > 0 else np.nan
        mean_all = np.mean(peak_freqs)

        return mean_top_3, low_freq_mean, mean_all

    except Exception as e:
        logger.error(f"Error in tempo estimation: {e}")
        return np.nan, np.nan, np.nan


def get_song_specs(audio_filename: str, onsets: np.ndarray, offsets: np.ndarray, params: SpectrogramParams,
                   split_syllables: bool = False, tempo: bool = False) -> ProcessingResult:
    """
    Extract spectrograms from audio file for given onset/offset pairs.

    Now returns a consistent ProcessingResult object instead of variable tuple length.
    """
    specs = []
    valid_inds = []
    audio_segs = []
    spec_t = []

    # Read audio file once
    try:
        audio, fs = read_audio_file(audio_filename)
    except Exception as e:
        logger.error(f"Failed to read audio file {audio_filename}: {e}")
        return ProcessingResult(specs, audio_segs, spec_t, valid_inds, None)

    # Process audio
    audio = rms_norm(audio)
    audio = butter_bandpass_filter_sos(audio, lowcut=params.min_freq, highcut=params.max_freq, fs=fs, order=5)

    # Calculate tempo if requested
    tempos = tempo_estimates(audio, fs) if tempo else None

    # Handle long syllables before processing
    if split_syllables and hasattr(params, 'max_dur') and params.max_dur:
        # Split long syllables
        max_dur_ms = params.max_dur * 1000
        processed_onsets, processed_offsets, syllable_to_original_mapping = split_long_syllables_with_mapping(
            onsets, offsets, max_dur_ms
        )
    else:
        # Filter out long syllables
        if hasattr(params, 'max_dur') and params.max_dur:
            max_dur_ms = params.max_dur * 1000
            durations = offsets - onsets
            valid_mask = durations <= max_dur_ms
            processed_onsets = onsets[valid_mask]
            processed_offsets = offsets[valid_mask]
            # Create mapping from processed index to original index
            syllable_to_original_mapping = np.where(valid_mask)[0]
        else:
            # No duration limit
            processed_onsets = onsets
            processed_offsets = offsets
            syllable_to_original_mapping = np.arange(len(onsets))

    # Process each syllable
    for i, (onset, offset) in enumerate(zip(processed_onsets, processed_offsets)):
        try:
            spec, audio_seg, t = get_song_spec(t1=onset / 1000, t2=offset / 1000, audio=audio, params=params, fs=fs)

            if not (np.max(spec) == np.max(audio_seg) == np.max(t) == 0.0):
                specs.append(spec)
                valid_inds.append(syllable_to_original_mapping[i])  # Map back to original indices
                audio_segs.append(audio_seg)
                spec_t.append(t)
        except Exception as e:
            logger.error(f"Failed to process syllable {i} (original {syllable_to_original_mapping[i]}) "
                         f"in {audio_filename}: {e}")

    return ProcessingResult(specs, audio_segs, spec_t, valid_inds, tempos)


def define_slice_on_off(onsets: np.ndarray, offsets: np.ndarray, time_bin_size: float):
    # create an array of time bin onsets and corresponding offsets
    slice_onsets = np.arange(onsets[0], offsets[-1], time_bin_size)
    slice_offsets = slice_onsets + time_bin_size
    return slice_onsets, slice_offsets


def label_slices(onsets: np.ndarray, offsets: np.ndarray, labels: np.ndarray, time_bin_size: float) -> (Tuple)[
    np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    if type(labels) is not list:
        labels = list(labels)
    # one way to simplify this... just to create the slices, without any syllable annotations!
    # can keep the original onsets and syllables in the new h5 file, do labelling at a later time (or at least in a
    # separate function...)

    slice_onsets, slice_offsets = define_slice_on_off(onsets, offsets, time_bin_size=time_bin_size)

    # create unique integer labels for each syllable/label
    uni_syls, _, int_labels = np.unique(labels, return_inverse=True, return_index=True)
    uni_ints = np.unique(int_labels)

    # create mapping of integer labels to original labels
    mapping = {inte: syl for inte, syl in zip(uni_ints, uni_syls)}

    # initialize arrays for slice indicators and labels (1 for presence, 0 for absence)
    slice_indicator = np.zeros(len(slice_onsets))
    slice_labels = np.full(len(slice_onsets), np.nan)

    # track previous indices and label to detect overlaps
    prev_idxs = np.array([100000])
    prev_label = None

    # create an inverse mapping to convert labels back to integers
    inv_mapping = {v: k for (v, k) in mapping.items()}

    # loop over each onset, offset, and label, assigning to time bins
    for onset, offset, label in zip(onsets, offsets, int_labels):
        # find which time bins overlap with the current segment
        #   either where the onset happens after syl onset but before the offset, or
        #   the onset happens before the onset, but the offset happens after
        bin_indices = np.where(((slice_onsets >= onset) & (slice_onsets < offset)) |
                               ((slice_onsets < onset) & (slice_offsets > onset)))[0]

        # handle potential overlap with the previous segment
        # first check for any overlap then...
        if len(prev_idxs) > len(bin_indices):
            # one strategy for aligning if prev_seg was longer than current
            prev_idxs = np.array(prev_idxs[-(len(bin_indices)):])
            overlap_flag = np.zeros(len(prev_idxs), dtype=bool)
            for idx in prev_idxs:
                overlap_flag = (idx == bin_indices) | overlap_flag
            overlap = prev_idxs[overlap_flag]
        else:
            # another strategy if current is longer than prev
            overlap_flag = np.zeros(len(prev_idxs), dtype=bool)
            for idx in bin_indices:
                overlap_flag = (idx == prev_idxs) | overlap_flag
            overlap = prev_idxs[overlap_flag]

        # assign the current label to the appropriate time bins
        slice_indicator[bin_indices] = 1
        slice_labels[bin_indices] = label

        # if there's an overlap, create a combined label
        if len(overlap) > 0:
            combined_label = mapping[prev_label] + mapping[label]
            if combined_label not in mapping.values():
                # create a new label if the combined label doesn't exist
                new_label = max(mapping.keys()) + 1
                mapping[new_label] = combined_label
                inv_mapping = {v: k for (k, v) in mapping.items()}
                ov_label = new_label
            else:
                # otherwise, reuse the existing label
                ov_label = inv_mapping[combined_label]
            slice_labels[overlap] = ov_label

        # update the previous indices and label for the next iteration
        prev_idxs = bin_indices
        prev_label = label

    # assign the label 'gap' for slices with no label
    slice_labels[np.isnan(slice_labels)] = -1
    mapping[-1] = 'gap'

    return slice_onsets, slice_offsets, slice_labels, mapping


def downsample_spec(spec: np.ndarray, params) -> np.ndarray:
    """Downsample spectrogram time axis to target shape."""
    target_shape = params.target_shape
    current_shape = spec.shape

    # Should only need to downsample time axis
    assert current_shape[0] == target_shape[0], f"Frequency bins mismatch: {current_shape[0]} vs {target_shape[0]}"

    # Only downsample time axis
    zoom_factors = (1.0, target_shape[1] / current_shape[1])

    downsampled_spec = zoom(spec, zoom_factors, order=1)  # bilinear is fine for time axis
    return downsampled_spec


def get_song_spec(t1: float, t2: float, audio: np.ndarray, params: SpectrogramParams, fs: int = 32000,
                  fill_value: float = -1 / EPSILON, downsample: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and normalize spectrogram from a birdsong syllable segment.

    Processes a time segment of audio to create a standardized spectrogram suitable
    for birdsong analysis and classification. Applies robust normalization to handle
    varying recording conditions across different birds and recording sessions.

    Parameters
    ----------
    t1 : float
        Start time of syllable segment in seconds.
    t2 : float
        End time of syllable segment in seconds.
    audio : np.ndarray
        Full audio signal array (1D).
    params : SpectrogramParams
        Spectrogram configuration parameters including STFT settings, target shape,
        frequency range, and normalization parameters.
    fs : int, optional
        Sampling frequency in Hz, by default 32000.
    fill_value : float, optional
        Value used for padding spectrograms to target duration, by default -1/EPSILON.
    downsample : bool, optional
        Whether to downsample to target_shape for memory efficiency, by default True.

    Returns
    -------
    spec : np.ndarray
        Normalized spectrogram array of shape target_shape if downsampled,
        or (nfft//2 + 1, n_time_bins) if not downsampled.
    audio_segment : np.ndarray
        Extracted audio segment corresponding to the time window.
    t : np.ndarray
        Time array for the spectrogram frames, adjusted to start at t1.

    Notes
    -----
    The normalization pipeline consists of:
    1. STFT computation with Hann window
    2. Log magnitude spectrogram
    3. Percentile-based normalization (5th-95th percentile) for robustness
    4. Padding/truncation to standardized duration
    5. Optional downsampling for computational efficiency

    Warnings are issued if syllable duration exceeds max_dur. Invalid segments
    (outside audio bounds) return arrays filled with fill_value.

    Examples
    --------
    >>> spec, audio_seg, time = get_song_spec(1.0, 1.2, audio_data, params)
    >>> print(spec.shape)
    (257, 320)
    """

    # warn if the segment duration exceeds max_dur
    if t2 - t1 > params.max_dur + 1e-4:
        warnings.warn(f"Segment longer than max_dur: {t2 - t1}s, max_dur = {params.max_dur}s")

    # convert onset and offset times to sample indices
    s1, s2 = int(round(t1 * fs)), int(round(t2 * fs))
    assert s1 < s2, f"s1: {s1}, s2: {s2}, t1: {t1}, t2: {t2}"

    # if the segment is too small or invalid, return empty spectrogram
    if (s1 < 0) or (s2 >= len(audio)):
        return (np.full(params.target_shape, fill_value), np.zeros(s2 - s1), np.array([]))
    else:
        # extract the audio segment, and remove dc offset if necessary
        audio_segment = audio[max(0, s1):min(len(audio), s2)]
        w = get_window('hann', Nx=params.nfft)
        STFT = ShortTimeFFT(w, hop=params.hop, fs=fs)
        Sx = STFT.stft(audio_segment)
        t = STFT.t(len(audio_segment))
        non_negative_time_indices = t >= 0
        non_negative_time_indices[-int(params.nfft / 2):] = False
        t = t[non_negative_time_indices]
        spec = np.log(abs(Sx[:, non_negative_time_indices]))
        t += max(0, t1)  # adjust time to start at t1

        p5, p95 = np.percentile(spec, [5, 95])  # before padding
        exp_spec = np.full((int(params.nfft / 2) + 1, int(np.ceil(params.max_dur / STFT.delta_t))), fill_value)
        if spec.shape[1] > exp_spec.shape[1]:
            logging.warning(f"Truncating spectrogram from {spec.shape[1]} to {exp_spec.shape[1]} frames. "
                            f"Syllable: {t2 - t1:.4f}s, max_dur: {params.max_dur}s")
            spec = spec[:, :exp_spec.shape[1]]
        exp_spec[:, :np.shape(spec)[1]] = spec[:, :]
        # Then normalize the whole thing
        exp_spec = (exp_spec - p5) / (p95 - p5)
        exp_spec = np.clip(exp_spec, 0, 1)

        if downsample:
            spec = downsample_spec(exp_spec, params)
        else:
            spec = exp_spec
        return spec, audio_segment, t


def read_metadata(metadata_file_path: str, read_songpath_from_metadata: bool) -> Tuple[
    str, float, np.ndarray, np.ndarray, np.ndarray]:
    metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)
    try:
        fs = metadata_matfile['Fs']
    except KeyError:
        fs = 32000.

    if not read_songpath_from_metadata:
        song_file_path = '.'.join(metadata_file_path.split('.')[0:2])
        wseg_offset = 0
    else:
        wseg_offset = (256 * 1000) / fs
        try:
            fname = metadata_matfile['fname']
        except KeyError:
            fname = metadata_matfile['fnamecell']
        fname = fname.replace('\\', '/') if platform == 'win32' else fname.replace('/', '\\')
        song_file_path = fix_mixture_of_separators(fname)

    syl_onsets = metadata_matfile['onsets'] + wseg_offset
    syl_offsets = metadata_matfile['offsets'] + wseg_offset
    labels = metadata_matfile['labels']
    return song_file_path, fs, syl_onsets, syl_offsets, labels


def save_spec_slices(metadata_file_paths: List[str], save_path: str, params: SpectrogramParams,
                     slice_length: Optional[float] = None, read_songpath_from_metadata: bool = True,
                     verbose: bool = False, ) -> None:
    problem_files = []

    for metadata_file_path in tqdm(metadata_file_paths, 'Saving slice spectrograms...'):
        if not os.path.exists(metadata_file_path):
            print(f"File not found: {metadata_file_path},")
            continue
        song_file_path, fs, syl_onsets, syl_offsets, labels = read_metadata(metadata_file_path,
                                                                            read_songpath_from_metadata)
        slice_onsets, slice_offsets, slice_labels, mapping = label_slices(onsets=syl_onsets, offsets=syl_offsets,
                                                                          labels=labels, time_bin_size=slice_length)
        specs, audio_segs, spec_t, valid_inds = get_song_specs(audio_filename=song_file_path, onsets=slice_onsets,
                                                               offsets=slice_offsets, params=params)
        # TODO there are instances here where the last syllable is truncated, not clear why
        try:
            if platform == 'win32':
                filename = song_file_path.split('\\')[-1]
            else:
                filename = song_file_path.split('/')[-1]
            [bird, day, time] = filename.split('_')[0:3]  # read bird, date, and time from filename
            time = time.split('.')[0]  # for time, have to separate from file extension
        except ValueError:
            # some files follow a different naming convention, detect this
            [bird, daytime] = filename.split('_')[0:2]
            day = daytime.split('.')[0][0:8]
            time = daytime.split('.')[0][8:]

        os.makedirs(os.path.join(save_path, bird), exist_ok=True)
        data_path = os.path.join(save_path, bird, 'data')
        os.makedirs(data_path, exist_ok=True)
        data_path = os.path.join(data_path, 'syllables')
        os.makedirs(data_path, exist_ok=True)
        h5file_save_path = os.path.join(data_path, f'syllables_{bird}_{day}_{time}.h5')
        syl_strs = [f'{h5file_save_path}_{i}' for i in valid_inds]
        hash_obj = hashlib.sha256()
        hashes = []
        for syl_str in syl_strs:
            hash_obj.update(syl_str.encode('utf-8'))
            hash_id = hash_obj.hexdigest()
            hashes.append(hash_id)
        try:
            with tables.open_file(h5file_save_path, "w") as f:
                f.create_array(f.root, 'audio_filename', np.array([song_file_path]).astype('S'))
                f.create_array(f.root, 'spectrograms', np.array(specs))
                f.create_array(f.root, 'offsets', np.array(slice_offsets))
                f.create_array(f.root, 'onsets', np.array(slice_onsets))
                f.create_array(f.root, 'manual', np.array(slice_labels[valid_inds]))
                f.create_array(f.root, 'hashes', np.array(hashes).astype('S'))
                f.create_array(f.root, 'syl_offsets', np.array(syl_offsets))
                f.create_array(f.root, 'syl_onsets', np.array(syl_onsets))
                f.create_array(f.root, 'waveforms', np.array(audio_segs))
                f.create_array(f.root, 'spec_t', np.array(spec_t))
                f.create_array(f.root, 'position_idxs', np.array(valid_inds))
            if verbose:
                print(f"Saved data to {data_path}")
        except Exception as e:
            print(f"Failed saving HDF5 for {metadata_file_path}: {e}")
            problem_files.append(metadata_file_path)
            continue
        finally:
            # clear memory
            del specs, slice_offsets, slice_labels, syl_offsets, syl_onsets, hashes, audio_segs, spec_t, valid_inds
            gc.collect()

    if problem_files and verbose:
        print(f"Problem files: {problem_files}")


if __name__ == '__main__':
    pass