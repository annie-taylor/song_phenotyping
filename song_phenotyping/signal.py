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
from datetime import datetime
import psutil
import re

from sys import platform
from tqdm import tqdm
from scipy.io import loadmat
from scipy.signal import get_window, ShortTimeFFT
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
from song_phenotyping.tools.audio_utils import read_audio_file
from song_phenotyping.tools.signal_utils import smooth, butter_bandpass_filter_sos
from song_phenotyping.tools.system_utils import fix_mixture_of_separators

import pyfftw.interfaces.scipy_fft

# Set globally when module is imported
scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)

EPSILON = 1e-9  # small constant to avoid log(0)
# Tempo estimation parameters
PEAK_HEIGHT_THRESHOLD = 2e14
LOW_FREQ_THRESHOLD = 3.0
FREQ_RANGE_MIN = 0.125
FREQ_RANGE_MAX = 30.0


# ============================================================================
# LOGGING AND UTILITY FUNCTIONS (consolidated from A_spec_saving.py)
# ============================================================================

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

    # Formatter with emoji support
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger for this module
logger = setup_logging()


def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024


@dataclass
class ProcessingResult:
    """Standardized return type for processing operations."""
    specs: List[np.ndarray]
    waveforms: List[np.ndarray]
    spec_times: List[np.ndarray]
    valid_indices: List[int]
    tempos: Optional[Tuple[float, float, float]] = None
    phase_features: Optional[List[dict]] = None


def parse_audio_filename(file_path: str) -> Dict[str, Any]:
    """
    Parse audio filename to extract bird, date, and time components.
    Handles multiple filename formats including wseg patterns.
    """
    try:
        if platform == 'win32':
            filename = file_path.split('\\')[-1]
        else:
            filename = file_path.split('/')[-1]

        # Remove .not.mat suffix if present (for wseg files)
        clean_filename = filename.replace('.not.mat', '')

        # Pattern 1: bk1bk3_170811_140945.wav (BIRD_YYMMDD_HHMMSS.wav) - actual timestamp
        pattern1 = r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)_(\d{6})_(\d{6})\.wav$'
        match1 = re.match(pattern1, clean_filename)
        if match1:
            bird, day, time = match1.groups()
            # Convert YYMMDD to full date format if needed
            if len(day) == 6:
                day = '20' + day  # Convert YY to 20YY
            return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

        # Pattern 2: bk1bk3.20081118-10.wav (BIRD.YYYYMMDD-SEQ.wav) - sequence number
        pattern2 = r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})-(\d+)\.wav$'
        match2 = re.match(pattern2, clean_filename)
        if match2:
            bird, day, seq = match2.groups()
            # Use sequence number as milliseconds offset from midnight to preserve order
            time = str(int(seq)).zfill(6)  # Convert to 6-digit string (microseconds)
            return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

        # Pattern 3: bk1bk3.20081118.wav (BIRD.YYYYMMDD.wav) - no sequence (index 0)
        pattern3 = r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})\.wav$'
        match3 = re.match(pattern3, clean_filename)
        if match3:
            bird, day = match3.groups()
            time = '000000'  # First song of the day (sequence 0)
            return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

        # Original patterns - keep for backward compatibility
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
            # Log the problematic filename for debugging
            logger.warning(f'🔍 Unrecognized filename format: {filename}')
            raise ValueError(f'Unrecognized filename format: {filename}')

        return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

    except Exception as e:
        logger.error(f'💥 Failed to parse filename {file_path}: {e}')
        return {'bird': None, 'day': None, 'time': None, 'filename': None, 'success': False}


def load_and_validate_metadata(metadata_file_path: str, wseg_offset: float = 0.0) -> Dict[str, Any]:
    """
    Load metadata from .mat file and validate required fields.
    Consolidated version from A_spec_saving.py
    """
    try:
        metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)

        # Extract sampling rate with fallback
        fs = metadata_matfile.get('Fs') or metadata_matfile.get('fs') or 32000.0

        # Extract required arrays
        raw_onsets = metadata_matfile.get('onsets')
        raw_offsets = metadata_matfile.get('offsets')
        labels = list(metadata_matfile.get('labels'))

        # Clean up metadata file from memory
        del metadata_matfile
        gc.collect()

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


def pad_waveforms_to_same_length(waveforms: List[np.ndarray],
                                 time_arrays: List[np.ndarray] = None,
                                 pad_value: float = np.nan) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Pad all waveforms (and optionally time arrays) to the same length.
    Consolidated from A_spec_saving.py
    """
    if not waveforms:
        return [], [] if time_arrays is not None else None

    # Validate inputs
    if time_arrays is not None and len(waveforms) != len(time_arrays):
        raise ValueError(f"Waveforms and time arrays must have same length: {len(waveforms)} vs {len(time_arrays)}")

    # Find maximum length
    max_length = max(len(wav) for wav in waveforms)

    # Pad waveforms
    padded_waveforms = []
    for wav in waveforms:
        if len(wav) == max_length:
            padded_waveforms.append(wav.copy())
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


def generate_syllable_hashes(base_identifier: str, indices: List[int]) -> List[str]:
    """
    Generate unique hash IDs for each syllable.
    Consolidated from A_spec_saving.py
    """
    hashes = []
    for idx in indices:
        unique_str = f"{base_identifier}_{idx}"
        hash_obj = hashlib.sha256()
        hash_obj.update(unique_str.encode('utf-8'))
        hashes.append(hash_obj.hexdigest())
    return hashes


def split_long_syllables_with_mapping(onsets: np.ndarray, offsets: np.ndarray,
                                      max_duration_ms: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split syllables longer than max_duration and create mapping to original indices.
    Consolidated from A_spec_saving.py
    """
    if len(onsets) == 0:
        return onsets.copy(), offsets.copy(), np.array([], dtype=int)

    new_onsets = []
    new_offsets = []
    mapping = []

    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        duration = offset - onset

        if duration <= max_duration_ms:
            new_onsets.append(onset)
            new_offsets.append(offset)
            mapping.append(i)
        else:
            # Split long syllable
            n_full_segments = int(duration // max_duration_ms)
            remainder = duration % max_duration_ms

            # Add full segments
            for seg in range(n_full_segments):
                seg_onset = onset + (seg * max_duration_ms)
                seg_offset = onset + ((seg + 1) * max_duration_ms)
                new_onsets.append(seg_onset)
                new_offsets.append(seg_offset)
                mapping.append(i)

            # Add remainder if it exists
            if remainder > 0:
                final_onset = onset + (n_full_segments * max_duration_ms)
                new_onsets.append(final_onset)
                new_offsets.append(offset)
                mapping.append(i)

    return np.array(new_onsets), np.array(new_offsets), np.array(mapping)


def create_output_paths(save_path: str, bird: str) -> Dict[str, str]:
    """Create and return standardized output directory paths."""
    from song_phenotyping.tools.pipeline_paths import SPECS_DIR, STAGES_DIR
    bird_dir = os.path.join(save_path, bird)
    paths = {
        'bird_dir':          bird_dir,
        'syllables_dir':     os.path.join(bird_dir, STAGES_DIR),
        'slices_dir':        os.path.join(bird_dir, STAGES_DIR),
        'syllable_specs_dir': os.path.join(bird_dir, SPECS_DIR),
        'slice_specs_dir':   os.path.join(bird_dir, SPECS_DIR),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


# ============================================================================
# CORE PROCESSING FUNCTIONS (updated versions)
# ============================================================================

def read_metadata(metadata_file_path: str, read_songpath_from_metadata: bool) -> Tuple[
    str, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read metadata and return song path, fs, onsets, offsets, labels.
    Updated to use consolidated utilities.
    """
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

        if platform == 'win32':
            fname = fname.replace('\\', '/')
            song_file_path = 'Z:\\' + '\\'.join(fname.split('/')[1:])
        else:
            fname = fname.replace('/', '\\')
            song_file_path = '/Volumes/users/' + '/'.join(fname.split('\\')[1:])

    syl_onsets = metadata_matfile['onsets'] + wseg_offset
    syl_offsets = metadata_matfile['offsets'] + wseg_offset
    labels = metadata_matfile['labels']

    # Clean up
    del metadata_matfile
    gc.collect()

    return song_file_path, fs, syl_onsets, syl_offsets, labels


def rms_norm(array):
    """RMS normalization with safe dtype handling"""
    array_f64 = array.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(array_f64)))
    return array_f64 / rms


def tempo_estimates(audio_norm: np.ndarray, fs: int) -> Tuple[float, float, float]:
    """
    Estimate tempo characteristics from normalized audio signal.
    Cleaned up version with better error handling.
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

        from matplotlib import pyplot as plt
        plt.plot(filtered_power)
        plt.show()

        # Find peaks in the power spectrum
        peak_indices = signal.find_peaks(filtered_power, height=PEAK_HEIGHT_THRESHOLD)[0]

        if len(peak_indices) == 0:
            logger.warning("🎵 No tempo peaks found in audio signal")
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
        logger.error(f"💥 Error in tempo estimation: {e}")
        return np.nan, np.nan, np.nan


def get_song_specs(audio_filename: str, onsets: np.ndarray, offsets: np.ndarray, params: SpectrogramParams,
                   split_syllables: bool = False, tempo: bool = False) -> ProcessingResult:
    """
    Extract spectrograms from audio file for given onset/offset pairs.
    Updated to return ProcessingResult and use consolidated utilities.
    """
    specs = []
    valid_inds = []
    audio_segs = []
    spec_t = []
    phase_features = []
    need_phase = getattr(params, 'save_inst_freq', False) or getattr(params, 'save_group_delay', False)

    # Read audio file once
    try:
        audio, fs = read_audio_file(audio_filename)
        logger.debug(f"🎵 Read audio file: {os.path.basename(audio_filename)}")
    except Exception as e:
        logger.error(f"💥 Failed to read audio file {audio_filename}: {e}")
        return ProcessingResult(specs, audio_segs, spec_t, valid_inds, None)

    # Process audio
    audio = rms_norm(audio)
    audio = butter_bandpass_filter_sos(audio, lowcut=params.min_freq, highcut=params.max_freq, fs=fs, order=3)

    # Calculate tempo if requested
    tempos = tempo_estimates(audio, fs) if tempo else None

    # Handle long syllables before processing
    if split_syllables and hasattr(params, 'max_dur') and params.max_dur:
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
            syllable_to_original_mapping = np.where(valid_mask)[0]
        else:
            processed_onsets = onsets
            processed_offsets = offsets
            syllable_to_original_mapping = np.arange(len(onsets))

    # Process each syllable
    for i, (onset, offset) in enumerate(zip(processed_onsets, processed_offsets)):
        try:
            result = get_song_spec(t1=onset / 1000, t2=offset / 1000, audio=audio, params=params, fs=fs)
            if need_phase:
                spec, phase_dict, audio_seg, t = result
            else:
                spec, audio_seg, t = result
                phase_dict = {}

            if not (np.max(spec) == np.max(audio_seg) == np.max(t) == 0.0):
                specs.append(spec)
                valid_inds.append(syllable_to_original_mapping[i])
                audio_segs.append(audio_seg)
                spec_t.append(t)
                if need_phase:
                    phase_features.append(phase_dict)
        except Exception as e:
            logger.error(f"💥 Failed to process syllable {i} (original {syllable_to_original_mapping[i]}) "
                        f"in {audio_filename}: {e}")

    return ProcessingResult(specs, audio_segs, spec_t, valid_inds, tempos,
                            phase_features if need_phase else None)


def save_spec_slices(candidate_files: List[Tuple[str, str]], save_path: str, params: SpectrogramParams,
                     slice_length: Optional[float] = None, read_songpath_from_metadata: bool = True,
                     verbose: bool = False, prefer_local: bool = True) -> None:
    """
    Process fixed-length slice spectrograms from metadata files.
    Updated to handle file pairs and prefer_local parameter.
    """
    # Set slice_length in params if provided
    if slice_length is not None:
        params.slice_length = slice_length

    if not hasattr(params, 'slice_length') or params.slice_length is None:
        raise ValueError("slice_length must be specified either in params or as argument")

    logger.info(f"🔪 Starting slice processing for {len(candidate_files)} files "
                f"with slice_length={params.slice_length}ms")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    problem_files = []
    processed_count = 0

    for idx, (metadata_file_path, audio_file_path) in enumerate(tqdm(candidate_files, '🔄 Processing slice spectrograms...'), 1):
        try:
            if not os.path.exists(metadata_file_path):
                logger.warning(f"📁 Metadata file not found: {metadata_file_path}")
                problem_files.append(metadata_file_path)
                continue

            # Handle audio file path resolution
            if read_songpath_from_metadata:
                # For wseg files, read from metadata but prefer provided audio_file_path
                song_file_path, fs, syl_onsets, syl_offsets, labels = read_metadata(
                    metadata_file_path, read_songpath_from_metadata
                )
                # Override with provided audio_file_path if it exists and prefer_local is True
                if audio_file_path and os.path.exists(audio_file_path) and prefer_local:
                    song_file_path = audio_file_path
            else:
                # For evsonganaly files, use provided audio_file_path directly
                song_file_path = audio_file_path
                # Still need to read metadata for onsets/offsets/labels
                metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)
                fs = metadata_matfile.get('Fs', 32000.0)
                syl_onsets = metadata_matfile['onsets']
                syl_offsets = metadata_matfile['offsets']
                labels = metadata_matfile['labels']
                del metadata_matfile
                gc.collect()

            if not os.path.exists(song_file_path):
                logger.warning(f"🎵 Audio file not found: {song_file_path}")
                problem_files.append(metadata_file_path)
                continue

            # Create slices and labels
            slice_onsets, slice_offsets, slice_labels, mapping = label_slices(
                onsets=syl_onsets, offsets=syl_offsets, labels=labels,
                time_bin_size=params.slice_length
            )

            # Extract spectrograms for slices
            result = get_song_specs(song_file_path, slice_onsets, slice_offsets, params=params)
            specs, audio_segs, spec_t, valid_inds = result.specs, result.waveforms, result.spec_times, result.valid_indices

            if not valid_inds:
                logger.warning(f"⚠️ No valid slices: {os.path.basename(metadata_file_path)}")
                problem_files.append(metadata_file_path)
                continue

            # Parse filename info
            file_info = parse_audio_filename(song_file_path)
            if not file_info['success']:
                logger.warning(f"📝 Could not parse filename: {song_file_path}")
                problem_files.append(metadata_file_path)
                continue

            # Create output path
            paths = create_output_paths(save_path, file_info['bird'])
            h5file_save_path = os.path.join(
                paths['slice_specs_dir'],
                f"slices_{file_info['bird']}_{file_info['day']}_{file_info['time']}.h5"
            )

            # Skip if file already exists
            if os.path.exists(h5file_save_path):
                logger.debug(f"⏭️ File already exists: {os.path.basename(h5file_save_path)}")
                continue

            # Generate hashes
            hashes = generate_syllable_hashes(h5file_save_path, valid_inds)

            # Save slice data
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

                processed_count += 1
                if verbose:
                    logger.info(f"✅ Saved {len(valid_inds)} slices: {os.path.basename(h5file_save_path)}")

            except Exception as e:
                logger.error(f"💾 Failed saving HDF5 for {metadata_file_path}: {e}")
                problem_files.append(metadata_file_path)
                continue

            # Clean up memory
            del specs, slice_offsets, slice_labels, syl_offsets, syl_onsets, hashes, audio_segs, spec_t, valid_inds
            gc.collect()

            # Periodic memory reporting
            if idx % 10 == 0:
                logger.info(f"📊 Progress {idx}/{len(candidate_files)}, "
                            f"memory: {get_memory_usage():.1f} MB")

        except Exception as e:
            logger.error(f"💥 Unexpected error processing {metadata_file_path}: {e}")
            problem_files.append(metadata_file_path)

    # Final summary
    logger.info(f"🎯 Slice processing complete:")
    logger.info(f"  ✅ Successfully processed: {processed_count}")
    logger.info(f"  ❌ Failed files: {len(problem_files)}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")

    if problem_files and verbose:
        logger.warning(f"Problem files: {[os.path.basename(f) for f in problem_files]}")


# ============================================================================
# EXISTING FUNCTIONS (keeping original signatures for compatibility)
# ============================================================================

def verify_save(filepath):
    """Verify HDF5 file was saved correctly"""
    with tables.open_file(filepath, mode='r') as f:
        specs = f.root.spectrograms.read()
        logger.debug(f"✅ Verification - Contains NaN: {np.isnan(specs).any()}")
        logger.debug(f"📊 Verification - Data shape: {specs.shape}")
    return specs


def save_segmented_audio_data(h5file_save_path, song_file_path, segmented_audio_data):
    """
    Save segmented audio data to HDF5 file.
    Updated with better logging and validation.
    """
    spec_data = np.array(segmented_audio_data['spectrograms'], dtype=np.float64)
    logger.debug(f"💾 Saving data: shape={spec_data.shape}, min={np.min(spec_data):.3f}, max={np.max(spec_data):.3f}")

    if np.isnan(spec_data).any():
        logger.warning("⚠️ NaN values detected before saving")

    atom = tables.Float64Atom()
    try:
        with tables.open_file(h5file_save_path, mode='w') as h5file:
            h5file.create_array(h5file.root, 'audio_filename', obj=np.array([song_file_path], dtype=np.str_))
            h5file.create_array(h5file.root, 'waveforms', obj=np.array(segmented_audio_data['waveforms']))
            h5file.create_array(h5file.root, 'spectrograms', obj=np.array(segmented_audio_data['spectrograms'],
                                                                          dtype=np.float64), atom=atom)
            h5file.create_array(h5file.root, 'spec_t', obj=np.array(segmented_audio_data['spec_t']))
            h5file.create_array(h5file.root, 'manual', obj=np.array(segmented_audio_data['manual'], dtype=np.str_))
            h5file.create_array(h5file.root, 'onsets', obj=np.array(segmented_audio_data['onsets']))
            h5file.create_array(h5file.root, 'offsets', obj=np.array(segmented_audio_data['offsets']))
            h5file.create_array(h5file.root, 'position_idxs', obj=np.array(segmented_audio_data['position_idxs']))
            h5file.create_array(h5file.root, 'hashes', obj=np.array(segmented_audio_data['hashes'], dtype=np.str_))
            # Duration metadata (always written when present)
            if 'durations' in segmented_audio_data and segmented_audio_data['durations'] is not None:
                h5file.create_array(h5file.root, 'durations',
                                    obj=np.array(segmented_audio_data['durations'], dtype=np.float64))
            # Optional phase-derived features
            if 'inst_freq' in segmented_audio_data and segmented_audio_data['inst_freq'] is not None:
                h5file.create_array(h5file.root, 'inst_freq',
                                    obj=np.array(segmented_audio_data['inst_freq'], dtype=np.float32))
            if 'group_delay' in segmented_audio_data and segmented_audio_data['group_delay'] is not None:
                h5file.create_array(h5file.root, 'group_delay',
                                    obj=np.array(segmented_audio_data['group_delay'], dtype=np.float32))
            h5file.flush()

        logger.debug(f"✅ Successfully saved to {os.path.basename(h5file_save_path)}")

        # Verify save
        spectrograms = verify_save(h5file_save_path)
        assert np.array_equal(spectrograms, segmented_audio_data['spectrograms'])

    except Exception as e:
        logger.error(f"💥 Failed to save {h5file_save_path}: {e}")
        raise


def define_slice_on_off(onsets: np.ndarray, offsets: np.ndarray, time_bin_size: float):
    """Create an array of time bin onsets and corresponding offsets"""
    slice_onsets = np.arange(onsets[0], offsets[-1], time_bin_size)
    slice_offsets = slice_onsets + time_bin_size
    return slice_onsets, slice_offsets


def label_slices(onsets: np.ndarray, offsets: np.ndarray, labels: np.ndarray, time_bin_size: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Label time slices based on syllable overlap.
    Keeping original implementation for compatibility.
    """
    if type(labels) is not list:
        labels = list(labels)

    slice_onsets, slice_offsets = define_slice_on_off(onsets, offsets, time_bin_size=time_bin_size)

    # Create unique integer labels for each syllable/label
    uni_syls, _, int_labels = np.unique(labels, return_inverse=True, return_index=True)
    uni_ints = np.unique(int_labels)

    # Create mapping of integer labels to original labels
    mapping = {inte: syl for inte, syl in zip(uni_ints, uni_syls)}

    # Initialize arrays for slice indicators and labels
    slice_indicator = np.zeros(len(slice_onsets))
    slice_labels = np.full(len(slice_onsets), np.nan)

    # Track previous indices and label to detect overlaps
    prev_idxs = np.array([100000])
    prev_label = None

    # Create an inverse mapping to convert labels back to integers
    inv_mapping = {v: k for (v, k) in mapping.items()}

    # Loop over each onset, offset, and label, assigning to time bins
    for onset, offset, label in zip(onsets, offsets, int_labels):
        # Find which time bins overlap with the current segment
        bin_indices = np.where(((slice_onsets >= onset) & (slice_onsets < offset)) |
                               ((slice_onsets < onset) & (slice_offsets > onset)))[0]

        # Handle potential overlap with the previous segment
        if len(prev_idxs) > len(bin_indices):
            prev_idxs = np.array(prev_idxs[-(len(bin_indices)):])
            overlap_flag = np.zeros(len(prev_idxs), dtype=bool)
            for idx in prev_idxs:
                overlap_flag = (idx == bin_indices) | overlap_flag
            overlap = prev_idxs[overlap_flag]
        else:
            overlap_flag = np.zeros(len(prev_idxs), dtype=bool)
            for idx in bin_indices:
                overlap_flag = (idx == prev_idxs) | overlap_flag
            overlap = prev_idxs[overlap_flag]

        # Assign the current label to the appropriate time bins
        slice_indicator[bin_indices] = 1
        slice_labels[bin_indices] = label

        # If there's an overlap, create a combined label
        if len(overlap) > 0:
            combined_label = mapping[prev_label] + mapping[label]
            if combined_label not in mapping.values():
                new_label = max(mapping.keys()) + 1
                mapping[new_label] = combined_label
                inv_mapping = {v: k for (k, v) in mapping.items()}
                ov_label = new_label
            else:
                ov_label = inv_mapping[combined_label]
            slice_labels[overlap] = ov_label

        # Update the previous indices and label for the next iteration
        prev_idxs = bin_indices
        prev_label = label

    # Assign the label 'gap' for slices with no label
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


def extract_phase_features(Sx, params):
    """Extract compact phase-derived features."""
    phase = np.angle(Sx)

    # Instantaneous frequency (phase derivative over time)
    inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)

    # Group delay (phase derivative over frequency)
    group_delay = -np.diff(np.unwrap(phase, axis=0), axis=0)

    # Downsample these too
    if params.use_warping:
        # Apply same interpolation logic to phase features
        inst_freq_ds = downsample_spec(inst_freq, params)
        group_delay_ds = downsample_spec(group_delay, params)
    else:
        inst_freq_ds = downsample_spec(inst_freq, params)
        group_delay_ds = downsample_spec(group_delay, params)

    return inst_freq_ds, group_delay_ds


def extract_minimal_phase_feature(complex_spec_ds):
    """Extract single most informative phase feature."""

    # Phase coherence across time (captures temporal structure)
    if complex_spec_ds.shape[1] > 1:
        phase_coherence = np.abs(np.mean(
            np.exp(1j * np.diff(np.angle(complex_spec_ds), axis=1)), axis=1
        ))
        return phase_coherence.reshape(-1, 1)  # Single column per frequency bin
    else:
        return np.zeros((complex_spec_ds.shape[0], 1))

#
# mag_features = []
# phase_features = []
#
# for audio_segment in audio_segments:
#     mag, phase_dict, _, _ = get_song_spec_with_phase(...)
#
#     mag_features.append(mag.flatten())
#     # Only keep the most important phase feature
#     phase_features.append(phase_dict['inst_freq'].flatten() * 0.2)  # Lower weight
#
# # Combine for UMAP
# combined_features = np.column_stack([
#     np.array(mag_features),
#     np.array(phase_features)
# ])

def extract_compact_phase_features(complex_spec_ds):
    """Extract minimal phase features from already-downsampled complex spectrogram."""

    # Only compute what you need, on the small downsampled version
    phase = np.angle(complex_spec_ds)

    # Instantaneous frequency (only if you have >1 time frame)
    if complex_spec_ds.shape[1] > 1:
        inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
        # Pad to match original size
        inst_freq = np.pad(inst_freq, ((0, 0), (0, 1)), mode='edge')
    else:
        inst_freq = np.zeros_like(phase)

    # Group delay (only if you have >1 frequency bin)
    if complex_spec_ds.shape[0] > 1:
        group_delay = -np.diff(np.unwrap(phase, axis=0), axis=0)
        # Pad to match original size
        group_delay = np.pad(group_delay, ((0, 1), (0, 0)), mode='edge')
    else:
        group_delay = np.zeros_like(phase)

    return {
        'inst_freq': inst_freq,
        'group_delay': group_delay
    }

def get_song_spec(t1: float, t2: float, audio: np.ndarray, params: SpectrogramParams, fs: int = 32000,
                  fill_value: float = -1 / EPSILON, downsample: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and normalize spectrogram from a birdsong syllable segment.
    Updated with better logging and error handling.
    """
    # warn if the segment duration exceeds max_dur
    if t2 - t1 > params.max_dur + 1e-4:
        logger.warning(f"⚠️ Segment longer than max_dur: {t2 - t1}s, max_dur = {params.max_dur}s")

    # convert onset and offset times to sample indices
    s1, s2 = int(round(t1 * fs)), int(round(t2 * fs))
    assert s1 < s2, f"s1: {s1}, s2: {s2}, t1: {t1}, t2: {t2}"

    need_phase = getattr(params, 'save_inst_freq', False) or getattr(params, 'save_group_delay', False)

    # if the segment is too small or invalid, return empty spectrogram
    if (s1 < 0) or (s2 > len(audio)):
        logger.debug(f"⚠️ Invalid segment bounds: s1={s1}, s2={s2}, audio_len={len(audio)}")
        empty = np.full(params.target_shape, fill_value)
        if need_phase:
            return empty, {}, np.zeros(s2 - s1), np.array([])
        return empty, np.zeros(s2 - s1), np.array([])
    else:
        # extract the audio segment, and remove dc offset if necessary
        audio_segment = audio[max(0, s1):min(len(audio), s2)]
        w = get_window('hann', Nx=params.nfft)
        STFT = ShortTimeFFT(w, hop=params.hop, fs=fs)
        Sx = STFT.stft(audio_segment)
        t = STFT.t(len(audio_segment))
        f = STFT.f
        non_negative_time_indices = t >= 0
        non_negative_time_indices[-int(params.nfft / 2):] = False
        t = t[non_negative_time_indices]
        complex_spec = Sx[:, non_negative_time_indices]
        spec = np.log(abs(Sx[:, non_negative_time_indices]))
        t += max(0, t1)  # adjust time to start at t1

        # begin DTW alternative block (regular interpolation over grid)
        if params.use_warping:
            # Create the interpolator using RegularGridInterpolator
            reasonable_fill = np.median(spec[np.isfinite(spec)])
            interp = RegularGridInterpolator((f, t), spec,
                                             method='linear',
                                             bounds_error=False,
                                             fill_value=reasonable_fill)

            target_freqs = np.linspace(max(params.min_freq, f.min()), min(params.max_freq, f.max()),
                                       params.target_shape[0])
            # Define target times
            duration = t2 - t1
            shoulder = 0.5 * (params.max_dur - duration)
            target_times = np.linspace(max(t1 - shoulder, t.min()), min(t2 + shoulder, t.max()),
                                       int(np.ceil(params.max_dur / STFT.delta_t)))
            logger.debug(f"  🎯 Target ranges:")
            logger.debug(
                f"    target_freqs: [{target_freqs.min():.1f}, {target_freqs.max():.1f}] Hz ({len(target_freqs)} points)")
            logger.debug(
                f"    target_times: [{target_times.min():.3f}, {target_times.max():.3f}] s ({len(target_times)} points)")

            # Create meshgrid for interpolation points
            target_freqs_grid, target_times_grid = np.meshgrid(target_freqs, target_times, indexing='ij')
            points = np.column_stack([target_freqs_grid.ravel(), target_times_grid.ravel()])

            # Interpolate magnitude
            interp_spec = interp(points).reshape(len(target_freqs), len(target_times))
            spec = interp_spec
            p5, p95 = np.percentile(spec, [2, 98])  # after interpolation

            # Interpolate complex spec (real and imag separately)
            if need_phase:
                real_part = np.real(complex_spec)
                imag_part = np.imag(complex_spec)
                interp_real = RegularGridInterpolator(
                    (f, t), real_part, method='linear', bounds_error=False,
                    fill_value=np.median(real_part[np.isfinite(real_part)]))
                interp_imag = RegularGridInterpolator(
                    (f, t), imag_part, method='linear', bounds_error=False,
                    fill_value=0.0)
                complex_spec_processed = (
                    interp_real(points).reshape(len(target_freqs), len(target_times))
                    + 1j * interp_imag(points).reshape(len(target_freqs), len(target_times))
                )

        else:
            p5, p95 = np.percentile(spec, [2, 98])  # before padding
            n_t_target = int(np.ceil(params.max_dur / STFT.delta_t))
            n_f = int(params.nfft / 2) + 1
            exp_spec = np.full((n_f, n_t_target), fill_value)

            if spec.shape[1] > exp_spec.shape[1]:
                logger.warning(f"✂️ Truncating spectrogram from {spec.shape[1]} to {exp_spec.shape[1]} frames. "
                               f"Syllable: {t2 - t1:.4f}s, max_dur: {params.max_dur}s")
                spec = spec[:, :exp_spec.shape[1]]
            exp_spec[:, :np.shape(spec)[1]] = spec[:, :]
            spec = exp_spec

            # Zero-pad complex spec to same target shape
            if need_phase:
                exp_complex = np.zeros((n_f, n_t_target), dtype=complex)
                cs = complex_spec[:, :n_t_target] if complex_spec.shape[1] > n_t_target else complex_spec
                exp_complex[:, :cs.shape[1]] = cs
                complex_spec_processed = exp_complex

        # Then normalize the whole thing
        exp_spec = (spec - p5) / (p95 - p5)
        exp_spec = np.clip(exp_spec, 0, 1)

        if downsample:
            spec = downsample_spec(exp_spec, params)
            if need_phase:
                real_ds = downsample_spec(np.real(complex_spec_processed), params)
                imag_ds = downsample_spec(np.imag(complex_spec_processed), params)
                complex_spec_ds = real_ds + 1j * imag_ds
        else:
            spec = exp_spec
            if need_phase:
                complex_spec_ds = complex_spec_processed

        # Compute phase-derived features from downsampled complex spec
        phase_dict = {}
        if need_phase:
            phase = np.angle(complex_spec_ds)  # (n_freq, n_time) = (513, 300)

            if getattr(params, 'save_inst_freq', False):
                inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)  # (513, 299)
                if_range = inst_freq.max() - inst_freq.min()
                phase_dict['inst_freq'] = (
                    ((inst_freq - inst_freq.min()) / if_range).astype(np.float32)
                    if if_range > 0 else np.zeros_like(inst_freq, dtype=np.float32)
                )

            if getattr(params, 'save_group_delay', False):
                group_delay = -np.diff(np.unwrap(phase, axis=0), axis=0)  # (512, 300)
                gd_range = group_delay.max() - group_delay.min()
                phase_dict['group_delay'] = (
                    ((group_delay - group_delay.min()) / gd_range).astype(np.float32)
                    if gd_range > 0 else np.zeros_like(group_delay, dtype=np.float32)
                )

        if need_phase:
            return spec, phase_dict, audio_segment, t
        return spec, audio_segment, t


if __name__ == '__main__':
    pass
