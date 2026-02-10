#import h5py

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
from typing import  List, Optional, Tuple, Dict, Any

from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file
from tools.signal_utils import smooth, butter_bandpass_filter_sos
from tools.system_utils import fix_mixture_of_separators
#from Z_add_syllable_database import fix_mixture_of_separators

import pyfftw.interfaces.scipy_fft

# Set globally when module is imported
scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)

EPSILON = 1e-9  # small constant to avoid log(0)
# Tempo estimation parameters
PEAK_HEIGHT_THRESHOLD = 5e33
LOW_FREQ_THRESHOLD = 3.0
FREQ_RANGE_MIN = 0.0
FREQ_RANGE_MAX = 30.0

def save_data_specs(metadata_file_paths: List[str], save_path: str, params: SpectrogramParams, verbose: bool = False,
                    read_songpath_from_metadata: bool = True,) -> None:

    problem_files = []
    # iterate through each audio file
    for metadata_file_path in tqdm(metadata_file_paths, "Reading metadata..."):
        # check if the file exists
        if not (os.path.exists(metadata_file_path)):
            print(f"File not found: {metadata_file_path},")
            continue
        else:
            metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)
            try:
                fs = metadata_matfile['Fs']
            except KeyError:
                fs = 32000.
            if not read_songpath_from_metadata:
                # if not reading songpath from metadata, infer it from metadata path
                song_file_path = '.'.join(metadata_file_path.split('.')[0:2])
                wseg_offset = 0
            else:  # this for whisperseg files
                # otherwise, read song_file_path from metadata
                wseg_offset = (256 * 1000) / fs
                try:
                    fname = metadata_matfile['fname']
                except KeyError:
                    #there are some old files where the audio filepath seems to be broken or corrupted
                    fname = metadata_matfile['fnamecell']
                    #print(f"Filename unreadable: {metadata_file_path},")
                    #break
                if platform == 'win32':
                    fname = fname.replace('\\', '/')  # to resolve issue where there is a mixture of separators
                    song_file_path = 'Z:\\' + '\\'.join(fname.split('/')[1:])  # use windows specific separators
                else:
                    fname = fname.replace('/', '\\')  # to resolve issue where there is a mixture of separators
                    song_file_path = '/Volumes/users/' + '/'.join(fname.split('\\')[1:])  # use iOS separators
                if not (os.path.exists(song_file_path)):
                    print(f"File not found: {song_file_path},")
                    break
            try:
                #re.match('..-..-....', song.split('/')[-2]) or re.match('....-..-..', song.split('/')[-2])
                if platform == 'win32':
                    filename = song_file_path.split('\\')[-1]
                else:
                    filename = song_file_path.split('/')[-1]
                split_filename = filename.split('.')
                if len(split_filename) > 4:
                    bird = split_filename[0]
                    daytime = split_filename[-1]
                    Warning('This may not be the right way to handle files with this naming convention.')
                split_filename = filename.split('_')
                if len(split_filename) == 3:
                    [bird, day, time] = split_filename[0:3]  # read bird, date, and time from filename
                    time = time.split('.')[0]  # for time, have to separate from file extension
                elif len(split_filename) == 2:
                    # some files follow a different naming convention, detect this
                    [bird, daytime] = filename.split('_')[0:2]
                    day = daytime.split('.')[0][0:8]
                    time = daytime.split('.')[0][8:]
                elif len(split_filename) == 4:  # for folders that have an extra modifier beyond bird name
                    [bird, _, day, time] = split_filename[0:4]  # read bird, date, and time from filename
                    time = time.split('.')[0]  # for time, have to separate from file extension
            except ValueError:
                Warning(f'Failed to detect filename format / naming convention for {filename}.')

            syl_onsets = metadata_matfile['onsets'] + wseg_offset
            syl_offsets = metadata_matfile['offsets'] + wseg_offset
            labels = metadata_matfile['labels']
            #labels = np.array([label for label in labels])
            try:
                assert len(labels) == len(syl_offsets)
                if len(syl_onsets) == 1:
                    # if only contains one syllable, don't read file
                    song = False
                else:
                    song = True
            except TypeError:
                # only one onset, not in list
                song = False

            # by default right now, if file already exists, do not reprocess date or overwrite
            os.makedirs(os.path.join(save_path, bird), exist_ok=True)  # check if dir for bird exists yet
            data_path = os.path.join(save_path, bird, 'data')
            os.makedirs(data_path, exist_ok=True)
            data_path = os.path.join(data_path, 'syllables')
            os.makedirs(data_path, exist_ok=True)
            h5file_save_path = os.path.join(data_path, f'syllables_{bird}_{day}_{time}.h5')
            h5file_exists = os.path.isfile(h5file_save_path)
            if not h5file_exists:
                if song and (syl_offsets[-1] - syl_onsets[0] > 2):  # also only read and save if longer than 2 seconds
                    # optional: handle slicing logic (e.g., label_slices) if needed
                    # dictionary to store segmented audio data
                    segmented_audio_data = {
                        'spectrograms': [],  # stores spectrograms of each valid syllable/slice
                        'waveforms': [],  # stores waveform of each valid syllable/slice
                        'spec_t': [],  # stores spectrogram reference times for each valid syllable/slice
                        'manual': [],  # stores corresponding syllable labels
                        'onsets': [],  # stores onset times of syllables/slices
                        'offsets': [],  # stores offset times of syllables/slices
                        'position_idxs': [],  # stores idx (relative position) of syllables/slices
                        'hashes': []
                    }

                    syl_lengths = syl_offsets - syl_onsets
                    long_mask = syl_lengths > params.max_dur * 1000
                    if np.any(long_mask):
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
                                n_splits = n_sub_syls[np.sum(long_mask[:i])]  # Get number of splits for this syllable
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
                        syl_onsets, syl_offsets, labels = new_onsets, new_offsets, new_labels

                    # extract spectrograms for syllables using get_specs()
                    specs, wavs, ts, valid_inds = get_song_specs(song_file_path, syl_onsets, syl_offsets, params=params)
                    onsets = [syl_onsets[i] for i in valid_inds]
                    offsets = [syl_offsets[i] for i in valid_inds]
                    labels = [labels[i] for i in valid_inds]
                    syl_idx = [i for i in valid_inds]
                    syl_strs = [f'{h5file_save_path}_{i}' for i in valid_inds]
                    hash_obj = hashlib.sha256()
                    hashes = []
                    for syl_str in syl_strs:
                        hash_obj.update(syl_str.encode('utf-8'))
                        hash_id = hash_obj.hexdigest()
                        hashes.append(hash_id)

                    segmented_audio_data['waveforms'] += wavs
                    # pad waveforms with NaNs s.t. all arrays are same length (have to redo this for every new batch)
                    # TODO(annie-taylor): clean this up a bit
                    max_l = 0
                    padded_waveforms = []
                    padded_ts = []
                    for wav in segmented_audio_data['waveforms']:
                        if len(wav) > max_l:
                            max_l = len(wav)
                    for wav, ts in zip(segmented_audio_data['waveforms'], segmented_audio_data['spec_t']):
                        if len(wav) != max_l:
                            padded_wav = np.full(max_l, np.nan)
                            padded_ts = np.full(max_l, np.nan)
                            padded_wav[0:len(wav)] = wav
                            padded_ts[0:len(ts)] = ts
                        else:
                            padded_wav = wav
                            padded_ts = ts
                        padded_waveforms.append(padded_wav)
                        padded_ts.append(padded_ts)

                    # filter and store valid syllables or slices
                    for i in range(len(valid_inds)):
                        segmented_audio_data['spectrograms'].append(specs[i])
                        segmented_audio_data['manual'].append(labels[i])
                        segmented_audio_data['onsets'].append(onsets[i])
                        segmented_audio_data['offsets'].append(offsets[i])
                        segmented_audio_data['position_idxs'].append(syl_idx[i])
                        segmented_audio_data['hashes'].append(hashes[i])
                    segmented_audio_data['spec_t'] = padded_ts
                    segmented_audio_data['waveforms'] = padded_waveforms
                    del specs, labels, onsets, offsets, padded_waveforms, padded_ts
                    if len(segmented_audio_data['onsets']) >0:
                        # save the segmented audio data to an HDF5 file
                        # don't overwrite files, unless specified (file will have already been removed)
                         save_segmented_audio_data(h5file_save_path, song_file_path, segmented_audio_data)
                    # Reset segmentation data for the next file
                    del segmented_audio_data
                    if verbose:
                        # print confirmation of saved data
                        print(f"Saved data to {data_path}")
            else:
                if not h5file_exists:
                    problem_files.append(song_file_path)
                    if problem_files and verbose:
                        print(f"Problem file: {song_file_path}")


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
        h5file.create_array(h5file.root, 'audio_filename', obj=np.array([song_file_path], dtype=np.str_))  # Save as string array
        h5file.create_array(h5file.root, 'waveforms', obj=np.array(segmented_audio_data['waveforms']))
        h5file.create_array(h5file.root, 'spectrograms', obj=np.array(segmented_audio_data['spectrograms'],
                                                                      dtype=np.float64), atom=atom)
        h5file.create_array(h5file.root, 'spec_t', obj=np.array(segmented_audio_data['spec_t']))
        h5file.create_array(h5file.root, 'manual', obj=np.array(segmented_audio_data['manual'], dtype=np.str_))  # Save as string array
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

    Parameters
    ----------
    audio_norm : np.ndarray
        Normalized audio signal array.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    mean_top_3 : float
        Mean frequency of top 3 peaks by relative height, or NaN if no peaks found.
    low_f_mean : float
        Mean frequency of low frequency peaks (< LOW_FREQ_THRESHOLD Hz), or NaN if none found.
    mean_all : float
        Mean frequency of all detected peaks, or NaN if no peaks found.
    """
    rect_audio_norm = smooth(audio_norm ** 2)
    rect_auto_corr_audio_norm = np.correlate(rect_audio_norm, rect_audio_norm, 'same')  # or keep 'full' if needed

    # Use rfft for real signals (2x faster)
    power_spectrum = np.abs(np.fft.rfft(rect_auto_corr_audio_norm)) ** 2
    freqs = np.fft.rfftfreq(rect_auto_corr_audio_norm.size, 1 / fs)

    # Direct boolean indexing
    freq_mask = (freqs > FREQ_RANGE_MIN) & (freqs < FREQ_RANGE_MAX)
    power_spectrum = power_spectrum[freq_mask]
    freqs = freqs[freq_mask]

    peaks = signal.find_peaks(power_spectrum, height=PEAK_HEIGHT_THRESHOLD)[0]

    if len(peaks) == 0:
        return np.nan, np.nan, np.nan

    rel_height = power_spectrum[peaks] / np.sum(power_spectrum[peaks])
    top_3_by_rel_height = np.argsort(rel_height)[:3]
    low_f_peak_idxs = freqs[peaks] < LOW_FREQ_THRESHOLD
    low_f_mean = np.mean(freqs[peaks[low_f_peak_idxs]]) if low_f_peak_idxs.any() else np.nan
    mean_all = np.mean(freqs[peaks])
    mean_top_3 = np.mean(freqs[peaks[top_3_by_rel_height]])

    return mean_top_3, low_f_mean, mean_all


def get_song_specs(audio_filename: str, onsets: np.ndarray, offsets: np.ndarray, params: SpectrogramParams,
                   split_syllables: bool = False, tempo: bool = False) -> (
        tuple[list[Any], list[Any], list[Any], list[Any]] | tuple[list[Any], list[Any], list[Any], list[Any],
tuple[floating[Any], floating[Any], floating[Any]]]):
    specs = []
    valid_inds = []
    audio_segs = []
    spec_t = []

    # Read audio file once
    try:
        audio, fs = read_audio_file(audio_filename)
    except Exception as e:
        print(f"Failed to read audio file {audio_filename}: {e}")
        return specs, audio_segs, spec_t, valid_inds

    # Process audio
    audio = rms_norm(audio)
    audio = butter_bandpass_filter_sos(audio, lowcut=params.min_freq, highcut=params.max_freq, fs=fs, order=5)

    if tempo:
        tempos = tempo_estimates(audio, fs)
    else:
        tempos = (np.nan, np.nan, np.nan)

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
            print(f"Failed to process syllable {i} (original {syllable_to_original_mapping[i]}): {e}")

    return specs, audio_segs, spec_t, valid_inds, tempos


def define_slice_on_off(onsets: np.ndarray, offsets: np.ndarray, time_bin_size: float):
    # create an array of time bin onsets and corresponding offsets
    slice_onsets = np.arange(onsets[0], offsets[-1], time_bin_size)
    slice_offsets = slice_onsets + time_bin_size
    return slice_onsets, slice_offsets


def label_slices(onsets: np.ndarray, offsets: np.ndarray, labels: np.ndarray, time_bin_size: float) -> (Tuple)[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
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
        t += max(0, t1)   # adjust time to start at t1

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


def read_metadata(metadata_file_path: str, read_songpath_from_metadata: bool) -> Tuple[str, float, np.ndarray, np.ndarray, np.ndarray]:
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
                     verbose: bool = False,) -> None:
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
            #TODO there are instances here where the last syllable is truncated, not clear why
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
                    f.create_array(f.root,'spec_t', np.array(spec_t))
                    f.create_array(f.root,'position_idxs', np.array(valid_inds))
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
