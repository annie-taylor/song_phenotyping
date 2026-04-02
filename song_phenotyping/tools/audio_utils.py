import numpy as np
from typing import Union, Tuple
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.ndimage import zoom
import warnings
from pathlib import Path

from song_phenotyping.tools.evfuncs import readrecf, load_cbin
from song_phenotyping.tools.system_utils import replace_macaw_root


def read_audio_file(audio_filename) -> Tuple[np.ndarray, int]:
    """
    Read an audio file and return its waveform and sampling rate.

    This function supports multiple audio file formats such as `.wav`, `.cbin`, and `.rec`. It reads
    the file, handles mono and stereo channels by extracting only the first channel, and returns the
    audio waveform and its sampling rate. For unsupported file formats, the function will return `None`
    for both the audio and sampling rate.

    Parameters
    ----------
    audio_filename : Union[str, Path]
        Path to the audio file to be read. Supported formats are `.wav`, `.cbin`, and `.rec`.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple (audio, fs) where:
            - audio (numpy.ndarray): The audio waveform as a NumPy array.
            - fs (int): The sampling rate of the audio file.

    Raises
    ------
    ValueError
        If the file type is not recognized or cannot be read.

    Examples
    --------
    >>> audio, fs = read_audio_file('birdsong.wav')
    >>> print(audio.shape, fs)
    """
    if type(audio_filename) is not Path: audio_filename = Path(audio_filename)
    if audio_filename.suffix == '.wav':
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=WavFileWarning)
            try:
                fs, audio = wavfile.read(audio_filename)
            except FileNotFoundError:
                audio_filename = replace_macaw_root(audio_filename)
                fs, audio = wavfile.read(audio_filename)
            if audio.ndim > 1:
                audio = audio[:, 0]
    elif audio_filename.suffix == '.cbin':
        try:
            audio, fs = load_cbin(audio_filename)
        except FileNotFoundError:
            audio_filename = replace_macaw_root(audio_filename)
            audio, fs = load_cbin(audio_filename)
        if audio.ndim > 1:
            audio = audio[:, 0]
    elif audio_filename.suffix == '.rec':
        try:
            audio, fs = readrecf(audio_filename)
        except FileNotFoundError:
            audio_filename = replace_macaw_root(audio_filename)
            audio, fs = readrecf(audio_filename)
        if audio.ndim > 1:
            audio = audio[:, 0]
    else:
        raise ValueError(f"Unsupported file type: {audio_filename.suffix}")
        audio = None
        fs = None
    return audio, fs


def downsample_spec(spec: np.ndarray, target_shape: Tuple[int, int] = (257, 320)) -> np.ndarray:
    """
    Downsample spectrogram array to a target shape using bilinear interpolation.

    Parameters
    ----------
    spec : np.ndarray
        Input spectrogram array.
    target_shape : Tuple[int, int], optional
        Target shape for the downsampled spectrogram, by default (257, 320).

    Returns
    -------
    np.ndarray
        Downsampled spectrogram.

    Examples
    --------
    >>> spec = np.random.randn(500, 400)
    >>> downsampled_spec = downsample_spec(spec)
    >>> print(downsampled_spec.shape)
    (257, 320)
    """
    current_shape = spec.shape
    zoom_factors = (target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])
    downsampled_spec = zoom(spec, zoom_factors, order=1)
    assert downsampled_spec.shape == target_shape, f"Expected shape {target_shape}, but got {downsampled_spec.shape}"
    return downsampled_spec

## keeping phase derived feature recommendations here for now:

def extract_phase_features(stft_complex, hop_time, freq_bins):
    """Extract phase-based features for birdsong classification"""

    magnitude = np.abs(stft_complex)
    phase = np.angle(stft_complex)

    features = {}

    # 1. Instantaneous Frequency Features (captures FM sweeps)
    phase_unwrapped = np.unwrap(phase, axis=1)  # Remove 2π jumps
    inst_freq = np.gradient(phase_unwrapped, hop_time, axis=1) / (2 * np.pi)

    features['fm_rate_mean'] = np.mean(np.std(inst_freq, axis=1))  # FM variability
    features['fm_direction_bias'] = np.mean(np.diff(inst_freq) > 0)  # Upsweep tendency
    features['fm_complexity'] = np.std(np.diff(inst_freq, 2))  # FM acceleration

    # 2. Phase Coherence Features (measures vocal control)
    temporal_coherence = np.abs(np.mean(np.exp(1j * phase), axis=1))
    features['phase_coherence_mean'] = np.mean(temporal_coherence)
    features['phase_coherence_std'] = np.std(temporal_coherence)
    features['vocal_control_index'] = np.mean(temporal_coherence > 0.7)  # High coherence fraction

    # 3. Phase Flux Features (captures transitions/onsets)
    phase_diff = np.abs(np.diff(phase_unwrapped, axis=1))
    magnitude_weights = (magnitude[:, :-1] + magnitude[:, 1:]) / 2  # Average adjacent frames
    weighted_flux = phase_diff * (magnitude_weights > np.percentile(magnitude, 25))  # Only where there's energy

    features['phase_flux_mean'] = np.mean(weighted_flux)
    features['phase_flux_peaks'] = len(np.where(np.sum(weighted_flux, axis=0) >
                                                np.percentile(np.sum(weighted_flux, axis=0), 90))[0])

    return features

#
# def get_song_spec_with_phase(t1, t2, audio, params, fs=32000, extract_phase=False):
#     # ... your existing STFT computation ...
#
#     phase_features = {}
#     if extract_phase:
#         # Extract phase features from complex STFT before taking magnitude
#         hop_time = params.hop / fs
#         phase_features = extract_phase_features(Sx[:, non_negative_time_indices],
#                                                 hop_time, f)
#
#     # Continue with your existing magnitude processing...
#     spec = np.log(abs(Sx[:, non_negative_time_indices]))
#
#     # ... rest of processing ...
#
#     return spec, audio_segment, t, phase_features

def combine_magnitude_and_phase_features(magnitude_features, phase_features):
    """Intelligently combine different feature types"""
    combined = {**magnitude_features, **phase_features}

    # Add interaction features
    combined['spectral_phase_ratio'] = (magnitude_features.get('spectral_centroid_mean', 0) /
                                        (phase_features.get('phase_coherence_mean', 1) + 1e-10))

    return combined

def prepare_complex_input(stft_complex):
    return np.stack([stft_complex.real, stft_complex.imag], axis=0)