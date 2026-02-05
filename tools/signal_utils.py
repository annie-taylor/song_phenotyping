from scipy.signal import butter, sosfiltfilt
import numpy as np
from scipy.signal import convolve
from typing import Tuple


def butter_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter_sos(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to data using second-order sections.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter, by default 5.

    Returns
    -------
    np.ndarray
        Filtered signal.

    Examples
    --------
    >>> data = np.random.randn(32000)
    >>> filtered_data = butter_bandpass_filter_sos(data, 500, 15000, 32000, order=5)
    >>> print(filtered_data)
    """
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def smooth(data: np.ndarray, window_size: float = 2.0, fs: float = 32000) -> np.ndarray:
    """
    Smooth data using a moving average filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.
    window_size : float, optional
        Window size in milliseconds, by default 2.0.
    fs : float, optional
        Sampling frequency in Hz, by default 32000.

    Returns
    -------
    np.ndarray
        Smoothed signal.

    Examples
    --------
    >>> data = np.random.randn(32000)
    >>> smoothed_data = smooth(data, window_size=2.0, fs=32000)
    >>> print(smoothed_data)
    """
    window_in_bins = int(np.ceil((window_size / 1000) * fs))
    window = np.ones(window_in_bins) / window_in_bins
    smoothed = convolve(data, window, 'same')
    return smoothed

def rms_norm(array: np.ndarray) -> np.ndarray:
    """
    Normalize an array using its root mean square (RMS) value.

    Parameters
    ----------
    array : np.ndarray
        Input array to be normalized.

    Returns
    -------
    np.ndarray
        RMS-normalized array.

    Examples
    --------
    >>> data = np.random.randn(32000)
    >>> normalized_data = rms_norm(data)
    >>> print(normalized_data)
    """
    rms = np.sqrt(np.mean(np.square(array)))
    return array / rms


def define_slice_on_off(onsets: np.ndarray, offsets: np.ndarray, time_bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define slice onsets and offsets based on time bin size.

    Parameters
    ----------
    onsets : np.ndarray
        Array of onset times (in seconds).
    offsets : np.ndarray
        Array of offset times (in seconds).
    time_bin_size : float
        Time bin size in seconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        slice_onsets : Array of slice onset times.
        slice_offsets : Array of slice offset times.

    Examples
    --------
    >>> onsets = np.array([0.0, 1.0, 2.5])
    >>> offsets = np.array([0.8, 2.0, 3.5])
    >>> time_bin_size = 0.1
    >>> slice_onsets, slice_offsets = define_slice_on_off(onsets, offsets, time_bin_size)
    >>> print(slice_onsets, slice_offsets)
    """
    slice_onsets = np.arange(onsets[0], offsets[-1], time_bin_size)
    slice_offsets = slice_onsets + time_bin_size
    return slice_onsets, slice_offsets