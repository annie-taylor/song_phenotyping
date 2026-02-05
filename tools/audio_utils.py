import numpy as np
from typing import Union, Tuple
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.ndimage import zoom
import warnings
from pathlib import Path

from tools.evfuncs import readrecf, load_cbin
from tools.system_utils import replace_macaw_root


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
