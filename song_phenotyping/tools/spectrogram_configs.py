"""Parameter dataclasses for spectrogram computation."""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd


@dataclass
class SpectrogramParams:
    """Parameters controlling spectrogram computation and syllable extraction.

    Parameters
    ----------
    nfft : int, optional
        FFT window size in samples. Determines frequency resolution.
        Default is 1024.
    hop : int, optional
        Hop size between successive FFT windows in samples. Default is 1.
    target_shape : tuple of int, optional
        ``(n_freq_bins, n_time_bins)`` to which each syllable spectrogram is
        resized. Defaults to ``(nfft // 2 + 1, 300)``.
    min_freq : float, optional
        Low-frequency cutoff in Hz applied when cropping spectrograms.
        Default is 200.0.
    max_freq : float, optional
        High-frequency cutoff in Hz. Default is 15000.0.
    max_dur : float or None, optional
        Maximum syllable duration in seconds. Syllables longer than this are
        discarded. ``None`` disables the duration filter. Default is 0.080.
    fs : float, optional
        Expected audio sample rate in Hz. Default is 32000.0.
    padding : float, optional
        Padding added around each syllable boundary in seconds. Default is 0.0.
    slice_length : float or None, optional
        Fixed window length in milliseconds for slice-based extraction
        (Stage A1). ``None`` disables fixed-window mode. Default is ``None``.
    songs_per_bird : int, optional
        Maximum number of songs to process per bird when using
        ``SpectrogramParams`` as the sole source of this limit. Default is 5.
    overwrite_existing : bool, optional
        If ``True``, recompute and overwrite previously saved spectrogram
        files. Default is ``False``.
    use_warping : bool, optional
        Apply dynamic time-warping alignment before saving. Default is
        ``False``.
    downsample : bool, optional
        Downsample audio to ``fs`` before computing spectrograms.
        Default is ``False``.
    save_inst_freq : bool, optional
        Compute and store instantaneous frequency (IF) alongside the
        magnitude spectrogram.  IF is the temporal derivative of unwrapped
        phase, capturing pitch-modulation structure.  Shape per syllable:
        ``(n_freq, n_time - 1)`` = ``(513, 299)`` with default settings.
        Default is ``False``.
    save_group_delay : bool, optional
        Compute and store group delay (GD) alongside the magnitude
        spectrogram.  GD is the negative frequency-derivative of unwrapped
        phase, capturing spectral dispersion.  Shape per syllable:
        ``(n_freq - 1, n_time)`` = ``(512, 300)`` with default settings.
        Default is ``False``.
    duration_feature_weight : float, optional
        When non-zero, the normalised syllable duration (relative to
        ``max_dur``) is tiled into ``n_freq`` extra features and appended
        to the flattened feature vector at Stage B.  Set to a value roughly
        comparable to the magnitude feature weight (experiment to tune).
        ``0.0`` (default) disables the feature entirely.
    warp_freq_sum : bool, optional
        Sum over frequency bins before DTW (deprecated). Default is ``True``.
    shift_lambdas : list of float, optional
        DTW shift penalty grid (deprecated). Default is ``[100, 10, 1, 0]``.
    slope_lambdas : list of float, optional
        DTW slope penalty grid (deprecated).
        Default is ``[inf, inf, 10, 1]``.

    Raises
    ------
    ValueError
        If ``slice_length`` is not positive, or ``songs_per_bird`` is not
        positive.

    Examples
    --------
    Default parameters for syllable-based extraction:

    >>> params = SpectrogramParams()

    Smoke-test configuration (3 songs, no warping):

    >>> params = SpectrogramParams(songs_per_bird=3)

    Fixed-window slice extraction at 50 ms:

    >>> params = SpectrogramParams(slice_length=50.0)
    """

    nfft: int = 1024
    hop: int = 1
    target_shape: tuple = (int((nfft / 2) + 1), 300)
    min_freq: float = 200.0
    max_freq: float = 15000.0
    max_dur: Optional[float] = 0.080
    fs: float = 32000.0
    padding: float = 0.0

    slice_length: Optional[float] = None
    songs_per_bird: int = 5
    overwrite_existing: bool = False

    use_warping: bool = False
    downsample: bool = False

    # Phase-derived feature flags (Stage A optional outputs)
    save_inst_freq: bool = False
    """Save instantaneous frequency alongside magnitude spectrograms."""
    save_group_delay: bool = False
    """Save group delay alongside magnitude spectrograms."""

    # Duration feature weight for UMAP (Stage B)
    duration_feature_weight: float = 0.0
    """Weight for duration token appended to flattened feature vector.
    Zero (default) disables the feature entirely."""

    # DTW parameters — largely deprecated; computationally expensive
    warp_freq_sum: bool = True
    shift_lambdas: List[float] = field(default_factory=lambda: [100, 10, 1, 0])
    slope_lambdas: List[float] = field(default_factory=lambda: [np.inf, np.inf, 10, 1])

    def __post_init__(self):
        if self.slice_length is not None and self.slice_length <= 0:
            raise ValueError("slice_length must be positive")
        if self.songs_per_bird <= 0:
            raise ValueError("songs_per_bird must be positive")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_params(self):
        """Raise ``ValueError`` if any parameter is out of range.

        Checks ``nfft``, ``hop``, ``min_freq``, ``max_freq``, ``max_dur``,
        ``fs``, and ``padding``.
        """
        if self.nfft <= 0:
            raise ValueError("nfft must be greater than 0.")
        if self.hop <= 0:
            raise ValueError("hop must be greater than 0.")
        if self.min_freq < 0:
            raise ValueError("min_freq must be non-negative.")
        if self.max_freq <= self.min_freq:
            raise ValueError("max_freq must be greater than min_freq.")
        if self.max_dur is not None and self.max_dur <= 0:
            raise ValueError("max_dur must be greater than 0.")
        if self.fs <= 0:
            raise ValueError("fs must be greater than 0.")
        if self.padding < 0:
            raise ValueError("padding must be non-negative.")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return core parameters as a plain dictionary.

        Returns
        -------
        dict
            Keys: ``nfft``, ``hop``, ``target_shape``, ``min_freq``,
            ``max_freq``, ``max_dur``, ``fs``, ``padding``.
        """
        return {
            "nfft": self.nfft,
            "hop": self.hop,
            "target_shape": self.target_shape,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "max_dur": self.max_dur,
            "fs": self.fs,
            "padding": self.padding,
        }

    def from_dict(self, params_dict: dict):
        """Update fields in-place from a dictionary.

        Parameters
        ----------
        params_dict : dict
            Keys matching field names on this dataclass; unrecognised keys
            are silently ignored. After updating, :meth:`validate_params` is
            called.
        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate_params()

    def save_to_hdf5(self, hdf5_path: str):
        """Persist parameters to an HDF5 file.

        Parameters
        ----------
        hdf5_path : str
            Destination file path. The parameters are stored under the key
            ``spectrogram_params``.
        """
        df = pd.DataFrame([self.to_dict()])
        df.to_hdf(hdf5_path, key="spectrogram_params", mode="w")

    @classmethod
    def load_from_hdf5(cls, hdf5_path: str) -> "SpectrogramParams":
        """Load parameters from an HDF5 file written by :meth:`save_to_hdf5`.

        Parameters
        ----------
        hdf5_path : str
            Path to the HDF5 file.

        Returns
        -------
        SpectrogramParams
            New instance populated from the stored values.
        """
        df = pd.read_hdf(hdf5_path, key="spectrogram_params")
        return cls(**df.to_dict(orient="records")[0])
