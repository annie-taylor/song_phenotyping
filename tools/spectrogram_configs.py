from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class SpectrogramParams:
    """
    """
    nfft: int = 1024
    hop: int = 1
    target_shape: tuple[int, int] = (int((nfft/2) + 1), 320)
    min_freq: float = 400.0
    max_freq: float = 10000.0
    max_dur: Optional[float] = 0.150
    fs: float = 32000.0
    padding: float = 0.0

    slice_length: Optional[float] = None  # in milliseconds
    songs_per_bird: int = 20
    overwrite_existing: bool = False
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.slice_length is not None and self.slice_length <= 0:
            raise ValueError("slice_length must be positive")
        if self.songs_per_bird <= 0:
            raise ValueError("songs_per_bird must be positive")

    def validate_params(self):
        """
        Validate the parameter values to ensure they are within acceptable ranges.
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

    def to_dict(self) -> dict:
        """
        Convert the parameters to a dictionary format.
        """
        return {
            "nfft": self.nfft,
            "hop": self.hop,
            "target_shape": self.target_shape,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "max_dur": self.max_dur,
            "fs": self.fs,
            "padding": self.padding
        }

    def from_dict(self, params_dict: dict):
        """
        Update the parameters from a dictionary format.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the spectrogram parameters.
        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate_params()

    def save_to_hdf5(self, hdf5_path: str):
        """Save the spectrogram parameters to an HDF5 file."""
        # Convert to dictionary and then to DataFrame
        params_dict = self.to_dict()
        df = pd.DataFrame([params_dict])

        # Write to HDF5
        df.to_hdf(hdf5_path, key='spectrogram_params', mode='w')

    @classmethod
    def load_from_hdf5(cls, hdf5_path: str) -> "SpectrogramParams":
        """Load the spectrogram parameters from an HDF5 file."""
        # Read DataFrame from HDF5
        df = pd.read_hdf(hdf5_path, key='spectrogram_params')

        # Convert DataFrame back to dictionary
        params_dict = df.to_dict(orient='records')[0]

        # Create a new instance of SpectrogramParams using the dictionary
        return cls(**params_dict)
