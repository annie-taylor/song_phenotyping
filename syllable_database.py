# syllable_database.py

import os
import logging
import warnings
import numpy as np
import pandas as pd
import librosa
import tables
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from tqdm import tqdm
import gc

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


@dataclass
class FeatureExtractionParams:
    """Parameters for acoustic feature extraction."""
    # Librosa parameters
    hop_length: int = 512
    n_fft: int = 2048
    n_mfcc: int = 13

    # Spectral feature parameters
    spectral_rolloff_percent: float = 0.85

    # F0 estimation parameters
    f0_method: str = 'pyin'  # 'pyin', 'yin', or 'piptrack'
    f0_fmin: float = 300.0  # Minimum F0 for birdsong
    f0_fmax: float = 8000.0  # Maximum F0 for birdsong

    # Audio preprocessing
    apply_preemphasis: bool = True
    preemphasis_coeff: float = 0.97


@dataclass
class SyllableRecord:
    """Complete record for a single syllable with all extracted features."""
    # Identifiers
    hash_id: str
    bird_name: str
    song_file: str
    syllable_index: int

    # Temporal features
    duration_ms: float
    start_time_ms: float
    end_time_ms: float
    position_in_song: int
    song_length_syllables: int

    # Librosa spectral features
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_std: float
    spectral_rolloff_mean: float
    spectral_rolloff_std: float
    spectral_contrast_mean: float
    spectral_contrast_std: float
    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float

    # MFCCs (statistics of coefficients 1-13)
    mfcc_means: List[float]  # 13 values
    mfcc_stds: List[float]  # 13 values

    # Fundamental frequency features
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    f0_range: float
    f0_voiced_fraction: float  # Fraction of frames with detected F0

    # Energy and amplitude features
    rms_energy_mean: float
    rms_energy_std: float

    # Temporal envelope features
    onset_strength_mean: float
    onset_strength_std: float
    tempo_estimate: float

    # Context features
    prev_syllable_gap_ms: float
    next_syllable_gap_ms: float

    # Labels
    manual_label: Optional[str]
    clustering_labels: Dict[str, int]  # {clustering_method_id: label}


class SyllableDatabase:
    """
    Database for syllable acoustic features and analysis.

    Extracts comprehensive acoustic features from syllable audio segments
    and provides analysis tools for clustering validation.
    """

    def __init__(self, bird_path: str, feature_params: FeatureExtractionParams = None):
        self.bird_path = bird_path
        self.bird_name = os.path.basename(bird_path)
        self.syllable_dir = os.path.join(bird_path, 'data', 'syllables')
        self.database_dir = os.path.join(bird_path, 'data', 'syllable_database')
        os.makedirs(self.database_dir, exist_ok=True)

        # Feature extraction parameters
        self.feature_params = feature_params or FeatureExtractionParams()

        # File paths
        self.hdf5_path = os.path.join(self.database_dir, 'syllable_features.h5')
        self.csv_path = os.path.join(self.database_dir, 'syllable_features.csv')
        self.params_path = os.path.join(self.database_dir, 'feature_params.json')


    def build_database(self, force_rebuild: bool = False) -> bool:
        """
        Build complete syllable feature database for this bird.

        Args:
            force_rebuild: If True, rebuild even if database exists

        Returns:
            bool: Success status
        """
        try:
            # Check if database already exists
            if not force_rebuild and os.path.exists(self.hdf5_path):
                logging.info(f"Database already exists for {self.bird_name}. Use force_rebuild=True to rebuild.")
                return True

            logging.info(f"Building syllable database for {self.bird_name}")

            # Get syllable files
            if not os.path.exists(self.syllable_dir):
                logging.error(f"Syllable directory not found: {self.syllable_dir}")
                return False

            syllable_files = [f for f in os.listdir(self.syllable_dir) if f.endswith('.h5')]
            if not syllable_files:
                logging.error(f"No syllable files found in {self.syllable_dir}")
                return False

            logging.info(f"Found {len(syllable_files)} syllable files to process")

            # Process each syllable file
            all_records = []
            total_syllables = 0

            for file_idx, filename in enumerate(tqdm(syllable_files, desc="Processing syllable files")):
                file_path = os.path.join(self.syllable_dir, filename)

                try:
                    records = self._process_syllable_file(file_path, filename)
                    all_records.extend(records)
                    total_syllables += len(records)
                    logging.debug(f"Processed {len(records)} syllables from {filename}")

                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
                    continue

            if not all_records:
                logging.error(f"No syllables processed for {self.bird_name}")
                return False

            logging.info(f"Processed {total_syllables} syllables from {len(syllable_files)} files")

            # Load clustering labels
            clustering_labels = self._load_clustering_labels()
            logging.info(f"Found {len(clustering_labels)} clustering methods")

            # Add clustering labels to records
            for record in all_records:
                record.clustering_labels = {}
                for method_id, hash_to_label in clustering_labels.items():
                    record.clustering_labels[method_id] = hash_to_label.get(record.hash_id, -1)

            # Save to both formats
            success_hdf5 = self._save_to_hdf5(all_records)
            success_csv = self._save_to_csv(all_records)

            # Save feature parameters
            self._save_feature_params()

            if success_hdf5 and success_csv:
                logging.info(f"Successfully built syllable database for {self.bird_name}")
                logging.info(f"  - HDF5: {self.hdf5_path}")
                logging.info(f"  - CSV: {self.csv_path}")
                return True
            else:
                logging.error("Failed to save database files")
                return False
        except Exception as e:
            logging.error(f"Error building database for {self.bird_name}: {e}")
            return False

    def _process_syllable_file(self, file_path: str, filename: str) -> List[SyllableRecord]:
        """
        Process a single syllable HDF5 file and extract features from all syllables.

        Args:
            file_path: Path to syllable HDF5 file
            filename: Name of the file for record keeping

        Returns:
            List[SyllableRecord]: Records for all syllables in the file
        """
        records = []

        try:
            with tables.open_file(file_path, 'r') as f:
                # Load data from HDF5 file
                waveforms = f.root.waveforms.read()
                onsets = f.root.onsets.read()
                offsets = f.root.offsets.read()
                hashes = f.root.hashes.read()
                position_idxs = f.root.position_idxs.read()

                # Load manual labels if available
                manual_labels = None
                if hasattr(f.root, 'manual'):
                    manual_labels = f.root.manual.read()
                    manual_labels = [label.decode('utf-8') if isinstance(label, (bytes, np.bytes_)) else str(label)
                                     for label in manual_labels]

                # Convert hashes to strings
                hashes = [h.decode('utf-8') if isinstance(h, (bytes, np.bytes_)) else str(h) for h in hashes]

                # Process each syllable
                for i, (waveform, onset, offset, hash_id, pos_idx) in enumerate(
                        zip(waveforms, onsets, offsets, hashes, position_idxs)
                ):
                    try:
                        # Skip empty waveforms
                        if len(waveform) == 0:
                            logging.debug(f"Skipping empty waveform for syllable {i} in {filename}")
                            continue

                        # Remove nans from waveform (all but the longest waveform will have this)
                        waveform = waveform[~np.isnan(waveform)]

                        # Extract features from waveform
                        features = self._extract_features_from_audio(waveform)

                        # Calculate temporal features
                        duration_ms = (offset - onset)  # Convert to milliseconds
                        start_time_ms = onset
                        end_time_ms = offset

                        # Calculate context features (gaps to previous/next syllables)
                        prev_gap_ms = self._calculate_prev_gap(i, onsets, offsets)
                        next_gap_ms = self._calculate_next_gap(i, onsets, offsets)

                        # Get manual label if available
                        manual_label = None
                        if manual_labels and i < len(manual_labels):
                            manual_label = manual_labels[i] if manual_labels[i] not in ['-', ''] else None

                        # Create syllable record
                        record = SyllableRecord(
                            # Identifiers
                            hash_id=hash_id,
                            bird_name=self.bird_name,
                            song_file=filename,
                            syllable_index=i,

                            # Temporal features
                            duration_ms=duration_ms,
                            start_time_ms=start_time_ms,
                            end_time_ms=end_time_ms,
                            position_in_song=int(pos_idx),
                            song_length_syllables=len(waveforms),

                            # Acoustic features (from extracted features dict)
                            **features,

                            # Context features
                            prev_syllable_gap_ms=prev_gap_ms,
                            next_syllable_gap_ms=next_gap_ms,

                            # Labels (clustering labels added later)
                            manual_label=manual_label,
                            clustering_labels={}
                        )

                        records.append(record)

                    except Exception as e:
                        logging.error(f"Error processing syllable {i} in {filename}: {e}")
                        continue

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

        return records

    def _extract_features_from_audio(self, waveform: np.ndarray, sr: int = 32000) -> Dict[str, Any]:
        """
        Extract comprehensive acoustic features from audio waveform using librosa.

        Args:
            waveform: Audio waveform array
            sr: Sample rate in Hz

        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        features = {}

        try:
            # Apply preemphasis if requested
            if self.feature_params.apply_preemphasis:
                waveform = librosa.effects.preemphasis(waveform, coef=self.feature_params.preemphasis_coeff)

            # Ensure waveform is not empty and has sufficient length
            if len(waveform) < self.feature_params.hop_length:
                # Return NaN features for very short segments
                return self._get_nan_features()

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=waveform, sr=sr, hop_length=self.feature_params.hop_length
            )[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=waveform, sr=sr, hop_length=self.feature_params.hop_length
            )[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=waveform, sr=sr, hop_length=self.feature_params.hop_length,
                roll_percent=self.feature_params.spectral_rolloff_percent
            )[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)

            spectral_contrast = librosa.feature.spectral_contrast(
                y=waveform, sr=sr, hop_length=self.feature_params.hop_length
            )
            features['spectral_contrast_mean'] = np.mean(spectral_contrast)
            features['spectral_contrast_std'] = np.std(spectral_contrast)

            zcr = librosa.feature.zero_crossing_rate(
                waveform, hop_length=self.feature_params.hop_length
            )[0]
            features['zero_crossing_rate_mean'] = np.mean(zcr)
            features['zero_crossing_rate_std'] = np.std(zcr)

            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=waveform, sr=sr, n_mfcc=self.feature_params.n_mfcc,
                hop_length=self.feature_params.hop_length, n_fft=self.feature_params.n_fft
            )
            features['mfcc_means'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_stds'] = np.std(mfccs, axis=1).tolist()

            # Fundamental frequency estimation
            f0_features = self._extract_f0_features(waveform, sr)
            features.update(f0_features)

            # Energy features
            rms_energy = librosa.feature.rms(
                y=waveform, hop_length=self.feature_params.hop_length
            )[0]
            features['rms_energy_mean'] = np.mean(rms_energy)
            features['rms_energy_std'] = np.std(rms_energy)

            # Onset strength and tempo features
            onset_strength = librosa.onset.onset_strength(
                y=waveform, sr=sr, hop_length=self.feature_params.hop_length
            )
            features['onset_strength_mean'] = np.mean(onset_strength)
            features['onset_strength_std'] = np.std(onset_strength)

            # Tempo estimation (may not be meaningful for single syllables, but included for completeness)
            try:
                tempo, _ = librosa.beat.beat_track(
                    onset_envelope=onset_strength, sr=sr, hop_length=self.feature_params.hop_length
                )
                # Fix for deprecation warning - ensure tempo is a scalar
                if isinstance(tempo, np.ndarray):
                    features['tempo_estimate'] = float(tempo.item()) if tempo.size > 0 else np.nan
                else:
                    features['tempo_estimate'] = float(tempo)
            except:
                features['tempo_estimate'] = np.nan

        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return self._get_nan_features()

        return features

    def _extract_f0_features(self, waveform: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract fundamental frequency features using librosa.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            Dict[str, float]: F0-related features
        """
        f0_features = {}

        try:
            if self.feature_params.f0_method == 'pyin':
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    waveform, fmin=self.feature_params.f0_fmin,
                    fmax=self.feature_params.f0_fmax, sr=sr
                )
            elif self.feature_params.f0_method == 'yin':
                f0 = librosa.yin(
                    waveform, fmin=self.feature_params.f0_fmin,
                    fmax=self.feature_params.f0_fmax, sr=sr
                )
                voiced_flag = ~np.isnan(f0)
            else:
                # Fallback to piptrack
                pitches, magnitudes = librosa.piptrack(
                    y=waveform, sr=sr, fmin=self.feature_params.f0_fmin,
                    fmax=self.feature_params.f0_fmax
                )
                f0 = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    f0.append(pitch if pitch > 0 else np.nan)
                f0 = np.array(f0)
                voiced_flag = ~np.isnan(f0)

            # Calculate F0 statistics
            valid_f0 = f0[~np.isnan(f0)]

            if len(valid_f0) > 0:
                f0_features['f0_mean'] = np.mean(valid_f0)
                f0_features['f0_std'] = np.std(valid_f0)
                f0_features['f0_min'] = np.min(valid_f0)
                f0_features['f0_max'] = np.max(valid_f0)
                f0_features['f0_range'] = np.max(valid_f0) - np.min(valid_f0)
                f0_features['f0_voiced_fraction'] = len(valid_f0) / len(f0)
            else:
                f0_features['f0_mean'] = np.nan
                f0_features['f0_std'] = np.nan
                f0_features['f0_min'] = np.nan
                f0_features['f0_max'] = np.nan
                f0_features['f0_range'] = np.nan
                f0_features['f0_voiced_fraction'] = 0.0

        except Exception as e:
            logging.debug(f"Error extracting F0 features: {e}")
            f0_features = {
                'f0_mean': np.nan,
                'f0_std': np.nan,
                'f0_min': np.nan,
                'f0_max': np.nan,
                'f0_range': np.nan,
                'f0_voiced_fraction': 0.0
            }

        return f0_features

    def _get_nan_features(self) -> Dict[str, Any]:
        """Return dictionary with NaN values for all features."""
        return {
            'spectral_centroid_mean': np.nan,
            'spectral_centroid_std': np.nan,
            'spectral_bandwidth_mean': np.nan,
            'spectral_bandwidth_std': np.nan,
            'spectral_rolloff_mean': np.nan,
            'spectral_rolloff_std': np.nan,
            'spectral_contrast_mean': np.nan,
            'spectral_contrast_std': np.nan,
            'zero_crossing_rate_mean': np.nan,
            'zero_crossing_rate_std': np.nan,
            'mfcc_means': [np.nan] * self.feature_params.n_mfcc,
            'mfcc_stds': [np.nan] * self.feature_params.n_mfcc,
            'f0_mean': np.nan,
            'f0_std': np.nan,
            'f0_min': np.nan,
            'f0_max': np.nan,
            'f0_range': np.nan,
            'f0_voiced_fraction': 0.0,
            'rms_energy_mean': np.nan,
            'rms_energy_std': np.nan,
            'onset_strength_mean': np.nan,
            'onset_strength_std': np.nan,
            'tempo_estimate': np.nan
        }

    def _calculate_prev_gap(self, syllable_idx: int, onsets: np.ndarray, offsets: np.ndarray) -> float:
        """Calculate gap to previous syllable in milliseconds."""
        if syllable_idx == 0:
            return np.nan  # No previous syllable

        prev_offset = offsets[syllable_idx - 1]
        current_onset = onsets[syllable_idx]
        gap_seconds = current_onset - prev_offset
        return gap_seconds

    def _calculate_next_gap(self, syllable_idx: int, onsets: np.ndarray, offsets: np.ndarray) -> float:
        """Calculate gap to next syllable in milliseconds."""
        if syllable_idx >= len(onsets) - 1:
            return np.nan  # No next syllable

        current_offset = offsets[syllable_idx]
        next_onset = onsets[syllable_idx + 1]
        gap_seconds = next_onset - current_offset
        return gap_seconds

    def _load_clustering_labels(self) -> Dict[str, Dict[str, int]]:
        """
        Load clustering labels from master_summary.csv and clustering result files.

        Returns:
            Dict[str, Dict[str, int]]: {method_id: {hash_id: label}}
        """
        clustering_labels = {}

        try:
            # Load master summary to get clustering result paths
            master_summary_path = os.path.join(self.bird_path, 'master_summary.csv')
            if not os.path.exists(master_summary_path):
                logging.warning(f"No master summary found at {master_summary_path}")
                return clustering_labels

            master_summary = pd.read_csv(master_summary_path)

            for idx, row in master_summary.iterrows():
                try:
                    label_path = row.get('label_path', '')
                    if not label_path or pd.isna(label_path):
                        continue

                    # Create method ID from clustering parameters
                    method_id = self._create_method_id(row, idx)

                    # Load clustering labels from HDF5 file
                    hash_to_label = self._load_clustering_file(label_path)
                    if hash_to_label:
                        clustering_labels[method_id] = hash_to_label

                except Exception as e:
                    logging.error(f"Error loading clustering labels for row {idx}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error loading clustering labels: {e}")

        return clustering_labels


    def _create_method_id(self, row: pd.Series, idx: int) -> str:
        """Create a unique method ID from clustering parameters."""
        try:
            # Extract key parameters for ID
            method = row.get('clustering_method', 'unknown')
            n_neighbors = row.get('n_neighbors', 'na')
            min_dist = row.get('min_dist', 'na')
            min_cluster_size = row.get('min_cluster_size', 'na')
            min_samples = row.get('min_samples', 'na')
            metric = row.get('metric', 'na')

            # Create readable ID
            method_id = f"rank{idx}_{method}_n{n_neighbors}_d{min_dist}_cs{min_cluster_size}_s{min_samples}_{metric}"
            return method_id.replace('.', 'p')  # Replace dots with 'p' for cleaner names

        except Exception as e:
            logging.error(f"Error creating method ID: {e}")
            return f"rank{idx}_unknown"


    def _load_clustering_file(self, label_path: str) -> Dict[str, int]:
        """
        Load clustering labels from HDF5 file.

        Args:
            label_path: Path to clustering labels HDF5 file

        Returns:
            Dict[str, int]: Hash to label mapping
        """
        hash_to_label = {}

        try:
            # Resolve path if needed
            resolved_path = self._resolve_file_path(label_path)

            with tables.open_file(resolved_path, 'r') as f:
                labels = f.root.labels.read()
                hashes = f.root.hashes.read()

                # Convert hashes to strings
                hashes = [h.decode('utf-8') if isinstance(h, (bytes, np.bytes_)) else str(h)
                          for h in hashes]

                # Create mapping
                hash_to_label = dict(zip(hashes, labels))

        except Exception as e:
            logging.error(f"Error loading clustering file {label_path}: {e}")

        return hash_to_label


    def _process_hdf5_array(self, array_data: np.ndarray, field_name: str, node) -> Union[np.ndarray, list]:
        """
        Process HDF5 array data based on its type and expected field structure.

        Args:
            array_data: Raw array data from HDF5
            field_name: Name of the field
            node: HDF5 node object for type information

        Returns:
            Processed data in the correct format for DataFrame
        """
        try:
            # Get array properties
            dtype = array_data.dtype
            shape = array_data.shape

            logging.debug(f"Processing field '{field_name}': dtype={dtype}, shape={shape}")

            # Handle string/Unicode fields (bytes need to be decoded)
            if dtype.kind in ['U', 'S', 'a']:  # Unicode, byte string, or void
                if field_name in ['hash_id', 'bird_name', 'song_file', 'manual_label', 'clustering_labels']:
                    # Convert bytes to strings if needed
                    if dtype.kind == 'S' or (dtype.kind == 'U' and array_data.dtype.char == 'S'):
                        processed = [item.decode('utf-8') if isinstance(item, bytes) else str(item)
                                     for item in array_data]
                    else:
                        processed = [str(item) for item in array_data]
                    return processed
                else:
                    # Generic string handling
                    return [item.decode('utf-8') if isinstance(item, bytes) else str(item)
                            for item in array_data]

            # Handle multi-dimensional numeric arrays (MFCC features)
            elif field_name in ['mfcc_means', 'mfcc_stds']:
                if array_data.ndim == 2:
                    # Convert 2D array to list of lists for DataFrame compatibility
                    return [row.tolist() for row in array_data]
                elif array_data.ndim == 1:
                    # Handle case where it might have been flattened
                    expected_length = self.feature_params.n_mfcc
                    if len(array_data) % expected_length == 0:
                        n_records = len(array_data) // expected_length
                        reshaped = array_data.reshape(n_records, expected_length)
                        return [row.tolist() for row in reshaped]
                    else:
                        # Fallback: return as list
                        return array_data.tolist()
                else:
                    return array_data.tolist()

            # Handle 1D numeric arrays
            elif array_data.ndim == 1:
                # Check for numeric types
                if dtype.kind in ['i', 'u', 'f', 'c']:  # integer, unsigned int, float, complex
                    return array_data
                else:
                    # Convert to appropriate Python types
                    return array_data.tolist()

            # Handle higher-dimensional arrays (shouldn't occur in our case, but safety)
            elif array_data.ndim > 2:
                logging.warning(f"Unexpected {array_data.ndim}D array for field '{field_name}'. Converting to list.")
                return array_data.tolist()

            # Handle 2D arrays (other than MFCC)
            elif array_data.ndim == 2:
                logging.warning(f"Unexpected 2D array for field '{field_name}'. Converting to list of lists.")
                return [row.tolist() for row in array_data]

            # Default case
            else:
                return array_data

        except Exception as e:
            logging.error(f"Error processing HDF5 array for field '{field_name}': {e}")
            logging.error(f"Array dtype: {array_data.dtype}, shape: {array_data.shape}")
            # Return as-is and let pandas handle it
            return array_data


    def _resolve_file_path(self, file_path: str) -> str:
        """
        Resolve file path, handling cross-platform and network path issues.
        """
        # If file exists as-is, return it
        if os.path.exists(file_path):
            return file_path

        try:
            # Handle network path resolution
            from tools.system_utils import check_sys_for_macaw_root
            path_to_macaw = check_sys_for_macaw_root()

            # Extract relative path (last 9 components as in original)
            path_parts = file_path.replace('\\', '/').split('/')
            if len(path_parts) >= 9:
                relative_path = '/'.join(path_parts[-9:])
                resolved_path = os.path.join(path_to_macaw, relative_path)
                if os.path.exists(resolved_path):
                    return resolved_path

            # If all else fails, return original path
            return file_path

        except Exception as e:
            logging.warning(f"Error resolving path {file_path}: {e}")
            return file_path

    def _save_to_hdf5(self, records: List[SyllableRecord]) -> bool:
        """
        Save syllable records to HDF5 format for efficient analysis.

        Args:
            records: List of syllable records to save

        Returns:
            bool: Success status
        """
        try:
            with tables.open_file(self.hdf5_path, 'w') as f:
                # Convert records to structured arrays for efficient storage
                data_dict = self._records_to_arrays(records)

                # Save each array to HDF5 with explicit type handling
                for key, array in data_dict.items():
                    try:
                        if isinstance(array, list):
                            # Handle list data (like MFCC coefficients)
                            array = np.array(array)

                        # Ensure proper data types for specific fields
                        if key in ['hash_id', 'bird_name', 'song_file', 'manual_label', 'clustering_labels']:
                            # String fields - ensure Unicode string type with sufficient length
                            if array.dtype.kind not in ['U', 'S']:
                                array = np.array([str(item) for item in array], dtype='U')
                            else:
                                # Ensure we have Unicode strings, not byte strings
                                max_len = max(len(str(item)) for item in array) if len(array) > 0 else 1
                                array = array.astype(f'U{max_len}')

                        elif key in ['mfcc_means', 'mfcc_stds']:
                            # MFCC fields - ensure 2D float array
                            if array.ndim == 1:
                                # Reshape if somehow flattened
                                expected_length = self.feature_params.n_mfcc
                                if len(array) % expected_length == 0:
                                    n_records = len(array) // expected_length
                                    array = array.reshape(n_records, expected_length)
                            array = array.astype(np.float64)

                        elif key in ['syllable_index', 'position_in_song', 'song_length_syllables']:
                            # Integer fields
                            array = array.astype(np.int32)

                        else:
                            # Numeric fields - ensure float64 for consistency
                            if array.dtype.kind in ['i', 'u', 'f']:
                                array = array.astype(np.float64)

                        # Create the array in HDF5
                        f.create_array('/', key, obj=array)
                        logging.debug(f"Saved field '{key}' with dtype {array.dtype} and shape {array.shape}")

                    except Exception as e:
                        logging.error(f"Error saving field '{key}': {e}")
                        logging.error(f"Array type: {type(array)}, dtype: {getattr(array, 'dtype', 'N/A')}")
                        raise

                # Save metadata with numpy type conversion
                metadata = {
                    'n_syllables': int(len(records)),
                    'bird_name': self.bird_name,
                    'feature_params': asdict(self.feature_params),
                    'creation_timestamp': pd.Timestamp.now().isoformat(),
                    'hdf5_version': tables.__version__
                }

                # Convert any numpy types in feature_params to native Python types
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj

                metadata = convert_numpy_types(metadata)

                # Save metadata as JSON string in Unicode array
                metadata_json = json.dumps(metadata)
                f.create_array('/', 'metadata', obj=np.array([metadata_json], dtype=f'U{len(metadata_json)}'))

            logging.info(f"Saved {len(records)} records to HDF5: {self.hdf5_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving to HDF5: {e}")
            return False


    def _save_to_csv(self, records: List[SyllableRecord]) -> bool:
        """
        Save syllable records to CSV format for easy inspection.

        Args:
            records: List of syllable records to save

        Returns:
            bool: Success status
        """
        try:
            # Convert records to DataFrame
            df = self._records_to_dataframe(records)

            # Save to CSV
            df.to_csv(self.csv_path, index=False)

            logging.info(f"Saved {len(records)} records to CSV: {self.csv_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return False

    def _records_to_arrays(self, records: List[SyllableRecord]) -> Dict[str, np.ndarray]:
        """Convert list of records to dictionary of arrays for HDF5 storage."""
        if not records:
            return {}

        data_dict = {}

        # Extract all fields from first record to get structure
        first_record = records[0]

        for field_name in first_record.__dataclass_fields__:
            values = []

            for record in records:
                value = getattr(record, field_name)

                if field_name == 'clustering_labels':
                    # Handle clustering labels dictionary
                    if value:
                        converted_labels = {}
                        for k, v in value.items():
                            # Convert numpy int64 to native Python int
                            if isinstance(v, np.integer):
                                converted_labels[k] = int(v)
                            else:
                                converted_labels[k] = v
                        values.append(json.dumps(converted_labels))
                    else:
                        values.append('{}')

                elif isinstance(value, list):
                    # Handle list fields (like MFCC coefficients)
                    if field_name in ['mfcc_means', 'mfcc_stds']:
                        expected_length = self.feature_params.n_mfcc
                        if len(value) != expected_length:
                            # Pad or truncate to expected length
                            padded_value = (value + [np.nan] * expected_length)[:expected_length]
                            values.append(padded_value)
                        else:
                            values.append(value)
                    else:
                        values.append(value)

                elif value is None:
                    # Handle None values appropriately
                    if field_name in ['manual_label']:
                        values.append('')  # Empty string for string fields
                    else:
                        values.append(np.nan)  # NaN for numeric fields

                else:
                    values.append(value)

            # Convert to appropriate numpy array with explicit type handling
            try:
                if field_name == 'clustering_labels':
                    # JSON strings - use Unicode with sufficient length
                    max_len = max(len(s) for s in values) if values else 1
                    data_dict[field_name] = np.array(values, dtype=f'U{max_len}')

                elif field_name in ['hash_id', 'bird_name', 'song_file', 'manual_label']:
                    # String fields - ensure all are strings and use Unicode
                    str_values = [str(v) if v is not None else '' for v in values]
                    max_len = max(len(s) for s in str_values) if str_values else 1
                    data_dict[field_name] = np.array(str_values, dtype=f'U{max_len}')

                elif field_name in ['mfcc_means', 'mfcc_stds']:
                    # 2D numeric arrays
                    array_data = np.array(values, dtype=np.float64)
                    logging.debug(f"Created {field_name} array with shape: {array_data.shape}")
                    data_dict[field_name] = array_data

                elif field_name in ['syllable_index', 'position_in_song', 'song_length_syllables']:
                    # Integer fields - ensure int32
                    int_values = [int(v) if not pd.isna(v) else -1 for v in values]
                    data_dict[field_name] = np.array(int_values, dtype=np.int32)

                else:
                    # All other numeric fields - use float64
                    numeric_values = []
                    for v in values:
                        if v is None or pd.isna(v):
                            numeric_values.append(np.nan)
                        elif isinstance(v, (int, float, np.number)):
                            numeric_values.append(float(v))
                        else:
                            # Try to convert to float, fallback to NaN
                            try:
                                numeric_values.append(float(v))
                            except (ValueError, TypeError):
                                numeric_values.append(np.nan)

                    data_dict[field_name] = np.array(numeric_values, dtype=np.float64)

            except Exception as e:
                logging.error(f"Error converting field {field_name} to array: {e}")
                logging.error(f"Sample values: {values[:3] if len(values) > 3 else values}")
                logging.error(f"Value types: {[type(v) for v in values[:3]]}")
                raise

        return data_dict


    def _records_to_dataframe(self, records: List[SyllableRecord]) -> pd.DataFrame:
        """Convert list of records to pandas DataFrame for CSV storage."""
        if not records:
            return pd.DataFrame()

        data_rows = []

        for record in records:
            row = {}

            # Add all fields except complex ones
            for field_name in record.__dataclass_fields__:
                value = getattr(record, field_name)

                if field_name == 'clustering_labels':
                    # Flatten clustering labels into separate columns
                    for method_id, label in value.items():
                        row[f'cluster_{method_id}'] = label
                elif field_name in ['mfcc_means', 'mfcc_stds']:
                    # Flatten MFCC coefficients into separate columns
                    for i, coeff in enumerate(value):
                        row[f'{field_name}_{i + 1}'] = coeff
                else:
                    row[field_name] = value

            data_rows.append(row)

        return pd.DataFrame(data_rows)

    def _save_feature_params(self) -> bool:
        """Save feature extraction parameters to JSON file."""
        try:
            with open(self.params_path, 'w') as f:
                json.dump(asdict(self.feature_params), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving feature parameters: {e}")
            return False

    def load_database(self) -> pd.DataFrame:
        """
        Load existing syllable database from HDF5 file.

        Returns:
            pd.DataFrame: Loaded syllable database
        """
        try:
            if not os.path.exists(self.hdf5_path):
                logging.error(f"Database file not found: {self.hdf5_path}")
                return pd.DataFrame()

            data_dict = {}

            with tables.open_file(self.hdf5_path, 'r') as f:
                # Load all arrays
                for node in f.list_nodes(f.root, classname='Array'):
                    if node._v_name == 'metadata':
                        continue

                    array_data = node.read()
                    field_name = node._v_name

                    # Handle different data types based on HDF5 node properties
                    processed_data = self._process_hdf5_array(array_data, field_name, node)
                    data_dict[field_name] = processed_data

            # Convert to DataFrame
            df = pd.DataFrame(data_dict)

            # Process clustering labels (convert from JSON strings)
            if 'clustering_labels' in df.columns:
                clustering_cols = {}
                for idx, labels_json in enumerate(df['clustering_labels']):
                    try:
                        # Handle both string and bytes
                        if isinstance(labels_json, (bytes, np.bytes_)):
                            labels_json = labels_json.decode('utf-8')

                        labels_dict = json.loads(labels_json) if labels_json != '{}' else {}
                        for method_id, label in labels_dict.items():
                            if method_id not in clustering_cols:
                                clustering_cols[method_id] = [np.nan] * len(df)
                            clustering_cols[method_id][idx] = label
                    except Exception as e:
                        logging.debug(f"Error processing clustering labels at index {idx}: {e}")
                        continue

                # Add clustering columns to DataFrame
                for method_id, labels in clustering_cols.items():
                    df[f'cluster_{method_id}'] = labels

            logging.info(f"Loaded {len(df)} syllable records from database")
            return df

        except Exception as e:
            logging.error(f"Error loading database: {e}")
            return pd.DataFrame()

    def analyze_clustering_quality(self, clustering_method_id: str) -> Dict[str, Any]:
        """Keep the comprehensive analysis but reduce logging verbosity."""

        # Temporarily reduce logging to avoid overflow
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)  # Only show errors during analysis

        try:
            # Load database
            df = self.load_database()
            if df.empty:
                return {}

            cluster_col = f'cluster_{clustering_method_id}'
            if cluster_col not in df.columns:
                return {}

            # Get feature columns (exclude identifiers and labels)
            feature_cols = [col for col in df.columns if col not in [
                'hash_id', 'bird_name', 'song_file', 'syllable_index', 'manual_label'
            ] and not col.startswith('cluster_')]

            # Remove rows with missing cluster labels
            analysis_df = df[df[cluster_col] != -1].copy()
            if analysis_df.empty:
                return {}

            unique_clusters = analysis_df[cluster_col].unique()
            n_clusters = len(unique_clusters)

            analysis_results = {
                'clustering_method': clustering_method_id,
                'n_clusters': n_clusters,
                'n_syllables': len(analysis_df),
                'cluster_sizes': {},
                'feature_analysis': {},
                'cluster_separation': {}
            }

            # Analyze each cluster
            for cluster_id in unique_clusters:
                cluster_data = analysis_df[analysis_df[cluster_col] == cluster_id]
                analysis_results['cluster_sizes'][str(cluster_id)] = len(cluster_data)

            # Analyze feature distributions (keep all your existing logic)
            separation_ratios = []
            for feature in feature_cols:
                if feature in ['mfcc_means', 'mfcc_stds']:
                    continue  # Skip complex features for now

                feature_data = analysis_df[feature].dropna()
                if len(feature_data) == 0:
                    continue

                # Your existing analysis logic here...
                # (keeping all the variance calculations)

                within_cluster_var = 0
                between_cluster_var = 0
                overall_mean = feature_data.mean()

                cluster_means = []
                cluster_vars = []

                for cluster_id in unique_clusters:
                    cluster_feature_data = analysis_df[
                        analysis_df[cluster_col] == cluster_id
                        ][feature].dropna()

                    if len(cluster_feature_data) > 1:
                        cluster_mean = cluster_feature_data.mean()
                        cluster_var = cluster_feature_data.var()
                        cluster_size = len(cluster_feature_data)

                        cluster_means.append(cluster_mean)
                        cluster_vars.append(cluster_var)
                        within_cluster_var += cluster_var * cluster_size
                        between_cluster_var += cluster_size * (cluster_mean - overall_mean) ** 2

                # Normalize variances
                total_samples = len(feature_data)
                within_cluster_var /= total_samples
                between_cluster_var /= total_samples

                # Calculate separation ratio
                separation_ratio = between_cluster_var / within_cluster_var if within_cluster_var > 0 else np.inf

                analysis_results['feature_analysis'][feature] = {
                    'within_cluster_variance': within_cluster_var,
                    'between_cluster_variance': between_cluster_var,
                    'separation_ratio': separation_ratio,
                    'cluster_means': cluster_means,
                    'cluster_variances': cluster_vars,
                    'overall_mean': overall_mean,
                    'overall_variance': feature_data.var()
                }

                if not np.isinf(separation_ratio):
                    separation_ratios.append(separation_ratio)

            # Calculate overall clustering quality metrics
            analysis_results['cluster_separation'] = {
                'mean_separation_ratio': np.mean(separation_ratios) if separation_ratios else 0,
                'median_separation_ratio': np.median(separation_ratios) if separation_ratios else 0,
                'min_separation_ratio': np.min(separation_ratios) if separation_ratios else 0,
                'max_separation_ratio': np.max(separation_ratios) if separation_ratios else 0
            }

            return analysis_results

        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)


    def compare_manual_vs_clustering(self, clustering_method_id: str) -> Dict[str, Any]:
        """
        Compare manual labels with clustering results using feature analysis.

        Args:
            clustering_method_id: ID of clustering method to compare

        Returns:
            Dict[str, Any]: Comparison results
        """
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

        try:
            # Your existing logic here, just remove the verbose logging statements
            df = self.load_database()
            if df.empty:
                return {}

            cluster_col = f'cluster_{clustering_method_id}'
            if cluster_col not in df.columns:
                return {}

            comparison_df = df[
                (df['manual_label'].notna()) &
                (df['manual_label'] != '') &
                (df['manual_label'] != 'None') &  # Exclude 'None' labels
                (df[cluster_col] != -1)
                ].copy()

            if comparison_df.empty:
                return {}

            # Analyze manual label clustering quality
            manual_analysis = self._analyze_label_feature_consistency(
                comparison_df, 'manual_label', 'Manual Labels'
            )

            # Analyze clustering label quality
            cluster_analysis = self._analyze_label_feature_consistency(
                comparison_df, cluster_col, f'Clustering ({clustering_method_id})'
            )

            comparison_results = {
                'clustering_method': clustering_method_id,
                'n_compared_syllables': len(comparison_df),
                'manual_analysis': manual_analysis,
                'clustering_analysis': cluster_analysis,
                'comparison_summary': {
                    'manual_mean_separation': manual_analysis.get('mean_separation_ratio', 0),
                    'clustering_mean_separation': cluster_analysis.get('mean_separation_ratio', 0),
                    'relative_quality': 0
                }
            }

            # Calculate relative quality score
            manual_sep = manual_analysis.get('mean_separation_ratio', 0)
            cluster_sep = cluster_analysis.get('mean_separation_ratio', 0)

            if manual_sep > 0:
                comparison_results['comparison_summary']['relative_quality'] = cluster_sep / manual_sep

            return comparison_results

        finally:
            # Restore original logging level
            logging.getLogger().setLevel(original_level)


    def _analyze_label_feature_consistency(self, df: pd.DataFrame, label_col: str, label_name: str) -> Dict[str, Any]:
        """
        Analyze how well a labeling scheme separates syllables based on acoustic features.

        Args:
            df: DataFrame with syllables and labels
            label_col: Column name containing labels
            label_name: Human-readable name for the labeling scheme

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get feature columns
            feature_cols = [col for col in df.columns if col not in [
                'hash_id', 'bird_name', 'song_file', 'syllable_index', 'manual_label'
            ] and not col.startswith('cluster_') and col not in ['mfcc_means', 'mfcc_stds']]

            unique_labels = df[label_col].unique()
            n_labels = len(unique_labels)

            analysis = {
                'label_name': label_name,
                'n_labels': n_labels,
                'label_sizes': {},
                'feature_separation': {},
                'mean_separation_ratio': 0
            }

            # Calculate label sizes
            for label in unique_labels:
                analysis['label_sizes'][str(label)] = len(df[df[label_col] == label])

            # Analyze feature separation for each feature
            separation_ratios = []

            for feature in feature_cols:
                feature_data = df[feature].dropna()
                if len(feature_data) < 2:
                    continue

                # Calculate within-label and between-label variance
                within_var = 0
                between_var = 0
                overall_mean = feature_data.mean()
                total_samples = len(feature_data)

                for label in unique_labels:
                    label_data = df[df[label_col] == label][feature].dropna()

                    if len(label_data) > 1:
                        label_mean = label_data.mean()
                        label_var = label_data.var()
                        label_size = len(label_data)

                        # Weighted within-label variance
                        within_var += label_var * label_size

                        # Between-label variance
                        between_var += label_size * (label_mean - overall_mean) ** 2

                # Normalize
                within_var /= total_samples
                between_var /= total_samples

                # Separation ratio
                sep_ratio = between_var / within_var if within_var > 0 else np.inf

                if not np.isinf(sep_ratio):
                    separation_ratios.append(sep_ratio)

                analysis['feature_separation'][feature] = {
                    'within_variance': within_var,
                    'between_variance': between_var,
                    'separation_ratio': sep_ratio
                }

            # Overall separation quality
            if separation_ratios:
                analysis['mean_separation_ratio'] = np.mean(separation_ratios)
                analysis['median_separation_ratio'] = np.median(separation_ratios)

            return analysis

        except Exception as e:
            return {}

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the syllable database.

        Returns:
            Dict[str, Any]: Summary information
        """
        try:
            df = self.load_database()
            if df.empty:
                return {'error': 'Database is empty or could not be loaded'}

            # Basic statistics
            summary = {
                'bird_name': self.bird_name,
                'total_syllables': len(df),
                'unique_songs': df['song_file'].nunique() if 'song_file' in df.columns else 0,
                'duration_stats': {},
                'manual_labels': {},
                'clustering_methods': [],
                'feature_coverage': {}
            }

            # Duration statistics
            if 'duration_ms' in df.columns:
                duration_data = df['duration_ms'].dropna()
                if len(duration_data) > 0:
                    summary['duration_stats'] = {
                        'mean_ms': float(duration_data.mean()),
                        'std_ms': float(duration_data.std()),
                        'min_ms': float(duration_data.min()),
                        'max_ms': float(duration_data.max()),
                        'median_ms': float(duration_data.median())
                    }

            # Manual label statistics
            if 'manual_label' in df.columns:
                manual_labels = df['manual_label'].dropna()
                manual_labels = manual_labels[manual_labels != '']

                if len(manual_labels) > 0:
                    label_counts = manual_labels.value_counts().to_dict()
                    summary['manual_labels'] = {
                        'n_labeled_syllables': len(manual_labels),
                        'n_unique_labels': len(label_counts),
                        'label_distribution': {str(k): int(v) for k, v in label_counts.items()},
                        'coverage_fraction': len(manual_labels) / len(df)
                    }

            # Clustering methods
            clustering_cols = [col for col in df.columns if col.startswith('cluster_')]
            for col in clustering_cols:
                method_id = col.replace('cluster_', '')
                cluster_data = df[col].dropna()
                cluster_data = cluster_data[cluster_data != -1]

                if len(cluster_data) > 0:
                    cluster_counts = cluster_data.value_counts().to_dict()
                    summary['clustering_methods'].append({
                        'method_id': method_id,
                        'n_clustered_syllables': len(cluster_data),
                        'n_clusters': len(cluster_counts),
                        'coverage_fraction': len(cluster_data) / len(df),
                        'cluster_sizes': {str(k): int(v) for k, v in cluster_counts.items()}
                    })

            # Feature coverage (percentage of non-NaN values)
            feature_cols = [col for col in df.columns if col not in [
                'hash_id', 'bird_name', 'song_file', 'syllable_index', 'manual_label'
            ] and not col.startswith('cluster_')]

            for feature in feature_cols:
                if feature in ['mfcc_means', 'mfcc_stds']:
                    continue  # Skip complex features

                non_nan_count = df[feature].notna().sum()
                summary['feature_coverage'][feature] = {
                    'coverage_fraction': float(non_nan_count / len(df)),
                    'n_valid_values': int(non_nan_count)
                }

            return summary

        except Exception as e:
            logging.error(f"Error generating database summary: {e}")
            return {'error': str(e)}

    def save_analysis_results(self, analysis_results: Dict[str, Any],
                              analysis_type: str = 'clustering_quality') -> str:
        """
        Save analysis results to JSON file.

        Args:
            analysis_results: Results dictionary to save
            analysis_type: Type of analysis for filename

        Returns:
            str: Path to saved file
        """
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{analysis_type}_{timestamp}.json'
            save_path = os.path.join(self.database_dir, filename)

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            converted_results = convert_numpy_types(analysis_results)

            with open(save_path, 'w') as f:
                json.dump(converted_results, f, indent=2)

            logging.info(f"Saved analysis results to: {save_path}")
            return save_path

        except Exception as e:
            logging.error(f"Error saving analysis results: {e}")
            return ""

    def analyze_top_clustering_methods(self, n_top_methods: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Analyze only the top N clustering methods instead of all methods.

        Args:
            n_top_methods: Number of top-ranked methods to analyze
        """
        # Load database to get available clustering methods
        df = self.load_database()
        if df.empty:
            logging.error("Cannot analyze: database is empty")
            return {}

        # Get clustering columns and sort by rank
        clustering_cols = [col for col in df.columns if col.startswith('cluster_rank')]
        if not clustering_cols:
            logging.warning("No clustering methods found in database")
            return {}

        clustering_cols.sort(key=lambda x: int(x.split('rank')[1].split('_')[0]))

        # Take only top N methods
        top_methods = clustering_cols[:n_top_methods]
        clustering_methods = [col.replace('cluster_', '') for col in top_methods]

        total_methods = len([c for c in df.columns if c.startswith('cluster_')])
        logging.info(
            f"Analyzing top {len(clustering_methods)} clustering methods (skipping {total_methods - len(clustering_methods)} lower-ranked methods)")

        all_results = {}
        for method_id in clustering_methods:
            try:
                # Use existing analysis functions (they work!)
                quality_results = self.analyze_clustering_quality(method_id)
                comparison_results = self.compare_manual_vs_clustering(method_id)

                all_results[method_id] = {
                    'clustering_quality': quality_results,
                    'manual_comparison': comparison_results,
                    'analysis_timestamp': pd.Timestamp.now().isoformat()
                }

            except Exception as e:
                logging.error(f"Error analyzing method {method_id}: {e}")
                all_results[method_id] = {'error': str(e)}

        return all_results

    def analyze_and_save_top_methods(self, n_top_methods: int = 3) -> Dict[str, Dict[str, Any]]:
        """Generate analysis for top methods only and save to single summary file."""

        # Analyze top methods
        results = self.analyze_top_clustering_methods(n_top_methods)

        if not results:
            logging.warning("No analysis results generated")
            return {}

        # Save single comprehensive summary (instead of 77 files)
        summary_path = self.save_analysis_results(results, 'top_clustering_analysis')

        # Also create a simple summary table for quick inspection
        summary_table = []
        for method_id, data in results.items():
            if 'error' not in data:
                quality = data.get('clustering_quality', {})
                comparison = data.get('manual_comparison', {})

                summary_table.append({
                    'method': method_id,
                    'n_clusters': quality.get('n_clusters', 0),
                    'n_syllables': quality.get('n_syllables', 0),
                    'mean_separation': quality.get('cluster_separation', {}).get('mean_separation_ratio', 0),
                    'relative_quality': comparison.get('comparison_summary', {}).get('relative_quality', 0),
                    'manual_separation': comparison.get('comparison_summary', {}).get('manual_mean_separation', 0),
                    'clustering_separation': comparison.get('comparison_summary', {}).get('clustering_mean_separation',
                                                                                          0)
                })

        # Save summary table as CSV for easy inspection
        if summary_table:
            summary_df = pd.DataFrame(summary_table)
            csv_path = os.path.join(self.database_dir, 'clustering_summary.csv')
            summary_df.to_csv(csv_path, index=False)
            logging.info(f"Saved summary table to: {csv_path}")

            # Log key results
            best_method = summary_table[0]  # First is rank 0
            logging.info(f"Best method ({best_method['method']}): "
                         f"{best_method['n_clusters']} clusters, "
                         f"separation: {best_method['mean_separation']:.3f}, "
                         f"vs manual: {best_method['relative_quality']:.3f}")

        return results

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_syllable_database(bird_path: str, force_rebuild: bool = False,
                            feature_params: FeatureExtractionParams = None) -> SyllableDatabase:
    """
    Convenience function to build syllable database for a bird.

    Args:
        bird_path: Path to bird directory
        force_rebuild: Whether to rebuild existing database
        feature_params: Custom feature extraction parameters

    Returns:
        SyllableDatabase: Database instance
    """
    db = SyllableDatabase(bird_path, feature_params)
    success = db.build_database(force_rebuild=force_rebuild)

    if success:
        logging.info(f"Successfully built database for {db.bird_name}")

        # Generate and save summary
        summary = db.get_database_summary()
        summary_path = db.save_analysis_results(summary, 'database_summary')

        return db
    else:
        logging.error(f"Failed to build database for {db.bird_name}")
        return db


def analyze_top_clustering_methods_only(bird_path: str, n_methods: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    Analyze only top clustering methods instead of all methods.

    Args:
        bird_path: Path to bird directory
        n_methods: Number of top methods to analyze (default=3)

    Returns:
        Dict[str, Dict[str, Any]]: Analysis results for top methods only
    """
    db = SyllableDatabase(bird_path)
    return db.analyze_and_save_top_methods(n_methods)


def main(bird_paths: List[str], force_rebuild: bool = False, n_methods: int = 3) -> None:
    """
    Main function to build syllable databases for multiple birds.

    Args:
        bird_paths: List of paths to bird directories
        force_rebuild: Whether to rebuild existing databases
        n_methods: Number of top clustering methods to analyze
    """
    try:
        logging.info(f"Building syllable databases for {len(bird_paths)} birds")
        logging.info(f"Will analyze top {n_methods} clustering methods per bird")

        successful_birds = []
        failed_birds = []
        analysis_summary = []

        for bird_path in tqdm(bird_paths, desc="Processing birds"):
            bird_name = os.path.basename(bird_path)

            try:
                logging.info(f"Processing bird: {bird_name}")

                # Build database (keep all features)
                db = build_syllable_database(bird_path, force_rebuild=force_rebuild)

                # Analyze only top N methods instead of all
                analysis_results = analyze_top_clustering_methods_only(bird_path, n_methods)

                if analysis_results:
                    successful_birds.append(bird_name)

                    # Extract key metrics for cross-bird summary
                    best_method_id = list(analysis_results.keys())[0]
                    if 'error' not in analysis_results[best_method_id]:
                        quality = analysis_results[best_method_id]['clustering_quality']
                        comparison = analysis_results[best_method_id]['manual_comparison']

                        analysis_summary.append({
                            'bird': bird_name,
                            'best_method': best_method_id,
                            'n_clusters': quality.get('n_clusters', 0),
                            'n_syllables': quality.get('n_syllables', 0),
                            'separation_ratio': quality.get('cluster_separation', {}).get('mean_separation_ratio', 0),
                            'relative_quality': comparison.get('comparison_summary', {}).get('relative_quality', 0)
                        })

                        logging.info(f"  → {quality.get('n_clusters', 0)} clusters, "
                                     f"separation: {quality.get('cluster_separation', {}).get('mean_separation_ratio', 0):.3f}, "
                                     f"vs manual: {comparison.get('comparison_summary', {}).get('relative_quality', 0):.3f}")
                else:
                    failed_birds.append(bird_name)
                    logging.warning(f"No analysis results for {bird_name}")

            except Exception as e:
                logging.error(f"Error processing {bird_name}: {e}")
                failed_birds.append(bird_name)
                continue

        # Save cross-bird summary
        if analysis_summary:
            summary_df = pd.DataFrame(analysis_summary)
            summary_path = 'all_birds_clustering_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            logging.info(f"Saved cross-bird summary to: {summary_path}")

            # Overall statistics
            logging.info(f"\nOverall Results:")
            logging.info(f"  Average separation ratio: {summary_df['separation_ratio'].mean():.3f}")
            logging.info(f"  Average relative quality: {summary_df['relative_quality'].mean():.3f}")
            logging.info(f"  Best performing bird: {summary_df.loc[summary_df['relative_quality'].idxmax(), 'bird']} "
                         f"(quality: {summary_df['relative_quality'].max():.3f})")

        # Report results
        logging.info(f"\nDatabase building complete:")
        logging.info(f"  Successful: {len(successful_birds)} birds")
        logging.info(f"  Failed: {len(failed_birds)} birds")

        if failed_birds:
            logging.warning(f"Failed birds: {failed_birds}")

    except Exception as e:
        logging.error(f"Error in main database building pipeline: {e}")
        raise


if __name__ == '__main__':
    # Setup logging
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'syllable_database.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting syllable database pipeline")

    # Test on example datasets
    test_paths = [
        os.path.join('/Volumes', 'Extreme SSD', 'wseg test'),
        os.path.join('/Volumes', 'Extreme SSD', 'evsong test'),
    ]

    bird_paths = []
    for dataset_path in test_paths:
        if os.path.exists(dataset_path):
            # Get all bird directories
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Check if it has syllable data
                    syllable_dir = os.path.join(item_path, 'data', 'syllables')
                    if os.path.exists(syllable_dir):
                        bird_paths.append(item_path)

    if bird_paths:
        logging.info(f"Found {len(bird_paths)} birds to process")
        main(bird_paths, force_rebuild=False)
    else:
        logging.error("No bird directories found")
