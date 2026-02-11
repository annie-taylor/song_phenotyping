# Standard library imports
import os
import logging
import traceback
from tqdm import tqdm
from typing import Union, List, Dict, Any, Tuple, Optional

# External library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tables  # PyTables for HDF5 file handling
from scipy.stats import skew, kurtosis
from dataclasses import dataclass, field
from enum import Enum

# Custom imports
from tools.system_utils import check_sys_for_macaw_root  # Resolves system paths

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class LabelType(Enum):
    """Enumeration for different label types in birdsong analysis."""
    MANUAL = "manual"
    AUTO = "auto"

@dataclass
class AnalysisConfig:
    """Configuration parameters for birdsong sequence analysis."""

    # Vocabulary filtering
    min_syllable_proportion: float = 0.02
    adaptive_threshold_factor: float = 0.25

    # Repeat analysis
    repeat_candidate_range: range = field(default_factory=lambda: range(2, 20))
    repeat_significance_threshold: float = 0.2
    dyad_threshold: float = 0.7

    # Statistical analysis
    importance_threshold_offset: float = 0.05 # For repeat_tendency calculation

    # Visualization
    figure_dpi: int = 500
    heatmap_annotation_size: int = 7
    max_example_specs: int = 100

    # File processing
    hdf5_compression_level: int = 5
    hdf5_compression_lib: str = 'zlib'

@dataclass
class AnalysisPaths:
    """Paths for birdsong analysis pipeline."""
    bird_folder: str
    data_folder: str
    figure_folder: str

    # Computed paths
    master_summary: str = field(init=False)
    phenotype_summary: str = field(init=False)
    sequence_data: Optional[str] = field(default=None, init=False)
    current_rank_folder: Optional[str] = field(default=None, init=False)

    # Image paths (populated during processing)
    transition_imgs: List[str] = field(default_factory=list, init=False)
    cluster_imgs: List[str] = field(default_factory=list, init=False)
    repeat_img: Optional[str] = field(default=None, init=False)
    conf_matrices: Optional[str] = field(default=None, init=False)
    graph_img: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize computed paths."""
        self.master_summary = os.path.join(self.bird_folder, 'master_summary.csv')
        self.phenotype_summary = os.path.join(self.bird_folder, 'phenotype_summary.csv')


@dataclass
class SequenceData:
    """Simple data container for sequence analysis results."""
    bird_name: str
    syl_type: str
    repertoire_size: int
    vocab: List[Union[str, int]]
    syl_counts: Dict[Union[str, int], int]
    syl_proportions: np.ndarray
    n_songs: int
    n_syllables: int
    all_auto_syllables: List[Union[str, int]]
    entropy: float
    entropy_scaled: float
    paths_to_songs: List[str]
    repeat_counts: Optional[pd.DataFrame] = None

    def save_to_pickle(self, path: str) -> None:
        """Save sequence data to pickle file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, path: str) -> 'SequenceData':
        """Load sequence data from pickle file."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

# ============================================================================
# TYPE HANDLING AND VALIDATION
# ============================================================================

class LabelHandler:
    """Unified handler for manual and automatic labels with consistent type conversion."""
    def __init__(self, label_type: LabelType):
        """
        Initialize label handler for specific label type.

        Parameters
        ----------
        label_type : LabelType
            Whether handling manual (string) or automatic (integer) labels
        """
        self.label_type = label_type

    @property
    def start_token(self) -> Union[str, int]:
        """Get appropriate start token for this label type."""
        return 's' if self.label_type == LabelType.MANUAL else -5

    @property
    def end_token(self) -> Union[str, int]:
        """Get appropriate end token for this label type."""
        return 'z' if self.label_type == LabelType.MANUAL else -3

    @property
    def non_syl_tokens(self) -> List[Union[str, int]]:
        """Get appropriate non-syllable tokens for this label type."""
        if self.label_type == LabelType.MANUAL:
            return ['s', 'z', '-', '\r']
        else:
            return [-5, -3, -1]  # -1 is HDBSCAN uncertain label

    def normalize_labels(self, raw_labels: List[Any]) -> List[Union[str, int]]:
        """
        Convert raw labels to consistent format based on type.

        Parameters
        ----------
        raw_labels : List[Any]
            Raw labels from file (may be bytes, strings, ints, etc.)

        Returns
        -------
        List[Union[str, int]]
            Normalized labels in consistent format
        """
        if self.label_type == LabelType.MANUAL:
            return [self._to_string(label) for label in raw_labels]
        else:
            return [self._to_int(label) for label in raw_labels]

    def _to_string(self, item: Any) -> str:
        """Convert any label format to string."""
        if isinstance(item, bytes):
            return item.decode('utf-8')
        elif isinstance(item, (np.str_, np.bytes_)):
            return str(item)
        return str(item)

    def _to_int(self, item: Any) -> int:
        """Convert any label format to integer."""
        if isinstance(item, bytes):
            decoded = item.decode('utf-8')
            return int(decoded)
        elif isinstance(item, (str, np.str_)):
            return int(item)
        return int(item)

    def add_sequence_tokens(self, labels: List[Union[str, int]]) -> List[Union[str, int]]:
        """
        Add start and end tokens to a sequence of labels.

        Parameters
        ----------
        labels : List[Union[str, int]]
            Sequence of syllable labels

        Returns
        -------
        List[Union[str, int]]
            Labels with start and end tokens added
        """
        return [self.start_token] + labels + [self.end_token]


def detect_label_type(labels: List[Any]) -> LabelType:
    """
    Automatically detect whether labels are manual or automatic.

    Parameters
    ----------
    labels : List[Any]
        List of labels to analyze

    Returns
    -------
    LabelType
        Detected label type (MANUAL for strings, AUTO for integers)

    Notes
    -----
    Manual labels are typically single characters ('a', 'b', etc.)
    Automatic labels are typically integers (0, 1, 2, etc.)
    The -1 label (HDBSCAN uncertain/noise) is treated as a legitimate automatic label.
    """
    if not labels:
        raise ValueError("Cannot detect type of empty label list")

    # Convert to list if it's a numpy array
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()

    # Check first few non-token labels
    sample_labels = []
    for l in labels[:10]:
        # Convert various types to standard Python types
        if isinstance(l, (bytes, np.bytes_)):
            l_val = l.decode('utf-8')
        elif isinstance(l, (np.str_, np.string_)):
            l_val = str(l)
        elif isinstance(l, (np.integer, np.int32, np.int64)):
            l_val = int(l)
        else:
            l_val = l

        # Check if it's a token
        if l_val not in ['s', 'z', -5, -3]:
            sample_labels.append(l_val)

    if not sample_labels:
        raise ValueError("No valid labels found for type detection")

    sample = sample_labels[0]

    # Determine type based on sample
    if isinstance(sample, str) and len(sample) == 1 and sample.isalpha():
        return LabelType.MANUAL
    elif isinstance(sample, (int, np.integer)) or (isinstance(sample, str) and sample.isdigit()):
        return LabelType.AUTO
    else:
        # Default to manual if unclear
        logging.warning(f"Ambiguous label type for sample '{sample}', defaulting to MANUAL")
        return LabelType.MANUAL


def validate_sequence_data(all_syls: List[Union[str, int]], label_handler: LabelHandler) -> Tuple[bool, List[str]]:
    """
    Validate syllable sequence data for consistency and completeness.

    Parameters
    ----------
    all_syls : List[Union[str, int]]
        Complete syllable sequence including tokens
    label_handler : LabelHandler
        Handler for the expected label type

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []

    # Check if sequence is empty
    if not all_syls:
        issues.append("Empty syllable sequence")
        return False, issues

    # Check for proper start/end tokens
    if all_syls[0] != label_handler.start_token:
        issues.append(f"Sequence should start with {label_handler.start_token}")

    if all_syls[-1] != label_handler.end_token:
        issues.append(f"Sequence should end with {label_handler.end_token}")

    # Check for type consistency
    expected_type = str if label_handler.label_type == LabelType.MANUAL else int
    non_token_syls = [s for s in all_syls if s not in label_handler.non_syl_tokens]

    for syl in non_token_syls:
        if not isinstance(syl, expected_type):
            issues.append(f"Inconsistent type: expected {expected_type.__name__}, got {type(syl).__name__} for '{syl}'")
            break  # Don't spam with type errors

    # Check for minimum sequence length
    if len(non_token_syls) < 1:
        issues.append("No actual syllables found (only tokens)")

    return len(issues) == 0, issues


def standardize_tokens_in_sequence(all_syls: List[Any], target_type: LabelType) -> List[Union[str, int]]:
    """
    Convert mixed token types in a sequence to standardized format.

    Parameters
    ----------
    all_syls : List[Any]
        Syllable sequence with potentially mixed token types
    target_type : LabelType
        Target label type to standardize to

    Returns
    -------
    List[Union[str, int]]
        Sequence with standardized tokens
    """
    handler = LabelHandler(target_type)

    # Token mapping for conversion
    if target_type == LabelType.MANUAL:
        token_map = {-5: 's', -3: 'z', 's': 's', 'z': 'z'}
    else:
        token_map = {'s': -5, 'z': -3, -5: -5, -3: -3}

    standardized = []
    for syl in all_syls:
        if syl in token_map:
            standardized.append(token_map[syl])
        else:
            # Convert regular syllables using handler
            try:
                if target_type == LabelType.MANUAL:
                    standardized.append(handler._to_string(syl))
                else:
                    standardized.append(handler._to_int(syl))
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not convert syllable '{syl}' to {target_type.value}: {e}")
                standardized.append(syl)  # Keep original if conversion fails
    return standardized


# ============================================================================
# DATA LOADING AND I/O
# ============================================================================


def load_umap_embeddings(embedding_path: str) -> Tuple[
    Optional[List[str]], Optional[np.ndarray], Optional[List[Union[str, int]]]]:
    """
    Load UMAP embeddings and optional metadata from HDF5 file using PyTables.

    Parameters
    ----------
    embedding_path : str
        Path to HDF5 embedding file

    Returns
    -------
    Tuple[Optional[List[str]], Optional[np.ndarray], Optional[List[Union[str, int]]]]
        (hashes, embeddings, labels) or (None, None, None) on error

    Notes
    -----
    This function handles both manual and automatic labels automatically,
    normalizing them to consistent formats.
    """
    try:
        with tables.open_file(embedding_path, mode='r') as f:
            # Read hashes and decode them into strings
            hashes_raw = f.root.hashes.read()
            hashes = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in hashes_raw]

            # Read embeddings
            embeddings = f.root.embeddings.read()

            # Read labels and handle type dynamically
            labels = None
            if 'labels' in [node._v_name for node in f.list_nodes(f.root)]:
                labels_raw = f.root.labels.read()

                # Detect label type and normalize
                if len(labels_raw) > 0:
                    label_type = detect_label_type(labels_raw)
                    handler = LabelHandler(label_type)
                    labels = handler.normalize_labels(labels_raw)

            return hashes, embeddings, labels
    except Exception as e:
        logging.error(f"Error loading embeddings from {embedding_path}: {e}")
        logging.error(traceback.format_exc())
        return None, None, None


def load_syllables_from_hdf5(path_to_bird_folder: str, config: AnalysisConfig) -> Tuple[Dict[str, List[Union[str, int]]], List[Union[str, int]]]:
    """
    Load syllable data from HDF5 files in the specified directory.

    Parameters
    ----------
    path_to_bird_folder : str
        Path to directory containing HDF5 syllable files
    config : AnalysisConfig
        Configuration object with analysis parameters

    Returns
    -------
    Tuple[Dict[str, List[Union[str, int]]], List[Union[str, int]]]
        (all_syls_dict, original_manual_syls)
        all_syls_dict contains sequences for each label type
        original_manual_syls contains the manual syllable sequence
    """
    songpaths = [os.path.join(path_to_bird_folder, file) for file in os.listdir(path_to_bird_folder)
                 if file.endswith('.h5')]

    all_syls = {'manual': [], 'kmeans': [], 'hdbscan': []}
    original_syls = []

    for songpath in songpaths:
        try:
            with tables.open_file(songpath, 'r') as f:
                for syl_type in ['manual', 'kmeans', 'hdbscan']:
                    if syl_type in [node._v_name for node in f.list_nodes(f.root)]:
                        # Determine label type and create handler
                        raw_labels = f.root._f_get_child(syl_type).read()

                        if len(raw_labels) > 0:
                            label_type = LabelType.MANUAL if syl_type == 'manual' else LabelType.AUTO
                            handler = LabelHandler(label_type)

                            # Normalize and add tokens
                            normalized_labels = handler.normalize_labels(raw_labels)
                            song_with_tokens = handler.add_sequence_tokens(normalized_labels)

                            # Validate sequence
                            is_valid, issues = validate_sequence_data(song_with_tokens, handler)
                            if not is_valid:
                                logging.warning(f"Validation issues in {songpath} for {syl_type}: {issues}")

                            # Check for token conflicts
                            if handler.start_token in normalized_labels or handler.end_token in normalized_labels:
                                logging.warning(f'Start/stop token conflict in {songpath} for {syl_type}')

                            # Add to collections
                            all_syls[syl_type].extend(song_with_tokens)
                            if syl_type == 'manual':
                                original_syls.extend(song_with_tokens)
        except Exception as e:
            logging.error(f'Error processing {songpath}: {e}')
    return all_syls, original_syls


def read_filenames_from_directory(path: str) -> List[str]:
    """
    Read HDF5 filenames from directory and convert to syllable filename format.

    Parameters
    ----------
    path : str
        Directory path to scan for files

    Returns
    -------
    List[str]
        List of syllable filenames (converted from flattened format)
    """
    try:
        files = os.listdir(path)
        filenames = []
        for file in files:
            if file.endswith('.h5'):
                # Convert flattened filename to syllables filename
                syllable_filename = file.replace('flattened.h5', 'syllables.h5')
                filenames.append(syllable_filename)
        return filenames
    except Exception as e:
        logging.error(f"Error reading filenames from {path}: {e}")
        return []


def get_best_label_path(bird_path: str, n_paths: int = 1) -> Tuple[List[str], List[pd.Series]]:
    """
    Find the best performing label schemes from master summary.

    Parameters
    ----------
    bird_path : str
        Path to bird directory containing master_summary.csv
    n_paths : int, default=1
        Number of top-performing label paths to return

    Returns
    -------
    Tuple[List[str], List[pd.Series]]
        (best_paths, best_rows) - paths to label files and corresponding summary rows
    """
    try:
        master_summary_path = os.path.join(bird_path, 'master_summary.csv')
        if not os.path.exists(master_summary_path):
            logging.error(f"Master summary not found at {master_summary_path}")
            return [], []

        master_summary = pd.read_csv(master_summary_path)

        # Sort by composite score (assuming higher is better)
        if 'composite_score' in master_summary.columns:
            master_summary = master_summary.sort_values('composite_score', ascending=False)

        best_rows = []
        best_paths = []

        for i in range(min(n_paths, len(master_summary))):
            best_row = master_summary.iloc[i]
            best_label_path = best_row['label_path']

            # Resolve path if needed (using your existing logic)
            resolved_path = _resolve_file_path(best_label_path)

            if os.path.exists(resolved_path):
                best_rows.append(best_row)
                best_paths.append(resolved_path)
            else:
                logging.warning(f"Label file not found: {resolved_path}")

        return best_paths, best_rows

    except Exception as e:
        logging.error(f"Error getting best label paths from {bird_path}: {e}")
        return [], []


def load_labels(label_save_path: str) -> Tuple[
    Optional[List[Union[str, int]]], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Load cluster labels, hashes, and evaluation scores from HDF5 file.
    Enhanced version with automatic type detection and normalization.

    Parameters
    ----------
    label_save_path : str
        Path to HDF5 label file

    Returns
    -------
    Tuple[Optional[List[Union[str, int]]], Optional[List[str]], Optional[Dict[str, Any]]]
        (labels, hashes, scores) or (None, None, None) on error
    """
    try:
        resolved_path = _resolve_file_path(label_save_path)
        with tables.open_file(resolved_path, mode='r') as f:
            # Load labels as raw array first
            labels_raw = f.root.labels.read()

            # Convert labels based on their actual type
            if len(labels_raw) > 0:
                # Check if labels are integers or need conversion
                if labels_raw.dtype.kind in ['i', 'u']:  # integer types
                    labels = labels_raw.tolist()
                elif labels_raw.dtype.kind in ['U', 'S']:  # string types
                    labels = [str(item) for item in labels_raw]
                else:
                    # Handle other types by converting to list first
                    labels = [item for item in labels_raw]
            else:
                labels = []

            # Load and normalize hashes
            hashes_raw = f.root.hashes.read()
            hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h) for h in hashes_raw]

            # Load scores (all arrays except labels and hashes)
            scores = {}
            for node in f.list_nodes(f.root, classname='Array'):
                if node._v_name not in ['labels', 'hashes']:
                    score_value = node.read()
                    # Handle scalar vs array values
                    scores[node._v_name] = float(score_value) if score_value.ndim == 0 else score_value

        return labels, hashes, scores
    except Exception as e:
        logging.error(f"Error loading labels from {label_save_path}: {e}")
        return None, None, None


def _resolve_file_path(file_path: str) -> str:
    """
    Resolve file path, handling cross-platform and network path issues.

    Parameters
    ----------
    file_path : str
        Original file path that may need resolution

    Returns
    -------
    str
        Resolved file path
    """
    # If file exists as-is, return it
    if os.path.exists(file_path):
        return file_path
    try:
        # Handle network path resolution
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


# ============================================================================
# VOCABULARY AND SEQUENCE PROCESSING
# ============================================================================


def generate_vocabulary(all_syls: List[Union[str, int]], label_handler: LabelHandler, config: AnalysisConfig) -> Tuple[
    List[Union[str, int]], int, Dict[Union[str, int], int], int, np.ndarray]:
    """
    Generate vocabulary and statistics from syllable sequences.

    Filters out non-syllable tokens and low-frequency syllables to create
    a clean vocabulary for downstream analysis.

    Parameters
    ----------
    all_syls : List[Union[str, int]]
        Complete sequence of syllable labels including start/stop tokens
    label_handler : LabelHandler
        Handler for the specific label type being processed
    config : AnalysisConfig
        Configuration object with filtering parameters

    Returns
    -------
    Tuple[List[Union[str, int]], int, Dict[Union[str, int], int], int, np.ndarray]
        vocabulary : List[Union[str, int]]
            Unique syllable labels excluding non-syllable tokens
        vocab_size : int
            Number of syllables contributing to phenotype analysis
            (excludes low-frequency syllables < min_syllable_proportion)
        syl_counts : Dict[Union[str, int], int]
            Count of occurrences for each syllable type
        n_syls_total : int
            Total number of syllables (excluding non-syllable tokens)
        syl_proportions : np.ndarray
            Proportion of each syllable type relative to total syllables

    Notes
    -----
    Syllables representing less than min_syllable_proportion of the total vocabulary
    are excluded from vocab_size to prevent rare syllables from affecting phenotype analysis.
    The -1 label (HDBSCAN uncertain/noise) is treated as a legitimate syllable category.
    """
    vocabulary = []
    syl_counts = {}
    non_syl_tokens = label_handler.non_syl_tokens

    # Get unique syllables excluding tokens
    unique_syls = np.unique(all_syls)

    # Build initial vocabulary and counts
    for syl in unique_syls:
        if syl not in non_syl_tokens:
            vocabulary.append(syl)
            syl_counts[syl] = 0

    # Count syllable occurrences
    for syl in all_syls:
        if syl not in non_syl_tokens:
            syl_counts[syl] += 1

    # Calculate total syllables and proportions
    n_syls_total = sum(syl_counts.values())
    if n_syls_total == 0:
        logging.warning("No syllables found after filtering tokens")
        return [], 0, {}, 0, np.array([])

    syl_proportions = np.array(list(syl_counts.values())) / n_syls_total

    # Calculate vocabulary size excluding rare syllables
    vocab_size = sum(1 for prop in syl_proportions if prop > config.min_syllable_proportion)

    # Remove the special case handling for -1 labels - treat them like any other syllable

    return vocabulary, vocab_size, syl_counts, n_syls_total, syl_proportions


def calculate_transition_counts(all_syls: List[Union[str, int]], label_handler: LabelHandler) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate syllable transition probabilities and higher-order matrices.

    Computes first, second, and third-order transition matrices from syllable
    sequences, providing measures of sequential structure in birdsong.

    Parameters
    ----------
    all_syls : List[Union[str, int]]
        Complete syllable sequence including start/stop tokens
    label_handler : LabelHandler
        Handler for the specific label type being processed

    Returns
    -------
    transition_counts_df : pd.DataFrame
        Raw counts of transitions between syllable pairs
    t_mat_df : pd.DataFrame
        First-order transition probability matrix P(syl_n | syl_n-1)
    t2_mat_df : pd.DataFrame
        Second-order transition probability matrix
    t3_mat_df : pd.DataFrame
        Third-order transition probability matrix

    Notes
    -----
    Start and stop tokens are placed at beginning and end of syllable ordering
    for consistent matrix interpretation. Transitions from end tokens to start
    tokens are excluded from the analysis.
    """
    if len(all_syls) < 2:
        logging.warning("Sequence too short for transition analysis")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df

    # Create syllable-to-integer mapping
    unique_syls = np.unique(all_syls)
    syl_to_int_map = {syl: idx for idx, syl in enumerate(unique_syls)}

    # Handle potential negative indices by remapping
    if any(isinstance(syl, int) and syl < 0 for syl in unique_syls):
        max_positive = max((syl for syl in unique_syls if isinstance(syl, int) and syl >= 0), default=-1)
        remap_start = max_positive + 1

        for syl in unique_syls:
            if isinstance(syl, int) and syl < 0:  # Remove the "and syl != -1" condition
                syl_to_int_map[syl] = remap_start
                remap_start += 1

    # Remove carriage return if present
    if '\r' in syl_to_int_map:
        del syl_to_int_map['\r']
    unique_syls = [s for s in unique_syls if s != '\r']

    # Initialize transition count matrix
    n_syls = len(syl_to_int_map)
    syl_transition_counts = np.zeros((n_syls, n_syls), dtype=np.int64)

    # Calculate transition counts
    start_token = label_handler.start_token
    end_token = label_handler.end_token

    for i in range(len(all_syls) - 1):
        current_syl = all_syls[i]
        next_syl = all_syls[i + 1]

        # Skip transitions from end token to start token (between songs)
        if current_syl == end_token and next_syl == start_token:
            continue

        if current_syl in syl_to_int_map and next_syl in syl_to_int_map:
            current_idx = syl_to_int_map[current_syl]
            next_idx = syl_to_int_map[next_syl]
            syl_transition_counts[next_idx, current_idx] += 1

    # Order syllables with start/end tokens at beginning/end
    ordered_syls = [s for s in unique_syls if s not in [start_token, end_token]]
    ordered_syls = [start_token] + sorted(ordered_syls) + [end_token]

    # Create reordered matrices
    n_ordered = len(ordered_syls)
    transition_mat = np.zeros((n_ordered, n_ordered), dtype=float)
    reordered_transition_counts = np.zeros((n_ordered, n_ordered), dtype=int)

    for i, syl_from in enumerate(ordered_syls):
        for j, syl_to in enumerate(ordered_syls):
            if syl_from in syl_to_int_map and syl_to in syl_to_int_map:
                from_idx = syl_to_int_map[syl_from]
                to_idx = syl_to_int_map[syl_to]

                count = syl_transition_counts[to_idx, from_idx]
                reordered_transition_counts[i, j] = count

                # Calculate probability (row sum for normalization)
                row_sum = np.sum(syl_transition_counts[from_idx, :])
                if row_sum > 0:
                    transition_mat[i, j] = count / row_sum

    # Calculate higher-order transition matrices
    t2_mat = np.linalg.matrix_power(transition_mat, 2)
    t3_mat = np.linalg.matrix_power(transition_mat, 3)

    # Create DataFrames with proper indexing
    transition_counts_df = pd.DataFrame(reordered_transition_counts, index=ordered_syls, columns=ordered_syls)
    t_mat_df = pd.DataFrame(transition_mat, index=ordered_syls, columns=ordered_syls)
    t2_mat_df = pd.DataFrame(t2_mat, index=ordered_syls, columns=ordered_syls)
    t3_mat_df = pd.DataFrame(t3_mat, index=ordered_syls, columns=ordered_syls)

    return transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df


def average_syllable_features(
    all_syls: List[Union[str, int]],
    all_durations: Dict[Union[str, int], List[float]],
    all_following_gaps: Dict[Union[str, int], List[float]],
    all_preceding_gaps: Dict[Union[str, int], List[float]],
    all_rel_pos: Dict[Union[str, int], List[float]],
    label_handler: LabelHandler
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Calculate average acoustic features for each syllable type.

    Parameters
    ----------
    all_syls : List[Union[str, int]]
        Complete syllable sequence
    all_durations : Dict[Union[str, int], List[float]]
        Duration measurements for each syllable occurrence
    all_following_gaps : Dict[Union[str, int], List[float]]
        Following gap measurements for each syllable occurrence
    all_preceding_gaps : Dict[Union[str, int], List[float]]
        Preceding gap measurements for each syllable occurrence
    all_rel_pos : Dict[Union[str, int], List[float]]
        Relative position measurements for each syllable occurrence
    label_handler : LabelHandler
        Handler for the specific label type

    Returns
    -------
    Tuple[Dict, Dict, Dict, Dict]
        (syl_durations, syl_follow_gaps, syl_preceding_gaps, syl_rel_pos)
        Average values for each syllable type
    """
    non_syl_tokens = label_handler.non_syl_tokens
    syls = [syl for syl in np.unique(all_syls) if syl not in non_syl_tokens]

    syl_durations = {}
    syl_follow_gaps = {}
    syl_preceding_gaps = {}
    syl_rel_pos = {}

    for syl in syls:
        if syl in all_durations and len(all_durations[syl]) > 0:
            syl_durations[syl] = np.mean(all_durations[syl])
            syl_follow_gaps[syl] = np.mean(all_following_gaps[syl])
            syl_preceding_gaps[syl] = np.mean(all_preceding_gaps[syl])
            syl_rel_pos[syl] = np.mean(all_rel_pos[syl])

    return syl_durations, syl_follow_gaps, syl_preceding_gaps, syl_rel_pos


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def calculate_entropy(
        t_mat_df: pd.DataFrame,
        syl_proportions: np.ndarray,
        label_handler: LabelHandler
) -> Tuple[float, float]:
    """
    Calculate the entropy of the transition matrix.

    Parameters
    ----------
    t_mat_df : pd.DataFrame
        Transition probability matrix
    syl_proportions : np.ndarray
        Proportion of each syllable type in the sequence
    label_handler : LabelHandler
        Handler for the specific label type (used to exclude start/stop tokens)

    Returns
    -------
    Tuple[float, float]
        (entropy, entropy_scaled) - raw and syllable-proportion-weighted entropy

    Notes
    -----
    Excludes start and stop tokens from entropy calculation to focus on
    within-song sequential structure.
    """
    # Exclude start/stop tokens (first and last rows/columns)
    if len(t_mat_df) < 3:  # Need at least start, one syllable, stop
        logging.warning("Transition matrix too small for entropy calculation")
        return 0.0, 0.0

    # Get core transition matrix (excluding start/stop tokens)
    core_matrix = t_mat_df.iloc[1:-1, 1:-1].to_numpy()

    if core_matrix.size == 0:
        return 0.0, 0.0

    # Calculate raw entropy
    nonzero_probs = core_matrix[core_matrix > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))

    # Calculate scaled entropy using syllable proportions
    # Make sure syl_proportions matches the core_matrix dimensions
    n_core_syls = core_matrix.shape[0]

    if len(syl_proportions) >= n_core_syls:
        # Use the first n_core_syls proportions (excluding start/stop tokens)
        core_syl_proportions = syl_proportions[:n_core_syls]
    else:
        # If we have fewer proportions, pad with zeros or handle appropriately
        logging.warning(f"Syllable proportions ({len(syl_proportions)}) don't match core matrix size ({n_core_syls})")
        core_syl_proportions = syl_proportions
        # Fall back to regular entropy
        entropy_scaled = entropy
        return entropy, entropy_scaled

    if n_core_syls > 0 and len(core_syl_proportions) == n_core_syls:
        syl_prop_mat = np.tile(core_syl_proportions.reshape(-1, 1), (1, n_core_syls))
        weighted_probs = core_matrix * syl_prop_mat
        nonzero_weighted = weighted_probs[weighted_probs > 0]
        if len(nonzero_weighted) > 0:
            entropy_scaled = -np.sum(nonzero_weighted * np.log(core_matrix[weighted_probs > 0]))
        else:
            entropy_scaled = entropy
    else:
        entropy_scaled = entropy

    return entropy, entropy_scaled


def count_repeats_optimized(
    all_syls: List[Union[str, int]],
    label_handler: LabelHandler,
    config: AnalysisConfig
) -> pd.DataFrame:
    """
    Count repeated sequences of syllables using optimized algorithm.

    Parameters
    ----------
    all_syls : List[Union[str, int]]
        Complete syllable sequence including start/stop tokens
    label_handler : LabelHandler
        Handler for the specific label type
    config : AnalysisConfig
        Configuration object with repeat analysis parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with repeat counts, indexed by repeat length and columned by syllable

    Notes
    -----
    This optimized version uses vectorized operations where possible and
    avoids redundant calculations from the original nested loop approach.
    """
    if len(all_syls) < 4:  # Need at least start + 2 syllables + stop for a repeat
        return pd.DataFrame()

    # Convert to numpy array for faster operations
    syls_array = np.array(all_syls)
    non_syl_tokens = set(label_handler.non_syl_tokens)

    # Get unique syllables excluding tokens
    unique_syls = [syl for syl in np.unique(syls_array) if syl not in non_syl_tokens]

    if not unique_syls:
        return pd.DataFrame()

    repeat_counts = {}
    candidate_lengths = list(config.repeat_candidate_range)

    # For each syllable type, find all its positions
    for syl in unique_syls:
        positions = np.where(syls_array == syl)[0]

        # For each position, check for consecutive repeats
        for pos in positions:
            max_run_length = 1

            # Find maximum consecutive run starting at this position
            while (
                pos + max_run_length < len(syls_array)
                and syls_array[pos + max_run_length] == syl
                and max_run_length < max(candidate_lengths)
            ):
                max_run_length += 1

            # Record all repeat lengths for this run (2, 3, 4, ... up to max_run_length)
            for run_len in range(2, max_run_length + 1):
                if run_len in candidate_lengths:
                    repeat_counts[(syl, run_len)] = repeat_counts.get((syl, run_len), 0) + 1

    # Convert to DataFrame format
    if not repeat_counts:
        return pd.DataFrame()

    # Create DataFrame with repeat lengths as index and syllables as columns
    repeat_df_data = {}
    for syl in unique_syls:
        repeat_df_data[syl] = [
            repeat_counts.get((syl, length), 0) for length in candidate_lengths
        ]

    repeat_df = pd.DataFrame(repeat_df_data, index=candidate_lengths)

    # Remove overlapping counts (subtract shorter repeats contained in longer ones)
    repeat_df = _remove_overlapping_repeats(repeat_df)

    # Remove empty rows and columns
    repeat_df = repeat_df.loc[(repeat_df != 0).any(axis=1), (repeat_df != 0).any(axis=0)]

    return repeat_df


def _remove_overlapping_repeats(repeat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove shorter repeats that are contained within longer repeats.

    Parameters
    ----------
    repeat_df : pd.DataFrame
        DataFrame with repeat counts before overlap removal

    Returns
    -------
    pd.DataFrame
        DataFrame with overlapping repeats removed
    """
    if repeat_df.empty:
        return repeat_df

    # Work from longest to shortest repeats
    lengths = sorted(repeat_df.index, reverse=True)

    for i, longer_len in enumerate(lengths[:-1]):
        for shorter_len in lengths[i + 1:]:
            # For each syllable, subtract overlaps
            for syl in repeat_df.columns:
                n_longer = repeat_df.loc[longer_len, syl]
                if n_longer > 0:
                    # Each longer repeat contains (longer_len - shorter_len + 1) shorter repeats
                    overlaps = n_longer * (longer_len - shorter_len + 1)
                    repeat_df.loc[shorter_len, syl] = max(
                        0, repeat_df.loc[shorter_len, syl] - overlaps
                    )

    return repeat_df


def repeat_significance(
    repeat_counts_df: pd.DataFrame,
    syl_counts: Dict[Union[str, int], int],
    config: AnalysisConfig,
    label_handler: LabelHandler
) -> Dict[Union[str, int], Tuple[bool, float]]:
    """
    Check for significant proportion of repeats from syllable instances.

    Parameters
    ----------
    repeat_counts_df : pd.DataFrame
        DataFrame with repeat counts by syllable and repeat length
    syl_counts : Dict[Union[str, int], int]
        Total count of each syllable type in the sequence
    config : AnalysisConfig
        Configuration object with significance threshold
    label_handler : LabelHandler
        Handler for the specific label type

    Returns
    -------
    Dict[Union[str, int], Tuple[bool, float]]
        Dictionary mapping syllable to (is_significant, repeat_proportion)
    """
    repeated_syls = repeat_counts_df.columns if not repeat_counts_df.empty else []
    significance = {}
    non_syl_tokens = set(label_handler.non_syl_tokens)

    for syl in repeated_syls:
        if syl not in non_syl_tokens and syl in syl_counts:
            n_total_syls = syl_counts[syl]

            # Calculate total syllables involved in repeats
            # Each repeat of length N involves N syllables
            n_repeated_syls = sum(
                repeat_len * repeat_count
                for repeat_len, repeat_count in repeat_counts_df[syl].items()
                if repeat_count > 0
            )

            repeat_prop = n_repeated_syls / n_total_syls if n_total_syls > 0 else 0
            is_significant = repeat_prop >= config.repeat_significance_threshold

            significance[syl] = (is_significant, repeat_prop)
        else:
            significance[syl] = (False, 0.0)

    return significance


def remove_insignificant_repeats(
    repeat_counts: pd.DataFrame,
    syl_counts: Dict[Union[str, int], int],
    config: AnalysisConfig,
    label_handler: LabelHandler
) -> pd.DataFrame:
    """
    Remove syllables with insignificant repeat patterns from analysis.

    Parameters
    ----------
    repeat_counts : pd.DataFrame
        DataFrame with repeat counts by syllable and repeat length
    syl_counts : Dict[Union[str, int], int]
        Total count of each syllable type in the sequence
    config : AnalysisConfig
        Configuration object with filtering parameters
    label_handler : LabelHandler
        Handler for the specific label type

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only significant repeats
    """
    if repeat_counts.empty:
        return repeat_counts

    # Calculate overall syllable proportions for adaptive threshold
    total_syls = sum(syl_counts.values())
    overall_syl_prop = {syl: count / total_syls for syl, count in syl_counts.items()}

    # Remove syllables with very low overall occurrence
    adaptive_threshold = config.adaptive_threshold_factor * (1 / len(syl_counts))
    columns_to_remove = []

    for syl in repeat_counts.columns:
        if syl in overall_syl_prop:
            if overall_syl_prop[syl] < adaptive_threshold:
                columns_to_remove.append(syl)

    # Remove low-occurrence syllables
    repeat_counts = repeat_counts.drop(columns=columns_to_remove, errors='ignore')

    # Remove syllables without significant repeat proportions
    significance = repeat_significance(repeat_counts, syl_counts, config, label_handler)
    insignificant_syls = [syl for syl, (is_sig, _) in significance.items() if not is_sig]
    repeat_counts = repeat_counts.drop(columns=insignificant_syls, errors='ignore')

    return repeat_counts


def repeat_phenotypes(
    repeat_counts: pd.DataFrame,
    config: AnalysisConfig,
    keep_individual_dists: bool = False
) -> Tuple[Dict[str, Any], Optional[Dict]]:
    """
    Calculate phenotypic statistics from repeat count data.

    Parameters
    ----------
    repeat_counts : pd.DataFrame
        DataFrame with repeat counts by syllable and repeat length
    config : AnalysisConfig
        Configuration object with analysis parameters
    keep_individual_dists : bool, default=False
        Whether to return individual syllable repeat distributions

    Returns
    -------
    Tuple[Dict[str, Any], Optional[Dict]]
        (repeat_stats, repeat_stats_all) - summary statistics and optional individual distributions
    """
    if repeat_counts.empty or len(repeat_counts.columns) == 0:
        return {
            'repeat_bool': False,
            'dyad_bool': False,
            'num_dyad': 0,
            'num_longer_reps': 0,
            'mean_repeat_syls': 0.0,
            'median_repeat_syls': 0.0,
            'var_repeat_syls': 0.0,
            'skew_repeat_syls': 0.0,
            'kurt_repeat_syls': 0.0
        }, None

    repeat_stats_all = {} if keep_individual_dists else None
    num_unique_repeaty_syls = len(repeat_counts.columns)
    dyad_bool = False
    num_dyads = 0
    combined_rep_list = []

    # Process each syllable's repeat distribution
    for syl in repeat_counts.columns:
        # Create list of repeat lengths weighted by their occurrence
        rep_list = []
        for repeat_len in repeat_counts.index:
            n_instances = repeat_counts.loc[repeat_len, syl]
            rep_list.extend([repeat_len] * n_instances)

        if not rep_list:
            continue

        # Check for dyad dominance (repeats of length 2)
        dyad_count = sum(1 for length in rep_list if length == 2)
        dyad_prop = dyad_count / len(rep_list) if rep_list else 0

        if dyad_prop > config.dyad_threshold:
            dyad_bool = True
            num_dyads += 1
            # Remove dyads from longer repeat analysis
            rep_list = [length for length in rep_list if length != 2]

        # Add to combined distribution for overall statistics
        combined_rep_list.extend(rep_list)

        # Store individual statistics if requested
        if keep_individual_dists and rep_list:
            mean_repeat = np.mean(rep_list)
            median_repeat = np.median(rep_list)
            var_repeat = np.var(rep_list)
            skew_repeat = skew(rep_list)
            kurt_repeat = kurtosis(rep_list)

            repeat_stats_all[syl] = (
                rep_list, mean_repeat, median_repeat,
                var_repeat, skew_repeat, kurt_repeat
            )

    # Calculate overall repeat statistics
    if combined_rep_list:
        mean_repeat_syls = np.mean(combined_rep_list)
        median_repeat_syls = np.median(combined_rep_list)
        var_repeat_syls = np.var(combined_rep_list)
        skew_repeat_syls = skew(combined_rep_list)
        kurt_repeat_syls = kurtosis(combined_rep_list)
    else:
        mean_repeat_syls = median_repeat_syls = var_repeat_syls = 0.0
        skew_repeat_syls = kurt_repeat_syls = 0.0

    repeat_stats = {
        'repeat_bool': True,
        'dyad_bool': dyad_bool,
        'num_dyad': num_dyads,
        'num_longer_reps': num_unique_repeaty_syls - num_dyads,
        'mean_repeat_syls': mean_repeat_syls,
        'median_repeat_syls': median_repeat_syls,
        'var_repeat_syls': var_repeat_syls,
        'skew_repeat_syls': skew_repeat_syls,
        'kurt_repeat_syls': kurt_repeat_syls
    }

    return repeat_stats, repeat_stats_all


def calculate_repeat_tendency(
        all_syls: List[Union[str, int]],
        syl_counts: Dict[Union[str, int], int],
        syl_proportions: np.ndarray,
        label_handler: LabelHandler,
        config: AnalysisConfig
) -> Dict[Union[str, int], bool]:
    """
    Calculate the repeat tendency for individual syllables.

    Parameters
    ----------
    all_syls : List[Union[str, int]]
        Complete syllable sequence
    syl_counts : Dict[Union[str, int], int]
        Count of each syllable type
    syl_proportions : np.ndarray
        Proportion of each syllable type
    label_handler : LabelHandler
        Handler for the specific label type
    config : AnalysisConfig
        Configuration object with analysis parameters

    Returns
    -------
    Dict[Union[str, int], bool]
        Dictionary indicating which syllables have high repeat tendency
    """
    repeat_ind = {}

    # Calculate importance threshold
    import_threshold = (1 / len(syl_counts)) - config.importance_threshold_offset

    # Get significant syllables
    syl_keys = list(syl_counts.keys())
    significant_mask = syl_proportions >= import_threshold
    syl_sig = [syl_keys[i] for i, is_sig in enumerate(significant_mask) if is_sig]

    # Remove tokens from consideration
    non_syl_tokens = set(label_handler.non_syl_tokens)
    syl_sig = [syl for syl in syl_sig if syl not in non_syl_tokens]

    if not syl_sig:
        return repeat_ind

    # Calculate transition counts for diagonal analysis
    transition_counts_df, _, _, _ = calculate_transition_counts(all_syls, label_handler)

    if transition_counts_df.empty:
        return repeat_ind

    # Check repeat tendency for each significant syllable
    for syl in syl_sig:
        if syl in transition_counts_df.index and syl in transition_counts_df.columns:
            # Get self-transition count (diagonal element)
            n_repeats = transition_counts_df.loc[syl, syl]

            # Get total transitions from this syllable
            total_transitions = transition_counts_df.loc[syl, :].sum()

            if total_transitions > 0:
                repeat_tendency = n_repeats / total_transitions
                repeat_ind[syl] = repeat_tendency >= 0.4  # 40% threshold for repeat tendency

    return repeat_ind


# ============================================================================
# COMPARISON AND VALIDATION
# ============================================================================


def syllable_confusion_matrix(
    labels1: List[Union[str, int]],
    labels2: List[Union[str, int]],
    paths: AnalysisPaths
) -> AnalysisPaths:
    """
    Generate confusion matrix comparing two label sets (e.g., manual vs automatic).

    Parameters
    ----------
    labels1 : List[Union[str, int]]
        First set of labels (e.g., manual labels)
    labels2 : List[Union[str, int]]
        Second set of labels (e.g., automatic labels)
    paths : AnalysisPaths
        Paths object to store output location

    Returns
    -------
    AnalysisPaths
        Updated paths object with confusion matrix path
    """
    # Convert to numpy arrays and remove tokens
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    # Remove non-syllable tokens from both label sets
    non_syl_tokens = {'s', 'z', -5, -3}
    mask = np.array(
        [l1 not in non_syl_tokens and l2 not in non_syl_tokens for l1, l2 in zip(labels1, labels2)]
    )

    labels1_clean = labels1[mask]
    labels2_clean = labels2[mask]

    if len(labels1_clean) == 0 or len(labels2_clean) == 0:
        logging.warning("No valid labels found for confusion matrix")
        return paths

    assert len(labels1_clean) == len(labels2_clean), "Label arrays must be same length"

    # Create label mappings for consistent integer indexing
    def create_label_mapping(labels):
        unique_labels = np.unique(labels)
        if isinstance(labels[0], (str, np.str_, bytes)):
            # String labels: map to consecutive integers
            return {label: idx for idx, label in enumerate(unique_labels)}
        else:
            # Integer labels: use as-is
            return {label: label for label in unique_labels}

    mapping_1 = create_label_mapping(labels1_clean)
    mapping_2 = create_label_mapping(labels2_clean)

    # Create confusion matrix
    n_labels1 = len(mapping_1)
    n_labels2 = len(mapping_2)
    confusion_matrix = np.zeros((n_labels2, n_labels1), dtype=int)

    # Populate confusion matrix
    for l1, l2 in zip(labels1_clean, labels2_clean):
        idx1 = mapping_1[l1]
        idx2 = mapping_2[l2]
        confusion_matrix[idx2, idx1] += 1

    # Save confusion matrix visualization
    if paths.current_rank_folder:
        paths.conf_matrices = os.path.join(paths.current_rank_folder, 'manual_auto_confmat.jpg')
        syl_comparison_mat(confusion_matrix, mapping_1, mapping_2, paths.conf_matrices)

    return paths

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def save_transition_images(
    transition_counts_df: pd.DataFrame,
    t_mat_df: pd.DataFrame,
    t2_mat_df: pd.DataFrame,
    t3_mat_df: pd.DataFrame,
    img_path: str,
    name: str,
    syl_type: str,
    config: AnalysisConfig
) -> List[str]:
    """
    Generate and save images of transition matrices.

    Parameters
    ----------
    transition_counts_df : pd.DataFrame
        Raw transition counts matrix
    t_mat_df : pd.DataFrame
        First-order transition probability matrix
    t2_mat_df : pd.DataFrame
        Second-order transition probability matrix
    t3_mat_df : pd.DataFrame
        Third-order transition probability matrix
    img_path : str
        Directory path for saving images
    name : str
        Base name for output files
    syl_type : str
        Label type identifier for filename
    config : AnalysisConfig
        Configuration object with visualization parameters

    Returns
    -------
    List[str]
        List of paths to generated image files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create output directory
    os.makedirs(img_path, exist_ok=True)

    # Define file paths
    file_paths = [
        f'{img_path}/{name}_{syl_type}_transition_counts_mat.jpg',
        f'{img_path}/{name}_{syl_type}_transition1x_mat.jpg',
        f'{img_path}/{name}_{syl_type}_transition2x_mat.jpg',
        f'{img_path}/{name}_{syl_type}_transition3x_mat.jpg'
    ]

    matrices = [
        (transition_counts_df.astype(int), "d", "transition counts"),
        (t_mat_df, "", "1st order transition matrix"),
        (t2_mat_df, "", "2nd order transition matrix"),
        (t3_mat_df, "", "3rd order transition matrix")
    ]

    # Generate each heatmap
    for i, (matrix, fmt, title) in enumerate(matrices):
        plt.figure(figsize=(10, 10))

        if fmt == "d":  # Integer format for counts
            annot = True
            fmt_str = "d"
        else:  # Formatted annotation for probabilities
            annot = format_annotation(matrix)
            fmt_str = ""

        heatmap = sns.heatmap(
            matrix,
            annot=annot,
            annot_kws={'size': config.heatmap_annotation_size},
            fmt=fmt_str,
            linewidths=0.2,
            square=True,
            cbar_kws={"shrink": 0.7}
        )

        # Set labels based on matrix type
        if i == 0:  # Counts matrix
            heatmap.set(xlabel='syllable n-1', ylabel='syllable n', title=title)
        elif i == 2:  # Second-order
            heatmap.set(xlabel='syllable n-2', ylabel='syllable n', title=title)
        elif i == 3:  # Third-order
            heatmap.set(xlabel='syllable n-3', ylabel='syllable n', title=title)
        else:  # First-order
            heatmap.set(xlabel='syllable n-1', ylabel='syllable n', title=title)

        fig = heatmap.get_figure()
        fig.tight_layout()
        fig.savefig(file_paths[i], dpi=config.figure_dpi, format='jpeg')
        plt.close(fig)

    return file_paths

def save_repeat_image(
    repeat_counts_df: pd.DataFrame,
    img_path: str,
    name: str,
    syl_type: str,
    config: AnalysisConfig
) -> str:
    """
    Generate and save an image of repeat counts.

    Parameters
    ----------
    repeat_counts_df : pd.DataFrame
        DataFrame with repeat counts by syllable and repeat length
    img_path : str
        Directory path for saving image
    name : str
        Base name for output file
    syl_type : str
        Label type identifier for filename
    config : AnalysisConfig
        Configuration object with visualization parameters

    Returns
    -------
    str
        Path to generated image file
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = f'{img_path}/{name}_{syl_type}_repeat_counts.jpg'

    plt.figure(figsize=(10, 10))

    if repeat_counts_df.empty:
        # Create placeholder for empty data
        plt.figure(figsize=(5, 5))
        heatmap = sns.heatmap(
            np.zeros((1, 1), dtype=int),
            annot=True,
            annot_kws={'size': config.heatmap_annotation_size},
            fmt="d",
            linewidths=0.2,
            square=True,
            cbar_kws={"shrink": 0.7}
        )
        heatmap.set(xlabel='', ylabel='', title='no repeated syllables')
    else:
        # Create heatmap of repeat counts
        heatmap = sns.heatmap(
            repeat_counts_df.astype(int),
            annot=True,
            annot_kws={'size': config.heatmap_annotation_size},
            fmt="d",
            linewidths=0.2,
            square=True,
            cbar_kws={"shrink": 0.7}
        )
        heatmap.set(xlabel='syllable', ylabel='n repeats', title='repeat counts')

    fig = heatmap.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.figure_dpi, format='jpeg')
    plt.close(fig)

    return output_path


def plot_repeat_distributions(
    repeat_stats_all: Dict[Union[str, int], Tuple],
    save_path: str,
    bird: str,
    label_str: str,
    config: AnalysisConfig
) -> None:
    """
    Plot stacked histogram of repeat length distributions by syllable.

    Parameters
    ----------
    repeat_stats_all : Dict[Union[str, int], Tuple]
        Dictionary mapping syllables to (rep_list, mean, median, var, skew, kurt)
    save_path : str
        Directory path for saving the plot
    bird : str
        Bird identifier for plot title and filename
    label_str : str
        Label scheme identifier for plot title and filename
    config : AnalysisConfig
        Configuration object with visualization parameters
    """
    import matplotlib.pyplot as plt

    if not repeat_stats_all:
        logging.warning("No repeat statistics to plot")
        return

    plt.figure(figsize=(12, 6))

    # Get the number of syllables and create color palette
    num_syllables = len(repeat_stats_all)
    colormap = plt.get_cmap('tab20')
    colors = [colormap(i / num_syllables) for i in range(num_syllables)]

    # Get all unique repeat lengths across all syllables
    all_lengths = []
    for syl_data in repeat_stats_all.values():
        rep_list = syl_data[0]  # First element is the repeat length list
        all_lengths.extend(rep_list)

    if not all_lengths:
        logging.warning("No repeat lengths found to plot")
        return

    max_length = max(all_lengths)
    bins = range(2, max_length + 2)

    # Create stacked histogram
    bottom = np.zeros(len(bins) - 1)

    for i, (syl, syl_data) in enumerate(repeat_stats_all.items()):
        rep_list = syl_data[0]
        if rep_list:  # Only plot if there are repeats
            counts, _ = np.histogram(rep_list, bins=bins)
            plt.bar(
                bins[:-1],
                counts,
                bottom=bottom,
                label=f'Syllable {syl}',
                color=colors[i % num_syllables],
                width=0.8
            )
            bottom += counts

    plt.xlabel('Number of Repeats')
    plt.ylabel('Count')
    plt.title(f'Repeat Distributions by Syllable\n{bird} - {label_str}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Save the plot
    output_path = os.path.join(save_path, f'{bird}_{label_str}_repeat_distributions.png')
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def syl_comparison_mat(
        conf_mat: np.ndarray,
        mapping1: Union[Dict, List],
        mapping2: Union[Dict, List],
        save_path: str,
        config: Optional[AnalysisConfig] = None
) -> None:
    """
    Plot and save a comparison matrix (confusion matrix) comparing label identity
    from two different labelling strategies as a heatmap.

    Parameters
    ----------
    conf_mat : np.ndarray
        The confusion matrix to visualize
    mapping1 : Union[Dict, List]
        Dictionary or list mapping labels for columns (x-axis)
    mapping2 : Union[Dict, List]
        Dictionary or list mapping labels for rows (y-axis)
    save_path : str
        Full path where the figure will be saved
    config : Optional[AnalysisConfig], default=None
        Configuration object with visualization parameters

    Notes
    -----
    Creates a heatmap visualization of the confusion matrix with appropriate
    tick labels for comparing manual vs automatic labeling strategies.
    """
    if config is None:
        config = AnalysisConfig()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(10, 8))

        # Process mappings to create sorted tick labels
        if isinstance(mapping1, dict):
            sorted_xticklabels = [mapping1[i] for i in sorted(mapping1.keys())]
        else:
            sorted_xticklabels = list(mapping1)

        if isinstance(mapping2, dict):
            sorted_yticklabels = [mapping2[i] for i in sorted(mapping2.keys())]
        else:
            sorted_yticklabels = list(mapping2)

        # Create heatmap
        ax = sns.heatmap(
            conf_mat,
            annot=True,
            fmt="d",
            yticklabels=sorted_yticklabels,
            xticklabels=sorted_xticklabels,
            cmap='Blues',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        # Set labels and title
        ax.set_xlabel('Manual Labels', fontsize=12)
        ax.set_ylabel('Automatic Labels', fontsize=12)
        ax.set_title('Label Comparison Matrix', fontsize=14, pad=20)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=config.figure_dpi, format='jpeg', bbox_inches='tight')
        plt.close()

        logging.info(f"Confusion matrix saved to: {save_path}")

    except Exception as e:
        logging.error(f"Error creating confusion matrix plot: {e}")
        logging.error(traceback.format_exc())
        # Ensure figure is closed even on error
        plt.close()


def format_annotation(array: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Format numerical values for heatmap annotations with appropriate precision.

    Parameters
    ----------
    array : Union[pd.DataFrame, np.ndarray]
        Array of numerical values to format

    Returns
    -------
    np.ndarray
        Array of formatted string representations

    Notes
    -----
    Uses intelligent formatting:
    - Values close to 0 or 1: integer format
    - Very small/large values: scientific notation
    - Otherwise: 2 decimal places, switching to scientific if too long
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(array, pd.DataFrame):
        array = array.to_numpy()

    r, c = array.shape
    annot_array = np.empty((r, c), dtype=object)

    # Format each value
    for i in range(r):
        for j in range(c):
            value = array[i, j]

            # Handle special cases
            if np.isnan(value):
                annot_array[i, j] = ''
            elif np.isclose(value, 0, atol=1e-2) or np.isclose(value, 1, atol=1e-2):
                annot_array[i, j] = f'{int(value)}'
            elif abs(value) < 0.01 or abs(value) >= 100:
                annot_array[i, j] = f'{value:.2e}'
            else:
                formatted_value = f'{value:.2f}'
                # Switch to scientific notation if formatted string is too long
                if len(formatted_value) > 4:
                    annot_array[i, j] = f'{value:.2e}'
                else:
                    annot_array[i, j] = formatted_value

    return annot_array


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def add_phenotype_to_df(
    bird: str,
    phenotypes: Dict[str, Any],
    df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add phenotype data for a bird to a DataFrame.

    Parameters
    ----------
    bird : str
        Bird identifier
    phenotypes : Dict[str, Any]
        Dictionary of phenotype measurements
    df : Optional[pd.DataFrame], default=None
        Existing DataFrame to append to, or None to create new

    Returns
    -------
    pd.DataFrame
        DataFrame with phenotype data added
    """
    if df is None:
        df = pd.DataFrame(phenotypes, index=[bird])
    else:
        # Add new row for this bird
        new_row = pd.Series(phenotypes, name=bird)
        df = pd.concat([df, new_row.to_frame().T])

    return df

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def _setup_phenotyping(
    paths: AnalysisPaths,
    bird: str,
    n_phenotypes: int,
    reset_phenotypes: bool,
    config: AnalysisConfig
) -> 'PhenotypingSetup':
    """
    Setup data structures and validate inputs for phenotyping process.

    Parameters
    ----------
    paths : AnalysisPaths
        Paths object with directory locations
    bird : str
        Bird identifier
    n_phenotypes : int
        Number of top phenotypes to process
    reset_phenotypes : bool
        Whether to reset existing phenotype data
    config : AnalysisConfig
        Configuration object

    Returns
    -------
    PhenotypingSetup
        Setup object with initialized data structures
    """
    from dataclasses import dataclass

    @dataclass
    class PhenotypingSetup:
        label_paths: List[str]
        master_summary: pd.DataFrame
        phenotypes: Dict[str, List]
        phenotype_csv_path: str

    # Load master summary
    if not os.path.exists(paths.master_summary):
        raise FileNotFoundError(f"Master summary not found: {paths.master_summary}")

    master_summary = pd.read_csv(paths.master_summary)
    phenotype_csv_path = paths.phenotype_summary

    # Initialize phenotype data structure
    phenotype_columns = [
        'label_str', 'nmi', 'composite_score', 'repertoire_sz',
        'sequence_entropy', 'entropy_scaled', 'repeat_bool', 'dyad_bool',
        'num_dyad', 'num_longer_reps', 'mean_repeat_syls', 'median_repeat_syls',
        'var_repeat_syls', 'skew_repeat_syls', 'kurt_repeat_syls'
    ]

    # Handle existing phenotype data
    if reset_phenotypes and os.path.exists(phenotype_csv_path):
        os.remove(phenotype_csv_path)

    if os.path.exists(phenotype_csv_path):
        phenotypes_df = pd.read_csv(phenotype_csv_path)
        phenotypes = {col: phenotypes_df[col].tolist() for col in phenotype_columns if col in phenotypes_df.columns}
    else:
        phenotypes = {col: [] for col in phenotype_columns}

    # Get top label paths to process
    label_paths = list(master_summary['label_path'])
    if len(label_paths) > n_phenotypes:
        label_paths = label_paths[:n_phenotypes]

    return PhenotypingSetup(
        label_paths=label_paths,
        master_summary=master_summary,
        phenotypes=phenotypes,
        phenotype_csv_path=phenotype_csv_path
    )


def phenotype_bird(
    paths: AnalysisPaths,
    bird: str,
    n_phenotypes: int = 5,
    reset_phenotypes: bool = False,
    config: AnalysisConfig = None
) -> bool:
    """
    Main function to phenotype a bird using multiple label schemes.

    Parameters
    ----------
    paths : AnalysisPaths
        Paths object with directory locations
    bird : str
        Bird identifier
    n_phenotypes : int, default=5
        Number of top phenotypes to process
    reset_phenotypes : bool, default=False
        Whether to reset existing phenotype data
    config : AnalysisConfig, default=None
        Configuration object for analysis parameters

    Returns
    -------
    bool
        True if phenotyping completed successfully
    """
    if config is None:
        config = AnalysisConfig()

    # Setup data structures for phenotyping
    setup_data = _setup_phenotyping(paths, bird, n_phenotypes, reset_phenotypes, config)

    # Process each label scheme
    for rank, label_path in enumerate(setup_data.label_paths):
        try:
            # Load sequence data
            sequence_data = _load_or_create_sequence_data(paths, rank, label_path, setup_data, config)

            # Calculate phenotypes
            phenotype_results = _calculate_phenotype_metrics(sequence_data, setup_data.master_summary, rank, config)

            # Generate visualizations
            _generate_visualizations(paths, sequence_data, rank, bird, config)

            # Update phenotype summary
            _update_phenotype_summary(setup_data.phenotypes, phenotype_results)

        except Exception as e:
            logging.error(f"Error processing label scheme at rank {rank}: {e}")
            logging.error(traceback.format_exc())

    # Save final phenotype results
    return _save_phenotype_results(paths, setup_data.phenotypes)


def _load_or_create_sequence_data(
    paths: AnalysisPaths,
    rank: int,
    label_path: str,
    setup_data: 'PhenotypingSetup',
    config: AnalysisConfig
) -> 'SequenceData':
    """
    Load or create sequence data for a given label scheme.

    Parameters
    ----------
    paths : AnalysisPaths
        Paths object with directory locations
    rank : int
        Rank of the label scheme
    label_path : str
        Path to the label file
    setup_data : PhenotypingSetup
        Setup data structure
    config : AnalysisConfig
        Configuration object for analysis parameters

    Returns
    -------
    SequenceData
        Loaded or newly created sequence data object
    """
    # Paths for sequence data
    paths.sequence_data = os.path.join(paths.data_folder, 'sequences', f'rank{rank}_sequence_data.pkl')
    os.makedirs(os.path.dirname(paths.sequence_data), exist_ok=True)

    # Load sequence data if already processed
    if os.path.isfile(paths.sequence_data):
        sequence_data = SequenceData.load_from_pickle(paths.sequence_data)
    else:
        # Create new sequence data object
        labels, hashes, _ = load_labels(label_path)
        hashed_labels = dict(zip(hashes, labels))

        all_auto_syls, all_manual_syls, all_specs = [], [], []
        song_paths = read_filenames_from_directory(os.path.join(paths.data_folder, 'syllables'))
        for song in song_paths:
            song_path = os.path.join(paths.data_folder, 'syllables', song)
            try:
                with tables.open_file(song_path, mode='r') as f:
                    hashes = [item.decode('utf-8') for item in f.root.hashes.read()]
                    manual_labels = [
                        item.decode('utf-8') if isinstance(item, bytes) else item for item in f.root.manual.read()
                    ]

                    for idx, hash in enumerate(hashes):
                        all_auto_syls.append(hashed_labels.get(hash, -1))
                        all_manual_syls.append(manual_labels[idx])

                    all_specs += list(f.root.spectrograms.read())

            except Exception as e:
                logging.error(f"Error reading from {song_path}: {e}")
                logging.error(traceback.format_exc())

        # Calculate sequence statistics
        label_handler = LabelHandler(LabelType.AUTO)
        vocabulary, vocab_size, syl_counts, n_syls_total, syl_proportions = generate_vocabulary(
            all_auto_syls, label_handler, config
        )
        transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df = calculate_transition_counts(all_auto_syls, label_handler)
        entropy, entropy_scaled = calculate_entropy(t_mat_df, syl_proportions, label_handler)

        # Extract bird name from the bird folder path (not from master_summary)
        bird_name = os.path.basename(paths.bird_folder)

        # Package sequence data
        sequence_data = SequenceData(
            bird_name=bird_name,  # Extract from folder path
            syl_type='auto',
            repertoire_size=vocab_size,
            vocab=vocabulary,
            syl_counts=syl_counts,
            syl_proportions=syl_proportions,
            n_songs=len(song_paths),
            n_syllables=n_syls_total,
            all_auto_syllables=all_auto_syls,
            entropy=entropy,
            entropy_scaled=entropy_scaled,
            paths_to_songs=song_paths
        )
        sequence_data.save_to_pickle(paths.sequence_data)

    return sequence_data


def _calculate_phenotype_metrics(
        sequence_data: 'SequenceData',
        master_summary: pd.DataFrame,
        rank: int,
        config: AnalysisConfig
) -> Dict[str, Any]:
    """
    Calculate phenotype metrics for a given sequence data object.

    Parameters
    ----------
    sequence_data : SequenceData
        Sequence data object
    master_summary : pd.DataFrame
        Master summary DataFrame
    rank : int
        Rank of the label scheme
    config : AnalysisConfig
        Configuration object for analysis parameters

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated phenotype metrics
    """
    repeat_counts_df = count_repeats_optimized(sequence_data.all_auto_syllables, LabelHandler(LabelType.AUTO), config)
    repeat_counts_df = remove_insignificant_repeats(repeat_counts_df, sequence_data.syl_counts, config,
                                                    LabelHandler(LabelType.AUTO))

    # Store the repeat counts in the sequence data object for later use in visualizations
    sequence_data.repeat_counts = repeat_counts_df

    repeat_stats, _ = repeat_phenotypes(repeat_counts_df, config)

    phenotype_results = {
        'label_str': f'rank{rank}',
        'nmi': master_summary.iloc[rank]['nmi'] if 'nmi' in master_summary.columns else np.nan,
        'composite_score': master_summary.iloc[rank][
            'composite_score'] if 'composite_score' in master_summary.columns else np.nan,
        'repertoire_sz': sequence_data.repertoire_size,
        'sequence_entropy': sequence_data.entropy,
        'entropy_scaled': sequence_data.entropy_scaled,
        **repeat_stats
    }

    return phenotype_results


def _generate_visualizations(
    paths: AnalysisPaths,
    sequence_data: 'SequenceData',
    rank: int,
    bird: str,
    config: AnalysisConfig
) -> None:
    """
    Generate visualizations for sequence data.

    Parameters
    ----------
    paths : AnalysisPaths
        Paths object with directory locations
    sequence_data : SequenceData
        Sequence data object
    rank : int
        Rank of the label scheme
    bird : str
        Bird identifier
    config : AnalysisConfig
        Configuration object with visualization parameters
    """
    img_path = os.path.join(paths.figure_folder, f'rank{rank}_labels')
    os.makedirs(img_path, exist_ok=True)

    # Save transition matrices
    transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df = calculate_transition_counts(sequence_data.all_auto_syllables, LabelHandler(LabelType.AUTO))
    paths.transition_imgs = save_transition_images(transition_counts_df, t_mat_df, t2_mat_df, t3_mat_df, img_path, bird, 'auto', config)

    # Save repeat distributions
    paths.repeat_img = save_repeat_image(sequence_data.repeat_counts, img_path, bird, 'auto', config)


def _update_phenotype_summary(
    phenotypes: Dict[str, List],
    phenotype_results: Dict[str, Any]
) -> None:
    """
    Update phenotype summary with new results.

    Parameters
    ----------
    phenotypes : Dict[str, List]
        Phenotype summary dictionary
    phenotype_results : Dict[str, Any]
        Dictionary of new phenotype results
    """
    for key, value in phenotype_results.items():
        phenotypes[key].append(value)


def _save_phenotype_results(
    paths: AnalysisPaths,
    phenotypes: Dict[str, List]
) -> bool:
    """
    Save phenotype results to CSV file.

    Parameters
    ----------
    paths : AnalysisPaths
        Paths object with directory locations
    phenotypes : Dict[str, List]
        Phenotype summary dictionary

    Returns
    -------
    bool
        True if results saved successfully
    """
    phenotype_df = pd.DataFrame(phenotypes)
    phenotype_df.to_csv(paths.phenotype_summary, index=False)
    return True


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================


def main(save_path: str) -> None:
    """Main function to run the phenotyping pipeline."""
    try:
        # Get available birds
        birds = _get_available_birds(save_path)
        if not birds:
            logging.error("No birds found for processing")
            return

        logging.info(f"Found {len(birds)} birds for processing: {birds}")

        # Process each bird
        successful_birds = []
        failed_birds = []

        for bird in tqdm(birds, desc='Processing birds'):
            success = phenotype_bird_wrapper(
                save_path=save_path,
                bird=bird,
                n_phenotypes=5,
                reset_phenotypes=False
            )

            if success:
                successful_birds.append(bird)
            else:
                failed_birds.append(bird)

        # Report results
        logging.info(f"Processing complete. Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
        if failed_birds:
            logging.warning(f"Failed birds: {failed_birds}")

        # Perform cross-bird analysis if we have successful results
        if len(successful_birds) >= 2:
            logging.info("Starting cross-bird phenotype analysis...")

            try:
                # Aggregate phenotype data across birds
                aggregated_phenotypes = aggregate_phenotypes_across_birds(
                    save_path=save_path,
                    birds=successful_birds
                )

                if not aggregated_phenotypes.empty:
                    # Analyze phenotype patterns
                    phenotype_analysis = analyze_phenotype_patterns(aggregated_phenotypes)

                    # Save cross-bird phenotype analysis
                    saved_files = save_cross_bird_phenotype_analysis(
                        aggregated_phenotypes=aggregated_phenotypes,
                        phenotype_analysis=phenotype_analysis,
                        save_path=save_path,
                        filename_prefix='cross_bird_phenotypes'
                    )

                    logging.info(f"Cross-bird phenotype analysis complete. Files saved: {list(saved_files.keys())}")
                else:
                    logging.warning("No phenotype data available for cross-bird analysis")

            except Exception as e:
                logging.error(f"Error in cross-bird phenotype analysis: {e}")
                logging.error(traceback.format_exc())

        else:
            logging.info("Insufficient successful birds for cross-bird analysis (need at least 2)")

        logging.info("Phenotyping pipeline execution complete!")

    except Exception as e:
        logging.error(f"Error in main phenotyping pipeline: {e}")
        logging.error(traceback.format_exc())
        raise


def phenotype_bird_wrapper(save_path: str, bird: str, n_phenotypes: int = 5, reset_phenotypes: bool = False) -> bool:
    """
    Wrapper function to phenotype a single bird with proper path setup.

    Parameters
    ----------
    save_path : str
        Root directory containing bird data
    bird : str
        Bird identifier
    n_phenotypes : int, default=5
        Number of top phenotypes to process
    reset_phenotypes : bool, default=False
        Whether to reset existing phenotype data

    Returns
    -------
    bool
        True if phenotyping completed successfully
    """
    try:
        # Setup paths
        bird_folder = os.path.join(save_path, bird)
        data_folder = os.path.join(bird_folder, 'data')
        figure_folder = os.path.join(bird_folder, 'figures')

        # Check if required directories exist
        if not os.path.exists(data_folder):
            logging.error(f"Data folder not found for bird {bird}: {data_folder}")
            return False

        # Check if master_summary.csv exists
        master_summary_path = os.path.join(bird_folder, 'master_summary.csv')
        if not os.path.exists(master_summary_path):
            logging.error(f"Master summary not found for bird {bird}: {master_summary_path}")
            return False

        # Create paths object
        paths = AnalysisPaths(
            bird_folder=bird_folder,
            data_folder=data_folder,
            figure_folder=figure_folder
        )

        # Create figure directory if it doesn't exist
        os.makedirs(figure_folder, exist_ok=True)

        # Create analysis config
        config = AnalysisConfig()

        # Run phenotyping
        logging.info(f"Starting phenotyping for bird: {bird}")
        success = phenotype_bird(
            paths=paths,
            bird=bird,
            n_phenotypes=n_phenotypes,
            reset_phenotypes=reset_phenotypes,
            config=config
        )

        if success:
            logging.info(f"✅ Successfully phenotyped bird: {bird}")
        else:
            logging.error(f"❌ Failed to phenotype bird: {bird}")

        return success

    except Exception as e:
        logging.error(f"Error in phenotype_bird_wrapper for {bird}: {e}")
        logging.error(traceback.format_exc())
        return False


def _get_available_birds(save_path: str) -> List[str]:
    """
    Get list of available birds for processing.

    Parameters
    ----------
    save_path : str
        Root directory containing bird data

    Returns
    -------
    List[str]
        List of bird identifiers
    """
    try:
        birds = []
        if not os.path.exists(save_path):
            logging.error(f"Save path does not exist: {save_path}")
            return birds

        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)
            if os.path.isdir(item_path):
                # Check if it's a valid bird directory (has data folder and master_summary.csv)
                data_folder = os.path.join(item_path, 'data')
                master_summary = os.path.join(item_path, 'master_summary.csv')

                if os.path.exists(data_folder) and os.path.exists(master_summary):
                    birds.append(item)
                else:
                    logging.debug(f"Skipping {item}: missing data folder or master_summary.csv")

        return sorted(birds)

    except Exception as e:
        logging.error(f"Error getting available birds from {save_path}: {e}")
        return []

def aggregate_phenotypes_across_birds(save_path: str, birds: List[str]) -> pd.DataFrame:
    """
    Aggregate phenotype data across multiple birds.

    Parameters
    ----------
    save_path : str
        Root directory containing bird data
    birds : List[str]
        List of bird identifiers to aggregate

    Returns
    -------
    pd.DataFrame
        Aggregated phenotype data with bird identifiers
    """
    all_phenotypes = []

    for bird in birds:
        try:
            phenotype_path = os.path.join(save_path, bird, 'phenotype_summary.csv')
            if os.path.exists(phenotype_path):
                phenotype_df = pd.read_csv(phenotype_path)
                phenotype_df['bird'] = bird
                all_phenotypes.append(phenotype_df)
            else:
                logging.warning(f"Phenotype summary not found for bird {bird}")

        except Exception as e:
            logging.error(f"Error loading phenotypes for bird {bird}: {e}")

    if all_phenotypes:
        return pd.concat(all_phenotypes, ignore_index=True)
    else:
        return pd.DataFrame()


def analyze_phenotype_patterns(aggregated_phenotypes: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in aggregated phenotype data.

    Parameters
    ----------
    aggregated_phenotypes : pd.DataFrame
        Aggregated phenotype data across birds

    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results
    """
    analysis = {}

    try:
        # Basic statistics
        numeric_columns = aggregated_phenotypes.select_dtypes(include=[np.number]).columns
        analysis['summary_stats'] = aggregated_phenotypes[numeric_columns].describe()

        # Phenotype correlations
        correlation_matrix = aggregated_phenotypes[numeric_columns].corr()
        analysis['correlations'] = correlation_matrix

        # Bird-level summaries
        bird_summaries = aggregated_phenotypes.groupby('bird')[numeric_columns].agg(['mean', 'std', 'count'])
        analysis['bird_summaries'] = bird_summaries

        # Identify birds with extreme phenotypes
        extreme_birds = {}
        for col in numeric_columns:
            if col in aggregated_phenotypes.columns:
                q95 = aggregated_phenotypes[col].quantile(0.95)
                q05 = aggregated_phenotypes[col].quantile(0.05)

                high_birds = aggregated_phenotypes[aggregated_phenotypes[col] >= q95]['bird'].unique()
                low_birds = aggregated_phenotypes[aggregated_phenotypes[col] <= q05]['bird'].unique()

                extreme_birds[col] = {
                    'high_performers': high_birds.tolist(),
                    'low_performers': low_birds.tolist()
                }

        analysis['extreme_phenotypes'] = extreme_birds

        # Repeat behavior analysis
        if 'repeat_bool' in aggregated_phenotypes.columns:
            repeat_stats = aggregated_phenotypes.groupby('bird')['repeat_bool'].agg(['sum', 'count', 'mean'])
            analysis['repeat_behavior'] = repeat_stats

        # Entropy analysis
        entropy_cols = [col for col in aggregated_phenotypes.columns if 'entropy' in col.lower()]
        if entropy_cols:
            entropy_analysis = {}
            for col in entropy_cols:
                entropy_analysis[col] = {
                    'mean': aggregated_phenotypes[col].mean(),
                    'std': aggregated_phenotypes[col].std(),
                    'range': aggregated_phenotypes[col].max() - aggregated_phenotypes[col].min()
                }
            analysis['entropy_patterns'] = entropy_analysis

        logging.info("Phenotype pattern analysis completed successfully")

    except Exception as e:
        logging.error(f"Error in phenotype pattern analysis: {e}")
        logging.error(traceback.format_exc())

    return analysis


def save_cross_bird_phenotype_analysis(
        aggregated_phenotypes: pd.DataFrame,
        phenotype_analysis: Dict[str, Any],
        save_path: str,
        filename_prefix: str = 'cross_bird_phenotypes'
) -> Dict[str, str]:
    """
    Save cross-bird phenotype analysis results to files.

    Parameters
    ----------
    aggregated_phenotypes : pd.DataFrame
        Aggregated phenotype data
    phenotype_analysis : Dict[str, Any]
        Analysis results from analyze_phenotype_patterns
    save_path : str
        Root directory for saving files
    filename_prefix : str, default='cross_bird_phenotypes'
        Prefix for output filenames

    Returns
    -------
    Dict[str, str]
        Dictionary mapping file types to saved file paths
    """
    saved_files = {}

    try:
        # Create output directory
        output_dir = os.path.join(save_path, 'cross_bird_analysis')
        os.makedirs(output_dir, exist_ok=True)

        # Save aggregated phenotype data
        aggregated_path = os.path.join(output_dir, f'{filename_prefix}_aggregated.csv')
        aggregated_phenotypes.to_csv(aggregated_path, index=False)
        saved_files['aggregated_data'] = aggregated_path

        # Save summary statistics
        if 'summary_stats' in phenotype_analysis:
            summary_path = os.path.join(output_dir, f'{filename_prefix}_summary_stats.csv')
            phenotype_analysis['summary_stats'].to_csv(summary_path)
            saved_files['summary_stats'] = summary_path

        # Save correlation matrix
        if 'correlations' in phenotype_analysis:
            corr_path = os.path.join(output_dir, f'{filename_prefix}_correlations.csv')
            phenotype_analysis['correlations'].to_csv(corr_path)
            saved_files['correlations'] = corr_path

        # Save bird summaries
        if 'bird_summaries' in phenotype_analysis:
            bird_summary_path = os.path.join(output_dir, f'{filename_prefix}_bird_summaries.csv')
            phenotype_analysis['bird_summaries'].to_csv(bird_summary_path)
            saved_files['bird_summaries'] = bird_summary_path

        # Generate and save visualizations
        viz_paths = generate_phenotype_visualizations(
            aggregated_phenotypes,
            phenotype_analysis,
            output_dir,
            filename_prefix
        )
        saved_files.update(viz_paths)

        # Save analysis report as JSON
        analysis_path = os.path.join(output_dir, f'{filename_prefix}_analysis.json')
        # Convert numpy arrays to lists for JSON serialization
        serializable_analysis = _make_json_serializable(phenotype_analysis)

        import json
        with open(analysis_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        saved_files['analysis_report'] = analysis_path

        logging.info(f"Cross-bird phenotype analysis saved to: {output_dir}")

    except Exception as e:
        logging.error(f"Error saving cross-bird phenotype analysis: {e}")
        logging.error(traceback.format_exc())

    return saved_files


def generate_phenotype_visualizations(
    aggregated_phenotypes: pd.DataFrame,
    phenotype_analysis: Dict[str, Any],
    output_dir: str,
    filename_prefix: str
) -> Dict[str, str]:
    """
    Generate visualizations for phenotype analysis.

    Parameters
    ----------
    aggregated_phenotypes : pd.DataFrame
        Aggregated phenotype data
    phenotype_analysis : Dict[str, Any]
        Analysis results
    output_dir : str
        Directory for saving plots
    filename_prefix : str
        Prefix for output filenames

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot types to file paths
    """
    viz_paths = {}

    try:
        # Correlation heatmap
        if 'correlations' in phenotype_analysis:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(phenotype_analysis['correlations'], dtype=bool))
            sns.heatmap(
                phenotype_analysis['correlations'],
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f'
            )
            plt.title('Phenotype Correlations Across Birds')
            plt.tight_layout()

            corr_viz_path = os.path.join(output_dir, f'{filename_prefix}_correlation_heatmap.png')
            plt.savefig(corr_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['correlation_heatmap'] = corr_viz_path

        # Distribution plots for key phenotypes
        numeric_columns = aggregated_phenotypes.select_dtypes(include=[np.number]).columns
        key_phenotypes = ['repertoire_sz', 'sequence_entropy', 'mean_repeat_syls', 'composite_score']
        available_phenotypes = [col for col in key_phenotypes if col in numeric_columns]

        if available_phenotypes:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            for i, phenotype in enumerate(available_phenotypes[:4]):
                if i < len(axes):
                    ax = axes[i]

                    # Box plot by bird
                    sns.boxplot(data=aggregated_phenotypes, y=phenotype, x='bird', ax=ax)
                    ax.set_title(f'{phenotype.replace("_", " ").title()} by Bird')
                    ax.tick_params(axis='x', rotation=45)

            # Hide unused subplots
            for j in range(len(available_phenotypes), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()

            dist_viz_path = os.path.join(output_dir, f'{filename_prefix}_phenotype_distributions.png')
            plt.savefig(dist_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['phenotype_distributions'] = dist_viz_path

        # Scatter plot matrix for key relationships
        if len(available_phenotypes) >= 2:
            plt.figure(figsize=(12, 10))

            # Create pairplot
            plot_data = aggregated_phenotypes[available_phenotypes + ['bird']].copy()
            g = sns.pairplot(plot_data, hue='bird', diag_kind='hist', plot_kws={'alpha': 0.7})
            g.fig.suptitle('Phenotype Relationships Across Birds', y=1.02)

            scatter_viz_path = os.path.join(output_dir, f'{filename_prefix}_pairplot.png')
            plt.savefig(scatter_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['pairplot'] = scatter_viz_path

        # Repeat behavior visualization
        if 'repeat_bool' in aggregated_phenotypes.columns:
            plt.figure(figsize=(10, 6))

            repeat_summary = aggregated_phenotypes.groupby('bird')['repeat_bool'].mean().reset_index()
            repeat_summary = repeat_summary.sort_values('repeat_bool', ascending=False)

            sns.barplot(data=repeat_summary, x='bird', y='repeat_bool')
            plt.title('Proportion of Label Schemes with Repeats by Bird')
            plt.ylabel('Proportion with Repeats')
            plt.xlabel('Bird')
            plt.xticks(rotation=45)
            plt.tight_layout()

            repeat_viz_path = os.path.join(output_dir, f'{filename_prefix}_repeat_behavior.png')
            plt.savefig(repeat_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['repeat_behavior'] = repeat_viz_path

        # Performance ranking visualization
        if 'composite_score' in aggregated_phenotypes.columns:
            plt.figure(figsize=(12, 8))

            # Get top performing label schemes for each bird
            top_schemes = aggregated_phenotypes.loc[aggregated_phenotypes.groupby('bird')['composite_score'].idxmax()]

            sns.scatterplot(data=aggregated_phenotypes, x='repertoire_sz', y='composite_score',
                          hue='bird', alpha=0.6, s=50)

            # Highlight top performers
            sns.scatterplot(data=top_schemes, x='repertoire_sz', y='composite_score',
                          hue='bird', s=200, marker='*', legend=False)

            plt.title('Composite Score vs Repertoire Size\n(Stars indicate best performing scheme per bird)')
            plt.xlabel('Repertoire Size')
            plt.ylabel('Composite Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            perf_viz_path = os.path.join(output_dir, f'{filename_prefix}_performance_ranking.png')
            plt.savefig(perf_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['performance_ranking'] = perf_viz_path

        logging.info(f"Generated {len(viz_paths)} phenotype visualizations")

    except Exception as e:
        logging.error(f"Error generating phenotype visualizations: {e}")
        logging.error(traceback.format_exc())

    return viz_paths


def _make_json_serializable(obj, path="root"):
    """
    Convert numpy arrays and other non-serializable objects to JSON-compatible formats.

    Parameters
    ----------
    obj : Any
        Object to make JSON serializable
    path : str
        Current path in the object tree (for debugging)

    Returns
    -------
    Any
        JSON-serializable version of the object
    """
    try:
        if isinstance(obj, dict):
            # Handle dictionaries with potentially non-string keys (like tuples)
            serializable_dict = {}
            for key, value in obj.items():
                # Convert any key to a string
                if isinstance(key, tuple):
                    str_key = f"tuple_{str(key)}"
                elif isinstance(key, (np.integer, np.floating)):
                    str_key = str(key.item())
                elif key is None:
                    str_key = "None"
                else:
                    str_key = str(key)

                # Ensure the key is valid for JSON
                if not isinstance(str_key, str):
                    str_key = str(str_key)

                serializable_dict[str_key] = _make_json_serializable(value, f"{path}.{str_key}")
            return serializable_dict
        elif isinstance(obj, (list, tuple)):
            return [_make_json_serializable(item, f"{path}[{i}]") for i, item in enumerate(obj)]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict with string keys
            return {str(k): v for k, v in obj.to_dict('records')}
        elif isinstance(obj, pd.Series):
            # Convert Series to dict with string keys
            return {str(k): v for k, v in obj.to_dict().items()}
        elif isinstance(obj, (pd.Index, pd.MultiIndex)):
            return [str(x) for x in obj.tolist()]
        elif hasattr(obj, 'tolist'):  # Catch any other numpy-like objects
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Catch numpy scalars
            return obj.item()
        else:
            return obj
    except Exception as e:
        logging.warning(f"Error serializing object at {path}: {e}, converting to string")
        return str(obj)


def clear_phenotyping_outputs(save_path: str, bird: str = None, confirm: bool = True) -> bool:
    """
    Clear phenotyping output files to avoid clutter.

    Parameters
    ----------
    save_path : str
        Root directory containing bird data
    bird : str, optional
        Specific bird to clear (all birds if None)
    confirm : bool, default=True
        Whether to ask for confirmation before deleting

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get birds to process
        if bird is not None:
            birds_to_clear = [bird]
        else:
            birds_to_clear = _get_available_birds(save_path)

        if not birds_to_clear:
            logging.info("No birds found to clear")
            return True

        # Calculate what will be removed
        total_items = 0
        paths_to_remove = []

        for bird_name in birds_to_clear:
            bird_path = os.path.join(save_path, bird_name)

            # Phenotype summary CSV
            phenotype_summary_path = os.path.join(bird_path, 'phenotype_summary.csv')
            if os.path.exists(phenotype_summary_path):
                paths_to_remove.append(('phenotype_summary', phenotype_summary_path))
                total_items += 1

            # Sequence data directory
            sequence_data_path = os.path.join(bird_path, 'data', 'sequences')
            if os.path.exists(sequence_data_path):
                paths_to_remove.append(('sequence_data', sequence_data_path))
                # Count files for reporting
                for root, dirs, files in os.walk(sequence_data_path):
                    total_items += len(files)

            # Phenotype figures directory
            phenotype_figures_path = os.path.join(bird_path, 'figures')
            if os.path.exists(phenotype_figures_path):
                # Look for rank-specific folders and phenotype-related images
                rank_folders = [d for d in os.listdir(phenotype_figures_path)
                               if os.path.isdir(os.path.join(phenotype_figures_path, d)) and d.startswith('rank')]

                for rank_folder in rank_folders:
                    rank_path = os.path.join(phenotype_figures_path, rank_folder)
                    paths_to_remove.append(('phenotype_figures', rank_path))
                    # Count files for reporting
                    for root, dirs, files in os.walk(rank_path):
                        total_items += len(files)

        # Also check for cross-bird analysis directory
        cross_bird_path = os.path.join(save_path, 'cross_bird_analysis')
        if os.path.exists(cross_bird_path):
            paths_to_remove.append(('cross_bird_analysis', cross_bird_path))
            for root, dirs, files in os.walk(cross_bird_path):
                total_items += len(files)

        if not paths_to_remove:
            logging.info("No phenotyping outputs found to clear")
            return True

        # Show what will be removed
        logging.info(f"Found {total_items} items to remove across {len(birds_to_clear)} bird(s):")
        for bird_name in birds_to_clear:
            bird_items = [path for path_type, path in paths_to_remove if bird_name in path]
            if bird_items:
                logging.info(f"  {bird_name}: {len(bird_items)} directories/files")

        # Show cross-bird analysis if present
        cross_bird_items = [path for path_type, path in paths_to_remove if 'cross_bird_analysis' in path]
        if cross_bird_items:
            logging.info(f"  Cross-bird analysis: {len(cross_bird_items)} directories")

        # Confirmation
        if confirm:
            response = input(f"\nAre you sure you want to delete {total_items} phenotyping output items? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logging.info("Deletion cancelled")
                return False

        # Remove items
        removed_count = 0
        failed_count = 0

        for path_type, path in paths_to_remove:
            try:
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                    logging.debug(f"Removed directory: {path}")
                else:
                    os.remove(path)
                    logging.debug(f"Removed file: {path}")
                removed_count += 1
            except Exception as e:
                logging.error(f"Failed to remove {path}: {e}")
                failed_count += 1

        # Report results
        if failed_count == 0:
            logging.info(f"✅ Successfully removed all {removed_count} items")
        else:
            logging.warning(f"⚠️ Removed {removed_count} items, failed to remove {failed_count} items")

        return failed_count == 0

    except Exception as e:
        logging.error(f"Error clearing phenotyping outputs: {e}")
        return False


if __name__ == '__main__':
    # Setup logging
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'phenotyping_pipeline.log'))
            logging.StreamHandler()
        ]
    )

    # Setup paths and parameters
    # Uncomment and modify these paths as needed:
    # path_to_macaw = check_sys_for_macaw_root()

    # Test with your data paths
    save_path = os.path.join('/Volumes', 'Extreme SSD', 'wseg test')
    if os.path.exists(save_path):
        logging.info(f"Processing wseg test dataset...")
        # Uncomment to clear previous results:
        # clear_phenotyping_outputs(save_path=save_path, confirm=False)
        main(save_path=save_path)
    else:
        logging.warning(f"Path does not exist: {save_path}")

    save_path = os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
    if os.path.exists(save_path):
        logging.info(f"Processing evsong test dataset...")
        # Uncomment to clear previous results:
        # clear_phenotyping_outputs(save_path=save_path, confirm=False)
        main(save_path=save_path)
    else:
        logging.warning(f"Path does not exist: {save_path}")
