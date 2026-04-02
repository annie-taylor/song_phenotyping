# slice_validation.py

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import os
import tables
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Import your existing functions
from tools.song_io import get_memory_usage, logger, get_song_spec, rms_norm, butter_bandpass_filter_sos
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file


@dataclass
class SliceValidationConfig:
    """Simple configuration for slice validation."""
    slice_length_ms: float = 50.0
    max_files_to_validate: int = 5
    figure_size: Tuple[float, float] = (12, 10)
    skip_existing: bool = True


class SliceValidator:
    """Simple slice validator - just checks if slices can reconstruct originals."""

    def __init__(self, project_directory: str, config: SliceValidationConfig = None):
        self.project_directory = Path(project_directory)
        self.config = config or SliceValidationConfig()
        self.birds = self._discover_birds()

    def _discover_birds(self) -> List[str]:
        """Find birds with slice data."""
        birds = []
        for item in self.project_directory.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data'):

                slices_dir = item / 'slice_data' / 'specs'
                if slices_dir.exists() and list(slices_dir.glob('*.h5')):
                    birds.append(item.name)

        logger.info(f"Found {len(birds)} birds with slice data")
        return sorted(birds)

    def validate_slice_reconstruction(self, birds: List[str] = None) -> Dict[str, str]:
        """Create validation PDFs showing original vs reconstructed spectrograms."""
        if birds is None:
            birds = self.birds

        logger.info(f"🔍 Validating slice reconstruction for {len(birds)} birds")

        results = {}
        for bird in birds:
            pdf_path = self._validate_single_bird(bird)
            results[bird] = pdf_path

        successful = sum(1 for path in results.values() if path is not None)
        logger.info(f"✅ Created {successful}/{len(birds)} validation PDFs")

        return results

    def _validate_single_bird(self, bird: str) -> Optional[str]:
        """Create validation PDF for a single bird."""
        bird_dir = self.project_directory / bird

        # Setup paths
        slices_dir = bird_dir / 'slice_data' / 'specs'
        pdf_dir = bird_dir / 'pdfs'
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / f'{bird}_slice_validation.pdf'

        # Skip if exists
        if self.config.skip_existing and pdf_path.exists():
            logger.info(f"⏭️ Skipping {bird} - validation PDF already exists")
            return str(pdf_path)

        try:
            # Find slice files
            slice_files = list(slices_dir.glob('*.h5'))
            if not slice_files:
                logger.warning(f"No slice files found for {bird}")
                return None

            # Limit files for validation
            slice_files = slice_files[:self.config.max_files_to_validate]

            # Create validation PDF
            self._create_validation_pdf(pdf_path, bird, slice_files)
            logger.info(f"✅ Created validation PDF for {bird}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error creating validation PDF for {bird}: {e}")
            return None

    def _create_validation_pdf(self, pdf_path: Path, bird: str, slice_files: List[Path]):
        """Create the validation PDF comparing original vs reconstructed."""

        with pdf_backend.PdfPages(pdf_path) as pdf:
            for slice_file in tqdm(slice_files, desc=f"Validating {bird}"):
                try:
                    # Load slice data and reconstruct
                    original_spec, reconstructed_spec, metadata = self._reconstruct_from_slices(slice_file)

                    if original_spec is not None and reconstructed_spec is not None:
                        # Create comparison plot
                        fig = self._create_comparison_plot(
                            original_spec, reconstructed_spec,
                            slice_file.name, metadata
                        )

                        if fig:
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)

                except Exception as e:
                    logger.error(f"Error validating {slice_file}: {e}")

    def _reconstruct_from_slices(self, slice_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """Reconstruct original spectrogram from slices."""
        metadata = {}

        try:
            with tables.open_file(str(slice_file), 'r') as f:
                # Load slice data
                slice_specs = f.root.spectrograms.read()
                slice_onsets = f.root.onsets.read()
                slice_offsets = f.root.offsets.read()

                # Get audio filename
                audio_filename_raw = f.root.audio_filename.read()
                if isinstance(audio_filename_raw[0], bytes):
                    audio_filename = audio_filename_raw[0].decode('utf-8')
                else:
                    audio_filename = str(audio_filename_raw[0])

                metadata['audio_filename'] = audio_filename
                metadata['n_slices'] = len(slice_specs)
                metadata['total_duration'] = (slice_offsets[-1] - slice_onsets[0]) / 1000.0

            # Generate original spectrogram
            original_spec = self._generate_original_spectrogram(
                audio_filename, slice_onsets[0], slice_offsets[-1]
            )

            # Reconstruct from slices
            reconstructed_spec = self._concatenate_slices(
                slice_specs, slice_onsets, slice_offsets
            )

            return original_spec, reconstructed_spec, metadata

        except Exception as e:
            logger.error(f"Error reconstructing from {slice_file}: {e}")
            return None, None, {}

    def _generate_original_spectrogram(self, audio_filename: str,
                                       start_ms: float, end_ms: float) -> Optional[np.ndarray]:
        """Generate original spectrogram for comparison."""
        try:
            # Handle path resolution
            if not os.path.exists(audio_filename):
                from tools.system_utils import replace_macaw_root
                audio_filename = replace_macaw_root(audio_filename)

            if not os.path.exists(audio_filename):
                logger.warning(f"Audio file not found: {audio_filename}")
                return None

                # Load and process audio
            audio, fs = read_audio_file(audio_filename)
            audio = rms_norm(audio)

            # Apply filtering
            params = SpectrogramParams(max_dur=np.ceil((end_ms - start_ms) / 1000.0))
            audio = butter_bandpass_filter_sos(
                audio, lowcut=params.min_freq, highcut=params.max_freq, fs=fs, order=3
            )

            # Generate spectrogram for the time window
            start_time = start_ms / 1000.0
            end_time = end_ms / 1000.0

            spec, _, _ = get_song_spec(
                t1=start_time, t2=end_time, audio=audio,
                params=params, fs=fs, downsample=False
            )

            return spec

        except Exception as e:
            logger.error(f"Error generating original spectrogram: {e}")
            return None

    def _concatenate_slices(self, slice_specs: np.ndarray, slice_onsets: np.ndarray,
                            slice_offsets: np.ndarray) -> np.ndarray:
        """Simple slice concatenation - just stitch them together."""

        if len(slice_specs) == 0:
            return np.array([])

        # Simple approach: just concatenate horizontally
        # This assumes slices are in temporal order
        try:
            concatenated = np.hstack(slice_specs)
            return concatenated
        except Exception as e:
            logger.error(f"Error concatenating slices: {e}")
            # Fallback: return first slice
            return slice_specs[0] if len(slice_specs) > 0 else np.array([])

    def _create_comparison_plot(self, original_spec: np.ndarray, reconstructed_spec: np.ndarray,
                                filename: str, metadata: Dict) -> Optional[plt.Figure]:
        """Create simple comparison plot."""

        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.config.figure_size)

            # Original spectrogram
            ax1.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f'Original: {filename}')
            ax1.set_ylabel('Frequency')
            ax1.set_xticks([])

            # Reconstructed spectrogram
            ax2.imshow(reconstructed_spec, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_title('Reconstructed from Slices')
            ax2.set_ylabel('Frequency')
            ax2.set_xticks([])

            # Difference
            if original_spec.shape == reconstructed_spec.shape:
                diff = np.abs(original_spec - reconstructed_spec)
                im3 = ax3.imshow(diff, aspect='auto', origin='lower', cmap='hot')
                plt.colorbar(im3, ax=ax3)

                # Calculate error metrics
                mse = np.mean((original_spec - reconstructed_spec) ** 2)
                max_diff = np.max(diff)

                ax3.set_title(f'Absolute Difference (MSE: {mse:.4f}, Max: {max_diff:.4f})')
            else:
                ax3.text(0.5, 0.5,
                         f'Shape mismatch:\nOriginal: {original_spec.shape}\nReconstructed: {reconstructed_spec.shape}',
                         ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Shape Mismatch - Cannot Compare')

            ax3.set_ylabel('Frequency')
            ax3.set_xlabel('Time')

            # Add metadata
            metadata_text = f"Slices: {metadata.get('n_slices', 'N/A')}, Duration: {metadata.get('total_duration', 'N/A'):.2f}s"
            fig.suptitle(metadata_text, fontsize=12)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
            return None

class SliceAnalyzer(SliceValidator):
    """Extended validator that also provides basic slice statistics."""

    def analyze_slice_patterns(self, birds: List[str] = None) -> Dict[str, str]:
        """Create simple analysis of slice patterns (duration, gaps, etc.)."""
        if birds is None:
            birds = self.birds

        logger.info(f"📊 Analyzing slice patterns for {len(birds)} birds")

        results = {}
        for bird in birds:
            pdf_path = self._analyze_single_bird_patterns(bird)
            results[bird] = pdf_path

        return results

    def _analyze_single_bird_patterns(self, bird: str) -> Optional[str]:
        """Create pattern analysis PDF for a single bird."""
        bird_dir = self.project_directory / bird

        # Setup paths
        slices_dir = bird_dir / 'data' / 'slices'
        pdf_dir = bird_dir / 'pdfs'
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / f'{bird}_slice_analysis.pdf'

        # Skip if exists
        if self.config.skip_existing and pdf_path.exists():
            logger.info(f"⏭️ Skipping {bird} - analysis PDF already exists")
            return str(pdf_path)

        try:
            slice_files = list(slices_dir.glob('*.h5'))[:3]  # Analyze first 3 files
            if not slice_files:
                return None

            with pdf_backend.PdfPages(pdf_path) as pdf:
                for slice_file in slice_files:
                    fig = self._create_analysis_plot(slice_file)
                    if fig:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

            logger.info(f"✅ Created analysis PDF for {bird}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error creating analysis PDF for {bird}: {e}")
            return None

    def _create_analysis_plot(self, slice_file: Path) -> Optional[plt.Figure]:
        """Create simple analysis plot with basic statistics."""

        try:
            with tables.open_file(str(slice_file), 'r') as f:
                slice_onsets = f.root.onsets.read()
                slice_offsets = f.root.offsets.read()
                slice_labels = f.root.manual.read()

                # Convert labels if needed
                if len(slice_labels) > 0 and isinstance(slice_labels[0], bytes):
                    slice_labels = [label.decode('utf-8') for label in slice_labels]

            # Calculate basic metrics
            durations = slice_offsets - slice_onsets
            gaps = slice_onsets[1:] - slice_offsets[:-1] if len(slice_onsets) > 1 else []

            # Create plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Slice Analysis: {slice_file.name}', fontsize=14)

            # Duration histogram
            ax1.hist(durations, bins=20, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Duration (ms)')
            ax1.set_ylabel('Count')
            ax1.set_title('Slice Duration Distribution')
            ax1.axvline(np.mean(durations), color='red', linestyle='--',
                        label=f'Mean: {np.mean(durations):.1f} ms')
            ax1.legend()

            # Gap analysis
            if len(gaps) > 0:
                ax2.hist(gaps, bins=20, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Gap Duration (ms)')
                ax2.set_ylabel('Count')
                ax2.set_title('Inter-slice Gaps')
                ax2.axvline(np.mean(gaps), color='red', linestyle='--',
                            label=f'Mean: {np.mean(gaps):.1f} ms')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No gap data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Inter-slice Gaps')

            # Timeline view
            for i, (onset, offset, label) in enumerate(zip(slice_onsets, slice_offsets, slice_labels)):
                ax3.barh(i, offset - onset, left=onset, height=0.8, alpha=0.7)
                ax3.text(onset + (offset - onset) / 2, i, str(label),
                         ha='center', va='center', fontsize=6)

            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Slice Index')
            ax3.set_title('Slice Timeline')

            # Label distribution
            from collections import Counter

            label_counts = Counter(slice_labels)

            if len(label_counts) <= 15:  # Only show if reasonable number
                ax4.bar(range(len(label_counts)), list(label_counts.values()), alpha=0.7)
                ax4.set_xlabel('Label')
                ax4.set_ylabel('Count')
                ax4.set_title('Label Distribution')
                ax4.set_xticks(range(len(label_counts)))
                ax4.set_xticklabels(list(label_counts.keys()), rotation=45)
            else:
                ax4.text(0.5, 0.5, f'{len(label_counts)} unique labels\n(too many to display)',
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Label Distribution')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating analysis plot for {slice_file}: {e}")
            return None


# Simple utility functions

def quick_slice_validation(project_directory: str,
                           birds: List[str] = None) -> Dict[str, str]:
    """
    Quick function to validate slice reconstruction with defaults.

    Args:
        project_directory: Path to project directory
        birds: Specific birds to validate (all if None)

    Returns:
        Dict mapping bird names to validation PDF paths
    """
    validator = SliceValidator(project_directory)
    return validator.validate_slice_reconstruction(birds)


def analyze_slice_patterns(project_directory: str,
                           birds: List[str] = None) -> Dict[str, str]:
    """
    Quick function to analyze slice patterns.

    Args:
        project_directory: Path to project directory
        birds: Specific birds to analyze (all if None)

    Returns:
        Dict mapping bird names to analysis PDF paths
    """
    analyzer = SliceAnalyzer(project_directory)
    return analyzer.analyze_slice_patterns(birds)


def validate_single_bird_slices(project_directory: str, bird: str) -> Optional[str]:
    """
    Validate slices for a single bird.

    Args:
        project_directory: Path to project directory
        bird: Bird name to validate

    Returns:
        Path to validation PDF or None if failed
    """
    validator = SliceValidator(project_directory)
    return validator._validate_single_bird(bird)


# Example usage
def example_usage():
    """Example of how to use the simple slice validation module."""

    from tools.project_config import ProjectConfig
    cfg = ProjectConfig.load()
    project_dir = str(cfg.local_cache / 'evsong slice test')

    # Basic validation - all birds
    results = quick_slice_validation(project_dir)
    print(f"Created validation PDFs for {len(results)} birds")

    # Pattern analysis
    analysis_results = analyze_slice_patterns(project_dir)
    print(f"Created analysis PDFs for {len(analysis_results)} birds")

    # Single bird
    pdf_path = validate_single_bird_slices(project_dir, "bird1")
    if pdf_path:
        print(f"Created validation PDF: {pdf_path}")

    # Custom configuration
    config = SliceValidationConfig(
        slice_length_ms=40.0,
        max_files_to_validate=3,
        figure_size=(14, 12)
    )

    validator = SliceValidator(project_dir, config)
    results = validator.validate_slice_reconstruction(['bird1', 'bird2'])


if __name__ == '__main__':
    example_usage()