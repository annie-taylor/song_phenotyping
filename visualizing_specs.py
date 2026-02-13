import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy import signal
import librosa
import os
from tqdm import tqdm
import numpy as np
import random
import tables
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import consolidated functions (similar to A_spec_saving.py)
from tools.song_io import (
    setup_logging,
    get_memory_usage,
    parse_audio_filename,
    get_song_spec,
    rms_norm,
    butter_bandpass_filter_sos,
    logger
)
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file
from tools.audio_path_management import load_audio_paths_mapping, get_audio_path
from phenotype_pdfs import create_dual_labeled_spectrogram  # Reuse this!


class SpectrogramVisualizer:
    """Consolidated spectrogram visualization with shared functionality."""

    def __init__(self, project_directory: str):
        self.project_directory = Path(project_directory)
        self.birds = self._discover_birds()

    def _discover_birds(self) -> List[str]:
        """Discover birds using consolidated logic."""
        birds = []
        if not self.project_directory.exists():
            return birds

        for item in self.project_directory.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data'):

                # Check for syllable data
                syllables_dir = item / 'data' / 'syllables'
                slices_dir = item / 'data' / 'slices'

                has_syllables = syllables_dir.exists() and list(syllables_dir.glob('*.h5'))
                has_slices = slices_dir.exists() and list(slices_dir.glob('*.h5'))

                if has_syllables or has_slices:
                    birds.append(item.name)

            return sorted(birds)

    def create_song_spectrograms_pdf(self, target_duration: float = 8.0,
                                     sample_rate: int = 32000, prefer_local: bool = True):
        """Create song spectrograms using consolidated audio path management."""

        logger.info(f"🎵 Creating song spectrograms for {len(self.birds)} birds")
        logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

        for bird in self.birds:
            bird_dir = self.project_directory / bird

            try:
                # Use consolidated audio path management
                mapping = load_audio_paths_mapping(str(bird_dir))
                if not mapping:
                    logger.warning(f"No audio paths mapping found for {bird}")
                    continue

                # Create output directories using same structure as other modules
                pdf_dir = bird_dir / 'pdfs'
                spec_dir = bird_dir / 'spectrograms' / 'songs'  # Distinguish from syllable spectrograms
                pdf_dir.mkdir(exist_ok=True)
                spec_dir.mkdir(parents=True, exist_ok=True)

                pdf_path = pdf_dir / f'{bird}_all_song_specs.pdf'

                with pdf_backend.PdfPages(pdf_path) as pdf:
                    for filename in tqdm(mapping.keys(), desc=f"🔄 Processing {bird}"):
                        try:
                            # Use consolidated audio path resolution
                            file_path = get_audio_path(str(bird_dir), filename, prefer_local)

                            # Use consolidated audio reading and processing
                            audio, sr = read_audio_file(file_path)
                            audio = rms_norm(audio)

                            # Apply same filtering as main pipeline
                            params = SpectrogramParams()
                            audio = butter_bandpass_filter_sos(
                                audio, lowcut=params.min_freq, highcut=params.max_freq,
                                fs=sr, order=5
                            )

                            # Pad or trim to target duration
                            target_samples = int(target_duration * sr)
                            if len(audio) < target_samples:
                                audio = np.pad(audio, (0, target_samples - len(audio)))
                            else:
                                audio = audio[:target_samples]

                            # Create spectrogram using same parameters as pipeline
                            spec, _, t = get_song_spec(
                                t1=0, t2=target_duration, audio=audio,
                                params=params, fs=sr, downsample=False
                            )

                            # Create plot with consistent styling
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.imshow(spec, aspect='auto', origin='lower',
                                      extent=[0, target_duration, 0, spec.shape[0]])

                            ax.set_ylabel('Frequency Bins')
                            ax.set_xlabel('Time (s)')
                            ax.set_title(f'Song Spectrogram: {filename}')

                            # Add metadata
                            path_type = "Local" if prefer_local and mapping[filename].get('local') else "Server"
                            display_text = f"{path_type}: {file_path}"
                            ax.text(0.02, 0.98, display_text, transform=ax.transAxes,
                                    fontsize=8, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                            # Save PNG
                            png_filename = f"{Path(filename).stem}.png"
                            png_path = spec_dir / png_filename
                            plt.savefig(png_path, dpi=150, bbox_inches='tight')

                            # Add to PDF
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)

                        except Exception as e:
                            logger.error(f"Failed to create spectrogram for {filename}: {e}")

                logger.info(f"✅ Created spectrogram PDF for {bird}: {pdf_path}")

            except Exception as e:
                logger.error(f"💥 Error processing bird {bird}: {e}")

    def create_syllable_sample_pdfs(self, syllables_per_bird: int = 40, rank: int = 0):
        """Create syllable PDFs using shared dual-labeled spectrogram function."""

        logger.info(f"🔤 Creating syllable sample PDFs for {len(self.birds)} birds")

        for bird in self.birds:
            bird_dir = self.project_directory / bird

            syllables_dir = bird_dir / 'data' / 'syllables'

            if not syllables_dir.exists():
                logger.warning(f"No syllables directory for {bird}")
                continue

            syllable_files = list(syllables_dir.glob('*.h5'))
            if not syllable_files:
                logger.warning(f"No syllable files found for {bird}")
                continue

            # Sample syllables consistently
            random.seed(42)  # Consistent sampling
            n_to_sample = min(syllables_per_bird, len(syllable_files))
            sampled_files = random.sample(syllable_files, n_to_sample)

            # Create output directory
            pdf_dir = bird_dir / 'pdfs'
            pdf_dir.mkdir(exist_ok=True)
            pdf_path = pdf_dir / f'{bird}_example_syllables_sampled.pdf'

            # Generate spectrograms using shared function
            spec_images = []
            for syl_file in sampled_files:
                try:
                    # Use the shared dual-labeled spectrogram function
                    spec_path = create_dual_labeled_spectrogram(
                        syl_file=syl_file,
                        bird_path=bird_dir,
                        rank=rank,
                        spectrograms_dir=bird_dir / 'spectrograms' / 'sampled',
                        overwrite=False,  # Use cached versions
                        duration=6.0
                    )
                    if spec_path and os.path.exists(spec_path):
                        spec_images.append(spec_path)

                except Exception as e:
                    logger.error(f"Error creating spectrogram for {syl_file}: {e}")

            if not spec_images:
                logger.warning(f"No spectrograms generated for {bird}")
                continue

            # Create PDF with grid layout (reuse layout logic from phenotype PDFs)
            with pdf_backend.PdfPages(pdf_path) as pdf:
                n_cols, n_rows = 4, 2
                syllables_per_page = n_cols * n_rows

                for page_start in range(0, len(spec_images), syllables_per_page):
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
                    fig.suptitle(f'{bird} - Sampled Syllables (Page {page_start // syllables_per_page + 1})',
                                 fontsize=16)

                    # Flatten axes for easier indexing
                    if n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    elif n_cols == 1:
                        axes = [[ax] for ax in axes]
                    else:
                        axes = axes
                    axes_flat = [ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])]

                    page_images = spec_images[page_start:page_start + syllables_per_page]

                    for i, img_path in enumerate(page_images):
                        ax = axes_flat[i]

                        # Load and display the dual-labeled spectrogram image
                        img = plt.imread(img_path)
                        ax.imshow(img)
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # Add filename as title
                        filename = Path(img_path).stem
                        ax.set_title(filename, fontsize=8)

                    # Hide unused subplots
                    for i in range(len(page_images), len(axes_flat)):
                        axes_flat[i].set_visible(False)

                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

            logger.info(f"✅ Created syllable sample PDF for {bird}: {pdf_path} ({n_to_sample} syllables)")


class SliceValidator(SpectrogramVisualizer):
    """Extended visualizer for slice validation and exploration."""

    def validate_slice_reconstruction(self, bird: str, slice_length_ms: float = 50.0):
        """Reconstruct original spectrograms from slices to validate slicing."""

        logger.info(f"🔍 Validating slice reconstruction for {bird}")

        bird_dir = self.project_directory / bird
        slices_dir = bird_dir / 'data' / 'slices'  # Look in slices directory

        if not slices_dir.exists():
            logger.warning(f"No slices directory for {bird}")
            return

        # Find slice files (assuming they follow your naming convention)
        slice_files = list(slices_dir.glob('*.h5'))  # All files in slices dir are slice files
        if not slice_files:
            logger.warning(f"No slice files found for {bird}")
            return

        validation_dir = bird_dir / 'spectrograms' / 'slice_validation'
        validation_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = bird_dir / 'pdfs' / f'{bird}_slice_validation.pdf'

        with pdf_backend.PdfPages(pdf_path) as pdf:
            for slice_file in slice_files[:5]:  # Validate first 5 files
                try:
                    original_spec, reconstructed_spec = self._reconstruct_from_slices(
                        slice_file, slice_length_ms
                    )

                    if original_spec is not None and reconstructed_spec is not None:
                        # Create comparison plot
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

                        # Original spectrogram
                        ax1.imshow(original_spec, aspect='auto', origin='lower')
                        ax1.set_title(f'Original: {slice_file.name}')
                        ax1.set_ylabel('Frequency')

                        # Reconstructed spectrogram
                        ax2.imshow(reconstructed_spec, aspect='auto', origin='lower')
                        ax2.set_title('Reconstructed from Slices')
                        ax2.set_ylabel('Frequency')

                        # Difference
                        diff = np.abs(original_spec - reconstructed_spec)
                        im3 = ax3.imshow(diff, aspect='auto', origin='lower', cmap='hot')
                        ax3.set_title('Absolute Difference')
                        ax3.set_ylabel('Frequency')
                        ax3.set_xlabel('Time')
                        plt.colorbar(im3, ax=ax3)

                        # Add statistics
                        mse = np.mean((original_spec - reconstructed_spec) ** 2)
                        max_diff = np.max(diff)
                        fig.suptitle(f'MSE: {mse:.4f}, Max Diff: {max_diff:.4f}')

                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                except Exception as e:
                    logger.error(f"Error validating {slice_file}: {e}")

        logger.info(f"✅ Created slice validation PDF: {pdf_path}")

    def _reconstruct_from_slices(self, slice_file: Path, slice_length_ms: float) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """Reconstruct original spectrogram from slices."""
        try:
            with tables.open_file(str(slice_file), 'r') as f:
                # Load slice data
                slice_specs = f.root.spectrograms.read()
                slice_onsets = f.root.onsets.read()
                slice_offsets = f.root.offsets.read()

                # Get original audio file info
                audio_filename_raw = f.root.audio_filename.read()
                if isinstance(audio_filename_raw[0], bytes):
                    audio_filename = audio_filename_raw[0].decode('utf-8')
                else:
                    audio_filename = str(audio_filename_raw[0])

                # Load original syllable onsets/offsets if available
                if hasattr(f.root, 'syl_onsets') and hasattr(f.root, 'syl_offsets'):
                    syl_onsets = f.root.syl_onsets.read()

                else:
                    # Estimate from slice data
                    syl_onsets = np.array([slice_onsets[0]])
                    syl_offsets = np.array([slice_offsets[-1]])

                # Generate original spectrogram for comparison
                params = SpectrogramParams()

                # Resolve audio file path (handle server/local paths)
                if not os.path.exists(audio_filename):
                    from tools.system_utils import replace_macaw_root

                    audio_filename = replace_macaw_root(audio_filename)

                if not os.path.exists(audio_filename):
                    logger.warning(f"Audio file not found: {audio_filename}")
                    return None, None

                # Load and process audio
                audio, fs = read_audio_file(audio_filename)
                audio = rms_norm(audio)
                audio = butter_bandpass_filter_sos(
                    audio, lowcut=params.min_freq, highcut=params.max_freq, fs=fs, order=5
                )

                # Generate original spectrogram for the same time window
                first_time = syl_onsets[0] / 1000
                last_time = syl_offsets[-1] / 1000

                original_spec, _, _ = get_song_spec(
                    t1=first_time, t2=last_time, audio=audio,
                    params=params, fs=fs, downsample=False
                )

                # Reconstruct spectrogram from slices
                reconstructed_spec = self._concatenate_slices(
                    slice_specs, slice_onsets, slice_offsets, slice_length_ms,
                    first_time * 1000, last_time * 1000, params
                )

                return original_spec, reconstructed_spec

        except Exception as e:
            logger.error(f"Error reconstructing from slices: {e}")
            return None, None


    def _concatenate_slices(self, slice_specs: np.ndarray, slice_onsets: np.ndarray,
                            slice_offsets: np.ndarray, slice_length_ms: float,
                            start_time_ms: float, end_time_ms: float,
                            params: SpectrogramParams) -> np.ndarray:
        """Concatenate slices back into full spectrogram."""

        # Calculate expected dimensions
        total_duration_ms = end_time_ms - start_time_ms
        expected_time_bins = int(total_duration_ms / slice_length_ms)
        freq_bins = slice_specs.shape[1] if len(slice_specs) > 0 else params.target_shape[0]

        # Initialize reconstruction matrix
        reconstructed = np.zeros((freq_bins, expected_time_bins))

        # Place each slice in correct position
        for i, (spec, onset, offset) in enumerate(zip(slice_specs, slice_onsets, slice_offsets)):
            # Calculate slice position in reconstruction
            relative_onset = onset - start_time_ms
            slice_start_bin = int(relative_onset / slice_length_ms)
            slice_end_bin = slice_start_bin + spec.shape[1]

            # Ensure we don't exceed bounds
            slice_end_bin = min(slice_end_bin, expected_time_bins)
            spec_width = slice_end_bin - slice_start_bin

            if slice_start_bin >= 0 and spec_width > 0:
                reconstructed[:, slice_start_bin:slice_end_bin] = spec[:, :spec_width]

        return reconstructed


    def explore_slice_overlap_patterns(self, bird: str):
        """Analyze and visualize slice overlap patterns and label distributions."""

        logger.info(f"🔍 Exploring slice patterns for {bird}")

        bird_dir = self.project_directory / bird
        slices_dir = bird_dir / 'data' / 'slices'

        slice_files = list(slices_dir.glob('*.h5'))  # All files in slices dir are slice files
        if not slice_files:
            logger.warning(f"No slice files found for {bird}")
            return

        pdf_path = bird_dir / 'pdfs' / f'{bird}_slice_analysis.pdf'

        with pdf_backend.PdfPages(pdf_path) as pdf:
            # Analyze each file
            for slice_file in slice_files[:3]:  # Analyze first 3 files
                try:
                    analysis_data = self._analyze_slice_file(slice_file)

                    if analysis_data:
                        # Create multi-panel analysis plot
                        fig = plt.figure(figsize=(16, 12))

                        # Panel 1: Slice timeline with labels
                        ax1 = plt.subplot(4, 2, (1, 2))
                        self._plot_slice_timeline(ax1, analysis_data)

                        # Panel 2: Label distribution
                        ax2 = plt.subplot(4, 2, 3)
                        self._plot_label_distribution(ax2, analysis_data)

                        # Panel 3: Slice duration distribution
                        ax3 = plt.subplot(4, 2, 4)
                        self._plot_duration_distribution(ax3, analysis_data)

                        # Panel 4: Gap analysis
                        ax4 = plt.subplot(4, 2, 5)
                        self._plot_gap_analysis(ax4, analysis_data)

                        # Panel 5: Overlap matrix
                        ax5 = plt.subplot(4, 2, 6)
                        self._plot_overlap_matrix(ax5, analysis_data)

                        # Panel 6: Sample spectrograms
                        ax6 = plt.subplot(4, 2, (7, 8))
                        self._plot_sample_slices(ax6, analysis_data)

                        fig.suptitle(f'Slice Analysis: {slice_file.name}', fontsize=16)
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                except Exception as e:
                    logger.error(f"Error analyzing {slice_file}: {e}")

        logger.info(f"✅ Created slice analysis PDF: {pdf_path}")


    def _analyze_slice_file(self, slice_file: Path) -> Optional[Dict[str, Any]]:
        """Extract analysis data from slice file."""
        try:
            with tables.open_file(str(slice_file), 'r') as f:
                slice_specs = f.root.spectrograms.read()
                slice_onsets = f.root.onsets.read()
                slice_offsets = f.root.offsets.read()
                slice_labels = f.root.manual.read()

                # Convert labels if needed
                if len(slice_labels) > 0 and isinstance(slice_labels[0], bytes):
                    slice_labels = [label.decode('utf-8') for label in slice_labels]

                # Calculate metrics
                durations = slice_offsets - slice_onsets
                gaps = slice_onsets[1:] - slice_offsets[:-1]

                return {
                    'specs': slice_specs,
                    'onsets': slice_onsets,
                    'offsets': slice_offsets,
                    'labels': slice_labels,
                    'durations': durations,
                    'gaps': gaps,
                    'filename': slice_file.name
                }

        except Exception as e:
            logger.error(f"Error analyzing slice file {slice_file}: {e}")
            return None


    def _plot_slice_timeline(self, ax, analysis_data: Dict[str, Any]):
        """Plot slice timeline with color-coded labels."""
        onsets = analysis_data['onsets']
        offsets = analysis_data['offsets']
        labels = analysis_data['labels']

        # Create color map for labels
        unique_labels = list(set(labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        label_colors = dict(zip(unique_labels, colors))

        # Plot each slice as a horizontal bar
        for i, (onset, offset, label) in enumerate(zip(onsets, offsets, labels)):
            color = label_colors.get(label, 'gray')
            ax.barh(i, offset - onset, left=onset, height=0.8,
                    color=color, alpha=0.7, edgecolor='black')

            # Add label text
            ax.text(onset + (offset - onset) / 2, i, str(label),
                    ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Slice Index')
        ax.set_title('Slice Timeline with Labels')
        ax.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=label)
                           for label, color in label_colors.items()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')


    def _plot_label_distribution(self, ax, analysis_data: Dict[str, Any]):
        """Plot distribution of slice labels."""
        labels = analysis_data['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)

        bars = ax.bar(range(len(unique_labels)), counts, alpha=0.7)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(unique_labels, rotation=45)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    str(count), ha='center', va='bottom', fontsize=8)


    def _plot_duration_distribution(self, ax, analysis_data: Dict[str, Any]):
        """Plot distribution of slice durations."""
        durations = analysis_data['durations']

        ax.hist(durations, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Slice Duration Distribution')
        ax.axvline(np.mean(durations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(durations):.1f} ms')
        ax.legend()
        ax.grid(True, alpha=0.3)


    def _plot_gap_analysis(self, ax, analysis_data: Dict[str, Any]):
        """Plot gaps between slices."""
        gaps = analysis_data['gaps']

        if len(gaps) > 0:
            ax.hist(gaps, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Gap Duration (ms)')
            ax.set_ylabel('Count')
            ax.set_title('Inter-slice Gap Distribution')

            # Add statistics
            mean_gap = np.mean(gaps)
            ax.axvline(mean_gap, color='red', linestyle='--',
                       label=f'Mean: {mean_gap:.1f} ms')

            # Highlight negative gaps (overlaps)
            negative_gaps = gaps[gaps < 0]
            if len(negative_gaps) > 0:
                ax.axvline(0, color='orange', linestyle='-',
                           label=f'Overlaps: {len(negative_gaps)}')

            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No gap data available',
                    ha='center', va='center', transform=ax.transAxes)


    def _plot_overlap_matrix(self, ax, analysis_data: Dict[str, Any]):
        """Plot overlap matrix between consecutive slices."""
        onsets = analysis_data['onsets']
        offsets = analysis_data['offsets']

        # Calculate overlap matrix
        n_slices = len(onsets)
        overlap_matrix = np.zeros((n_slices, n_slices))

        for i in range(n_slices):
            for j in range(n_slices):
                if i != j:
                    # Calculate overlap between slice i and slice j
                    overlap_start = max(onsets[i], onsets[j])
                    overlap_end = min(offsets[i], offsets[j])
                    overlap = max(0, overlap_end - overlap_start)
                    overlap_matrix[i, j] = overlap

        # Plot heatmap
        im = ax.imshow(overlap_matrix, cmap='viridis', aspect='auto')
        ax.set_xlabel('Slice Index')
        ax.set_ylabel('Slice Index')
        ax.set_title('Slice Overlap Matrix (ms)')

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.6)


    def _plot_sample_slices(self, ax, analysis_data: Dict[str, Any]):
        """Plot sample slice spectrograms."""
        specs = analysis_data['specs']
        labels = analysis_data['labels']

        if len(specs) == 0:
            ax.text(0.5, 0.5, 'No spectrograms available',
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Select up to 5 representative slices
        n_samples = min(5, len(specs))
        sample_indices = np.linspace(0, len(specs) - 1, n_samples, dtype=int)

        # Create subplot grid
        for i, idx in enumerate(sample_indices):
            spec = specs[idx]
            label = labels[idx]

            # Calculate subplot position
            subplot_width = 1.0 / n_samples
            left = i * subplot_width

            # Create mini-axes within the main axes
            mini_ax = ax.inset_axes([left, 0, subplot_width * 0.9, 1])
            mini_ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            mini_ax.set_title(f'Slice {idx}\nLabel: {label}', fontsize=8)
            mini_ax.set_xticks([])
            mini_ax.set_yticks([])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Sample Slice Spectrograms', pad=20)


    def compare_syllable_vs_slice_approaches(self, bird: str):
        """Compare syllable-based vs slice-based spectrograms side by side."""

        logger.info(f"🔍 Comparing syllable vs slice approaches for {bird}")

        bird_dir = self.project_directory / bird
        syllables_dir = bird_dir / 'data' / 'syllables'
        slices_dir = bird_dir / 'data' / 'slices'

        syllable_files = list(syllables_dir.glob('*.h5')) if syllables_dir.exists() else []
        slice_files = list(slices_dir.glob('*.h5')) if slices_dir.exists() else []

        if not syllable_files or not slice_files:
            logger.warning(f"Missing syllable or slice files for {bird}")
            return

        pdf_path = bird_dir / 'pdfs' / f'{bird}_syllable_vs_slice_comparison.pdf'

        with pdf_backend.PdfPages(pdf_path) as pdf:
            # Compare first few files that might be from same audio
            for syl_file in syllable_files[:3]:
                try:
                    # Find corresponding slice file (match by base filename)
                    syl_base = self._extract_base_filename(syl_file.name)
                    matching_slice = None

                    for slice_file in slice_files:
                        slice_base = self._extract_base_filename(slice_file.name)
                        if syl_base == slice_base:
                            matching_slice = slice_file
                            break

                    if matching_slice is None:
                        continue

                    # Load both datasets
                    syl_data = self._load_syllable_data(syl_file)
                    slice_data = self._load_slice_data(matching_slice)

                    if syl_data and slice_data:
                        # Create comparison plot
                        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

                        # Row 1: Raw spectrograms comparison
                        self._plot_spectrogram_comparison(axes[0], syl_data, slice_data, 'Raw Spectrograms')

                        # Row 2: Label distribution comparison
                        self._plot_label_comparison(axes[1], syl_data, slice_data)

                        # Row 3: Temporal structure comparison
                        self._plot_temporal_comparison(axes[2], syl_data, slice_data)

                        fig.suptitle(f'Syllable vs Slice Comparison: {syl_base}', fontsize=16)
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                except Exception as e:
                    logger.error(f"Error comparing {syl_file}: {e}")

        logger.info(f"✅ Created comparison PDF: {pdf_path}")


    def _extract_base_filename(self, filename: str) -> str:
        """Extract base filename for matching syllable and slice files."""
        # Remove common prefixes/suffixes
        base = filename.replace('syllables_', '').replace('_slice', '').replace('.h5', '')
        return base


    def _load_syllable_data(self, syl_file: Path) -> Optional[Dict[str, Any]]:
        """Load syllable-based data."""
        try:
            with tables.open_file(str(syl_file), 'r') as f:
                return {
                    'specs': f.root.spectrograms.read(),
                    'onsets': f.root.onsets.read(),
                    'offsets': f.root.offsets.read(),
                    'labels': [label.decode('utf-8') if isinstance(label, bytes) else str(label)
                               for label in f.root.manual.read()],
                    'type': 'syllable'
                }
        except Exception as e:
            logger.error(f"Error loading syllable data: {e}")
            return None


    def _load_slice_data(self, slice_file: Path) -> Optional[Dict[str, Any]]:
        """Load slice-based data."""
        try:
            with tables.open_file(str(slice_file), 'r') as f:
                return {
                    'specs': f.root.spectrograms.read(),
                    'onsets': f.root.onsets.read(),
                    'offsets': f.root.offsets.read(),
                    'labels': [label.decode('utf-8') if isinstance(label, bytes) else str(label)
                               for label in f.root.manual.read()],
                    'type': 'slice'
                }
        except Exception as e:
            logger.error(f"Error loading slice data: {e}")
            return None


    def _plot_spectrogram_comparison(self, axes, syl_data: Dict, slice_data: Dict, title: str):
        """Plot side-by-side spectrogram comparison."""
        # Syllable approach (left)
        if len(syl_data['specs']) > 0:
            # Concatenate first few syllable spectrograms
            sample_specs = syl_data['specs'][:5]  # First 5 syllables
            concat_syl = np.hstack(sample_specs) if len(sample_specs) > 1 else sample_specs[0]

            axes[0].imshow(concat_syl, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title(f'Syllable-based ({len(syl_data["specs"])} syllables)')
            axes[0].set_ylabel('Frequency')

        # Slice approach (right)
        if len(slice_data['specs']) > 0:
            # Concatenate first few slice spectrograms
            sample_specs = slice_data['specs'][:10]  # First 10 slices
            concat_slice = np.hstack(sample_specs) if len(sample_specs) > 1 else sample_specs[0]

            axes[1].imshow(concat_slice, aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title(f'Slice-based ({len(slice_data["specs"])} slices)')

        # Add overall title
        axes[0].text(0.5, 1.15, title, transform=axes[0].transAxes,
                     ha='center', fontsize=12, fontweight='bold')


    def _plot_label_comparison(self, axes, syl_data: Dict, slice_data: Dict):
        """Compare label distributions between approaches."""
        # Syllable labels
        syl_labels, syl_counts = np.unique(syl_data['labels'], return_counts=True)
        axes[0].bar(range(len(syl_labels)), syl_counts, alpha=0.7)
        axes[0].set_title('Syllable Label Distribution')
        axes[0].set_xticks(range(len(syl_labels)))
        axes[0].set_xticklabels(syl_labels, rotation=45)
        axes[0].set_ylabel('Count')

        # Slice labels
        slice_labels, slice_counts = np.unique(slice_data['labels'], return_counts=True)
        axes[1].bar(range(len(slice_labels)), slice_counts, alpha=0.7, color='orange')
        axes[1].set_title('Slice Label Distribution')
        axes[1].set_xticks(range(len(slice_labels)))
        axes[1].set_xticklabels(slice_labels, rotation=45)


    def _plot_temporal_comparison(self, axes, syl_data: Dict, slice_data: Dict):
        """Compare temporal structure between approaches."""
        # Syllable durations
        # Syllable durations
        syl_durations = syl_data['offsets'] - syl_data['onsets']
        axes[0].hist(syl_durations, bins=15, alpha=0.7, edgecolor='black')
        axes[0].set_title('Syllable Duration Distribution')
        axes[0].set_xlabel('Duration (ms)')
        axes[0].set_ylabel('Count')
        axes[0].axvline(np.mean(syl_durations), color='red', linestyle='--',
                        label=f'Mean: {np.mean(syl_durations):.1f} ms')
        axes[0].legend()

        # Slice durations
        slice_durations = slice_data['offsets'] - slice_data['onsets']
        axes[1].hist(slice_durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title('Slice Duration Distribution')
        axes[1].set_xlabel('Duration (ms)')
        axes[1].axvline(np.mean(slice_durations), color='red', linestyle='--',
                        label=f'Mean: {np.mean(slice_durations):.1f} ms')
        axes[1].legend()


def load_spectrogram_from_hash(target_hash: str, bird_path: str) -> np.ndarray:
    """
    Load original spectrogram for a specific hash from syllable HDF5 files.

    Args:
        target_hash: Hash identifier to find
        bird_path: Path to bird directory (containing data/syllables)

    Returns:
        numpy array of spectrogram or None if not found
    """
    try:
        syllables_dir = os.path.join(bird_path, 'data', 'syllables')

        if not os.path.exists(syllables_dir):
            logging.warning(f"Syllables directory not found: {syllables_dir}")
            return None

        # Get all syllable HDF5 files
        syllable_files = [f for f in os.listdir(syllables_dir)
                          if f.endswith('.h5') and f.startswith('syllables_')]

        for filename in syllable_files:
            file_path = os.path.join(syllables_dir, filename)

            try:
                with tables.open_file(file_path, mode='r') as f:
                    # Read hashes from this file
                    hashes_raw = f.root.hashes.read()
                    hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h)
                              for h in hashes_raw]

                    # Check if our target hash is in this file
                    if target_hash in hashes:
                        # Find the index
                        hash_idx = hashes.index(target_hash)

                        # Load the original spectrogram
                        spectrogram = f.root.spectrograms[hash_idx]

                        return spectrogram

            except Exception as e:
                logging.debug(f"Error reading {filename}: {e}")
                continue

        logging.warning(f"Hash {target_hash} not found in any syllable file")
        return None

    except Exception as e:
        logging.error(f"Error loading spectrogram for hash {target_hash}: {e}")
        return None


def interactive_plot_umap_v2(embeddings: np.ndarray, hashes: list,
                             ground_truth_labels: list = None,
                             cluster_labels: np.ndarray = None,
                             bird: str = '', bird_path: str = None):
    """
    Interactive UMAP plot adapted for new data structure.

    Args:
        embeddings: UMAP embeddings array (N x 2)
        hashes: List of sample hash identifiers
        ground_truth_labels: Ground truth labels from original data (optional)
        cluster_labels: Cluster assignments from clustering (optional)
        bird: Bird identifier for plot titles
        bird_path: Path to bird directory for loading spectrograms
    """
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        plt.ion()

        # Determine what labels to use for coloring
        plot_labels = None
        label_type = "None"

        if cluster_labels is not None:
            plot_labels = cluster_labels
            label_type = "Cluster"
        elif ground_truth_labels is not None:
            # Convert string labels to integers for coloring
            unique_labels, plot_labels = np.unique(ground_truth_labels, return_inverse=True)
            label_type = "Ground Truth"

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))

        if plot_labels is not None:
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                                 s=20, c=plot_labels, cmap='tab10', alpha=0.7)
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label=f'{label_type} Labels')
        else:
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                                 s=20, alpha=0.7, color='blue')

        ax.set_title(f'Interactive UMAP - {bird}\nClick points to view spectrograms')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        def on_click(event):
            try:
                if event.xdata is None or event.ydata is None:
                    return

                x_click, y_click = event.xdata, event.ydata

                # Find closest point
                distances = np.sqrt((embeddings[:, 0] - x_click) ** 2 +
                                    (embeddings[:, 1] - y_click) ** 2)
                closest_idx = np.argmin(distances)

                # Get information about the clicked point
                clicked_hash = hashes[closest_idx]
                clicked_gt_label = ground_truth_labels[closest_idx] if ground_truth_labels else "N/A"
                clicked_cluster = cluster_labels[closest_idx] if cluster_labels is not None else "N/A"

                print(f"Clicked point {closest_idx}: Hash={clicked_hash[:12]}..., "
                      f"GT_Label={clicked_gt_label}, Cluster={clicked_cluster}")

                # Load and display spectrogram
                if bird_path:
                    spec = load_spectrogram_from_hash(clicked_hash, bird_path)

                    if spec is not None:
                        # Create new figure for spectrogram
                        spec_fig, spec_ax = plt.subplots(figsize=(12, 8))

                        # Display spectrogram
                        im = spec_ax.imshow(spec, aspect='auto', origin='lower',
                                            cmap='viridis', interpolation='nearest')

                        # Add colorbar
                        plt.colorbar(im, ax=spec_ax, label='Amplitude')

                        # Set title with all available information
                        title_parts = [f'{bird} - Point {closest_idx}']
                        title_parts.append(f'Hash: {clicked_hash[:12]}...')
                        if clicked_gt_label != "N/A":
                            title_parts.append(f'GT Label: {clicked_gt_label}')
                        if clicked_cluster != "N/A":
                            title_parts.append(f'Cluster: {clicked_cluster}')

                        spec_ax.set_title(' | '.join(title_parts))
                        spec_ax.set_xlabel('Time Bins')
                        spec_ax.set_ylabel('Frequency Bins')

                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"Could not load spectrogram for hash: {clicked_hash}")
                else:
                    print("No bird_path provided - cannot load spectrogram")

            except Exception as e:
                print(f"Error in click handler: {e}")
                logging.error(f"Error in click handler: {e}")

        # Connect click event
        fig.canvas.mpl_connect('button_press_event', on_click)

        # Add instructions
        ax.text(0.02, 0.98, 'Click on points to view spectrograms',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')

        plt.show()

    except Exception as e:
        logging.error(f"Error in interactive plot: {e}")


def load_data_for_interactive_plot(embeddings_path: str, cluster_labels_path: str = None,
                                   bird_path: str = None):
    """
    Convenience function to load all data needed for interactive plotting.

    Args:
        embeddings_path: Path to UMAP embeddings HDF5 file
        cluster_labels_path: Path to cluster labels HDF5 file (optional)
        bird_path: Path to bird directory (optional, for spectrogram loading)

    Returns:
        dict: Dictionary with loaded data ready for interactive plotting
    """
    try:
        # Load embeddings and ground truth labels
        embeddings, hashes, ground_truth_labels = load_umap_embeddings(embeddings_path)
        if embeddings is None:
            logging.error(f"Could not load embeddings from {embeddings_path}")
            return None

        data = {
            'embeddings': embeddings,
            'hashes': hashes,
            'ground_truth_labels': ground_truth_labels,
            'cluster_labels': None,
            'bird_path': bird_path
        }

        # Load cluster labels if provided
        if cluster_labels_path:
            cluster_labels, cluster_hashes, scores = load_labels(cluster_labels_path)
            if cluster_labels is not None:
                # Verify hash alignment
                if len(hashes) == len(cluster_hashes) and all(h1 == h2 for h1, h2 in zip(hashes, cluster_hashes)):
                    data['cluster_labels'] = cluster_labels
                else:
                    logging.warning("Hash mismatch between embeddings and cluster labels")

        return data

    except Exception as e:
        logging.error(f"Error loading data for interactive plot: {e}")
        return None


def create_label_spectrograms_pdf(embeddings_path: str, cluster_labels_path: str,
                                  bird_path: str, save_path: str, bird_str: str,
                                  max_samples_per_label: int = 1000):
    """
    Create PDFs with spectrogram examples for each cluster label.

    Args:
        embeddings_path: Path to UMAP embeddings HDF5 file
        cluster_labels_path: Path to cluster labels HDF5 file
        bird_path: Path to bird directory (containing data/syllables)
        save_path: Directory to save PDF files
        bird_str: Bird identifier for filenames
        max_samples_per_label: Maximum number of spectrograms per label
    """
    try:
        # Load embeddings and hashes
        embeddings, hashes, ground_truth_labels = load_umap_embeddings(embeddings_path)
        if embeddings is None:
            logging.error(f"Could not load embeddings from {embeddings_path}")
            return False

        # Load cluster labels
        cluster_labels, cluster_hashes, scores = load_labels(cluster_labels_path)
        if cluster_labels is None:
            logging.error(f"Could not load cluster labels from {cluster_labels_path}")
            return False

        # Ensure hashes match between embeddings and cluster labels
        if len(hashes) != len(cluster_hashes) or not all(h1 == h2 for h1, h2 in zip(hashes, cluster_hashes)):
            logging.warning("Hash mismatch between embeddings and cluster labels")
            # You might want to implement hash alignment here

        # Get unique cluster labels (exclude noise if using HDBSCAN)
        unique_clusters = np.unique(cluster_labels)
        non_noise_clusters = unique_clusters[unique_clusters != -1]  # Remove noise cluster

        logging.info(f"Creating PDFs for {len(non_noise_clusters)} clusters")

        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Process each cluster
        for cluster_id in tqdm(non_noise_clusters, desc="Creating cluster PDFs"):
            try:
                # Get indices for this cluster
                cluster_indices = np.where(cluster_labels == cluster_id)[0]

                if len(cluster_indices) == 0:
                    continue

                # Sample if too many spectrograms
                if len(cluster_indices) > max_samples_per_label:
                    sampled_indices = np.random.choice(cluster_indices,
                                                       size=max_samples_per_label,
                                                       replace=False)
                else:
                    sampled_indices = cluster_indices

                # Get hashes for this cluster
                cluster_hashes_subset = [hashes[i] for i in sampled_indices]

                # Load spectrograms for this cluster
                spectrograms = []
                valid_hashes = []
                valid_gt_labels = []

                for i, hash_id in enumerate(cluster_hashes_subset):
                    spec = load_spectrogram_from_hash(hash_id, bird_path)
                    if spec is not None:
                        spectrograms.append(spec)
                        valid_hashes.append(hash_id)
                        # Add ground truth label if available
                        if ground_truth_labels:
                            original_idx = sampled_indices[i]
                            valid_gt_labels.append(ground_truth_labels[original_idx])

                if not spectrograms:
                    logging.warning(f"No spectrograms found for cluster {cluster_id}")
                    continue

                # Create PDF for this cluster
                pdf_filename = f'{bird_str}_cluster_{cluster_id}.pdf'
                pdf_path = os.path.join(save_path, pdf_filename)

                create_spectrogram_pdf(
                    spectrograms=spectrograms,
                    output_pdf_path=pdf_path,
                    title=f'{bird_str} - Cluster {cluster_id}',
                    hashes=valid_hashes,
                    ground_truth_labels=valid_gt_labels if valid_gt_labels else None
                )

                logging.info(f"Created PDF for cluster {cluster_id}: {len(spectrograms)} spectrograms")

            except Exception as e:
                logging.error(f"Error processing cluster {cluster_id}: {e}")
                continue

        return True

    except Exception as e:
        logging.error(f"Error creating label spectrogram PDFs: {e}")
        return False


def create_spectrogram_pdf(spectrograms: list, output_pdf_path: str,
                           title: str = "", hashes: list = None,
                           ground_truth_labels: list = None,
                           images_per_page: int = 25):
    """
    Create a multi-page PDF with spectrogram grids.

    Args:
        spectrograms: List of 2D numpy arrays (spectrograms)
        output_pdf_path: Path for output PDF file
        title: Title for the PDF pages
        hashes: Optional list of hash IDs for labeling
        ground_truth_labels: Optional list of ground truth labels
        images_per_page: Number of spectrograms per page
    """
    try:
        if not spectrograms:
            logging.warning("No spectrograms provided for PDF creation")
            return
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(images_per_page)))

        # Remove existing file if it exists
        if os.path.exists(output_pdf_path):
            try:
                os.remove(output_pdf_path)
            except PermissionError:
                logging.warning(f"Could not remove existing file: {output_pdf_path}")

        with PdfPages(output_pdf_path) as pdf:
            for page_start in range(0, len(spectrograms), images_per_page):
                page_specs = spectrograms[page_start:page_start + images_per_page]
                page_hashes = hashes[page_start:page_start + images_per_page] if hashes else None
                page_gt_labels = ground_truth_labels[
                    page_start:page_start + images_per_page] if ground_truth_labels else None

                fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

                # Handle single subplot case
                if grid_size == 1:
                    axes = [[axes]]
                elif len(axes.shape) == 1:
                    axes = axes.reshape(1, -1)

                # Plot spectrograms
                for i, spec in enumerate(page_specs):
                    row = i // grid_size
                    col = i % grid_size
                    ax = axes[row][col]

                    # Plot spectrogram
                    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                    ax.axis('off')

                    # Create title with available information
                    title_parts = []
                    if page_hashes and i < len(page_hashes):
                        title_parts.append(f'{page_hashes[i][:8]}...')
                    if page_gt_labels and i < len(page_gt_labels):
                        title_parts.append(f'GT:{page_gt_labels[i]}')

                    if title_parts:
                        ax.set_title(' | '.join(title_parts), fontsize=8)

                # Hide unused subplots
                for i in range(len(page_specs), grid_size * grid_size):
                    row = i // grid_size
                    col = i % grid_size
                    axes[row][col].axis('off')

                # Add main title
                if title:
                    page_num = (page_start // images_per_page) + 1
                    total_pages = int(np.ceil(len(spectrograms) / images_per_page))
                    fig.suptitle(f'{title} - Page {page_num}/{total_pages}',
                                 fontsize=16, y=0.98)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close(fig)

    except Exception as e:
        logging.error(f"Error creating spectrogram PDF: {e}")


def create_all_cluster_pdfs_for_bird(bird_path: str, bird_name: str,
                                     embedding_filename: str = None,
                                     cluster_filename: str = None):
    """
    Convenience function to create PDFs for all clusters of a bird.

    Args:
        bird_path: Path to bird directory
        bird_name: Bird identifier
        embedding_filename: Specific embedding file to use (best one if None)
        cluster_filename: Specific cluster file to use (best one if None)
    """
    try:
        data_path = os.path.join(bird_path, 'data')

        # Find embedding file
        if embedding_filename is None:
            embeddings_dir = os.path.join(data_path, 'embeddings')
            embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
            if not embedding_files:
                logging.error(f"No embedding files found for {bird_name}")
                return False
            embedding_filename = embedding_files[0]  # Use first one or implement selection logic

        embeddings_path = os.path.join(data_path, 'embeddings', embedding_filename)

        # Find cluster file
        if cluster_filename is None:
            # Look for best cluster file in master summary
            master_summary_path = os.path.join(bird_path, 'master_summary.csv')
            if os.path.exists(master_summary_path):
                summary_df = pd.read_csv(master_summary_path)
                if not summary_df.empty:
                    # Get the best performing cluster file
                    best_row = summary_df.iloc[0]  # Assuming sorted by performance
                    cluster_filename = os.path.basename(str(best_row['label_path']))
                else:
                    logging.error(f"Empty master summary for {bird_name}")
                    return False
            else:
                logging.error(f"No master summary found for {bird_name}")
                return False

        # Find the cluster file path
        labelling_dir = os.path.join(data_path, 'labelling')
        cluster_labels_path = None

        # Search through labelling subdirectories
        for root, dirs, files in os.walk(labelling_dir):
            if cluster_filename in files:
                cluster_labels_path = os.path.join(root, cluster_filename)
                break

        if cluster_labels_path is None:
            logging.error(f"Could not find cluster file {cluster_filename} for {bird_name}")
            return False

        # Create output directory
        pdf_save_path = os.path.join(bird_path, 'figures', 'cluster_spectrograms')

        # Create the PDFs
        success = create_label_spectrograms_pdf(
            embeddings_path=embeddings_path,
            cluster_labels_path=cluster_labels_path,
            bird_path=bird_path,
            save_path=pdf_save_path,
            bird_str=bird_name
        )

        if success:
            logging.info(f"Successfully created cluster PDFs for {bird_name}")

        return success

    except Exception as e:
        logging.error(f"Error creating cluster PDFs for {bird_name}: {e}")
        return False


def run_interactive_plot_from_results(bird_path: str, bird_name: str,
                                      embedding_filename: str = None,
                                      cluster_filename: str = None):
    """
    Convenience function to launch interactive plot from clustering results.

    Args:
        bird_path: Path to bird directory
        bird_name: Bird identifier
        embedding_filename: Specific embedding file (best one if None)
        cluster_filename: Specific cluster file (best one if None)
    """
    try:
        data_path = os.path.join(bird_path, 'data')

        # Find embedding file (same logic as above)
        if embedding_filename is None:
            embeddings_dir = os.path.join(data_path, 'embeddings')
            embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.h5')]
            if not embedding_files:
                logging.error(f"No embedding files found for {bird_name}")
                return False
            embedding_filename = embedding_files[0]

        embeddings_path = os.path.join(data_path, 'embeddings', embedding_filename)

        # Load data for plotting
        cluster_labels_path = None
        if cluster_filename is not None:
            # Find cluster file path (same logic as above)
            # Find cluster file path (same logic as above)
            labelling_dir = os.path.join(data_path, 'labelling')
            for root, dirs, files in os.walk(labelling_dir):
                if cluster_filename in files:
                    cluster_labels_path = os.path.join(root, cluster_filename)
                    break

            # Load data for interactive plotting
        plot_data = load_data_for_interactive_plot(
            embeddings_path=embeddings_path,
            cluster_labels_path=cluster_labels_path,
            bird_path=bird_path
        )

        if plot_data is None:
            logging.error(f"Could not load data for interactive plot for {bird_name}")
            return False

        # Launch interactive plot
        interactive_plot_umap_v2(
            embeddings=plot_data['embeddings'],
            hashes=plot_data['hashes'],
            ground_truth_labels=plot_data['ground_truth_labels'],
            cluster_labels=plot_data['cluster_labels'],
            bird=bird_name,
            bird_path=bird_path
        )

        return True

    except Exception as e:
        logging.error(f"Error running interactive plot for {bird_name}: {e}")
        return False


def run_interactive_plot_from_best_results(bird_path: str, bird_name: str):
    """
    Launch interactive plot using the best clustering results for a bird.

    Args:
        bird_path: Path to bird directory
        bird_name: Bird identifier
    """
    try:
        # Load master summary to find best results
        master_summary_path = os.path.join(bird_path, 'master_summary.csv')
        if not os.path.exists(master_summary_path):
            logging.error(f"No master summary found for {bird_name}")
            return False

        summary_df = pd.read_csv(master_summary_path)
        if summary_df.empty:
            logging.error(f"Empty master summary for {bird_name}")
            return False

        # Get the best result (first row, assuming sorted by performance)
        best_row = summary_df.iloc[0]

        # Extract embedding parameters to find embedding file
        n_neighbors = best_row['n_neighbors']
        min_dist = best_row['min_dist']
        metric = best_row['metric']
        embedding_filename = f'{metric}_{int(n_neighbors)}neighbors_{min_dist}dist.h5'

        # Get cluster filename
        cluster_filename = os.path.basename(str(best_row['label_path']))

        logging.info(f"Using best results for {bird_name}:")
        logging.info(f"  Embedding: {embedding_filename}")
        logging.info(f"  Clusters: {cluster_filename}")
        logging.info(f"  Composite Score: {best_row.get('composite_score', 'N/A')}")

        return run_interactive_plot_from_results(
            bird_path=bird_path,
            bird_name=bird_name,
            embedding_filename=embedding_filename,
            cluster_filename=cluster_filename
        )

    except Exception as e:
        logging.error(f"Error running interactive plot from best results for {bird_name}: {e}")
        return False


def example_usage_interactive_plot():
"""Example of how to use the interactive plotting functions."""

# Example 1: Plot from specific files
bird_path = "/path/to/bird/directory"
bird_name = "bu85bu97"

# Use specific files
success = run_interactive_plot_from_results(
    bird_path=bird_path,
    bird_name=bird_name,
    embedding_filename="euclidean_20neighbors_0.1dist.h5",
    cluster_filename="hdbscan_min_cluster_size20_min_samples5_labels.h5"
)

# Example 2: Use best results automatically
success = run_interactive_plot_from_best_results(
    bird_path=bird_path,
    bird_name=bird_name
)


def example_usage_create_pdfs():
    """Example of how to create cluster PDFs."""

    bird_path = "/path/to/bird/directory"
    bird_name = "bu85bu97"

    # Create PDFs for all clusters using best results
    success = create_all_cluster_pdfs_for_bird(
        bird_path=bird_path,
        bird_name=bird_name
    )

    if success:
        print(f"PDFs created successfully for {bird_name}")
    else:
        print(f"Failed to create PDFs for {bird_name}")


def main():
    """Updated main function using consolidated functionality."""
    # Use the centralized logger
    logger.info("🚀 Starting enhanced spectrogram visualization pipeline")

    # Test directories
    test_directories = [
        '/Volumes/Extreme SSD/evsong test',
        '/Volumes/Extreme SSD/wseg test'
    ]

    for test_dir in test_directories:
        if not os.path.exists(test_dir):
            logger.warning(f"Test directory not found: {test_dir}")
            continue

        logger.info(f"🔍 Processing {os.path.basename(test_dir)}")

        # Standard visualization
        visualizer = SpectrogramVisualizer(test_dir)

        # Create song spectrograms (using consolidated functions)
        visualizer.create_song_spectrograms_pdf(prefer_local=True)

        # Create syllable samples (using shared dual-labeled function)
        visualizer.create_syllable_sample_pdfs(syllables_per_bird=20)

        # Enhanced slice validation
        slice_validator = SliceValidator(test_dir)

        # Validate slice reconstruction for each bird
        for bird in slice_validator.birds[:2]:  # Test first 2 birds
            logger.info(f"🔍 Validating slices for {bird}")

            # Reconstruct original from slices
            slice_validator.validate_slice_reconstruction(bird)

            # Explore slice patterns
            slice_validator.explore_slice_overlap_patterns(bird)

            # Compare syllable vs slice approaches
            slice_validator.compare_syllable_vs_slice_approaches(bird)

    logger.info("✅ Enhanced visualization pipeline complete!")


if __name__ == '__main__':
    # Setup logging using consolidated function
    logger = setup_logging()
    main()