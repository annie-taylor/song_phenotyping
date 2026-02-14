# song_visualization.py

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Import your existing consolidated functions
from tools.song_io import (
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


@dataclass
class SongVisualizationConfig:
    """Configuration for song visualization parameters."""
    target_duration: float = 8.0
    sample_rate: int = 32000
    prefer_local: bool = True

    # PDF settings
    figure_size: Tuple[float, float] = (12, 6)
    dpi: int = 150

    # Spectrogram appearance
    colormap: str = 'viridis'
    show_metadata: bool = True
    metadata_fontsize: int = 8

    # Output options
    save_individual_pngs: bool = True
    create_pdf: bool = True

    # Processing options
    max_files_per_bird: Optional[int] = None
    skip_existing: bool = True


class SongVisualizer:
    """Focused song spectrogram visualization with configurable options."""

    def __init__(self, project_directory: str, config: SongVisualizationConfig = None):
        self.project_directory = Path(project_directory)
        self.config = config or SongVisualizationConfig()
        self.birds = self._discover_birds()

    def _discover_birds(self) -> List[str]:
        """Discover birds with available audio data."""
        birds = []
        if not self.project_directory.exists():
            logger.warning(f"Project directory not found: {self.project_directory}")
            return birds

        for item in self.project_directory.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data'):

                # Check for audio path mapping (indicates audio data available)
                mapping_file = item / 'audio_paths.json'
                if mapping_file.exists():
                    birds.append(item.name)

        logger.info(f"Found {len(birds)} birds with audio data: {birds}")
        return sorted(birds)

    def create_song_spectrograms(self, birds: List[str] = None) -> Dict[str, Dict[str, any]]:
        """
        Create song spectrograms for specified birds.

        Args:
            birds: List of bird names to process (all if None)

        Returns:
            Dict with processing results for each bird
        """
        if birds is None:
            birds = self.birds

        logger.info(f"🎵 Creating song spectrograms for {len(birds)} birds")
        logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

        results = {}

        for bird in birds:
            logger.info(f"🔄 Processing {bird}")
            bird_result = self._process_single_bird(bird)
            results[bird] = bird_result

        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(f"✅ Song visualization complete: {successful}/{len(birds)} birds successful")

        return results

    def _process_single_bird(self, bird: str) -> Dict[str, any]:
        """Process song spectrograms for a single bird."""
        bird_dir = self.project_directory / bird
        result = {
            'success': False,
            'files_processed': 0,
            'files_failed': 0,
            'pdf_path': None,
            'png_directory': None,
            'error': None
        }

        try:
            # Load audio path mapping
            mapping = load_audio_paths_mapping(str(bird_dir))
            if not mapping:
                result['error'] = "No audio paths mapping found"
                return result

            # Setup output directories
            output_paths = self._setup_output_directories(bird_dir, bird)
            result['pdf_path'] = output_paths['pdf_path']
            result['png_directory'] = output_paths['png_dir']

            # Check if PDF already exists and skip if configured
            if self.config.skip_existing and output_paths['pdf_path'].exists():
                logger.info(f"⏭️ Skipping {bird} - PDF already exists")
                result['success'] = True
                return result

            # Select files to process
            files_to_process = self._select_files_to_process(mapping)

            # Process files
            if self.config.create_pdf:
                result = self._create_pdf_with_spectrograms(
                    bird, bird_dir, files_to_process, output_paths, result
                )
            else:
                result = self._create_individual_spectrograms(
                    bird, bird_dir, files_to_process, output_paths, result
                )

        except Exception as e:
            logger.error(f"💥 Error processing {bird}: {e}")
            result['error'] = str(e)

        return result

    def _setup_output_directories(self, bird_dir: Path, bird: str) -> Dict[str, Path]:
        """Setup output directories for a bird."""
        pdf_dir = bird_dir / 'pdfs'
        png_dir = bird_dir / 'spectrograms' / 'songs'

        pdf_dir.mkdir(exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = pdf_dir / f'{bird}_song_spectrograms.pdf'

        return {
            'pdf_dir': pdf_dir,
            'png_dir': png_dir,
            'pdf_path': pdf_path
        }

    def _select_files_to_process(self, mapping: Dict) -> List[str]:
        """Select which files to process based on configuration."""
        filenames = list(mapping.keys())

        if self.config.max_files_per_bird is not None:
            # Sort for consistent selection
            filenames.sort()
            filenames = filenames[:self.config.max_files_per_bird]

        return filenames

    def _create_pdf_with_spectrograms(self, bird: str, bird_dir: Path,
                                      filenames: List[str], output_paths: Dict,
                                      result: Dict) -> Dict:
        """Create PDF with all spectrograms for a bird."""

        with pdf_backend.PdfPages(output_paths['pdf_path']) as pdf:
            for filename in tqdm(filenames, desc=f"🔄 Processing {bird}"):
                try:
                    # Process single file
                    spec_result = self._process_single_file(
                        filename, bird_dir, output_paths['png_dir']
                    )

                    if spec_result['success']:
                        # Add to PDF
                        pdf.savefig(spec_result['figure'], bbox_inches='tight')
                        plt.close(spec_result['figure'])
                        result['files_processed'] += 1
                    else:
                        result['files_failed'] += 1
                        logger.warning(f"Failed to process {filename}: {spec_result['error']}")

                except Exception as e:
                    result['files_failed'] += 1
                    logger.error(f"Error processing {filename}: {e}")

                result['success'] = result['files_processed'] > 0
                logger.info(f"✅ Created PDF for {bird}: {result['files_processed']} files processed")
                return result

    def _create_individual_spectrograms(self, bird: str, bird_dir: Path,
                                        filenames: List[str], output_paths: Dict,
                                        result: Dict) -> Dict:
        """Create individual PNG files without PDF."""

        for filename in tqdm(filenames, desc=f"🔄 Processing {bird}"):
            try:
                spec_result = self._process_single_file(
                    filename, bird_dir, output_paths['png_dir']
                )

                if spec_result['success']:
                    result['files_processed'] += 1
                else:
                    result['files_failed'] += 1

                # Always close figure
                if 'figure' in spec_result:
                    plt.close(spec_result['figure'])

            except Exception as e:
                result['files_failed'] += 1
                logger.error(f"Error processing {filename}: {e}")

        result['success'] = result['files_processed'] > 0
        return result

    def _process_single_file(self, filename: str, bird_dir: Path,
                             png_dir: Path) -> Dict[str, any]:
        """Process a single audio file to create spectrogram."""
        result = {
            'success': False,
            'figure': None,
            'png_path': None,
            'error': None
        }

        try:
            # Resolve audio file path
            file_path = get_audio_path(str(bird_dir), filename, self.config.prefer_local)

            # Load and process audio
            audio_data = self._load_and_process_audio(file_path)
            if audio_data is None:
                result['error'] = "Failed to load audio"
                return result

            # Create spectrogram
            spec_data = self._create_spectrogram(audio_data)
            if spec_data is None:
                result['error'] = "Failed to create spectrogram"
                return result

            # Create visualization
            fig = self._create_spectrogram_plot(
                spec_data, filename, file_path
            )
            result['figure'] = fig

            # Save PNG if configured
            if self.config.save_individual_pngs:
                png_filename = f"{Path(filename).stem}_song.png"
                png_path = png_dir / png_filename
                fig.savefig(png_path, dpi=self.config.dpi, bbox_inches='tight')
                result['png_path'] = png_path

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing file {filename}: {e}")

        return result

    def _load_and_process_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file."""
        try:
            # Load audio
            audio, sr = read_audio_file(file_path)

            # Normalize
            audio = rms_norm(audio)

            # Apply filtering (use same params as main pipeline)
            params = SpectrogramParams()
            audio = butter_bandpass_filter_sos(
                audio,
                lowcut=params.min_freq,
                highcut=params.max_freq,
                fs=sr,
                order=5
            )

            # Pad or trim to target duration
            target_samples = int(self.config.target_duration * sr)
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio = audio[:target_samples]

            return audio

        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            return None

    def _create_spectrogram(self, audio: np.ndarray) -> Optional[Dict]:
        """Create spectrogram from audio data."""
        try:
            params = SpectrogramParams()

            # Generate spectrogram using same method as pipeline
            spec, freqs, t = get_song_spec(
                t1=0,
                t2=self.config.target_duration,
                audio=audio,
                params=params,
                fs=self.config.sample_rate,
                downsample=False
            )

            return {
                'spectrogram': spec,
                'frequencies': freqs,
                'times': t,
                'params': params
            }

        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            return None

    def _create_spectrogram_plot(self, spec_data: Dict, filename: str,
                                 file_path: str) -> plt.Figure:
        """Create the actual spectrogram plot."""

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Plot spectrogram
        spec = spec_data['spectrogram']
        im = ax.imshow(
            spec,
            aspect='auto',
            origin='lower',
            extent=[0, self.config.target_duration, 0, spec.shape[0]],
            cmap=self.config.colormap
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude', fontsize=10)

        # Labels and title
        ax.set_ylabel('Frequency Bins', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_title(f'Song Spectrogram: {filename}', fontsize=14, fontweight='bold')

        # Add metadata if configured
        if self.config.show_metadata:
            self._add_metadata_to_plot(ax, filename, file_path, spec_data)

        plt.tight_layout()
        return fig

    def _add_metadata_to_plot(self, ax: plt.Axes, filename: str,
                              file_path: str, spec_data: Dict):
        """Add metadata annotations to the plot."""

        # File path info
        path_type = "Local" if self.config.prefer_local else "Server"
        display_text = f"{path_type}: {file_path}"

        # Add text box with metadata
        metadata_lines = [
            f"Duration: {self.config.target_duration}s",
            f"Sample Rate: {self.config.sample_rate}Hz",
            f"Shape: {spec_data['spectrogram'].shape}",
            display_text
        ]

        metadata_text = '\n'.join(metadata_lines)

        ax.text(
            0.02, 0.98, metadata_text,
            transform=ax.transAxes,
            fontsize=self.config.metadata_fontsize,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                alpha=0.8,
                edgecolor='gray'
            )
        )

class SongComparisonVisualizer(SongVisualizer):
    """Extended visualizer for comparing songs across birds or conditions."""

    def create_cross_bird_comparison(self, birds: List[str] = None,
                                     max_songs_per_bird: int = 3) -> str:
        """
        Create a comparison PDF showing songs from multiple birds side by side.

        Args:
            birds: List of birds to compare (all if None)
            max_songs_per_bird: Maximum songs to include per bird

        Returns:
            Path to created comparison PDF
        """
        if birds is None:
            birds = self.birds[:5]  # Limit to 5 birds for readability

        logger.info(f"🔄 Creating cross-bird song comparison for {len(birds)} birds")

        # Setup output
        comparison_dir = self.project_directory / 'comparisons'
        comparison_dir.mkdir(exist_ok=True)
        pdf_path = comparison_dir / f"song_comparison_{len(birds)}birds.pdf"

        with pdf_backend.PdfPages(pdf_path) as pdf:
            # Process each song position (1st song, 2nd song, etc.)
            for song_idx in range(max_songs_per_bird):
                try:
                    fig = self._create_comparison_page(birds, song_idx)
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                except Exception as e:
                    logger.error(f"Error creating comparison page {song_idx}: {e}")

        logger.info(f"✅ Created cross-bird comparison: {pdf_path}")
        return str(pdf_path)

    def _create_comparison_page(self, birds: List[str], song_idx: int) -> Optional[plt.Figure]:
        """Create a single comparison page showing one song from each bird."""

        # Collect spectrograms for this song index
        spectrograms = []
        bird_labels = []

        for bird in birds:
            bird_dir = self.project_directory / bird

            try:
                # Get audio files for this bird
                mapping = load_audio_paths_mapping(str(bird_dir))
                if not mapping:
                    continue

                filenames = sorted(list(mapping.keys()))
                if song_idx >= len(filenames):
                    continue

                # Process the song at this index
                filename = filenames[song_idx]
                spec_result = self._process_single_file(
                    filename, bird_dir, bird_dir / 'temp'
                )

                if spec_result['success']:
                    # Extract spectrogram data from the figure
                    # This is a bit hacky - better to refactor _process_single_file
                    # to return spectrogram data separately
                    spectrograms.append({
                        'bird': bird,
                        'filename': filename,
                        'figure': spec_result['figure']
                    })
                    bird_labels.append(bird)

                # Clean up
                if 'figure' in spec_result:
                    plt.close(spec_result['figure'])

            except Exception as e:
                logger.warning(f"Could not process song {song_idx} for {bird}: {e}")

        if not spectrograms:
            logger.warning(f"No spectrograms collected for song index {song_idx}")
            return None

        # Create comparison figure
        n_birds = len(spectrograms)
        fig, axes = plt.subplots(n_birds, 1, figsize=(12, 3 * n_birds))

        # Handle single bird case
        if n_birds == 1:
            axes = [axes]

        fig.suptitle(f'Song Comparison - Position {song_idx + 1}', fontsize=16, fontweight='bold')

        for i, spec_data in enumerate(spectrograms):
            ax = axes[i]

            # Re-create spectrogram for this subplot
            # This is inefficient - ideally we'd store the spec data
            bird_dir = self.project_directory / spec_data['bird']
            file_path = get_audio_path(str(bird_dir), spec_data['filename'], self.config.prefer_local)

            try:
                audio_data = self._load_and_process_audio(file_path)
                if audio_data is not None:
                    spec_result = self._create_spectrogram(audio_data)
                    if spec_result is not None:
                        spec = spec_result['spectrogram']

                        im = ax.imshow(
                            spec,
                            aspect='auto',
                            origin='lower',
                            extent=[0, self.config.target_duration, 0, spec.shape[0]],
                            cmap=self.config.colormap
                        )

                        ax.set_ylabel(f'{spec_data["bird"]}\nFreq. Bins', fontsize=10)
                        ax.set_title(f'{spec_data["filename"]}', fontsize=10)

                        if i == n_birds - 1:  # Last subplot
                            ax.set_xlabel('Time (s)', fontsize=10)
                        else:
                            ax.set_xticks([])

            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{spec_data["bird"]}',
                        ha='center', va='center', transform=ax.transAxes)
                logger.error(f"Error creating comparison plot for {spec_data['bird']}: {e}")

        plt.tight_layout()
        return fig

def create_song_visualization_summary(project_directory: str,
                                      output_file: str = None) -> Dict:
    """
    Create a summary report of song visualization results.

    Args:
        project_directory: Path to project directory
        output_file: Path to save summary (optional)

    Returns:
        Dictionary with summary statistics
    """

    project_path = Path(project_directory)
    summary = {
        'project_directory': str(project_path),
        'birds_processed': [],
        'total_songs': 0,
        'successful_birds': 0,
        'failed_birds': 0,
        'pdf_files': [],
        'errors': []
    }

    # Scan for birds with song visualization results
    for bird_dir in project_path.iterdir():
        if not bird_dir.is_dir() or bird_dir.name.startswith('.'):
            continue

        bird_name = bird_dir.name
        pdf_dir = bird_dir / 'pdfs'

        if pdf_dir.exists():
            song_pdfs = list(pdf_dir.glob('*song*.pdf'))
            if song_pdfs:
                summary['birds_processed'].append(bird_name)
                summary['successful_birds'] += 1
                summary['pdf_files'].extend([str(p) for p in song_pdfs])

                # Count songs from PNG directory if available
                png_dir = bird_dir / 'spectrograms' / 'songs'
                if png_dir.exists():
                    song_pngs = list(png_dir.glob('*.png'))
                    summary['total_songs'] += len(song_pngs)

    # Save summary if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Summary saved to {output_file}")

    return summary


# Utility functions for easy usage
def quick_song_visualization(project_directory: str,
                             birds: List[str] = None,
                             target_duration: float = 8.0,
                             max_files_per_bird: int = 10) -> Dict:
    """
    Quick function to create song visualizations with sensible defaults.

    Args:
        project_directory: Path to project directory
        birds: Specific birds to process (all if None)
        target_duration: Duration of song clips in seconds
        max_files_per_bird: Maximum files to process per bird

    Returns:
        Processing results dictionary
    """

    config = SongVisualizationConfig(
        target_duration=target_duration,
        max_files_per_bird=max_files_per_bird,
        skip_existing=True
    )

    visualizer = SongVisualizer(project_directory, config)
    return visualizer.create_song_spectrograms(birds)


def create_song_comparison(project_directory: str,
                           birds: List[str] = None,
                           max_songs_per_bird: int = 3) -> str:
    """
    Quick function to create cross-bird song comparison.

    Args:
        project_directory: Path to project directory
        birds: Birds to compare (first 5 if None)
        max_songs_per_bird: Songs per bird to include

    Returns:
        Path to comparison PDF
    """

    config = SongVisualizationConfig()
    visualizer = SongComparisonVisualizer(project_directory, config)
    return visualizer.create_cross_bird_comparison(birds, max_songs_per_bird)


# Example usage and testing
def example_usage():
    """Example of how to use the refactored song visualization module."""

    project_dir = "/Volumes/Extreme SSD/evsong test"

    # Basic usage with defaults
    results = quick_song_visualization(project_dir)
    print(f"Processed {len(results)} birds")

    # Custom configuration
    config = SongVisualizationConfig(
        target_duration=10.0,
        max_files_per_bird=5,
        figure_size=(14, 8),
        colormap='plasma',
        save_individual_pngs=True,
        create_pdf=True
    )

    visualizer = SongVisualizer(project_dir, config)
    results = visualizer.create_song_spectrograms(['bird1', 'bird2'])

    # Create comparison
    comparison_pdf = create_song_comparison(project_dir, ['bird1', 'bird2', 'bird3'])
    print(f"Comparison saved to: {comparison_pdf}")

    # Generate summary
    summary = create_song_visualization_summary(project_dir, 'song_summary.json')
    print(f"Summary: {summary['successful_birds']} birds, {summary['total_songs']} songs")


if __name__ == '__main__':
    # Test the module
    example_usage()