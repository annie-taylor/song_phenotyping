# song_visualization.py

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Import your existing functions
from tools.song_io import get_memory_usage, get_song_spec, rms_norm, butter_bandpass_filter_sos, logger
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file
from tools.audio_path_management import load_audio_paths_mapping, get_audio_path


@dataclass
class SongVisualizationConfig:
    """Simple configuration for song visualization."""
    target_duration: float = 8.0
    figure_size: Tuple[float, float] = (12, 6)
    colormap: str = 'viridis'
    max_files_per_bird: int = 10
    skip_existing: bool = True


class SongVisualizer:
    """Simple song spectrogram visualizer - just creates PDFs of song spectrograms."""

    def __init__(self, project_directory: str, config: SongVisualizationConfig = None):
        self.project_directory = Path(project_directory)
        self.config = config or SongVisualizationConfig()
        self.birds = self._discover_birds()

    def _discover_birds(self) -> List[str]:
        """Find birds with audio data."""
        birds = []
        for item in self.project_directory.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data'):

                # Check for audio path mapping
                mapping_file = item / 'audio_paths.json'
                if mapping_file.exists():
                    birds.append(item.name)

        logger.info(f"Found {len(birds)} birds with audio data")
        return sorted(birds)

    def create_song_pdfs(self, birds: List[str] = None) -> Dict[str, str]:
        """Create song spectrogram PDFs for specified birds."""
        if birds is None:
            birds = self.birds

        logger.info(f"🎵 Creating song PDFs for {len(birds)} birds")

        results = {}
        for bird in birds:
            pdf_path = self._create_single_bird_pdf(bird)
            results[bird] = pdf_path

        successful = sum(1 for path in results.values() if path is not None)
        logger.info(f"✅ Created {successful}/{len(birds)} song PDFs")

        return results

    def _create_single_bird_pdf(self, bird: str) -> Optional[str]:
        """Create PDF for a single bird."""
        bird_dir = self.project_directory / bird

        # Setup paths
        pdf_dir = bird_dir / 'pdfs'
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / f'{bird}_song_spectrograms.pdf'

        # Skip if exists
        if self.config.skip_existing and pdf_path.exists():
            logger.info(f"⏭️ Skipping {bird} - PDF already exists")
            return str(pdf_path)

        try:
            # Load audio mapping
            mapping = load_audio_paths_mapping(str(bird_dir))
            if not mapping:
                logger.warning(f"No audio mapping found for {bird}")
                return None

            # Select files to process
            filenames = list(mapping.keys())[:self.config.max_files_per_bird]

            # Create PDF
            with pdf_backend.PdfPages(pdf_path) as pdf:
                for filename in tqdm(filenames, desc=f"Processing {bird}"):
                    fig = self._create_song_spectrogram(filename, bird_dir)
                    if fig:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

            logger.info(f"✅ Created PDF for {bird}: {len(filenames)} songs")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error creating PDF for {bird}: {e}")
            return None

    def _create_song_spectrogram(self, filename: str, bird_dir: Path) -> Optional[plt.Figure]:
        """Create spectrogram for a single song."""

        try:
            # Get audio file path
            file_path = get_audio_path(str(bird_dir), filename, prefer_local=True)

            # Load and process audio
            audio, sr = read_audio_file(file_path)
            audio = rms_norm(audio)

            # Apply filtering
            params = SpectrogramParams()
            audio = butter_bandpass_filter_sos(
                audio, lowcut=params.min_freq, highcut=params.max_freq, fs=sr, order=5
            )

            # Trim or pad to target duration
            target_samples = int(self.config.target_duration * sr)
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio = audio[:target_samples]

            # Create spectrogram
            spec, _, _ = get_song_spec(
                t1=0, t2=self.config.target_duration, audio=audio,
                params=params, fs=sr, downsample=False
            )

            # Create plot
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.imshow(spec, aspect='auto', origin='lower',
                      extent=[0, self.config.target_duration, 0, spec.shape[0]],
                      cmap=self.config.colormap)

            ax.set_ylabel('Frequency Bins')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Song Spectrogram: {filename}')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.warning(f"Failed to create spectrogram for {filename}: {e}")
            return None


# Simple utility functions

def quick_song_pdfs(project_directory: str, birds: List[str] = None,
                    target_duration: float = 8.0, max_files: int = 10) -> Dict[str, str]:
    """
    Quick function to create song PDFs with defaults.

    Args:
        project_directory: Path to project directory
        birds: Specific birds to process (all if None)
        target_duration: Duration of song clips in seconds
        max_files: Maximum files to process per bird

    Returns:
        Dict mapping bird names to PDF paths
    """
    config = SongVisualizationConfig(
        target_duration=target_duration,
        max_files_per_bird=max_files
    )

    visualizer = SongVisualizer(project_directory, config)
    return visualizer.create_song_pdfs(birds)


# Example usage
def example_usage():
    """Example of how to use the simplified song visualization module."""

    project_dir = "/Volumes/Extreme SSD/evsong test"

    # Basic usage - all birds, default settings
    results = quick_song_pdfs(project_dir)
    print(f"Created PDFs for {len(results)} birds")

    # Custom duration and file limit
    results = quick_song_pdfs(project_dir, target_duration=10.0, max_files=5)

    # Single bird
    pdf_path = create_song_pdf_for_bird(project_dir, "bird1", target_duration=6.0)
    if pdf_path:
        print(f"Created PDF: {pdf_path}")

    # Custom configuration
    config = SongVisualizationConfig(
        target_duration=12.0,
        figure_size=(15, 8),
        colormap='plasma',
        max_files_per_bird=8
    )

    visualizer = SongVisualizer(project_dir, config)
    results = visualizer.create_song_pdfs(['bird1', 'bird2'])


if __name__ == '__main__':
    example_usage()