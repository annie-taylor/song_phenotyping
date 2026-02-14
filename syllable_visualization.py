# syllable_visualization.py

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import os
import random
import tables
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Import your existing functions
from tools.song_io import get_memory_usage, logger


@dataclass
class SyllableSamplingConfig:
    """Simple configuration for syllable sampling."""
    syllables_per_bird: int = 40
    random_seed: int = 42
    pdf_grid_size: Tuple[int, int] = (4, 2)  # (n_cols, n_rows)
    figure_size: Tuple[float, float] = (16, 8)
    skip_existing: bool = True


class SyllableSampler:
    """Simple syllable sampler focused on creating sample PDFs."""

    def __init__(self, project_directory: str, config: SyllableSamplingConfig = None):
        self.project_directory = Path(project_directory)
        self.config = config or SyllableSamplingConfig()
        self.birds = self._discover_birds()

    def _discover_birds(self) -> List[str]:
        """Find birds with syllable data."""
        birds = []
        for item in self.project_directory.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data'):

                syllables_dir = item / 'data' / 'syllables'
                if syllables_dir.exists() and list(syllables_dir.glob('*.h5')):
                    birds.append(item.name)

        logger.info(f"Found {len(birds)} birds with syllable data")
        return sorted(birds)

    def create_sample_pdfs(self, birds: List[str] = None) -> Dict[str, str]:
        """Create syllable sample PDFs for specified birds."""
        if birds is None:
            birds = self.birds

        logger.info(f"🔤 Creating syllable sample PDFs for {len(birds)} birds")

        results = {}
        for bird in birds:
            pdf_path = self._create_single_bird_pdf(bird)
            results[bird] = pdf_path

        successful = sum(1 for path in results.values() if path is not None)
        logger.info(f"✅ Created {successful}/{len(birds)} syllable PDFs")

        return results

    def _create_single_bird_pdf(self, bird: str) -> Optional[str]:
        """Create PDF for a single bird."""
        bird_dir = self.project_directory / bird

        # Setup paths
        syllables_dir = bird_dir / 'data' / 'syllables'
        pdf_dir = bird_dir / 'pdfs'
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / f'{bird}_syllable_samples.pdf'

        # Skip if exists
        if self.config.skip_existing and pdf_path.exists():
            logger.info(f"⏭️ Skipping {bird} - PDF already exists")
            return str(pdf_path)

        try:
            # Load syllables
            syllable_files = list(syllables_dir.glob('*.h5'))
            if not syllable_files:
                logger.warning(f"No syllable files found for {bird}")
                return None

            sampled_syllables = self._sample_syllables(syllable_files)
            if not sampled_syllables:
                logger.warning(f"No syllables could be sampled for {bird}")
                return None

            # Create PDF
            self._create_pdf(pdf_path, bird, sampled_syllables)
            logger.info(f"✅ Created PDF for {bird}: {len(sampled_syllables)} syllables")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error creating PDF for {bird}: {e}")
            return None

    def _sample_syllables(self, syllable_files: List[Path]) -> List[Dict]:
        """Sample syllables from files."""
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)

        # Collect all syllables
        all_syllables = []
        for file_path in syllable_files:
            syllables = self._load_syllables_from_file(file_path)
            all_syllables.extend(syllables)

        if not all_syllables:
            return []

        # Sample
        n_to_sample = min(self.config.syllables_per_bird, len(all_syllables))
        return random.sample(all_syllables, n_to_sample)

    def _load_syllables_from_file(self, file_path: Path) -> List[Dict]:
        """Load syllables from HDF5 file."""
        syllables = []

        try:
            with tables.open_file(str(file_path), 'r') as f:
                spectrograms = f.root.spectrograms.read()
                labels = f.root.manual.read()
                hashes = f.root.hashes.read()

                # Convert to strings if needed
                if len(labels) > 0 and isinstance(labels[0], bytes):
                    labels = [label.decode('utf-8') for label in labels]
                if len(hashes) > 0 and isinstance(hashes[0], bytes):
                    hashes = [hash_id.decode('utf-8') for hash_id in hashes]

                # Create syllable objects
                for i in range(len(spectrograms)):
                    syllable = {
                        'spectrogram': spectrograms[i],
                        'label': labels[i] if i < len(labels) else 'unknown',
                        'hash': hashes[i] if i < len(hashes) else f'hash_{i}',
                        'filename': file_path.stem
                    }
                    syllables.append(syllable)

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

        return syllables

    def _create_pdf(self, pdf_path: Path, bird: str, syllables: List[Dict]):
        """Create the actual PDF with syllable grids."""
        n_cols, n_rows = self.config.pdf_grid_size
        syllables_per_page = n_cols * n_rows

        with pdf_backend.PdfPages(pdf_path) as pdf:
            for page_start in range(0, len(syllables), syllables_per_page):
                page_syllables = syllables[page_start:page_start + syllables_per_page]

                fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figure_size)

                # Handle subplot array shapes
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                # Page title
                page_num = (page_start // syllables_per_page) + 1
                total_pages = (len(syllables) + syllables_per_page - 1) // syllables_per_page
                fig.suptitle(f'{bird} - Syllable Samples (Page {page_num}/{total_pages})',
                             fontsize=16, fontweight='bold')
                # Plot syllables
                for i, syl in enumerate(page_syllables):
                    ax = axes[i]

                    # Plot spectrogram
                    ax.imshow(syl['spectrogram'], aspect='auto', origin='lower', cmap='viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add title with label and hash
                    title = f"{syl['label']}\n{syl['hash'][:8]}..."
                    ax.set_title(title, fontsize=8)

                # Hide unused subplots
                for i in range(len(page_syllables), len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


# Simple utility functions

def quick_syllable_sampling(project_directory: str,
                            birds: List[str] = None,
                            syllables_per_bird: int = 40) -> Dict[str, str]:
    """
    Quick function to create syllable sample PDFs with defaults.

    Args:
        project_directory: Path to project directory
        birds: Specific birds to process (all if None)
        syllables_per_bird: Number of syllables to sample

    Returns:
        Dict mapping bird names to PDF paths
    """
    config = SyllableSamplingConfig(syllables_per_bird=syllables_per_bird)
    sampler = SyllableSampler(project_directory, config)
    return sampler.create_sample_pdfs(birds)


def create_syllable_samples_for_bird(project_directory: str, bird: str,
                                     syllables_per_bird: int = 40) -> Optional[str]:
    """
    Create syllable samples for a single bird.

    Args:
        project_directory: Path to project directory
        bird: Bird name to process
        syllables_per_bird: Number of syllables to sample

    Returns:
        Path to created PDF or None if failed
    """
    config = SyllableSamplingConfig(syllables_per_bird=syllables_per_bird)
    sampler = SyllableSampler(project_directory, config)
    return sampler._create_single_bird_pdf(bird)


# Example usage
def example_usage():
    """Example of how to use the simple syllable sampling module."""

    test_paths = [
        os.path.join('/Volumes', 'Extreme SSD', 'wseg test'),
        os.path.join('/Volumes', 'Extreme SSD', 'evsong test'),
        os.path.join('/Volumes', 'Extreme SSD', 'evsong test warp')
    ]

    for project_dir in test_paths:

        # Basic usage - all birds, default settings
        results = quick_syllable_sampling(project_dir)
        print(f"Created PDFs for {len(results)} birds")



if __name__ == '__main__':
    example_usage()