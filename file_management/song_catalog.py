import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import glob
import random
import string

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import librosa
import soundfile as sf
from scipy import signal


@dataclass
class SimpleSpectrogramConfig:
    """Configuration for simple spectrogram generation."""
    n_spectrograms: int = 30
    spectrograms_per_page: int = 4
    duration: float = 6.0
    sample_rate: int = 32000
    n_fft: int = 512
    hop_length: int = 64
    freq_range: Tuple[int, int] = (500, 8000)  # Frequency range to display
    anonymize: bool = False  # NEW: Flag to anonymize bird names and timestamps


class SimpleBirdSpectrogramGenerator:
    def __init__(self, config: SimpleSpectrogramConfig = None):
        self.config = config or SimpleSpectrogramConfig()
        # For anonymization - generate consistent random IDs per session
        self.anonymization_map = {}

    def _get_anonymous_id(self, bird_name: str) -> str:
        """Get consistent anonymous ID for a bird name."""
        if bird_name not in self.anonymization_map:
            # Generate a simple anonymous ID like "Bird_A", "Bird_B", etc.
            id_num = len(self.anonymization_map) + 1
            self.anonymization_map[bird_name] = f"Bird_{chr(64 + id_num)}"  # A, B, C, etc.
        return self.anonymization_map[bird_name]

    def _format_bird_name(self, bird_name: str) -> str:
        """Return bird name or anonymous ID based on config."""
        if self.config.anonymize:
            return self._get_anonymous_id(bird_name)
        return bird_name

    def _format_timestamp_info(self, timestamp: Optional[datetime], filename: str) -> str:
        """Format timestamp information based on anonymization setting."""
        if self.config.anonymize:
            return "Date/Time: [Hidden]"

        if timestamp:
            return f"Date: {timestamp.strftime('%Y-%m-%d')} | Time: {timestamp.strftime('%H:%M:%S')}"
        else:
            return f"File: {os.path.basename(filename)}"

    def find_audio_files(self, directory_path: str) -> List[str]:
        """Find all audio files in a directory and subdirectories."""
        audio_extensions = ['.wav', '.WAV', '.cbin', '.rec']
        audio_files = []

        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(root, file))
        except Exception as e:
            logging.error(f"Error searching directory {directory_path}: {e}")

        return audio_files

    def read_audio_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Read audio file using librosa or soundfile."""
        try:
            # Try librosa first
            audio, sr = librosa.load(file_path, sr=self.config.sample_rate)
            return audio, sr
        except:
            try:
                # Try soundfile for other formats
                audio, sr = sf.read(file_path)
                if sr != self.config.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                    sr = self.config.sample_rate
                return audio, sr
            except Exception as e:
                logging.warning(f"Could not read audio file {file_path}: {e}")
                return None, None

    def create_spectrogram(self, audio: np.ndarray, sr: int, title: str = "") -> plt.Figure:
        """Create a spectrogram plot."""
        # Limit duration if needed
        max_samples = int(self.config.duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            audio,
            fs=sr,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length
        )

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot spectrogram
        im = ax.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]],
            cmap='viridis'
        )

        # Set frequency limits
        ax.set_ylim(self.config.freq_range)

        # Labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Power (dB)')

        plt.tight_layout()
        return fig

    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from filename if possible."""
        # Common patterns for bird song files
        patterns = [
            r'(\d{8})_(\d{6})',  # YYYYMMDD_HHMMSS
            r'(\d{6})_(\d{6})',  # DDMMYY_HHMMSS
            r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                date_str, time_str = match.groups()
                try:
                    # Try different date formats
                    if len(date_str) == 8 and '-' not in date_str:
                        if date_str.startswith('20'):  # YYYYMMDD
                            dt_str = f"{date_str}_{time_str}"
                            return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
                        else:  # DDMMYY
                            dt_str = f"{date_str}_{time_str}"
                            return datetime.strptime(dt_str, "%d%m%y_%H%M%S")
                    elif '-' in date_str:  # YYYY-MM-DD format
                        dt_str = f"{date_str}_{time_str.replace('-', ':')}"
                        return datetime.strptime(dt_str, "%Y-%m-%d_%H:%M:%S")
                except ValueError:
                    continue
        return None

    def generate_bird_pdf(self, bird_name: str, directories: List[str], output_dir: str = ".") -> str:
        """Generate PDF with spectrograms for a single bird."""
        # Collect all audio files from all directories for this bird
        all_audio_files = []
        for directory in directories:
            if os.path.exists(directory):
                audio_files = self.find_audio_files(directory)
                all_audio_files.extend(audio_files)

        if not all_audio_files:
            logging.warning(f"No audio files found for {bird_name}")
            return ""

        # Sort files by timestamp if possible (but randomize if anonymizing)
        files_with_timestamps = []
        for file_path in all_audio_files:
            filename = os.path.basename(file_path)
            timestamp = self.extract_timestamp_from_filename(filename)
            files_with_timestamps.append((file_path, timestamp))

        if self.config.anonymize:
            # Randomize order to prevent temporal bias
            random.shuffle(files_with_timestamps)
        else:
            # Sort by timestamp, putting None values at the end
            files_with_timestamps.sort(key=lambda x: x[1] if x[1] else datetime.max)

        # Limit to requested number
        files_with_timestamps = files_with_timestamps[:self.config.n_spectrograms]

        # Create output filename
        display_name = self._format_bird_name(bird_name)
        if self.config.anonymize:
            output_filename = f"{display_name}_spectrograms_anonymous.pdf"
        else:
            output_filename = f"{bird_name}_spectrograms.pdf"

        output_path = os.path.join(output_dir, output_filename)

        with PdfPages(output_path) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.text(0.5, 0.7, f"{display_name} - Song Spectrograms",
                    ha='center', va='center', fontsize=24, fontweight='bold')

            # Add generation info
            generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if self.config.anonymize:
                ax.text(0.5, 0.6, "ANONYMIZED VERSION",
                        ha='center', va='center', fontsize=16, fontweight='bold', color='red')
                ax.text(0.5, 0.55, "Bird identity and timestamps hidden for unbiased analysis",
                        ha='center', va='center', fontsize=12, style='italic')
                ax.text(0.5, 0.4, f"Generated: {generation_time}",
                        ha='center', va='center', fontsize=12)
            else:
                ax.text(0.5, 0.6, f"Generated: {generation_time}",
                        ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.55, f"Total spectrograms: {len(files_with_timestamps)}",
                        ha='center', va='center', fontsize=12)
                ax.text(0.5, 0.5, f"Duration per spectrogram: {self.config.duration}s",
                        ha='center', va='center', fontsize=12)

            # Add directory info (only if not anonymizing)
            if not self.config.anonymize:
                ax.text(0.5, 0.3, "Source directories:",
                        ha='center', va='center', fontsize=12, fontweight='bold')
                dir_text = "\n".join([f"• {d}" for d in directories[:5]])  # Show first 5
                if len(directories) > 5:
                    dir_text += f"\n... and {len(directories) - 5} more"
                ax.text(0.5, 0.15, dir_text,
                        ha='center', va='center', fontsize=10, family='monospace')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Generate spectrograms
            current_page_count = 0
            for i, (file_path, timestamp) in enumerate(files_with_timestamps):
                try:
                    # Read audio
                    audio, sr = self.read_audio_file(file_path)
                    if audio is None:
                        continue

                    # Create title
                    timestamp_info = self._format_timestamp_info(timestamp, file_path)
                    if self.config.anonymize:
                        title = f"{display_name} - Spectrogram {i + 1}\n{timestamp_info}"
                    else:
                        title = f"{bird_name} - Spectrogram {i + 1}\n{timestamp_info}"

                    # Create spectrogram
                    fig = self.create_spectrogram(audio, sr, title)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

                    current_page_count += 1

                    if current_page_count % 10 == 0:
                        logging.info(f"Generated {current_page_count} spectrograms for {display_name}")

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    continue

        logging.info(f"Generated PDF: {output_path} with {current_page_count} spectrograms")
        return output_path


def parse_foster_directories(file_path: str) -> Dict[str, List[str]]:
    """Parse the xfoster_directories.txt file to extract bird directories."""
    bird_directories = {}

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Split by bird entries
        bird_sections = re.split(r'^([a-z]{2}\d+[a-z]{2}\d+):', content, flags=re.MULTILINE)

        for i in range(1, len(bird_sections), 2):
            bird_name = bird_sections[i].strip()
            directory_section = bird_sections[i + 1].strip()

            # Extract directory paths
            directories = []
            for line in directory_section.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    dir_path = line[2:].strip()
                    # Convert network paths to local paths if needed
                    if dir_path.startswith('\\\\macaw.ucsf.edu'):
                        # You might need to adjust this mapping based on your setup
                        dir_path = dir_path.replace('\\\\macaw.ucsf.edu\\users\\', '/path/to/mounted/')
                    elif dir_path.startswith('Z:'):
                        dir_path = dir_path.replace('Z:', '/path/to/mounted')

                    directories.append(dir_path)

            if directories:
                bird_directories[bird_name] = directories

    except Exception as e:
        logging.error(f"Error parsing foster directories file: {e}")

    return bird_directories


def generate_foster_bird_pdfs(directories_file: str, output_dir: str,
                              anonymize: bool = False, n_birds: Optional[int] = None):
    """Generate PDFs for all foster birds."""

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse directories
    bird_directories = parse_foster_directories(directories_file)

    if not bird_directories:
        logging.error("No bird directories found in file")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure generator
    config = SimpleSpectrogramConfig(
        n_spectrograms=30,
        spectrograms_per_page=4,
        duration=6.0,
        anonymize=anonymize
    )

    generator = SimpleBirdSpectrogramGenerator(config)

    # Limit number of birds if specified
    bird_names = list(bird_directories.keys())
    if n_birds:
        bird_names = bird_names[:n_birds]

    logging.info(f"Processing {len(bird_names)} birds (anonymize={anonymize})")

    successful_pdfs = []
    failed_birds = []

    for i, bird_name in enumerate(bird_names):
        try:
            logging.info(f"Processing bird {i + 1}/{len(bird_names)}: {bird_name}")

            directories = bird_directories[bird_name]
            pdf_path = generator.generate_bird_pdf(bird_name, directories, output_dir)

            if pdf_path:
                successful_pdfs.append(pdf_path)
                display_name = generator._format_bird_name(bird_name)
                logging.info(f"✓ Generated PDF for {display_name}: {pdf_path}")
            else:
                failed_birds.append(bird_name)
                logging.warning(f"✗ Failed to generate PDF for {bird_name}")

        except Exception as e:
            failed_birds.append(bird_name)
            logging.error(f"✗ Error processing {bird_name}: {e}")
            continue

    # Summary
    logging.info(f"\n=== SUMMARY ===")
    logging.info(f"Successfully generated: {len(successful_pdfs)} PDFs")
    logging.info(f"Failed: {len(failed_birds)} birds")

    if failed_birds:
        logging.info(f"Failed birds: {', '.join(failed_birds[:10])}...")

    if anonymize:
        logging.info(f"Anonymization mapping:")
        for original, anonymous in generator.anonymization_map.items():
            logging.info(f"  {original} -> {anonymous}")


def generate_sample_pdfs(directories_file: str, output_dir: str, n_sample: int = 3):
    """Generate both anonymous and identified PDFs for a sample of birds."""

    # Parse directories to get bird names
    bird_directories = parse_foster_directories(directories_file)
    bird_names = list(bird_directories.keys())[:n_sample]

    logging.info(f"Generating sample PDFs for {len(bird_names)} birds")

    # Generate identified versions
    logging.info("Generating IDENTIFIED versions...")
    generate_foster_bird_pdfs(
        directories_file=directories_file,
        output_dir=os.path.join(output_dir, "identified"),
        anonymize=False,
        n_birds=n_sample
    )

    # Generate anonymous versions
    logging.info("Generating ANONYMOUS versions...")
    generate_foster_bird_pdfs(
        directories_file=directories_file,
        output_dir=os.path.join(output_dir, "anonymous"),
        anonymize=True,
        n_birds=n_sample
    )

    logging.info("Sample generation complete!")


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # File paths - adjust these to your setup
    directories_file = "xfoster_directories.txt"
    output_dir = "foster_bird_spectrograms"

    # Example usage:

    # 1. Generate sample PDFs (both anonymous and identified)
    print("Generating sample PDFs...")
    generate_sample_pdfs(
        directories_file=directories_file,
        output_dir=output_dir,
        n_sample=3
    )

    # 2. Generate anonymous PDFs for all birds
    # print("Generating anonymous PDFs for all birds...")
    # generate_foster_bird_pdfs(
    #     directories_file=directories_file,
    #     output_dir=os.path.join(output_dir, "all_anonymous"),
    #     anonymize=True
    # )

    # 3. Generate identified PDFs for all birds
    # print("Generating identified PDFs for all birds...")
    # generate_foster_bird_pdfs(
    #     directories_file=directories_file,
    #     output_dir=os.path.join(output_dir, "all_identified"),
    #     anonymize=False
    # )