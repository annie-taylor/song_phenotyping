import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import glob
from sys import platform
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import librosa
import soundfile as sf
from scipy import signal
from tools.logging_utils import setup_logger

logger = setup_logger(__name__, 'song_catalog_unlabeled.log')

@dataclass
class SimpleSpectrogramConfig:
    """Configuration for simple spectrogram generation."""
    n_spectrograms: int = 30
    spectrograms_per_page: int = 4
    duration: float = 6.0
    sample_rate: int = 32000
    n_fft: int = 512
    hop_length: int = 64
    freq_range: Tuple[int, int] = (500, 8000)
    anonymize: bool = False
    # New parameters for improved functionality
    max_workers: int = 4  # For parallel processing
    quality_threshold: float = 0.1  # Minimum audio quality threshold
    output_format: str = 'pdf'  # Could support 'png', 'jpg' in future

class SimpleBirdSpectrogramGenerator:
    def __init__(self, config: SimpleSpectrogramConfig = None):
        self.config = config or SimpleSpectrogramConfig()
        self.anonymization_map = {}
        self.processed_files = set()  # Track processed files to avoid duplicates

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

    def _validate_audio_quality(self, audio: np.ndarray) -> bool:
        """Check if audio meets minimum quality standards."""
        if len(audio) == 0:
            return False

        # Check for sufficient signal strength
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.config.quality_threshold:
            logging.debug(f"Audio RMS ({rms:.3f}) below threshold ({self.config.quality_threshold})")
            return False

        # Check for clipping (too many samples at max value)
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        if clipping_ratio > 0.1:  # More than 10% clipped
            logging.debug(f"Audio clipping detected ({clipping_ratio:.1%})")
            return False

        return True

    def _enhance_spectrogram_display(self, fig: plt.Figure, ax: plt.Axes,
                                   Sxx_db: np.ndarray, f: np.ndarray, t: np.ndarray) -> None:
        """Add enhanced visualization features to spectrogram."""
        # Add frequency grid lines at common bird song frequencies
        important_freqs = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        for freq in important_freqs:
            if self.config.freq_range[0] <= freq <= self.config.freq_range[1]:
                ax.axhline(y=freq, color='white', alpha=0.3, linewidth=0.5, linestyle='--')

        for t_mark in range(1, int(self.config.duration)):
            ax.axvline(x=t_mark, color='white', alpha=0.3, linewidth=0.5, linestyle='--')

    def create_spectrogram(self, audio: np.ndarray, sr: int, title: str = "") -> Optional[plt.Figure]:
        """Create a spectrogram plot with enhanced features."""
        # Validate audio quality first
        if not self._validate_audio_quality(audio):
            logging.debug(f"Audio quality check failed for: {title}")
            return None

        # Limit duration if needed
        max_samples = int(self.config.duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        try:
            # Compute spectrogram with better parameters
            f, t, Sxx = signal.spectrogram(
                audio,
                fs=sr,
                nperseg=self.config.n_fft,
                noverlap=self.config.n_fft - self.config.hop_length,
                window='hann'  # Better window function
            )

            # Convert to dB with better dynamic range
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            # Create figure with better styling
            fig, ax = plt.subplots(figsize=(12, 6))  # Wider for better visibility

            # Plot spectrogram with better colormap
            im = ax.imshow(
                Sxx_db,
                aspect='auto',
                origin='lower',
                extent=[t[0], t[-1], f[0], f[-1]],
                cmap='plasma',  # Better colormap for bird songs
                vmin=np.percentile(Sxx_db, 5),  # Better contrast
                vmax=np.percentile(Sxx_db, 95)
            )

            # Set frequency limits
            ax.set_ylim(self.config.freq_range)

            # Enhanced labels and styling
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title(title, fontsize=14, pad=20)

            # Add enhancement features
            self._enhance_spectrogram_display(fig, ax, Sxx_db, f, t)

            # Add colorbar with better positioning
            cbar = plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

            plt.tight_layout()
            return fig

        except Exception as e:
            logging.error(f"Error creating spectrogram for {title}: {e}")
            return None

    def generate_summary_statistics(self, files_with_timestamps: List[Tuple[str, Optional[datetime]]],
                                  bird_name: str) -> Dict:
        """Generate summary statistics for the bird's audio files."""
        stats = {
            'total_files': len(files_with_timestamps),
            'files_with_timestamps': sum(1 for _, ts in files_with_timestamps if ts is not None),
            'date_range': None,
            'total_duration_estimate': len(files_with_timestamps) * self.config.duration,
            'file_formats': {}
        }

        # Analyze file formats
        for file_path, _ in files_with_timestamps:
            ext = os.path.splitext(file_path)[1].lower()
            stats['file_formats'][ext] = stats['file_formats'].get(ext, 0) + 1

        # Analyze date range
        timestamps = [ts for _, ts in files_with_timestamps if ts is not None]
        if timestamps:
            timestamps.sort()
            stats['date_range'] = (timestamps[0], timestamps[-1])

        return stats

    def create_enhanced_title_page(self, bird_name: str, stats: Dict, pdf: PdfPages) -> None:
        """Create an enhanced title page with statistics."""
        fig, ax = plt.subplots(figsize=(11, 8.5))

        display_name = self._format_bird_name(bird_name)

        # Main title
        ax.text(0.5, 0.85, f"{display_name} - Song Spectrograms",
                ha='center', va='center', fontsize=24, fontweight='bold')

        # Anonymization notice
        if self.config.anonymize:
            ax.text(0.5, 0.75, "ANONYMIZED VERSION",
                    ha='center', va='center', fontsize=18, fontweight='bold',
                    color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
            ax.text(0.5, 0.68, "Bird identity and timestamps hidden for unbiased analysis",
                    ha='center', va='center', fontsize=12, style='italic')

        # Statistics section
        y_pos = 0.55
        ax.text(0.5, y_pos, "Analysis Summary",
                ha='center', va='center', fontsize=16, fontweight='bold')

        y_pos -= 0.08
        stats_text = [
            f"Total spectrograms: {stats['total_files']}",
            f"Duration per spectrogram: {self.config.duration}s",
            f"Estimated total duration: {stats['total_duration_estimate']:.1f}s",
            f"Files with timestamps: {stats['files_with_timestamps']}"
        ]

        if not self.config.anonymize and stats['date_range']:
            start_date, end_date = stats['date_range']
            stats_text.append(f"Recording period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        for line in stats_text:
            ax.text(0.5, y_pos, line, ha='center', va='center', fontsize=12)
            y_pos -= 0.05

        # File format breakdown
        if stats['file_formats']:
            y_pos -= 0.03
            ax.text(0.5, y_pos, "File formats:", ha='center', va='center',
                   fontsize=12, fontweight='bold')
            y_pos -= 0.04
            format_text = ", ".join([f"{ext}: {count}" for ext, count in stats['file_formats'].items()])
            ax.text(0.5, y_pos, format_text, ha='center', va='center', fontsize=11)

        # Generation info
        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.5, 0.15, f"Generated: {generation_time}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        # Configuration details
        config_text = (f"Config: {self.config.sample_rate}Hz, "
                      f"{self.config.freq_range[0]}-{self.config.freq_range[1]}Hz, "
                      f"FFT={self.config.n_fft}")
        ax.text(0.5, 0.05, config_text, ha='center', va='center',
                fontsize=10, family='monospace', alpha=0.7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def generate_bird_pdf(self, bird_name: str, directories: List[str],
                         output_dir: str = ".") -> str:
        """Generate PDF with spectrograms for a single bird - enhanced version."""
        # Collect all audio files from all directories for this bird
        all_audio_files = []
        for directory in directories:
            if os.path.exists(directory):
                audio_files = self.find_audio_files(directory)
                all_audio_files.extend(audio_files)

        if not all_audio_files:
            logging.warning(f"No audio files found for {bird_name}")
            return ""

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in all_audio_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)

        # Sort files by timestamp if possible
        files_with_timestamps = []
        for file_path in unique_files:
            filename = os.path.basename(file_path)
            timestamp = self.extract_timestamp_from_filename(filename)
            files_with_timestamps.append((file_path, timestamp))

        if self.config.anonymize:
            random.shuffle(files_with_timestamps)
        else:
            files_with_timestamps.sort(key=lambda x: x[1] if x[1] else datetime.max)

        # Limit to requested number
        files_with_timestamps = files_with_timestamps[:self.config.n_spectrograms]

        # Generate statistics
        stats = self.generate_summary_statistics(files_with_timestamps, bird_name)

        # Create output filename
        display_name = self._format_bird_name(bird_name)
        if self.config.anonymize:
            output_filename = f"{display_name}_spectrograms_anonymous.pdf"
        else:
            output_filename = f"{bird_name}_spectrograms.pdf"

        output_path = os.path.join(output_dir, output_filename)

        with PdfPages(output_path) as pdf:
            # Enhanced title page
            self.create_enhanced_title_page(bird_name, stats, pdf)

            # Sequential processing (keeping it simple for now)
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
                    if fig is not None:
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



def parse_audio_filename(file_path: str) -> Dict[str, Any]:
    """
    Parse audio filename to extract bird, date, and time components.
    Handles multiple filename formats including wseg patterns.
    """
    try:
        if platform == 'win32':
            filename = file_path.split('\\')[-1]
        else:
            filename = file_path.split('/')[-1]

        # Remove .not.mat suffix if present (for wseg files)
        clean_filename = filename.replace('.not.mat', '')

        # Pattern 1: bk1bk3_170811_140945.wav (BIRD_YYMMDD_HHMMSS.wav) - actual timestamp
        pattern1 = r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)_(\d{6})_(\d{6})\.wav$'
        match1 = re.match(pattern1, clean_filename)
        if match1:
            bird, day, time = match1.groups()
            # Convert YYMMDD to full date format if needed
            if len(day) == 6:
                day = '20' + day  # Convert YY to 20YY
            return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

        # Pattern 2: bk1bk3.20081118-10.wav (BIRD.YYYYMMDD-SEQ.wav) - sequence number
        pattern2 = r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})-(\d+)\.wav$'
        match2 = re.match(pattern2, clean_filename)
        if match2:
            bird, day, seq = match2.groups()
            # Use sequence number as milliseconds offset from midnight to preserve order
            time = str(int(seq)).zfill(6)  # Convert to 6-digit string (microseconds)
            return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

        # Pattern 3: bk1bk3.20081118.wav (BIRD.YYYYMMDD.wav) - no sequence (index 0)
        pattern3 = r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})\.wav$'
        match3 = re.match(pattern3, clean_filename)
        if match3:
            bird, day = match3.groups()
            time = '000000'  # First song of the day (sequence 0)
            return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

        # Original patterns - keep for backward compatibility
        split_filename = filename.split('_')
        if len(split_filename) == 3:
            bird, day, time = split_filename[0:3]
            time = time.split('.')[0]
        elif len(split_filename) == 2:
            bird, daytime = filename.split('_')[0:2]
            day = daytime.split('.')[0][0:8]
            time = daytime.split('.')[0][8:]
        elif len(split_filename) == 4:
            bird, _, day, time = split_filename[0:4]
            time = time.split('.')[0]
        else:
            # Log the problematic filename for debugging
            logger.warning(f'🔍 Unrecognized filename format: {filename}')
            raise ValueError(f'Unrecognized filename format: {filename}')

        return {'bird': bird, 'day': day, 'time': time, 'filename': filename, 'success': True}

    except Exception as e:
        logger.error(f'💥 Failed to parse filename {file_path}: {e}')
        return {'bird': None, 'day': None, 'time': None, 'filename': None, 'success': False}

def parse_foster_directories(file_path: str) -> Dict[str, List[str]]:
    """Parse the xfoster_directories.txt file to extract bird directories."""
    bird_directories = {}

    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_bird = None
        current_directories = []

        print(f"Processing {len(lines)} lines...")

        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check if this is a bird name line (pattern: bird_name: X unique directories)
            bird_match = re.match(r'^([a-z]{2}\d+[a-z]{2}\d*): (\d+) unique directories', line)
            if bird_match:
                # Save previous bird if it exists (even if no directories)
                if current_bird is not None:
                    bird_directories[current_bird] = list(current_directories)
                    print(f"Added {current_bird} with {len(current_directories)} directories")

                # Start new bird
                current_bird = bird_match.group(1)
                current_directories = []
                continue

            # Check if this is a directory line (starts with "  - ")
            elif original_line.startswith('  - '):  # Use original_line, not stripped line!
                if current_bird is not None:
                    dir_path = original_line[4:].strip()  # Remove "  - " prefix

                    print(f"  Found directory for {current_bird}: {dir_path}")

                    # Convert network paths but keep Z: drive as is
                    if dir_path.startswith('\\\\macaw.ucsf.edu'):
                        # Convert network paths to Z: drive equivalent
                        dir_path = dir_path.replace('\\\\macaw.ucsf.edu\\users\\', 'Z:\\')

                    # Normalize to Windows backslashes for Z: drive
                    if dir_path.startswith('Z:'):
                        dir_path = dir_path.replace('/', '\\')

                    current_directories.append(dir_path)

        # Don't forget the last bird
        if current_bird is not None:
            bird_directories[current_bird] = list(current_directories)
            print(f"Added {current_bird} with {len(current_directories)} directories")

    except Exception as e:
        print(f"Error parsing foster directories file: {e}")
        import traceback
        traceback.print_exc()
        return {}

    print(f"Successfully parsed {len(bird_directories)} birds from directories file")
    return bird_directories


def debug_parse_foster_directories(file_path: str):
    """Debug version to see what's in the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Total lines: {len(lines)}")
        print("\nFirst 20 lines:")
        for i, line in enumerate(lines[:20]):
            print(f"{i + 1:3d}: {repr(line)}")

        print("\nLooking for bird name patterns...")
        bird_pattern = r'^([a-z]{2}\d+[a-z]{2}\d*): (\d+) unique directories'

        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            line = line.strip()
            if ':' in line and 'directories' in line:
                print(f"Line {i + 1}: {repr(line)}")
                match = re.match(bird_pattern, line)
                if match:
                    print(f"  ✓ MATCHES: bird={match.group(1)}, count={match.group(2)}")
                else:
                    print(f"  ✗ NO MATCH")
                    # Try different patterns
                    if re.match(r'^[a-z]{2}\d+[a-z]{2}\d*:', line):
                        print(f"    - Bird name pattern matches but full line doesn't")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def debug_audio_search(bird_name: str, directories: List[str], max_dirs_to_check: int = 3):
    """Debug function to see what's in the directories."""
    print(f"\n=== DEBUGGING AUDIO SEARCH for {bird_name} ===")

    audio_extensions = ['.wav', '.WAV', '.cbin', '.rec']

    for i, directory in enumerate(directories[:max_dirs_to_check]):
        print(f"\nDirectory {i + 1}: {directory}")

        # Check if directory exists
        if not os.path.exists(directory):
            print(f"  ❌ Directory does not exist")
            continue

        print(f"  ✅ Directory exists")

        # Check if it's accessible
        try:
            files = os.listdir(directory)
            print(f"  📁 Contains {len(files)} items")

            # Look for audio files
            audio_files = []
            for file in files[:10]:  # Check first 10 files
                if any(file.endswith(ext) for ext in audio_extensions):
                    audio_files.append(file)

            if audio_files:
                print(f"  🎵 Audio files found: {audio_files}")
            else:
                print(f"  📄 No audio files found. Sample files: {files[:5]}")

        except PermissionError:
            print(f"  🚫 Permission denied accessing directory")
        except Exception as e:
            print(f"  ❌ Error accessing directory: {e}")


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


# Additional utility functions
def validate_directories_file(file_path: str) -> bool:
    """Validate that the directories file exists and is readable."""
    if not os.path.exists(file_path):
        logging.error(f"Directories file not found: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(100)  # Read first 100 chars
        return True
    except Exception as e:
        logging.error(f"Cannot read directories file: {e}")
        return False


def create_batch_summary(output_dir: str, successful_pdfs: List[str],
                        failed_birds: List[str], anonymize: bool) -> str:
    """Create a summary report of the batch processing."""
    summary_path = os.path.join(output_dir, "batch_summary.txt")

    with open(summary_path, 'w') as f:
        f.write("BIRD SPECTROGRAM GENERATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Anonymized: {'Yes' if anonymize else 'No'}\n\n")

        f.write(f"SUCCESSFUL: {len(successful_pdfs)} PDFs\n")
        f.write("-" * 30 + "\n")
        for pdf_path in successful_pdfs:
            f.write(f"✓ {os.path.basename(pdf_path)}\n")

        if failed_birds:
            f.write(f"\nFAILED: {len(failed_birds)} birds\n")
            f.write("-" * 30 + "\n")
            for bird in failed_birds:
                f.write(f"✗ {bird}\n")

        f.write(f"\nTotal processed: {len(successful_pdfs) + len(failed_birds)}\n")
        if len(successful_pdfs) + len(failed_birds) > 0:
            f.write(f"Success rate: {len(successful_pdfs) / (len(successful_pdfs) + len(failed_birds)) * 100:.1f}%\n")

    return summary_path

def test_single_bird(directories_file: str, bird_name: str, output_dir: str = "test_output"):
    """Test spectrogram generation for a single bird."""
    logging.basicConfig(level=logging.DEBUG)

    # Parse directories
    bird_directories = parse_foster_directories(directories_file)

    if bird_name not in bird_directories:
        logging.error(f"Bird {bird_name} not found in directories file")
        available_birds = list(bird_directories.keys())[:10]
        logging.info(f"Available birds: {', '.join(available_birds)}...")
        return

    # Create test output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test both anonymous and identified versions
    for anonymize in [False, True]:
        config = SimpleSpectrogramConfig(
            n_spectrograms=5,  # Just a few for testing
            anonymize=anonymize
        )

        generator = SimpleBirdSpectrogramGenerator(config)

        logging.info(f"Testing {bird_name} (anonymize={anonymize})")
        directories = bird_directories[bird_name]

        # Debug directory access
        debug_audio_search(bird_name, directories, max_dirs_to_check=2)

        # Generate PDF
        pdf_path = generator.generate_bird_pdf(bird_name, directories, output_dir)

        if pdf_path:
            logging.info(f"✓ Test successful: {pdf_path}")
        else:
            logging.error(f"✗ Test failed for {bird_name}")


def main():
    """Main execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Configuration
    directories_file = "xfoster_directories.txt"
    output_base_dir = "foster_bird_spectrograms"

    # Validate input file
    if not validate_directories_file(directories_file):
        return

    try:
        # Option 1: Generate sample PDFs for testing
        print("\n=== GENERATING SAMPLE PDFs ===")
        sample_dir = os.path.join(output_base_dir, "samples")
        generate_sample_pdfs(
            directories_file=directories_file,
            output_dir=sample_dir,
            n_sample=3
        )

        # Option 2: Generate all anonymous PDFs (commented out for safety)
        # print("\n=== GENERATING ALL ANONYMOUS PDFs ===")
        # generate_foster_bird_pdfs(
        #     directories_file=directories_file,
        #     output_dir=os.path.join(output_base_dir, "all_anonymous"),
        #     anonymize=True
        # )

        # Option 3: Generate all identified PDFs (commented out for safety)
        # print("\n=== GENERATING ALL IDENTIFIED PDFs ===")
        # generate_foster_bird_pdfs(
        #     directories_file=directories_file,
        #     output_dir=os.path.join(output_base_dir, "all_identified"),
        #     anonymize=False
        # )

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# Updated main execution block
if __name__ == '__main__':
    # Setup logging for all operations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 1. Debug parsing (always safe to run first)
    print("=== DEBUGGING DIRECTORY PARSING ===")
    debug_parse_foster_directories("xfoster_directories.txt")

    # 2. Test a single bird (safe for testing)
    # Uncomment the next few lines to test a specific bird
    # print("\n=== TESTING SINGLE BIRD ===")
    # bird_directories = parse_foster_directories("xfoster_directories.txt")
    # if bird_directories:
    #     first_bird = list(bird_directories.keys())[0]
    #     test_single_bird("xfoster_directories.txt", first_bird, "single_bird_test")

    # 3. Generate sample PDFs (safe, limited output)
    print("\n=== GENERATING SAMPLE PDFs ===")
    generate_sample_pdfs(
        directories_file="xfoster_directories.txt",
        output_dir="foster_bird_spectrograms_samples",
        n_sample=10
    )

    # 4. Full generation (uncomment when ready for production run)
    # print("\n=== FULL GENERATION ===")
    # main()

    print("\n=== SCRIPT EXECUTION COMPLETE ===")
    print("Check the output directories for generated files.")
    print("Review the logs for any errors or warnings.")