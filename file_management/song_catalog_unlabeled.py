import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import librosa
import soundfile as sf
import random
from scipy import signal
from tools.song_io import get_song_spec
from tools.spectrogram_configs import SpectrogramParams

# Import your existing modules
from tools.evfuncs import load_cbin, readrecf
from tools.logging_utils import setup_logger

logger = setup_logger(__name__, 'bird_spectrogram_generator.log')


class BirdSpectrogramGenerator:
    def __init__(self, spec_params=None, n_spectrograms: int = 30, anonymize: bool = False, duration: int = 6):
        # Create default params if none provided
        if spec_params is None:
            spec_params = SpectrogramParams(max_dur=duration, downsample=False)

        self.spec_params = spec_params
        self.n_spectrograms = n_spectrograms
        self.anonymize = anonymize
        self.anonymization_map = {}

        # Extract what you need from spec_params
        self.duration = spec_params.max_dur
        self.freq_range = (spec_params.min_freq, spec_params.max_freq)

    def _get_anonymous_id(self, bird_name: str) -> str:
        """Get consistent anonymous ID for a bird name."""
        if bird_name not in self.anonymization_map:
            id_num = len(self.anonymization_map) + 1
            self.anonymization_map[bird_name] = f"Bird{id_num}"  # Bird1, Bird2, etc.
        return self.anonymization_map[bird_name]

    def _format_bird_name(self, bird_name: str) -> str:
        """Return bird name or anonymous ID based on """
        return self._get_anonymous_id(bird_name) if self.anonymize else bird_name

    def find_audio_files(self, directory_path: str) -> List[str]:
        """Find all audio files including those without extensions."""
        if not os.path.exists(directory_path):
            return []

        audio_files = []
        audio_extensions = ['.wav', '.WAV', '.cbin', '.rec']

        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Check for files with audio extensions
                    if any(file.endswith(ext) for ext in audio_extensions):
                        audio_files.append(file_path)

                    # Check for files matching bird naming pattern without extension
                    # Pattern: bird.YYYYMMDD.NNNN (like pk46bk46.20080610.2115)
                    elif re.match(r'^[a-z]{2}\d+[a-z]{2}\d*\.\d{8}\.\d+$', file):
                        # Verify it's actually an audio file by checking if .cbin exists
                        potential_cbin = file_path + '.cbin'
                        if os.path.exists(potential_cbin):
                            audio_files.append(potential_cbin)
                        else:
                            # Assume it's a .wav file
                            audio_files.append(file_path)

        except Exception as e:
            logger.error(f"Error searching directory {directory_path}: {e}")

        return audio_files

    def load_audio_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file using appropriate method based on extension."""
        try:
            if file_path.endswith('.cbin'):
                # Use your evfuncs module
                audio, fs = load_cbin(file_path)
                audio = audio.astype(np.float32) / 32768.0  # Convert to float
                return audio, fs  # Remove resampling
            else:
                # Try librosa first - load at native sample rate
                audio, fs = librosa.load(file_path, sr=None)  # fs=None keeps native rate
                return audio, fs

        except Exception as e:
            try:
                # Fallback to soundfile
                audio, fs = sf.read(file_path)
                return audio, fs  # Remove resampling
            except Exception as e2:
                logger.warning(f"Could not read audio file {file_path}: {e2}")
                return None, None

    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse filename to extract bird, date, and time information."""
        patterns = [
            # Pattern 1: bird_YYMMDD_HHMMSS.wav
            (r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)_(\d{6})_(\d{6})\.(?:wav|cbin)$', 'timestamp'),
            # Pattern 2: bird.YYYYMMDD-SEQ.wav
            (r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})-(\d+)\.(?:wav|cbin)$', 'sequence'),
            # Pattern 3: bird.YYYYMMDD.wav
            (r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})\.(?:wav|cbin)$', 'date_only'),
            # Pattern 4: bird.YYYYMMDD.NNNN (no extension)
            (r'^([a-zA-Z]+\d+[a-zA-Z]*\d*)\.(\d{8})\.(\d+)$', 'sequence_no_ext'),
        ]

        for pattern, pattern_type in patterns:
            match = re.match(pattern, filename)
            if match:
                groups = match.groups()
                bird = groups[0]

                if pattern_type == 'timestamp':
                    day = '20' + groups[1] if len(groups[1]) == 6 else groups[1]
                    time = groups[2]
                elif pattern_type in ['sequence', 'sequence_no_ext']:
                    day = groups[1]
                    time = groups[2].zfill(6)
                else:  # date_only
                    day = groups[1]
                    time = '000000'

                return {
                    'bird': bird, 'day': day, 'time': time,
                    'filename': filename, 'success': True
                }

        logger.warning(f'Unrecognized filename format: {filename}')
        return {'bird': None, 'day': None, 'time': None, 'filename': filename, 'success': False}

    def extract_timestamp(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from filename."""
        parsed = self.parse_filename(filename)
        if not parsed['success']:
            return None

        try:
            day = parsed['day']
            time = parsed['time']

            # Handle different date formats
            if len(day) == 8:  # YYYYMMDD
                date_str = f"{day}{time}"
                return datetime.strptime(date_str, "%Y%m%d%H%M%S")
            elif len(day) == 6:  # YYMMDD
                date_str = f"20{day}{time}"
                return datetime.strptime(date_str, "%Y%m%d%H%M%S")
        except ValueError as e:
            logger.debug(f"Could not parse timestamp from {filename}: {e}")

        return None

    def create_spectrogram_plot(self, audio: np.ndarray, sr: int, title: str = "") -> Optional[plt.Figure]:
        """Create spectrogram plot using your existing get_song_spec function."""
        try:

            params = self.spec_params

            # Limit audio duration
            max_samples = int(self.duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Get spectrogram using your function
            spec, audio_segment, t = get_song_spec(
                0.0, len(audio) / sr, audio, params, fs=sr
            )

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Display spectrogram
            im = ax.imshow(
                spec,
                aspect='auto',
                origin='lower',
                extent=[0, self.duration, self.freq_range[0], self.freq_range[1]],
                cmap='plasma'
            )

            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_ylim(self.freq_range)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Normalized Power', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating spectrogram for {title}: {e}")
            return None

    def create_title_page(self, bird_name: str, stats: Dict, pdf: PdfPages) -> None:
        """Create title page with summary statistics."""
        fig, ax = plt.subplots(figsize=(11, 8.5))

        display_name = self._format_bird_name(bird_name)

        # Main title
        ax.text(0.5, 0.85, f"{display_name} - Song Spectrograms",
                ha='center', va='center', fontsize=24, fontweight='bold')

        # Anonymization notice
        if self.anonymize:
            ax.text(0.5, 0.75, "ANONYMIZED VERSION",
                    ha='center', va='center', fontsize=18, fontweight='bold',
                    color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

        # Statistics
        y_pos = 0.6
        stats_text = [
            f"Total spectrograms: {stats['total_files']}",
            f"Duration per spectrogram: {self.duration}s",
            f"Files with timestamps: {stats['files_with_timestamps']}"
        ]

        for line in stats_text:
            ax.text(0.5, y_pos, line, ha='center', va='center', fontsize=12)
            y_pos -= 0.05

        # Generation info
        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.5, 0.15, f"Generated: {generation_time}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def generate_bird_pdf(self, bird_name: str, directories: List[str], output_dir: str = ".") -> str:
        """Generate PDF with spectrograms for a single bird."""
        # Collect all audio files
        all_audio_files = []
        for directory in directories:
            audio_files = self.find_audio_files(directory)
            all_audio_files.extend(audio_files)

        if not all_audio_files:
            logger.warning(f"No audio files found for {bird_name}")
            return ""

        # Remove duplicates and sort
        unique_files = list(dict.fromkeys(all_audio_files))  # Preserves order

        # Extract timestamps and sort
        files_with_timestamps = []
        for file_path in unique_files:
            filename = os.path.basename(file_path)
            timestamp = self.extract_timestamp(filename)
            files_with_timestamps.append((file_path, timestamp))

        # Sort by timestamp or randomize if anonymizing
        if self.anonymize:
            import random
            random.shuffle(files_with_timestamps)
        else:
            files_with_timestamps.sort(key=lambda x: x[1] if x[1] else datetime.max)

        # Limit to requested number
        files_with_timestamps = files_with_timestamps[:self.n_spectrograms]

        # Generate statistics
        stats = {
            'total_files': len(files_with_timestamps),
            'files_with_timestamps': sum(1 for _, ts in files_with_timestamps if ts is not None)
        }

        # Create output filename
        display_name = self._format_bird_name(bird_name)
        suffix = "_anonymous" if self.anonymize else ""
        output_filename = f"{display_name}_spectrograms{suffix}.pdf"
        output_path = os.path.join(output_dir, output_filename)

        # Generate PDF
        with PdfPages(output_path) as pdf:
            # Title page
            self.create_title_page(bird_name, stats, pdf)

            # Generate spectrograms
            successful_count = 0
            for i, (file_path, timestamp) in enumerate(files_with_timestamps):
                try:
                    # Load audio
                    audio, sr = self.load_audio_file(file_path)
                    if audio is None:
                        continue

                    # Create title
                    if self.anonymize:
                        title = f"{display_name} - Spectrogram {i + 1}\nDate/Time: [Hidden]"
                    else:
                        if timestamp:
                            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            title = f"{bird_name} - Spectrogram {i + 1}\n{time_str}"
                        else:
                            title = f"{bird_name} - Spectrogram {i + 1}\nFile: {os.path.basename(file_path)}"

                    # Create spectrogram
                    fig = self.create_spectrogram_plot(audio, sr, title)
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        successful_count += 1

                        if successful_count % 10 == 0:
                            logger.info(f"Generated {successful_count} spectrograms for {display_name}")

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

    def generate_anonymization_key(self, output_dir: str) -> str:
        """Generate CSV file mapping real bird names to anonymous IDs."""
        if not self.anonymization_map:
            return ""

        key_filename = "anonymization_key.csv"
        key_path = os.path.join(output_dir, key_filename)

        try:
            with open(key_path, 'w', newline='') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(['Real_Bird_Name', 'Anonymous_ID'])

                # Sort by anonymous ID for cleaner output
                sorted_items = sorted(self.anonymization_map.items(), key=lambda x: x[1])
                for real_name, anonymous_id in sorted_items:
                    writer.writerow([real_name, anonymous_id])

            logger.info(f"Generated anonymization key: {key_path}")
            return key_path

        except Exception as e:
            logger.error(f"Error generating anonymization key: {e}")
            return ""

        logger.info(f"Generated PDF: {output_path} with {successful_count} spectrograms")
        return output_path

def parse_directories_file(file_path: str) -> Dict[str, List[str]]:
    """Parse the directories file to extract bird directories."""
    bird_directories = {}

    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_bird = None
        current_directories = []

        for line in lines:
            original_line = line
            line = line.strip()

            if not line:
                continue

            # Check for bird name line
            bird_match = re.match(r'^([a-z]{2}\d+[a-z]{2}\d*): (\d+) unique directories', line)
            if bird_match:
                # Save previous bird
                if current_bird is not None:
                    bird_directories[current_bird] = list(current_directories)

                # Start new bird
                current_bird = bird_match.group(1)
                current_directories = []
                continue

            # Check for directory line
            elif original_line.startswith('  - '):
                if current_bird is not None:
                    dir_path = original_line[4:].strip()

                    # Convert network paths
                    if dir_path.startswith('\\\\macaw.ucsf.edu'):
                        dir_path = dir_path.replace('\\\\macaw.ucsf.edu\\users\\', 'Z:\\')

                    # Normalize paths
                    if dir_path.startswith('Z:'):
                        dir_path = dir_path.replace('/', '\\')

                    current_directories.append(dir_path)

        # Don't forget the last bird
        if current_bird is not None:
            bird_directories[current_bird] = list(current_directories)

    except Exception as e:
        logger.error(f"Error parsing directories file: {e}")
        return {}

    logger.info(f"Successfully parsed {len(bird_directories)} birds from directories file")
    return bird_directories

def generate_pdfs_for_birds(directories_file: str, output_dir: str,
                            anonymize: bool = False, n_birds: Optional[int] = None):
    """Generate PDFs for birds from directories file."""

    # Parse directories
    bird_directories = parse_directories_file(directories_file)
    if not bird_directories:
        logger.error("No bird directories found")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    generator = BirdSpectrogramGenerator(n_spectrograms=30, duration=6.0, anonymize=anonymize)

    # Limit number of birds if specified
    bird_names = list(bird_directories.keys())
    if n_birds:
        bird_names = bird_names[:n_birds]

    logger.info(f"Processing {len(bird_names)} birds (anonymize={anonymize})")

    successful_pdfs = []
    failed_birds = []

    for i, bird_name in enumerate(bird_names):
        try:
            logger.info(f"Processing bird {i + 1}/{len(bird_names)}: {bird_name}")

            directories = bird_directories[bird_name]
            pdf_path = generator.generate_bird_pdf(bird_name, directories, output_dir)

            if pdf_path:
                successful_pdfs.append(pdf_path)
                display_name = generator._format_bird_name(bird_name)
                logger.info(f"✓ Generated PDF for {display_name}: {pdf_path}")
            else:
                failed_birds.append(bird_name)
                logger.warning(f"✗ Failed to generate PDF for {bird_name}")

        except Exception as e:
            failed_birds.append(bird_name)
            logger.error(f"✗ Error processing {bird_name}: {e}")
            continue

        # Summary
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Successfully generated: {len(successful_pdfs)} PDFs")
        logger.info(f"Failed: {len(failed_birds)} birds")

        # Generate anonymization key if anonymizing
        if anonymize and generator.anonymization_map:
            key_path = generator.generate_anonymization_key(output_dir)
            if key_path:
                logger.info(f"Anonymization key saved to: {key_path}")

            logger.info(f"Anonymization mapping:")
            for original, anonymous in generator.anonymization_map.items():
                logger.info(f"  {original} -> {anonymous}")

def test_single_bird(directories_file: str, bird_name: str, output_dir: str = "test_output"):
    """Test spectrogram generation for a single bird."""

    # Parse directories
    bird_directories = parse_directories_file(directories_file)

    if bird_name not in bird_directories:
        logger.error(f"Bird {bird_name} not found in directories file")
        available_birds = list(bird_directories.keys())[:10]
        logger.info(f"Available birds: {', '.join(available_birds)}...")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Test both versions
    for anonymize in [False, True]:

        generator = BirdSpectrogramGenerator(n_spectrograms=5,  # Just a few for testing
            anonymize=anonymize)

        logger.info(f"Testing {bird_name} (anonymize={anonymize})")
        directories = bird_directories[bird_name]

        pdf_path = generator.generate_bird_pdf(bird_name, directories, output_dir)

        if pdf_path:
            logger.info(f"✓ Test successful: {pdf_path}")
        else:
            logger.error(f"✗ Test failed for {bird_name}")


def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    directories_file = "xfoster_directories.txt"
    output_base_dir = "foster_bird_spectrograms"

    # Validate input file
    if not os.path.exists(directories_file):
        logger.error(f"Directories file not found: {directories_file}")
        return

    try:
        # Generate sample PDFs for testing (3 birds)
        logger.info("=== GENERATING SAMPLE PDFs ===")
        sample_dir = os.path.join(output_base_dir, "samples")

        # # Generate identified versions
        # generate_pdfs_for_birds(
        #     directories_file=directories_file,
        #     output_dir=os.path.join(sample_dir, "identified"),
        #     anonymize=False,
        #     n_birds=3
        # )
        #
        # # Generate anonymous versions
        # generate_pdfs_for_birds(
        #     directories_file=directories_file,
        #     output_dir=os.path.join(sample_dir, "anonymous"),
        #     anonymize=True,
        #     n_birds=3
        # )

        # Uncomment below for full generation
        # logger.info("=== GENERATING ALL ANONYMOUS PDFs ===")
        generate_pdfs_for_birds(
            directories_file=directories_file,
            output_dir=os.path.join(output_base_dir, "all_anonymous"),
            anonymize=True
        )

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # For initial testing, run a single bird test
    print("=== TESTING SINGLE BIRD ===")

    # Parse directories to get first available bird
    bird_directories = parse_directories_file("xfoster_directories.txt")
    if bird_directories:
        first_bird = list(bird_directories.keys())[0]
        print(f"Testing with bird: {first_bird}")
        test_single_bird("xfoster_directories.txt", first_bird, "single_bird_test")

    # Uncomment below to run full sample generation
    print("\n=== RUNNING MAIN SAMPLE GENERATION ===")
    main()

    print("\n=== SCRIPT EXECUTION COMPLETE ===")