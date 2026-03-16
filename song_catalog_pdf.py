import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import tables
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from PIL import Image

from tools.song_io import get_song_spec, rms_norm, butter_bandpass_filter_sos
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file
from tools.system_utils import replace_macaw_root
from phenotype_pdfs import create_dual_labeled_spectrogram


@dataclass
class CatalogConfig:
    """Configuration for bird catalog PDF generation."""
    n_spectrograms: int = 30  # Reduced from 100
    spectrograms_per_page: int = 4
    include_manual_labels: bool = True
    include_automated_labels: bool = True
    overwrite_spectrograms: bool = False
    duration: float = 6.0
    page_margin: int = 50
    image_height: int = 150


class BirdCatalogPDFGenerator:
    def __init__(self, bird_path: str, config: CatalogConfig = None):
        self.bird_path = Path(bird_path)
        self.bird_name = self.bird_path.name
        self.config = config or CatalogConfig()

        # Directory paths
        self.syllable_dir = self.bird_path / 'syllable_data' / 'specs'
        self.spectrograms_dir = self.bird_path / 'spectrograms' / 'labelled'
        self.pdf_output_dir = self.bird_path / 'syllable_data' / 'pdfs'

        # Create directories
        self.spectrograms_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)

        # Add database caching
        self.syllable_db = None
        self._load_syllable_database()

    def _load_syllable_database(self):
        """Load and cache the syllable database once."""
        try:
            syllable_db_path = self.bird_path / 'syllable_data' / 'syllable_database' / 'syllable_features.csv'
            if syllable_db_path.exists():
                self.syllable_db = pd.read_csv(syllable_db_path)
                logging.info(f"Loaded syllable database: {len(self.syllable_db)} rows")
            else:
                logging.warning(f"Syllable database not found: {syllable_db_path}")
                self.syllable_db = None
        except Exception as e:
            logging.error(f"Error loading syllable database: {e}")
            self.syllable_db = None

    def generate_catalog_pdf(self, rank: int = 0, overwrite: bool = True) -> str:
        """
        Generate catalog PDF for specified clustering rank (defaults to rank 0).
        """
        try:
            output_path = self.pdf_output_dir / f'{self.bird_name}_catalog_rank{rank}.pdf'

            if output_path.exists() and not overwrite:
                logging.info(f"Catalog PDF already exists: {output_path}")
                return str(output_path)

            # Get syllable files with chronological sorting
            syllable_files_with_times = self._get_chronologically_sorted_files()

            if not syllable_files_with_times:
                logging.warning(f"No syllable files found for {self.bird_name}")
                return ""

            # Limit to requested number of spectrograms
            syllable_files_with_times = syllable_files_with_times[:self.config.n_spectrograms]

            # Generate or reuse spectrograms using shared function
            spectrogram_paths = []
            for syl_file, timestamp_info in syllable_files_with_times:
                spec_path = create_dual_labeled_spectrogram(
                    syl_file=syl_file,
                    bird_path=self.bird_path,
                    rank=rank,
                    spectrograms_dir=self.spectrograms_dir,
                    overwrite=self.config.overwrite_spectrograms,
                    duration=self.config.duration,
                    syllable_db=self.syllable_db
                )
                if spec_path:
                    spectrogram_paths.append((spec_path, timestamp_info))

            if not spectrogram_paths:
                logging.warning(f"No spectrograms generated for {self.bird_name}")
                return ""

            # Create PDF
            self._create_pdf(spectrogram_paths, output_path, rank)

            logging.info(f"Generated catalog PDF: {output_path} ({len(spectrogram_paths)} spectrograms)")
            return str(output_path)

        except Exception as e:
            logging.error(f"Error generating catalog PDF for rank {rank}: {e}")
            return ""

    def generate_all_rank_catalogs(self, max_ranks: int = 1, overwrite: bool = True) -> Dict[int, str]:
        """
        Generate catalog PDFs for clustering ranks (defaults to rank 0 only).
        """
        generated_pdfs = {}

        # Default to rank 0 only for efficiency
        if max_ranks == 1:
            try:
                pdf_path = self.generate_catalog_pdf(rank=0, overwrite=overwrite)
                if pdf_path:
                    generated_pdfs[0] = pdf_path
                    logging.info(f"Generated catalog for rank 0: {pdf_path}")
            except Exception as e:
                logging.error(f"Error generating catalog for rank 0: {e}")

            return generated_pdfs

        # Multi-rank processing if specifically requested
        available_ranks = self._get_available_ranks()
        if not available_ranks:
            logging.warning(f"No clustering ranks found for {self.bird_name}")
            return {}

        for rank in available_ranks[:max_ranks]:
            try:
                pdf_path = self.generate_catalog_pdf(rank=rank, overwrite=overwrite)
                if pdf_path:
                    generated_pdfs[rank] = pdf_path
                    logging.info(f"Generated catalog for rank {rank}: {pdf_path}")
            except Exception as e:
                logging.error(f"Error generating catalog for rank {rank}: {e}")
                continue

        return generated_pdfs

    def _create_pdf(self, spectrogram_data: List[Tuple[str, Dict[str, Any]]],
                    output_path: Path, rank: int):
        """
        Create PDF with spectrograms using ReportLab.

        Args:
            spectrogram_data: List of (spectrogram_path, timestamp_info) tuples
            output_path: Path where PDF should be saved
            rank: Clustering rank for title
        """
        try:
            # Create PDF canvas
            c = canvas.Canvas(str(output_path), pagesize=landscape(letter))
            width, height = landscape(letter)

            # Layout parameters
            margin = self.config.page_margin
            img_width = width - 2 * margin
            img_height = self.config.image_height
            images_per_page = self.config.spectrograms_per_page
            y_position = height - margin

            # Add title page
            self._add_title_page(c, width, height, rank, len(spectrogram_data))
            c.showPage()

            # Reset position for content pages
            y_position = height - margin

            for i, (spec_path, timestamp_info) in enumerate(spectrogram_data):
                # Start new page if needed
                if i % images_per_page == 0 and i > 0:
                    c.showPage()
                    y_position = height - margin

                try:
                    # Create metadata text
                    if timestamp_info['datetime']:
                        text = (f"Bird: {timestamp_info['bird_name']} | "
                                f"Date: {timestamp_info['formatted_date']} | "
                                f"Time: {timestamp_info['formatted_time']} | "
                                f"Rank: {rank}")
                    else:
                        text = f"Bird: {timestamp_info['bird_name']} | Rank: {rank} | File: {Path(spec_path).stem}"

                    # Load and add image
                    if os.path.exists(spec_path):
                        img = Image.open(spec_path)
                        aspect = img.width / img.height

                        # Calculate image dimensions while maintaining aspect ratio
                        if aspect > img_width / img_height:
                            actual_width = img_width
                            actual_height = img_width / aspect
                        else:
                            actual_height = img_height
                            actual_width = img_height * aspect

                        # Draw image
                        c.drawImage(spec_path, margin, y_position - actual_height,
                                    width=actual_width, height=actual_height)

                        # Add metadata text below image
                        c.setFont("Helvetica", 10)
                        c.drawString(margin, y_position - actual_height - 15, text)

                        # Update y_position for next image
                        y_position -= (actual_height + 40)
                    else:
                        logging.warning(f"Spectrogram file not found: {spec_path}")

                except Exception as e:
                    logging.error(f"Error adding spectrogram {spec_path} to PDF: {e}")
                    continue

            # Save PDF
            c.save()
            logging.info(f"PDF created successfully at {output_path}")

        except Exception as e:
            logging.error(f"Error creating PDF: {e}")
            raise

    def _add_title_page(self, canvas_obj: canvas.Canvas, width: float, height: float,
                        rank: int, n_spectrograms: int):
        """Add title page with catalog information and legend."""
        try:
            # Main title
            canvas_obj.setFont("Helvetica-Bold", 24)
            title = f"{self.bird_name} - Song Catalog (Rank {rank})"
            title_width = canvas_obj.stringWidth(title, "Helvetica-Bold", 24)
            canvas_obj.drawString((width - title_width) / 2, height - 100, title)

            # Subtitle
            canvas_obj.setFont("Helvetica", 16)
            subtitle = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            subtitle_width = canvas_obj.stringWidth(subtitle, "Helvetica", 16)
            canvas_obj.drawString((width - subtitle_width) / 2, height - 140, subtitle)

            # Statistics
            canvas_obj.setFont("Helvetica", 14)
            stats_text = [
                f"Total Spectrograms: {n_spectrograms}",
                f"Clustering Rank: {rank}",
                f"Duration per Spectrogram: {self.config.duration}s",
                f"Spectrograms per Page: {self.config.spectrograms_per_page}"
            ]

            y_pos = height - 200
            for stat in stats_text:
                stat_width = canvas_obj.stringWidth(stat, "Helvetica", 14)
                canvas_obj.drawString((width - stat_width) / 2, y_pos, stat)
                y_pos -= 25

            # Legend section
            canvas_obj.setFont("Helvetica-Bold", 16)
            legend_title = "Label Legend"
            legend_width = canvas_obj.stringWidth(legend_title, "Helvetica-Bold", 16)
            canvas_obj.drawString((width - legend_width) / 2, y_pos - 40, legend_title)

            canvas_obj.setFont("Helvetica", 12)
            legend_text = [
                "• Manual labels appear above spectrograms",
                "• Automated labels appear below spectrograms",
                "• Colors correspond to syllable types",
                "• Spectrograms are sorted chronologically by recording time"
            ]

            y_pos -= 80
            for legend_item in legend_text:
                legend_item_width = canvas_obj.stringWidth(legend_item, "Helvetica", 12)
                canvas_obj.drawString((width - legend_item_width) / 2, y_pos, legend_item)
                y_pos -= 20

            # Footer
            canvas_obj.setFont("Helvetica", 10)
            footer = f"Bird Path: {self.bird_path}"
            canvas_obj.drawString(50, 50, footer)

        except Exception as e:
            logging.error(f"Error creating title page: {e}")

    def _get_chronologically_sorted_files(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Get syllable files sorted chronologically by extracting timestamps from audio filenames.

        Returns:
            List of (syllable_file_path, timestamp_info) tuples sorted by time
        """
        if not self.syllable_dir.exists():
            return []

        syllable_files = list(self.syllable_dir.glob('*.h5'))
        if not syllable_files:
            return []

        files_with_timestamps = []

        for syl_file in syllable_files:
            try:
                timestamp_info = self._extract_timestamp_from_syllable_file(syl_file)
                if timestamp_info:
                    files_with_timestamps.append((syl_file, timestamp_info))
            except Exception as e:
                logging.debug(f"Could not extract timestamp from {syl_file}: {e}")
                # Add file without timestamp info for inclusion at end
                files_with_timestamps.append((syl_file, {
                    'datetime': None,
                    'formatted_date': 'Unknown',
                    'formatted_time': 'Unknown',
                    'bird_name': self.bird_name
                }))

        # Sort by datetime, putting None values at the end
        files_with_timestamps.sort(key=lambda x: x[1]['datetime'] if x[1]['datetime'] else datetime.max)

        return files_with_timestamps

    def _extract_timestamp_from_syllable_file(self, syl_file: Path) -> Optional[Dict[str, Any]]:
        """
        Extract timestamp information from syllable file's audio filename.

        Returns:
            Dictionary with datetime and formatted strings, or None if extraction fails
        """
        try:
            with tables.open_file(str(syl_file), 'r') as f:
                audio_filename_raw = f.root.audio_filename.read()
                if isinstance(audio_filename_raw[0], bytes):
                    audio_filename = audio_filename_raw[0].decode('utf-8')
                else:
                    audio_filename = str(audio_filename_raw[0])

            # Extract filename without path and extension
            audio_basename = Path(audio_filename).stem

            # Try different timestamp patterns
            timestamp_info = self._parse_timestamp_patterns(audio_basename)

            return timestamp_info

        except Exception as e:
            logging.debug(f"Error extracting timestamp from {syl_file}: {e}")
            return None

    def _parse_timestamp_patterns(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse various timestamp patterns from audio filenames.

        Handles patterns like:
        - bird_YYYYMMDD_HHMMSS
        - bird_DDMMYY_HHMMSS
        - YYYYMMDD_HHMMSS_bird
        - And other common variations

        Returns:
            Dictionary with parsed datetime and formatted strings
        """
        # Common patterns to try
        patterns = [
            # bird_YYYYMMDD_HHMMSS
            r'(\w+)_(\d{8})_(\d{6})',
            # bird_DDMMYY_HHMMSS
            r'(\w+)_(\d{6})_(\d{6})',
            # YYYYMMDD_HHMMSS_bird
            r'(\d{8})_(\d{6})_(\w+)',
            # DDMMYY_HHMMSS_bird
            r'(\d{6})_(\d{6})_(\w+)',
            # bird_YYYYMMDD-HHMMSS (with dash)
            r'(\w+)_(\d{8})-(\d{6})',
            # bird_DDMMYY-HHMMSS
            r'(\w+)_(\d{6})-(\d{6})',
            # TODO: add date format name.date-int (int starts at 2, idx 1/0 is unmarked)
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()

                # Determine which group contains what based on pattern
                if pattern.startswith(r'(\d'):  # Date first patterns
                    date_str, time_str, bird_name = groups
                else:  # Bird first patterns
                    bird_name, date_str, time_str = groups

                try:
                    # Parse date
                    date_format = self._detect_date_format(date_str)
                    date_obj = datetime.strptime(date_str, date_format)

                    # Parse time
                    time_str = self._sanitize_time_str(time_str)
                    time_obj = datetime.strptime(time_str, "%H%M%S")

                    # Combine date and time
                    full_datetime = datetime.combine(date_obj.date(), time_obj.time())

                    return {
                        'datetime': full_datetime,
                        'formatted_date': date_obj.strftime('%Y-%m-%d'),
                        'formatted_time': time_obj.strftime('%H:%M:%S'),
                        'bird_name': bird_name
                    }

                except ValueError as e:
                    logging.debug(f"Could not parse timestamp from {filename}: {e}")
                    continue

        return None

    def _detect_date_format(self, date_str: str) -> str:
        """
        Detect whether a date string uses "%d%m%y" or "%Y%m%d" format.
        Returns the correct format string.
        """
        current_year = datetime.now().year

        # Try YYYYMMDD format
        if len(date_str) == 8:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])

            if 1900 <= year <= current_year and self._is_valid_date(year, month, day):
                return "%Y%m%d"

        # Try DDMMYY format
        elif len(date_str) == 6:
            day = int(date_str[:2])
            month = int(date_str[2:4])
            year = int(date_str[4:6])

            # Convert 2-digit year to 4-digit year
            if year < 50:  # Assuming years 00-49 are 2000-2049
                year += 2000
            else:  # Assuming years 50-99 are 1950-1999
                year += 1900

            if self._is_valid_date(year, month, day):
                return "%d%m%y"

        raise ValueError(f"Invalid date format: {date_str}")

    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        """Check if a date is valid, including month lengths and leap years."""
        try:
            datetime(year, month, day)
            return True
        except ValueError:
            return False

    def _sanitize_time_str(self, time_str: str) -> str:
        """
        Sanitize time string by fixing common issues.

        Args:
            time_str: Raw time string (HHMMSS format expected)

        Returns:
            Sanitized time string
        """
        # Remove any non-digit characters
        time_str = re.sub(r'\D', '', time_str)

        # Ensure it's 6 digits
        if len(time_str) < 6:
            time_str = time_str.ljust(6, '0')
        elif len(time_str) > 6:
            time_str = time_str[:6]

        # Validate hour, minute, second ranges
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        # Fix invalid values
        if hour >= 24:
            hour = hour % 24
        if minute >= 60:
            minute = minute % 60
        if second >= 60:
            second = second % 60

        return f"{hour:02d}{minute:02d}{second:02d}"

    def _get_available_ranks(self) -> List[int]:
        """
        Get list of available clustering ranks from syllable database.

        Returns:
            Sorted list of available rank numbers
        """
        try:
            syllable_db_path = self.bird_path / 'data' / 'syllable_database' / 'syllable_features.csv'
            if not syllable_db_path.exists():
                logging.warning(f"Syllable database not found: {syllable_db_path}")
                return []

            df = pd.read_csv(syllable_db_path)

            # Find all cluster columns
            cluster_cols = [col for col in df.columns if col.startswith('cluster_rank')]

            # Extract rank numbers
            ranks = []
            for col in cluster_cols:
                # Extract rank number from column name like 'cluster_rank0_...'
                match = re.search(r'cluster_rank(\d+)_', col)
                if match:
                    rank_num = int(match.group(1))
                    if rank_num not in ranks:
                        ranks.append(rank_num)

            return sorted(ranks)

        except Exception as e:
            logging.error(f"Error getting available ranks: {e}")
            return []


def generate_bird_catalogs(bird_path: str, config: CatalogConfig = None,
                           max_ranks: int = 1, overwrite: bool = True) -> Dict[int, str]:
    """
    Convenience function to generate catalog PDFs for a bird (defaults to rank 0 only).
    """
    try:
        generator = BirdCatalogPDFGenerator(bird_path, config)
        return generator.generate_all_rank_catalogs(max_ranks=max_ranks, overwrite=overwrite)
    except Exception as e:
        logging.error(f"Error generating catalogs for {bird_path}: {e}")
        return {}


def batch_generate_catalogs(bird_paths: List[str], config: CatalogConfig = None,
                            max_ranks: int = 1, overwrite: bool = True) -> Dict[str, Dict[int, str]]:
    """
    Generate catalog PDFs for multiple birds (defaults to rank 0 only).
    """
    all_catalogs = {}

    for bird_path in bird_paths:
        bird_name = os.path.basename(bird_path)

        try:
            bird_catalogs = generate_bird_catalogs(
                bird_path=bird_path,
                config=config,
                max_ranks=max_ranks,
                overwrite=overwrite
            )

            if bird_catalogs:
                all_catalogs[bird_name] = bird_catalogs
                logging.info(f"Generated {len(bird_catalogs)} catalogs for {bird_name}")
            else:
                logging.warning(f"No catalogs generated for {bird_name}")

        except Exception as e:
            logging.error(f"Error generating catalogs for {bird_name}: {e}")
            continue

    return all_catalogs


def get_bird_list(project_directory: str) -> List[str]:
    """
    Get list of valid bird directories from project directory.

    Args:
        project_directory: Path to project root directory

    Returns:
        List of bird directory names
    """
    try:
        if not os.path.exists(project_directory):
            return []

        all_items = os.listdir(project_directory)
        birds = []
        if 'copied_data' in all_items: all_items.remove('copied_data')

        for item in all_items:
            item_path = os.path.join(project_directory, item)

            # Check if it's a valid bird directory
            if (os.path.isdir(item_path) and
                    not item.startswith('.') and
                    not item.endswith('.png') and
                    not item.endswith('.jpg') and
                    not item.endswith('.csv')):

                # Verify it has syllable data
                syllable_dir = os.path.join(item_path, 'syllable_data', 'specs')
                if os.path.exists(syllable_dir):
                    syllable_files = [f for f in os.listdir(syllable_dir) if f.endswith('.h5')]
                    if syllable_files:
                        birds.append(item)

        return sorted(birds)

    except Exception as e:
        logging.error(f"Error getting bird list from {project_directory}: {e}")
        return []


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test on example datasets
    test_paths = [
        # os.path.join('/Volumes', 'Extreme SSD', 'wseg test'),
        # os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
        os.path.join('E:', 'xfosters')
    ]

    for dataset_path in test_paths:
        if os.path.exists(dataset_path):
            try:
                print(f"\nGenerating Bird Catalog PDFs for {os.path.basename(dataset_path)}...")

                birds = get_bird_list(dataset_path)   # Get available birds
                if not birds:
                    print(f"No birds found in {dataset_path}")
                    continue

                print(f"Found {len(birds)} birds: {birds[:3]}...")  # Show first 3

                for bird in birds:
                    bird_path = os.path.join(dataset_path, bird)

                    print(f"\nGenerating catalog for {bird}...")

                    # Create config for testing
                    config = CatalogConfig(
                        n_spectrograms=20,  # Small number for testing
                        spectrograms_per_page=2,
                        overwrite_spectrograms=True  # Reuse existing spectrograms
                    )

                    # Generate catalog (rank 0 only by default)
                    catalogs = generate_bird_catalogs(
                        bird_path=bird_path,
                        config=config,
                        max_ranks=1,  # Only rank 0
                        overwrite=True
                    )

                    if catalogs:
                        print(f"✓ Generated catalog PDF:")
                        for rank, pdf_path in catalogs.items():
                            print(f"  - Rank {rank}: {pdf_path}")
                    else:
                        print("✗ No catalog PDF generated")

            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
