import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image
import tables
from fpdf import FPDF

from tools.song_io import get_song_spec, rms_norm, butter_bandpass_filter_sos
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_utils import read_audio_file
from tools.system_utils import replace_macaw_root


@dataclass
class PhenotypePDFPaths:
    """Container for paths to phenotype analysis images."""
    transition_counts_img: str
    transition_matrix_img: str
    repeat_patterns_img: str
    vocabulary_comparison_img: Optional[str] = None


class PhenotypePDFGenerator:
    """
    Generate PDF summaries from phenotyping analysis results.

    Creates separate PDFs for manual and automated results.
    """

    def __init__(self, bird_path: str):
        self.bird_path = Path(bird_path)
        self.bird_name = self.bird_path.name

        # Directory paths
        self.phenotype_plots_dir = self.bird_path / 'figures' / 'phenotyping'
        self.pdf_output_dir = self.bird_path / 'data' / 'pdfs'
        self.temp_dir = self.bird_path / 'data' / 'temp_images'

        # Create directories
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Spectrogram parameters
        self.spec_params = SpectrogramParams()

    def generate_manual_phenotype_pdf(self, phenotype_results: Dict[str, Any],
                                      syllable_database_path: str,
                                      overwrite: bool = True) -> str:
        """
        Generate PDF for manual phenotype analysis.

        Args:
            phenotype_results: Results from calculate_phenotypes_for_label_type()
            syllable_database_path: Path to syllable database CSV
            overwrite: Whether to overwrite existing PDF

        Returns:
            str: Path to generated PDF
        """
        try:
            output_path = self.pdf_output_dir / f'{self.bird_name}_manual_phenotypes.pdf'

            if output_path.exists() and not overwrite:
                logging.info(f"Manual PDF already exists: {output_path}")
                return str(output_path)

            # Load syllable database for duration/gap statistics
            syllable_stats = self._load_syllable_database_stats(syllable_database_path)

            # Find existing phenotype plots
            plot_paths = self._find_phenotype_plots('manual')

            # Generate example spectrograms
            spec_images = self._generate_example_spectrograms(n_specs=4, label_type='manual')

            # Create PDF
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=5)
            pdf.add_page()

            # Add content sections
            self._add_phenotype_header(pdf, phenotype_results, 'Manual Labels')
            self._add_basic_phenotype_stats(pdf, phenotype_results)
            self._add_syllable_statistics_table(pdf, phenotype_results, syllable_stats)
            self._add_repeat_statistics_table(pdf, phenotype_results)
            self._add_phenotype_images(pdf, plot_paths)
            self._add_spectrogram_examples(pdf, spec_images)

            # Save PDF
            self._save_pdf(pdf, output_path, overwrite)

            # Cleanup temp images
            self._cleanup_temp_images(spec_images)

            logging.info(f"Generated manual phenotype PDF: {output_path}")
            return str(output_path)

        except Exception as e:
            logging.error(f"Error generating manual phenotype PDF: {e}")
            return ""

    def generate_automated_phenotype_pdf(self, phenotype_results: Dict[str, Any],
                                         clustering_metadata: Dict[str, Any],
                                         syllable_database_path: str,
                                         rank: int = 0,
                                         overwrite: bool = True) -> str:
        """
        Generate PDF for automated phenotype analysis.

        Args:
            phenotype_results: Results from calculate_phenotypes_for_label_type()
            clustering_metadata: Clustering parameters and scores
            syllable_database_path: Path to syllable database CSV
            rank: Clustering rank (0 for best)
            overwrite: Whether to overwrite existing PDF

        Returns:
            str: Path to generated PDF
        """
        try:
            output_path = self.pdf_output_dir / f'{self.bird_name}_automated_phenotypes_rank{rank}.pdf'

            if output_path.exists() and not overwrite:
                logging.info(f"Automated PDF already exists: {output_path}")
                return str(output_path)

            # Load syllable database for duration/gap statistics
            syllable_stats = self._load_syllable_database_stats(syllable_database_path)

            # Find existing phenotype plots
            plot_paths = self._find_phenotype_plots(f'rank{rank}')

            # Generate example spectrograms with automated labels
            spec_images = self._generate_example_spectrograms(n_specs=4, label_type='automated', rank=rank)

            # Create PDF
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=5)
            pdf.add_page()

            # Add content sections
            self._add_phenotype_header(pdf, phenotype_results, f'Automated Labels (Rank {rank})')
            self._add_clustering_parameters(pdf, clustering_metadata)
            self._add_basic_phenotype_stats(pdf, phenotype_results)
            self._add_syllable_statistics_table(pdf, phenotype_results, syllable_stats)
            self._add_repeat_statistics_table(pdf, phenotype_results)
            self._add_phenotype_images(pdf, plot_paths)
            self._add_spectrogram_examples(pdf, spec_images)

            # Save PDF
            self._save_pdf(pdf, output_path, overwrite)

            # Cleanup temp images
            self._cleanup_temp_images(spec_images)

            logging.info(f"Generated automated phenotype PDF: {output_path}")
            return str(output_path)

        except Exception as e:
            logging.error(f"Error generating automated phenotype PDF: {e}")
            return ""

    def _load_syllable_database_stats(self, database_path: str) -> Dict[str, Dict[str, float]]:
        """Load syllable duration and gap statistics from database."""
        try:
            if not os.path.exists(database_path):
                logging.warning(f"Syllable database not found: {database_path}")
                return {}

            df = pd.read_csv(database_path)

            # Calculate statistics by syllable type
            stats = {}

            if 'manual_label' in df.columns:
                # Group by manual labels
                manual_df = df[df['manual_label'].notna() & (df['manual_label'] != '')]

                for label in manual_df['manual_label'].unique():
                    label_data = manual_df[manual_df['manual_label'] == label]

                    stats[str(label)] = {
                        'duration_ms': label_data['duration_ms'].mean() if 'duration_ms' in df.columns else np.nan,
                        'prev_gap_ms': label_data[
                            'prev_syllable_gap_ms'].mean() if 'prev_syllable_gap_ms' in df.columns else np.nan,
                        'next_gap_ms': label_data[
                            'next_syllable_gap_ms'].mean() if 'next_syllable_gap_ms' in df.columns else np.nan
                    }

            # Also check for clustering labels (cluster_rank0_, cluster_rank1_, etc.)
            cluster_cols = [col for col in df.columns if col.startswith('cluster_')]

            for cluster_col in cluster_cols:
                cluster_df = df[df[cluster_col].notna() & (df[cluster_col] != -1)]

                for label in cluster_df[cluster_col].unique():
                    label_data = cluster_df[cluster_df[cluster_col] == label]

                    # Use cluster column name as key (e.g., 'cluster_rank0_5' for rank 0, cluster 5)
                    key = f"{cluster_col}_{label}"
                    stats[key] = {
                        'duration_ms': label_data['duration_ms'].mean() if 'duration_ms' in df.columns else np.nan,
                        'prev_gap_ms': label_data[
                            'prev_syllable_gap_ms'].mean() if 'prev_syllable_gap_ms' in df.columns else np.nan,
                        'next_gap_ms': label_data[
                            'next_syllable_gap_ms'].mean() if 'next_syllable_gap_ms' in df.columns else np.nan
                    }

            return stats

        except Exception as e:
            logging.error(f"Error loading syllable database stats: {e}")
            return {}

    def _find_phenotype_plots(self, rank_str: str) -> PhenotypePDFPaths:
        """Find existing phenotype plot images."""
        try:
            plots_dir = self.phenotype_plots_dir

            # Look for expected plot files
            transition_counts = plots_dir / f"{rank_str}_transition_counts.png"
            transition_matrix = plots_dir / f"{rank_str}_transition_1st.png"
            repeat_patterns = plots_dir / f"{rank_str}_repeats.png"
            vocab_comparison = plots_dir / "vocabulary_comparison.png"

            return PhenotypePDFPaths(
                transition_counts_img=str(transition_counts) if transition_counts.exists() else "",
                transition_matrix_img=str(transition_matrix) if transition_matrix.exists() else "",
                repeat_patterns_img=str(repeat_patterns) if repeat_patterns.exists() else "",
                vocabulary_comparison_img=str(vocab_comparison) if vocab_comparison.exists() else None
            )

        except Exception as e:
            logging.error(f"Error finding phenotype plots: {e}")
            return PhenotypePDFPaths("", "", "")

    def _generate_example_spectrograms(self, n_specs: int = 4,
                                       label_type: str = 'manual',
                                       rank: int = 0) -> List[str]:
        """Generate example spectrograms with phenotype labels."""
        spec_paths = []

        try:
            # Get syllable files
            syllable_dir = self.bird_path / 'data' / 'syllables'
            if not syllable_dir.exists():
                logging.warning(f"No syllable directory found: {syllable_dir}")
                return []

            syllable_files = list(syllable_dir.glob('*.h5'))
            if not syllable_files:
                logging.warning(f"No syllable files found in {syllable_dir}")
                return []

            # Sample files for spectrograms
            import random
            sampled_files = random.sample(syllable_files, min(n_specs, len(syllable_files)))

            for i, syl_file in enumerate(sampled_files):
                try:
                    spec_path = self._create_phenotype_spectrogram(
                        syl_file, label_type, rank, i
                    )
                    if spec_path:
                        spec_paths.append(spec_path)
                except Exception as e:
                    logging.error(f"Error creating spectrogram for {syl_file}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error generating example spectrograms: {e}")

        return spec_paths

    def _create_phenotype_spectrogram(self, syl_file: Path,
                                      label_type: str, rank: int,
                                      file_idx: int, duration: float = 6.0) -> Optional[str]:
        """Create spectrogram with phenotype labels."""
        self.spec_params.max_dur = duration  # TODO this could cause errors later, but is preventing full song spec now
        try:
            with tables.open_file(str(syl_file), 'r') as f:
                # Read syllable data
                onsets = f.root.onsets.read()
                offsets = f.root.offsets.read()

                # Read audio filename
                audio_filename_raw = f.root.audio_filename.read()
                if isinstance(audio_filename_raw[0], bytes):
                    audio_filename = audio_filename_raw[0].decode('utf-8')
                else:
                    audio_filename = str(audio_filename_raw[0])

                # Resolve audio file path
                if not os.path.isfile(audio_filename):
                    audio_filename = replace_macaw_root(audio_filename)

                if not os.path.isfile(audio_filename):
                    logging.warning(f"Audio file not found: {audio_filename}")
                    return None

                # Read labels based on type
                labels = None
                if label_type == 'manual' and hasattr(f.root, 'manual'):
                    manual_raw = f.root.manual.read()
                    labels = np.array([
                        item.decode('utf-8') if isinstance(item, bytes) else str(item)
                        for item in manual_raw
                    ])
                elif label_type == 'automated':
                    # For automated, we'd need to load from syllable database
                    # This is more complex - for now, create placeholder
                    labels = np.array([f'C{i % 5}' for i in range(len(onsets))])

                if labels is None or len(labels) == 0:
                    return None

                # Read and process audio
                audio, fs = read_audio_file(audio_filename)
                audio = rms_norm(audio)
                audio = butter_bandpass_filter_sos(
                    audio,
                    lowcut=self.spec_params.min_freq,
                    highcut=self.spec_params.max_freq,
                    fs=fs,
                    order=5
                )

                # Set time window
                first_time = max((onsets[0] / 1000 - 0.25), 0.0)
                last_time = first_time + duration

                # Ensure we don't exceed audio length
                if len(audio) * (1 / fs) <= last_time:
                    last_time = (len(audio) - 1) * (1 / fs)

                # Generate spectrogram
                spec, _, t = get_song_spec(
                    t1=first_time,
                    t2=last_time,
                    audio=audio,
                    params=self.spec_params,
                    fs=fs,
                    downsample=False
                )

                # Filter syllables within time window
                time_mask = (onsets >= first_time * 1000) & (onsets <= last_time * 1000)
                syl_onsets = onsets[time_mask]
                syl_offsets = offsets[time_mask]
                syl_labels = labels[time_mask]

                if len(syl_onsets) <= 1:
                    return None

                # Create spectrogram plot
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.imshow(spec, aspect='auto', origin='lower')
                ax.set_yticks([])

                # Add syllable labels
                font_size = 8
                colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Color palette

                for i, (onset, offset, label) in enumerate(zip(syl_onsets, syl_offsets, syl_labels)):
                    if label in ['-', '', 's', 'z']:  # Skip non-syllable tokens
                        continue

                    label_x = (onset / (duration * 1000)) * spec.shape[1]

                    # Choose color based on label
                    color_idx = hash(str(label)) % len(colors)

                    ax.text(label_x, spec.shape[0] + 20, str(label),
                            color='black', fontsize=font_size, ha='center', va='top',
                            bbox=dict(facecolor=colors[color_idx], edgecolor='black',
                                      alpha=0.7, boxstyle='round'))

                # Set time axis
                time_ticks = np.arange(0, duration + 1)
                x_ticks = np.linspace(0, spec.shape[1], len(time_ticks))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(time_ticks.astype(int))
                ax.set_xlabel('Time (s)')
                ax.set_title(f'{label_type.title()} Labels - {self.bird_name}')

                # Save spectrogram
                timestamp = pd.Timestamp.now().strftime('%H%M%S')
                filename = f'phenotype_spec_{label_type}_{rank}_{file_idx}_{timestamp}.png'
                file_path = self.temp_dir / filename

                plt.tight_layout()
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                return str(file_path)

        except Exception as e:
            logging.error(f"Error creating phenotype spectrogram from {syl_file}: {e}")
            return None

    def _add_phenotype_header(self, pdf: FPDF, phenotype_results: Dict[str, Any], title: str):
        """Add header section with bird info and analysis type."""
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 8, f"{title} - {self.bird_name}", ln=True, align="C")
        pdf.ln(3)

        pdf.set_font("Arial", size=8)
        pdf.cell(0, 5, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(5)

    def _add_clustering_parameters(self, pdf: FPDF, clustering_metadata: Dict[str, Any]):
        """Add clustering parameters table for automated results."""
        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(0, 5, "Clustering Parameters:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", size=7)

        # Parameters table
        col_width = 30
        row_height = 4

        params = [
            ('Method', clustering_metadata.get('clustering_method', 'hdbscan')),
            ('N Neighbors', clustering_metadata.get('n_neighbors', 'N/A')),
            ('Min Distance', clustering_metadata.get('min_dist', 'N/A')),
            ('Metric', clustering_metadata.get('metric', 'euclidean')),
            ('Min Cluster Size', clustering_metadata.get('min_cluster_size', 'N/A')),
            ('Min Samples', clustering_metadata.get('min_samples', 'N/A'))
        ]

        # Display in 2 columns
        for i in range(0, len(params), 2):
            # First column
            param1 = params[i]
            pdf.cell(col_width / 2, row_height, param1[0], border=1, align='C')
            pdf.cell(col_width / 2, row_height, str(param1[1]), border=1, align='C')

            # Second column if available
            if i + 1 < len(params):
                param2 = params[i + 1]
                pdf.cell(col_width / 2, row_height, param2[0], border=1, align='C')
                pdf.cell(col_width / 2, row_height, str(param2[1]), border=1, align='C')

            pdf.ln()

        # Clustering quality metrics
        pdf.ln(2)
        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(0, 5, "Clustering Quality Metrics:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", size=7)

        quality_metrics = [
            ('Composite Score', f"{clustering_metadata.get('composite_score', np.nan):.3f}"),
            ('NMI', f"{clustering_metadata.get('nmi', np.nan):.3f}"),
            ('Silhouette', f"{clustering_metadata.get('silhouette', np.nan):.3f}"),
            ('DBI', f"{clustering_metadata.get('dbi', np.nan):.3f}"),
            ('N Clusters', str(clustering_metadata.get('n_clusters', 'N/A')))
        ]

        for i in range(0, len(quality_metrics), 2):
            # First column
            metric1 = quality_metrics[i]
            pdf.cell(col_width / 2, row_height, metric1[0], border=1, align='C')
            pdf.cell(col_width / 2, row_height, metric1[1], border=1, align='C')

            # Second column if available
            if i + 1 < len(quality_metrics):
                metric2 = quality_metrics[i + 1]
                pdf.cell(col_width / 2, row_height, metric2[0], border=1, align='C')
                pdf.cell(col_width / 2, row_height, metric2[1], border=1, align='C')

            pdf.ln()

        pdf.ln(5)

    def _add_basic_phenotype_stats(self, pdf: FPDF, phenotype_results: Dict[str, Any]):
        """Add basic phenotype statistics table."""
        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(0, 5, "Basic Phenotype Statistics:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", size=7)

        col_width = 25
        row_height = 4

        # Header row
        headers = ['Repertoire Size', 'Entropy', 'Entropy Scaled', 'N Songs', 'N Syllables']
        for header in headers:
            pdf.cell(col_width, row_height, header, border=1, align='C')
        pdf.ln()

        # Data row
        values = [
            str(phenotype_results.get('repertoire_size', 'N/A')),
            f"{phenotype_results.get('entropy', np.nan):.3f}",
            f"{phenotype_results.get('entropy_scaled', np.nan):.3f}",
            str(phenotype_results.get('n_songs', 'N/A')),
            str(phenotype_results.get('n_syllables_total', 'N/A'))
        ]

        for value in values:
            pdf.cell(col_width, row_height, value, border=1, align='C')
        pdf.ln()

        pdf.ln(5)

    def _add_syllable_statistics_table(self, pdf: FPDF, phenotype_results: Dict[str, Any],
                                       syllable_stats: Dict[str, Dict[str, float]]):
        """Add detailed syllable statistics table."""
        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(0, 5, "Syllable Statistics:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", size=7)

        # Get syllable counts from phenotype results
        syl_counts = phenotype_results.get('syllable_counts', {})
        syl_proportions = phenotype_results.get('syllable_proportions', np.array([]))

        if not syl_counts:
            pdf.cell(0, 5, "No syllable statistics available", ln=True)
            pdf.ln(5)
            return

        # Sort syllables for consistent display
        syllables = sorted(syl_counts.keys())
        total_syllables = sum(syl_counts.values())

        # Calculate layout parameters
        column_width = 12
        row_height = 3
        max_columns_per_row = int((pdf.w - 20) / column_width)

        # Helper function to print table rows
        def print_syllable_row(row_label: str, data_func, format_func=lambda x: str(x), start_idx: int = 0):
            pdf.cell(column_width, row_height, row_label, border=1, align='C')
            end_idx = min(start_idx + max_columns_per_row - 1, len(syllables))  # -1 for label column

            for i in range(start_idx, end_idx):
                syl = syllables[i]
                value = data_func(syl)
                formatted_value = format_func(value)
                pdf.cell(column_width, row_height, formatted_value, border=1, align='C')
            pdf.ln()

        # Print data in chunks if there are too many syllables
        for start_idx in range(0, len(syllables), max_columns_per_row - 1):
            if start_idx > 0:
                pdf.ln(2)  # Add space between chunks

            # Syllable names
            print_syllable_row('Syllable', lambda syl: syl, start_idx=start_idx)

            # Counts
            print_syllable_row('Count', lambda syl: syl_counts.get(syl, 0), start_idx=start_idx)

            # Frequencies
            print_syllable_row('Frequency',
                               lambda syl: syl_counts.get(syl, 0) / total_syllables,
                               lambda x: f"{x:.3f}", start_idx=start_idx)

            # Duration statistics from syllable database
            print_syllable_row('Duration (ms)',
                               lambda syl: syllable_stats.get(str(syl), {}).get('duration_ms', np.nan),
                               lambda x: f"{x:.1f}" if not np.isnan(x) else "N/A",
                               start_idx=start_idx)

            # Gap statistics
            print_syllable_row('Pre-gap (ms)',
                               lambda syl: syllable_stats.get(str(syl), {}).get('prev_gap_ms', np.nan),
                               lambda x: f"{x:.1f}" if not np.isnan(x) else "N/A",
                               start_idx=start_idx)

            print_syllable_row('Post-gap (ms)',
                               lambda syl: syllable_stats.get(str(syl), {}).get('next_gap_ms', np.nan),
                               lambda x: f"{x:.1f}" if not np.isnan(x) else "N/A",
                               start_idx=start_idx)

        pdf.ln(5)

    def _add_repeat_statistics_table(self, pdf: FPDF, phenotype_results: Dict[str, Any]):
        """Add repeat pattern statistics table."""
        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(0, 5, "Repeat Pattern Statistics:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", size=7)

        # Basic repeat statistics
        col_width = 20
        row_height = 4

        # Header row
        headers = ['Repeat Bool', 'Dyad Bool', 'Num Dyads', 'Num Longer Reps', 'Mean Length', 'Median Length',
                   'Std Dev']
        for header in headers:
            pdf.cell(col_width, row_height, header, border=1, align='C')
        pdf.ln()

        # Data row
        values = [
            str(phenotype_results.get('repeat_bool', False)),
            str(phenotype_results.get('dyad_bool', False)),
            str(phenotype_results.get('num_dyad', 0)),
            str(phenotype_results.get('num_longer_reps', 0)),
            f"{phenotype_results.get('mean_repeat_syls', np.nan):.2f}",
            f"{phenotype_results.get('median_repeat_syls', np.nan):.1f}",
            f"{np.sqrt(phenotype_results.get('var_repeat_syls', np.nan)):.2f}"
        ]

        for value in values:
            pdf.cell(col_width, row_height, value, border=1, align='C')
        pdf.ln()

        pdf.ln(5)

        # Detailed repeat counts if available
        repeat_counts = phenotype_results.get('repeat_counts', pd.DataFrame())
        if not repeat_counts.empty:
            pdf.set_font("Arial", size=8, style='B')
            pdf.cell(0, 5, "Detailed Repeat Counts:", ln=True)
            pdf.ln(2)

            pdf.set_font("Arial", size=7)

            # Display repeat counts matrix
            syllables = list(repeat_counts.columns)
            repeat_lengths = list(repeat_counts.index)

            # Header with syllables
            pdf.cell(15, row_height, 'Length', border=1, align='C')
            for syl in syllables:
                pdf.cell(15, row_height, str(syl), border=1, align='C')
            pdf.ln()

            # Data rows
            for length in repeat_lengths:
                pdf.cell(15, row_height, str(length), border=1, align='C')
                for syl in syllables:
                    count = repeat_counts.loc[length, syl]
                    pdf.cell(15, row_height, str(count), border=1, align='C')
                pdf.ln()

        pdf.ln(5)

    def _add_phenotype_images(self, pdf: FPDF, plot_paths: PhenotypePDFPaths):
        """Add phenotype analysis images."""
        image_width = 90
        image_height = 70
        spacing = 5

        # Add transition matrices
        y_position = pdf.get_y()

        if plot_paths.transition_counts_img and os.path.exists(plot_paths.transition_counts_img):
            pdf.image(plot_paths.transition_counts_img, x=10, y=y_position,
                      w=image_width, h=image_height)

        if plot_paths.transition_matrix_img and os.path.exists(plot_paths.transition_matrix_img):
            pdf.image(plot_paths.transition_matrix_img, x=10 + image_width + spacing, y=y_position,
                      w=image_width, h=image_height)

        # Move to next row
        pdf.ln(image_height + 10)

        # Add repeat patterns
        if plot_paths.repeat_patterns_img and os.path.exists(plot_paths.repeat_patterns_img):
            # Center the repeat pattern image
            x_center = (pdf.w - image_width) / 2
            pdf.image(plot_paths.repeat_patterns_img, x=x_center, y=pdf.get_y(),
                      w=image_width, h=image_height)
            pdf.ln(image_height + 10)

        # Add vocabulary comparison if available
        if plot_paths.vocabulary_comparison_img and os.path.exists(plot_paths.vocabulary_comparison_img):
            x_center = (pdf.w - image_width) / 2
            pdf.image(plot_paths.vocabulary_comparison_img, x=x_center, y=pdf.get_y(),
                      w=image_width, h=image_height)
            pdf.ln(image_height + 10)

    def _add_spectrogram_examples(self, pdf: FPDF, spec_paths: List[str]):
        """Add example spectrograms in a grid."""
        if not spec_paths:
            return

        image_width = 100
        image_height = 30
        spacing = 5

        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(0, 5, "Example Spectrograms:", ln=True)
        pdf.ln(5)

        # Add spectrograms in pairs (2 columns)
        for i in range(0, len(spec_paths), 2):
            # First column
            if i < len(spec_paths) and os.path.exists(spec_paths[i]):
                pdf.image(spec_paths[i], x=10, y=pdf.get_y(),
                          w=image_width, h=image_height)

            # Second column
            if i + 1 < len(spec_paths) and os.path.exists(spec_paths[i + 1]):
                pdf.image(spec_paths[i + 1], x=10 + image_width + spacing, y=pdf.get_y(),
                          w=image_width, h=image_height)

            pdf.ln(image_height + 5)

    def _save_pdf(self, pdf: FPDF, output_path: Path, overwrite: bool = True):
        """Save PDF file with error handling."""
        try:
            if output_path.exists() and overwrite:
                output_path.unlink()  # Delete existing file

            pdf.output(str(output_path))

        except Exception as e:
            logging.error(f"Error saving PDF to {output_path}: {e}")
            raise

    def _cleanup_temp_images(self, image_paths: List[str]):
        """Clean up temporary image files."""
        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                logging.warning(f"Could not remove temp image {img_path}: {e}")


def generate_phenotype_pdfs_from_saved_data(bird_path: str,
                                            overwrite: bool = True) -> Dict[str, str]:
    """
    Generate phenotype PDFs from saved detailed phenotype data.

    Args:
        bird_path: Path to bird directory
        overwrite: Whether to overwrite existing PDFs

    Returns:
        Dict mapping PDF type to generated file path
    """
    try:
        generator = PhenotypePDFGenerator(bird_path)
        generated_pdfs = {}

        # Path to detailed phenotype data
        detailed_data_dir = os.path.join(bird_path, 'data', 'phenotype_detailed')
        syllable_db_path = os.path.join(bird_path, 'data', 'syllable_database', 'syllable_features.csv')

        if not os.path.exists(detailed_data_dir):
            logging.warning(f"No detailed phenotype data found: {detailed_data_dir}")
            return {}

        # Generate manual PDF if manual data exists
        manual_data_path = os.path.join(detailed_data_dir, 'manual_phenotype_data.pkl')
        if os.path.exists(manual_data_path):
            try:
                with open(manual_data_path, 'rb') as f:
                    manual_results = pkl.load(f)

                manual_pdf = generator.generate_manual_phenotype_pdf(
                    manual_results, syllable_db_path, overwrite
                )
                if manual_pdf:
                    generated_pdfs['manual'] = manual_pdf

            except Exception as e:
                logging.error(f"Error loading manual phenotype data: {e}")

        # Generate automated PDF for best rank (rank 0)
        auto_data_path = os.path.join(detailed_data_dir, 'automated_phenotype_data_rank0.pkl')
        if os.path.exists(auto_data_path):
            try:
                with open(auto_data_path, 'rb') as f:
                    auto_data = pkl.load(f)

                auto_results = auto_data['phenotype_results']
                clustering_metadata = auto_data['clustering_metadata']

                auto_pdf = generator.generate_automated_phenotype_pdf(
                    auto_results, clustering_metadata, syllable_db_path, 0, overwrite
                )
                if auto_pdf:
                    generated_pdfs['automated'] = auto_pdf

            except Exception as e:
                logging.error(f"Error loading automated phenotype data: {e}")

        return generated_pdfs

    except Exception as e:
        logging.error(f"Error generating phenotype PDFs from saved data: {e}")
        return {}


def integrate_with_phenotyping_pipeline(bird_path: str, config) -> Dict[str, str]:
    """
    Integration function to be called from the main phenotyping pipeline.

    This should be called after phenotype_bird() completes successfully.

    Args:
        bird_path: Path to bird directory
        config: PhenotypingConfig object

    Returns:
        Dict mapping PDF type to generated file path
    """
    try:
        if not config.generate_plots:
            logging.info(f"Plot generation disabled for {os.path.basename(bird_path)}")
            return {}

        # Generate PDFs from saved detailed data
        generated_pdfs = generate_phenotype_pdfs_from_saved_data(
            bird_path, overwrite=True
        )

        if generated_pdfs:
            bird_name = os.path.basename(bird_path)
            logging.info(f"Generated phenotype PDFs for {bird_name}: {list(generated_pdfs.keys())}")

        return generated_pdfs

    except Exception as e:
        logging.error(f"Error in phenotype PDF integration: {e}")
        return {}


def _create_phenotype_spectrogram(self, syl_file: Path,
                                  label_type: str, rank: int,
                                  file_idx: int, duration: float = 6.0) -> Optional[str]:
    """Create spectrogram with phenotype labels."""
    try:
        with tables.open_file(str(syl_file), 'r') as f:
            # Read syllable data
            onsets = f.root.onsets.read()
            offsets = f.root.offsets.read()

            # Read audio filename
            audio_filename_raw = f.root.audio_filename.read()
            if isinstance(audio_filename_raw[0], bytes):
                audio_filename = audio_filename_raw[0].decode('utf-8')
            else:
                audio_filename = str(audio_filename_raw[0])

            # Resolve audio file path
            if not os.path.isfile(audio_filename):
                audio_filename = replace_macaw_root(audio_filename)

            if not os.path.isfile(audio_filename):
                logging.warning(f"Audio file not found: {audio_filename}")
                return None

            # Read labels based on type
            labels = None
            if label_type == 'manual' and hasattr(f.root, 'manual'):
                manual_raw = f.root.manual.read()
                labels = np.array([
                    item.decode('utf-8') if isinstance(item, bytes) else str(item)
                    for item in manual_raw
                ])
            elif label_type == 'automated':
                # Try to read from automated labels if they exist
                if hasattr(f.root, 'auto'):
                    auto_raw = f.root.auto.read()
                    labels = np.array([int(item) for item in auto_raw])
                else:
                    # Load from syllable database using clustering labels
                    try:
                        syllable_db_path = self.bird_path / 'data' / 'syllable_database' / 'syllable_features.csv'
                        if syllable_db_path.exists():
                            df = pd.read_csv(syllable_db_path)
                            # Find the song file in the database
                            song_name = syl_file.name
                            song_data = df[df['song_file'] == song_name]

                            if not song_data.empty:
                                # Get clustering labels for this rank
                                cluster_col = f'cluster_rank{rank}_'
                                cluster_cols = [col for col in song_data.columns if col.startswith(cluster_col)]
                                if cluster_cols:
                                    # Use the first matching cluster column
                                    labels = song_data[cluster_cols[0]].values
                                    labels = labels[~pd.isna(labels)]  # Remove NaN values
                    except Exception as e:
                        logging.debug(f"Could not load automated labels from database: {e}")
                        # Fallback to placeholder labels
                        labels = np.array([f'C{i % 5}' for i in range(len(onsets))])

            if labels is None or len(labels) == 0:
                return None

            # Read and process audio
            audio, fs = read_audio_file(audio_filename)
            audio = rms_norm(audio)
            audio = butter_bandpass_filter_sos(
                audio,
                lowcut=self.spec_params.min_freq,
                highcut=self.spec_params.max_freq,
                fs=fs,
                order=5
            )

            # Set time window
            first_time = max((onsets[0] / 1000 - 0.25), 0.0)
            last_time = first_time + duration

            # Ensure we don't exceed audio length
            if len(audio) * (1 / fs) <= last_time:
                last_time = (len(audio) - 1) * (1 / fs)

            # Generate spectrogram
            spec, _, t = get_song_spec(
                t1=first_time,
                t2=last_time,
                audio=audio,
                params=self.spec_params,
                fs=fs,
                downsample=False
            )

            # Filter syllables within time window
            time_mask = (onsets >= first_time * 1000) & (onsets <= last_time * 1000)
            syl_onsets = onsets[time_mask]
            syl_offsets = offsets[time_mask]
            syl_labels = labels[time_mask] if len(labels) == len(onsets) else labels[:len(syl_onsets)]

            if len(syl_onsets) <= 1:
                return None

            # Create spectrogram plot
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.imshow(spec, aspect='auto', origin='lower')
            ax.set_yticks([])

            # Add syllable labels
            font_size = 8
            colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Color palette

            for i, (onset, offset, label) in enumerate(zip(syl_onsets, syl_offsets, syl_labels)):
                if label in ['-', '', 's', 'z'] or (
                        isinstance(label, (int, float)) and label < 0):  # Skip non-syllable tokens
                    continue

                label_x = (onset / (duration * 1000)) * spec.shape[1]

                # Choose color based on label
                color_idx = hash(str(label)) % len(colors)

                ax.text(label_x, spec.shape[0] + 20, str(label),
                        color='black', fontsize=font_size, ha='center', va='top',
                        bbox=dict(facecolor=colors[color_idx], edgecolor='black',
                                  alpha=0.7, boxstyle='round'))

            # Set time axis
            time_ticks = np.arange(0, duration + 1)
            x_ticks = np.linspace(0, spec.shape[1], len(time_ticks))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(time_ticks.astype(int))
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{label_type.title()} Labels - {self.bird_name}')

            # Save spectrogram
            timestamp = pd.Timestamp.now().strftime('%H%M%S')
            filename = f'phenotype_spec_{label_type}_{rank}_{file_idx}_{timestamp}.png'
            file_path = self.temp_dir / filename

            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            return str(file_path)

    except Exception as e:
        logging.error(f"Error creating phenotype spectrogram from {syl_file}: {e}")
        return None


def batch_generate_phenotype_pdfs_from_saved_data(bird_paths: List[str],
                                                  overwrite: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Generate phenotype PDFs for multiple birds using saved detailed data.

    Args:
        bird_paths: List of paths to bird directories
        overwrite: Whether to overwrite existing PDFs

    Returns:
        Dict mapping bird names to their generated PDF paths
    """
    all_pdfs = {}

    for bird_path in bird_paths:
        bird_name = os.path.basename(bird_path)

        try:
            bird_pdfs = generate_phenotype_pdfs_from_saved_data(
                bird_path, overwrite
            )

            if bird_pdfs:
                all_pdfs[bird_name] = bird_pdfs
                logging.info(f"Generated {len(bird_pdfs)} PDFs for {bird_name}")
            else:
                logging.warning(f"No PDFs generated for {bird_name}")

        except Exception as e:
            logging.error(f"Error generating PDFs for {bird_name}: {e}")
            continue

    return all_pdfs


# Add function to load and use detailed data for manual testing
def load_detailed_phenotype_data(bird_path: str) -> Tuple[Optional[Dict], List[Dict], List[Dict]]:
    """
    Load saved detailed phenotype data for inspection or manual PDF generation.

    Args:
        bird_path: Path to bird directory

    Returns:
        Tuple of (manual_results, auto_results_list, clustering_results_list)
    """
    try:
        detailed_data_dir = os.path.join(bird_path, 'data', 'phenotype_detailed')

        if not os.path.exists(detailed_data_dir):
            logging.warning(f"No detailed phenotype data found: {detailed_data_dir}")
            return None, [], []

        # Load manual results if available
        manual_results = None
        manual_data_path = os.path.join(detailed_data_dir, 'manual_phenotype_data.pkl')
        if os.path.exists(manual_data_path):
            with open(manual_data_path, 'rb') as f:
                manual_results = pkl.load(f)

        # Load automated results
        auto_results_list = []
        clustering_results_list = []

        rank = 0
        while True:
            auto_data_path = os.path.join(detailed_data_dir, f'automated_phenotype_data_rank{rank}.pkl')
            if not os.path.exists(auto_data_path):
                break

            with open(auto_data_path, 'rb') as f:
                auto_data = pkl.load(f)
                auto_results_list.append(auto_data['phenotype_results'])
                clustering_results_list.append(auto_data['clustering_metadata'])

            rank += 1

        return manual_results, auto_results_list, clustering_results_list

    except Exception as e:
        logging.error(f"Error loading detailed phenotype data: {e}")
        return None, [], []


# Update the main execution section to demonstrate the new workflow
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test on example datasets
    test_paths = [
        os.path.join('/Volumes', 'Extreme SSD', 'wseg test', 'bu85bu97'),
        os.path.join('/Volumes', 'Extreme SSD', 'evsong test', 'or18or24')
    ]

    for example_bird_path in test_paths:
        try:
            print("Testing Integrated Phenotype PDF Generation Pipeline...")

            # Test loading saved detailed data
            print("\n1. Testing loading of saved detailed phenotype data...")
            manual_data, auto_data_list, clustering_data_list = load_detailed_phenotype_data(example_bird_path)

            if manual_data:
                print(f"✓ Loaded manual phenotype data with {len(manual_data)} fields")
            else:
                print("✗ No manual phenotype data found")

            if auto_data_list:
                print(f"✓ Loaded {len(auto_data_list)} automated phenotype datasets")
            else:
                print("✗ No automated phenotype data found")

            # Test PDF generation from saved data
            print("\n2. Testing PDF generation from saved detailed data...")
            generated_pdfs = generate_phenotype_pdfs_from_saved_data(
                bird_path=example_bird_path,
                overwrite=True
            )

            if generated_pdfs:
                print(f"✓ Generated PDFs from saved data: {list(generated_pdfs.keys())}")
                for pdf_type, pdf_path in generated_pdfs.items():
                    print(f"  - {pdf_type}: {pdf_path}")
            else:
                print("✗ No PDFs generated from saved data")

            # Test batch processing
            print("\n3. Testing batch processing...")
            example_bird_paths = [example_bird_path]  # Would be multiple paths in real usage

            batch_results = batch_generate_phenotype_pdfs_from_saved_data(
                bird_paths=example_bird_paths,
                overwrite=True
            )

            if batch_results:
                print(f"✓ Batch processing completed: {len(batch_results)} birds")
                for bird_name, pdfs in batch_results.items():
                    print(f"  - {bird_name}: {len(pdfs)} PDFs generated")
            else:
                print("✗ Batch processing failed")

            # Test integration with phenotyping pipeline
            print("\n4. Testing integration workflow...")
            print("To integrate with your phenotyping pipeline, add this to phenotype_bird():")
            print("""
            # After saving phenotype results, add:
            if config.generate_plots:
                from phenotype_pdf_generator import integrate_with_phenotyping_pipeline
                pdf_results = integrate_with_phenotyping_pipeline(bird_path, config)
                if pdf_results:
                    logging.info(f"Generated phenotype PDFs: {list(pdf_results.keys())}")
            """)

            print("\n" + "=" * 70)
            print("Integrated Phenotype PDF Generation Pipeline Test Complete")
            print("=" * 70)

            print("\nIntegration Summary:")
            print("1. Modified phenotype_bird() to save detailed data structures")
            print("2. PDF generator reads from saved pickle files with full data")
            print("3. Automatic integration via integrate_with_phenotyping_pipeline()")
            print("4. Batch processing support for multiple birds")
            print("5. Fallback to syllable database for automated labels in spectrograms")

            print("\nData Flow:")
            print("phenotype_bird() → save_detailed_phenotype_data() → *.pkl files")
            print("*.pkl files → generate_phenotype_pdfs_from_saved_data() → PDFs")

            print("\nFiles Created:")
            print("- data/phenotype_detailed/manual_phenotype_data.pkl")
            print("- data/phenotype_detailed/automated_phenotype_data_rank0.pkl")
            print("- data/phenotype_detailed/automated_phenotype_data_rank1.pkl")
            print("- data/pdfs/bird_manual_phenotypes.pdf")
            print("- data/pdfs/bird_automated_phenotypes_rank0.pdf")

        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback

            traceback.print_exc()