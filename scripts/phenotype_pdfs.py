import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, black, white
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from PIL import Image
import tables
import gc

from song_phenotyping.signal import get_song_spec, rms_norm, butter_bandpass_filter_sos
from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
from song_phenotyping.tools.audio_utils import read_audio_file
from song_phenotyping.tools.system_utils import replace_macaw_root


@dataclass
class PhenotypePDFPaths:
    """Container for paths to phenotype analysis images."""
    transition_counts_img: str
    transition_matrix_img: str
    repeat_patterns_img: str
    vocabulary_comparison_img: Optional[str] = None


def create_dual_labeled_spectrogram(syl_file: Path, bird_path: Path, rank: int = 0,
                                    spectrograms_dir: Path = None,
                                    overwrite: bool = False,
                                    duration: float = 6.0,
                                    syllable_db: pd.DataFrame = None) -> Optional[str]:  # Add this parameter
    """
    Create spectrogram with both manual and automated labels (shared function).
    """
    try:
        # Set up directories
        if spectrograms_dir is None:
            spectrograms_dir = bird_path / 'spectrograms' / 'labelled'
        spectrograms_dir.mkdir(parents=True, exist_ok=True)

        # Extract base name for output filename
        audio_base = syl_file.stem

        # Smart caching - check for existing spectrograms first
        existing_pattern = f"{audio_base}_dual_rank{rank}_*.png"
        existing_files = list(spectrograms_dir.glob(existing_pattern))

        if existing_files and not overwrite:
            # Return most recent existing file
            most_recent = max(existing_files, key=lambda p: p.stat().st_mtime)
            logging.debug(f"Reusing cached spectrogram: {most_recent}")
            return str(most_recent)

        # Set up spectrogram parameters
        spec_params = SpectrogramParams()
        spec_params.max_dur = duration

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

            # Load manual labels
            manual_labels = None
            if hasattr(f.root, 'manual'):
                manual_raw = f.root.manual.read()
                manual_labels = np.array([
                    item.decode('utf-8') if isinstance(item, bytes) else str(item)
                    for item in manual_raw
                ])

            # Load automated labels from syllable database
            automated_labels = None
            try:
                if syllable_db is not None:  # Use cached database instead of loading
                    song_name = syl_file.name
                    song_data = syllable_db[syllable_db['song_file'] == song_name]

                    if not song_data.empty:
                        cluster_col = f'cluster_rank{rank}_'
                        cluster_cols = [col for col in song_data.columns if col.startswith(cluster_col)]
                        if cluster_cols:
                            labels = song_data[cluster_cols[0]].values
                            automated_labels = np.array([
                                int(label) if not pd.isna(label) and label != -1 else -1
                                for label in labels
                            ])
                else:
                    logging.debug("No syllable database available for automated labels")
            except Exception as e:
                logging.debug(f"Could not load automated labels: {e}")

            # Skip if no labels available
            if manual_labels is None and automated_labels is None:
                logging.debug(f"No labels available for {syl_file}")
                return None

            # Read and process audio
            audio, fs = read_audio_file(audio_filename)
            audio = rms_norm(audio)
            audio = butter_bandpass_filter_sos(
                audio,
                lowcut=spec_params.min_freq,
                highcut=spec_params.max_freq,
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
                params=spec_params,
                fs=fs,
                downsample=False
            )

            # Filter syllables within time window
            time_mask = (onsets >= first_time * 1000) & (onsets <= last_time * 1000)
            syl_onsets = onsets[time_mask]
            syl_offsets = offsets[time_mask]

            # Filter labels to match syllables in time window
            manual_syl_labels = None
            automated_syl_labels = None

            if manual_labels is not None and len(manual_labels) > 0:
                manual_syl_labels = manual_labels[time_mask] if len(manual_labels) == len(
                    onsets) else manual_labels[:len(syl_onsets)]

            if automated_labels is not None and len(automated_labels) > 0:
                automated_syl_labels = automated_labels[time_mask] if len(automated_labels) == len(
                    onsets) else automated_labels[:len(syl_onsets)]

            if len(syl_onsets) <= 1:
                logging.debug(f"Insufficient syllables in time window for {syl_file}")
                return None

            # Create spectrogram plot with better spacing
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(spec, aspect='auto', origin='lower', cmap='plasma', extent=[0, duration, 0, spec.shape[0]])

            # Remove frame/spines but keep time axis
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)  # Keep bottom for time axis

            ax.set_yticks([])
            ax.set_xlabel('Time (s)')

            # Color palette for labels
            colors = plt.cm.Set1(np.linspace(0, 1, 10))
            font_size = 9

            # Add syllable labels with proper centering and spacing
            for i, (onset, offset) in enumerate(zip(syl_onsets, syl_offsets)):
                # Calculate center position of syllable in time
                syllable_center_time = ((onset + offset) / 2 / 1000) - first_time

                # Ensure label is within spectrogram bounds
                if 0 <= syllable_center_time <= duration:
                    # Add manual label (above spectrogram with more spacing)
                    if manual_syl_labels is not None and i < len(manual_syl_labels):
                        manual_label = manual_syl_labels[i]
                        if manual_label not in ['s', 'z']:
                            color_idx = hash(str(manual_label)) % len(colors)
                            ax.text(syllable_center_time, spec.shape[0] + 15, str(manual_label),  # Increased spacing
                                    color='black', fontsize=font_size, ha='center', va='bottom',
                                    bbox=dict(facecolor=colors[color_idx], edgecolor='black',
                                              alpha=0.8, boxstyle='round,pad=0.2'))

                    # Add automated label (below spectrogram with more spacing)
                    if automated_syl_labels is not None and i < len(automated_syl_labels):
                        auto_label = automated_syl_labels[i]
                        color_idx = hash(str(auto_label)) % len(colors)
                        ax.text(syllable_center_time, -15, str(auto_label),  # Increased spacing
                                color='black', fontsize=font_size, ha='center', va='top',
                                bbox=dict(facecolor=colors[color_idx], edgecolor='black',
                                          alpha=0.8, boxstyle='round,pad=0.2'))

            # Create title with more spacing
            title_parts = []
            if manual_labels is not None:
                title_parts.append("Manual")
            if automated_labels is not None:
                title_parts.append(f"Auto(R{rank})")
            title = f"{' + '.join(title_parts)} Labels - {bird_path.name}"
            ax.set_title(title, pad=20)  # Added padding above title

            # Set proper axis limits with more space for labels
            ax.set_xlim(0, duration)
            ax.set_ylim(-25, spec.shape[0] + 25)  # Increased spacing for labels

            # Save spectrogram with timestamp for uniqueness
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{audio_base}_dual_rank{rank}_{timestamp}.png'
            file_path = spectrograms_dir / filename

            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            gc.collect()

            logging.debug(f"Created new spectrogram: {file_path}")
            return str(file_path)

    except Exception as e:
        logging.error(f"Error creating dual spectrogram from {syl_file}: {e}")
        return None


class PhenotypePDFGenerator:
    """
    Generate PDF summaries from phenotyping analysis results using ReportLab.

    Creates separate PDFs for manual and automated results with dual-labeled spectrograms.
    """

    def __init__(self, bird_path: str):
        self.bird_path = Path(bird_path)
        self.bird_name = self.bird_path.name

        # Directory paths
        self.phenotype_plots_dir = self.bird_path / 'figures' / 'phenotyping'
        from song_phenotyping.tools.pipeline_paths import PLOTS_DIR
        self.pdf_output_dir = self.bird_path / PLOTS_DIR
        self.spectrograms_dir = self.bird_path / 'spectrograms' / 'labelled'

        # Create directories
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
        self.spectrograms_dir.mkdir(parents=True, exist_ok=True)

        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # Center
        )
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=12
        )

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
                cluster_df = df[df[cluster_col].notna()]

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

    def _add_syllable_statistics_table(self, c: canvas.Canvas, width: float, height: float,
                                       phenotype_results: Dict[str, Any], syllable_stats: Dict[str, Dict[str, float]]):
        """
        Add syllable statistics table with syllables as columns, wrapping to new rows if needed.
        FIXED: Better column organization, table wrapping, and safer DataFrame checks.
        """
        y_position = height - 320  # Position below basic stats

        # Section title
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Syllable Statistics:")
        y_position -= 25

        # Get syllable counts from phenotype results
        syllable_counts = phenotype_results.get('syllable_counts', {})

        if not syllable_counts or len(syllable_counts) == 0:
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, "No syllable statistics available")
            return

        # Filter and sort syllables
        valid_syllables = [syl for syl in sorted(syllable_counts.keys()) if syl not in ['s', 'z', '-', '']]
        total_syllables = sum(syllable_counts.values())

        if not valid_syllables or len(valid_syllables) == 0:
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, "No valid syllables found")
            return

        # Calculate table dimensions
        max_cols_per_table = 8  # Maximum syllables per table before wrapping
        col_width = 60
        row_height = 18

        # Split syllables into chunks for multiple tables if needed
        syllable_chunks = [valid_syllables[i:i + max_cols_per_table]
                           for i in range(0, len(valid_syllables), max_cols_per_table)]

        current_y = y_position

        for chunk_idx, syllable_chunk in enumerate(syllable_chunks):
            # Create table data with syllables as columns
            table_data = [
                ['Metric'] + [str(syl) for syl in syllable_chunk],  # Header row
                ['Count'] + [str(syllable_counts.get(syl, 0)) for syl in syllable_chunk],
                ['Frequency'] + [
                    f"{syllable_counts.get(syl, 0) / total_syllables:.3f}" if total_syllables > 0 else "0.000" for syl
                    in syllable_chunk]
            ]

            # Add duration and gap statistics if available
            duration_row = ['Duration (ms)']
            pre_gap_row = ['Pre-gap (ms)']
            post_gap_row = ['Post-gap (ms)']

            for syl in syllable_chunk:
                syl_stat = syllable_stats.get(str(syl), {})

                duration = syl_stat.get('duration_ms', np.nan)
                duration_row.append(f"{duration:.1f}" if not np.isnan(duration) else "N/A")

                pre_gap = syl_stat.get('prev_gap_ms', np.nan)
                pre_gap_row.append(f"{pre_gap:.1f}" if not np.isnan(pre_gap) else "N/A")

                post_gap = syl_stat.get('next_gap_ms', np.nan)
                post_gap_row.append(f"{post_gap:.1f}" if not np.isnan(post_gap) else "N/A")

            table_data.extend([duration_row, pre_gap_row, post_gap_row])

            # Draw the table
            col_widths = [80] + [col_width] * len(syllable_chunk)  # Wider first column for metric names
            self._draw_table(c, table_data, x=50, y=current_y, col_widths=col_widths)

            # Move y position down for next table (if any)
            current_y -= (len(table_data) * row_height + 40)  # Extra space between tables

            # Add continuation label if this isn't the last chunk
            if chunk_idx < len(syllable_chunks) - 1:
                c.setFont("Helvetica", 9)
                c.drawString(50, current_y + 10, "(continued below)")
                current_y -= 10

    def _add_repeat_statistics_table(self, c: canvas.Canvas, width: float, height: float,
                                     phenotype_results: Dict[str, Any]):
        """
        Add repeat pattern statistics table with improved layout.
        FIXED: Better positioning and formatting, proper DataFrame checks.
        """
        y_position = height - 500  # Adjusted position to account for larger syllable table

        # Section title
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Repeat Pattern Statistics:")
        y_position -= 25

        # Get repeat counts DataFrame
        repeat_counts = phenotype_results.get('repeat_counts')
        has_repeat_data = (repeat_counts is not None and
                           not (hasattr(repeat_counts, 'empty') and repeat_counts.empty))

        # Check for basic repeat boolean
        has_repeat_bool = phenotype_results.get('repeat_bool', False)

        if not has_repeat_data and not has_repeat_bool:
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, "No repeat pattern statistics available")
            return

        # Basic repeat stats table
        basic_repeat_data = [
            ['Repeat Bool', 'Dyad Bool', 'Num Dyads', 'Num Longer Reps', 'Mean Length', 'Median Length', 'Std Dev'],
            [
                str(phenotype_results.get('repeat_bool', False)),
                str(phenotype_results.get('dyad_bool', False)),
                str(phenotype_results.get('num_dyad', 0)),
                str(phenotype_results.get('num_longer_reps', 0)),
                f"{phenotype_results.get('mean_repeat_syls', np.nan):.2f}" if not np.isnan(
                    phenotype_results.get('mean_repeat_syls', np.nan)) else "N/A",
                f"{phenotype_results.get('median_repeat_syls', np.nan):.1f}" if not np.isnan(
                    phenotype_results.get('median_repeat_syls', np.nan)) else "N/A",
                f"{phenotype_results.get('var_repeat_syls', np.nan):.2f}" if not np.isnan(
                    phenotype_results.get('var_repeat_syls', np.nan)) else "N/A"
            ]
        ]

        # Draw basic repeat stats table
        self._draw_table(c, basic_repeat_data, x=50, y=y_position, col_widths=[60, 60, 60, 80, 70, 70, 60])
        y_position -= (len(basic_repeat_data) * 20 + 30)

        # Detailed repeat counts if available (FIXED: proper DataFrame check)
        if has_repeat_data:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y_position, "Detailed Repeat Counts:")
            y_position -= 20

            # Convert DataFrame to table format with syllables as columns
            syllables = list(repeat_counts.columns)[:6]  # Limit to 6 syllables for space
            repeat_lengths = list(repeat_counts.index)[:8]  # Limit to 8 lengths for space

            # Create detailed counts table
            detailed_data = [['Length'] + [str(syl) for syl in syllables]]

            for length in repeat_lengths:
                row = [str(length)]
                for syl in syllables:
                    count = repeat_counts.loc[length, syl] if syl in repeat_counts.columns else 0
                    row.append(str(count))
                detailed_data.append(row)

            # Draw detailed counts table
            col_widths = [50] + [50] * len(syllables)
            self._draw_table(c, detailed_data, x=50, y=y_position, col_widths=col_widths)


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

    def _generate_example_spectrograms(self, n_specs: int = 8,
                                       label_type: str = 'manual',
                                       rank: int = 0,
                                       overwrite_spectrograms: bool = False) -> List[str]:
        """Generate example spectrograms with improved caching and error handling."""
        spec_paths = []

        try:
            # Get syllable files
            from song_phenotyping.tools.pipeline_paths import SPECS_DIR
            syllable_dir = self.bird_path / SPECS_DIR
            if not syllable_dir.exists():
                logging.warning(f"No syllable directory found: {syllable_dir}")
                return []

            syllable_files = list(syllable_dir.glob('*.h5'))
            if not syllable_files:
                logging.warning(f"No syllable files found in {syllable_dir}")
                return []

            # Sample files for spectrograms - use consistent sampling for caching
            import random
            random.seed(42)  # Consistent sampling across runs
            sampled_files = random.sample(syllable_files, min(n_specs, len(syllable_files)))

            for syl_file in sampled_files:
                try:
                    spec_path = create_dual_labeled_spectrogram(
                        syl_file=syl_file,
                        bird_path=self.bird_path,
                        rank=rank,
                        spectrograms_dir=self.spectrograms_dir,
                        overwrite=overwrite_spectrograms,
                        duration=6.0
                    )
                    if spec_path and os.path.exists(spec_path):
                        spec_paths.append(spec_path)

                except Exception as e:
                    logging.error(f"Error creating spectrogram for {syl_file}: {e}")
                    continue

            logging.info(f"Generated/cached {len(spec_paths)} spectrograms for {self.bird_name}")
            return spec_paths

        except Exception as e:
            logging.error(f"Error generating example spectrograms: {e}")
            return []

    def _add_phenotype_header(self, c: canvas.Canvas, width: float, height: float,
                              phenotype_results: Dict[str, Any], title: str,
                              clustering_metadata: Dict[str, Any] = None):
        """
        Add header section with bird info, analysis type, and clustering info if automated.
        ENHANCED: Clustering parameters moved to title page.
        """
        # Main title
        c.setFont("Helvetica-Bold", 16)
        title_width = c.stringWidth(title, "Helvetica-Bold", 16)
        c.drawString((width - title_width) / 2, height - 50, title)

        # Bird name
        c.setFont("Helvetica", 12)
        bird_text = f"Bird: {self.bird_name}"
        bird_width = c.stringWidth(bird_text, "Helvetica", 12)
        c.drawString((width - bird_width) / 2, height - 75, bird_text)

        # Generation timestamp
        c.setFont("Helvetica", 10)
        timestamp = f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        timestamp_width = c.stringWidth(timestamp, "Helvetica", 10)
        c.drawString((width - timestamp_width) / 2, height - 95, timestamp)

        # Add clustering parameters if this is an automated analysis
        if clustering_metadata is not None:
            y_pos = height - 130

            # Clustering parameters section
            c.setFont("Helvetica-Bold", 12)
            cluster_title = "Clustering Parameters:"
            cluster_width = c.stringWidth(cluster_title, "Helvetica-Bold", 12)
            c.drawString((width - cluster_width) / 2, y_pos, cluster_title)
            y_pos -= 25

            # Parameters in a compact format
            c.setFont("Helvetica", 10)
            params_text = [
                f"Method: {clustering_metadata.get('clustering_method', 'hdbscan')} | "
                f"N Neighbors: {clustering_metadata.get('n_neighbors', 'N/A')} | "
                f"Min Distance: {clustering_metadata.get('min_dist', 'N/A')}",

                f"Min Cluster Size: {clustering_metadata.get('min_cluster_size', 'N/A')} | "
                f"Min Samples: {clustering_metadata.get('min_samples', 'N/A')} | "
                f"Metric: {clustering_metadata.get('metric', 'euclidean')}"
            ]

            for param_line in params_text:
                param_width = c.stringWidth(param_line, "Helvetica", 10)
                c.drawString((width - param_width) / 2, y_pos, param_line)
                y_pos -= 15

            # Quality metrics
            y_pos -= 10
            c.setFont("Helvetica-Bold", 11)
            quality_title = "Quality Metrics:"
            quality_width = c.stringWidth(quality_title, "Helvetica-Bold", 11)
            c.drawString((width - quality_width) / 2, y_pos, quality_title)
            y_pos -= 20

            c.setFont("Helvetica", 10)
            metrics_text = [
                f"Composite Score: {clustering_metadata.get('composite_score', np.nan):.3f} | "
                f"NMI: {clustering_metadata.get('nmi', np.nan):.3f} | "
                f"Silhouette: {clustering_metadata.get('silhouette', np.nan):.3f}",

                f"DBI: {clustering_metadata.get('dbi', np.nan):.3f} | "
                f"N Clusters: {clustering_metadata.get('n_clusters', 'N/A')}"
            ]

            for metric_line in metrics_text:
                metric_width = c.stringWidth(metric_line, "Helvetica", 10)
                c.drawString((width - metric_width) / 2, y_pos, metric_line)
                y_pos -= 15

    def _add_basic_phenotype_stats(self, c: canvas.Canvas, width: float, height: float,
                                   phenotype_results: Dict[str, Any]):
        """Add basic phenotype statistics table."""
        y_position = height - 200

        # Section title
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Basic Phenotype Statistics:")
        y_position -= 25

        # Create basic stats table
        stats_data = [
            ['Repertoire Size', 'Entropy', 'Entropy Scaled', 'N Songs', 'N Syllables'],
            [str(phenotype_results.get('repertoire_size', 'N/A')),
             f"{phenotype_results.get('entropy', np.nan):.3f}",
             f"{phenotype_results.get('entropy_scaled', np.nan):.3f}",
             str(phenotype_results.get('n_songs', 'N/A')),
             str(phenotype_results.get('n_syllables_total', 'N/A'))]
        ]

        # Draw basic stats table
        self._draw_table(c, stats_data, x=50, y=y_position, col_widths=[70, 70, 70, 70, 70])

    def _get_all_possible_labels(self, phenotype_results: Dict[str, Any],
                                 label_type: str, rank: int = 0) -> List[str]:
        """Get all possible labels for the legend."""
        try:
            if label_type == 'manual':
                # Get all syllable types from syllable counts
                syllable_counts = phenotype_results.get('syllable_counts', {})
                labels = list(syllable_counts.keys())
                # Filter out non-syllable tokens
                labels = [label for label in labels if label not in ['s', 'z', '-', '']]
                return sorted(labels)
            else:
                # For automated labels, try to get from syllable database
                from song_phenotyping.tools.pipeline_paths import STAGES_DIR
                syllable_db_path = self.bird_path / STAGES_DIR / 'syllable_database' / 'syllable_features.csv'
                if syllable_db_path.exists():
                    df = pd.read_csv(syllable_db_path)
                    cluster_col = f'cluster_rank{rank}_'
                    cluster_cols = [col for col in df.columns if col.startswith(cluster_col)]
                    if cluster_cols:
                        unique_labels = df[cluster_cols[0]].dropna().unique()
                        # Convert to strings
                        labels = [str(int(label)) for label in unique_labels ]
                        return sorted(labels, key=lambda x: int(x) if x.isdigit() else float('inf'))

                # Fallback to syllable counts if available
                syllable_counts = phenotype_results.get('syllable_counts', {})
                labels = [str(label) for label in syllable_counts.keys()]
                return sorted(labels, key=lambda x: int(x) if str(x).isdigit() else float('inf'))
        except Exception as e:
            logging.error(f"Error getting possible labels: {e}")
            return []

    def _draw_table(self, c: canvas.Canvas, table_data: List[List[str]],
                    x: float, y: float, col_widths: List[float]):
        """Draw a table using ReportLab canvas."""
        try:
            if not table_data:
                return

            row_height = 18

            # Draw table
            for row_idx, row in enumerate(table_data):
                current_y = y - (row_idx * row_height)
                current_x = x

                for col_idx, cell in enumerate(row):
                    if col_idx < len(col_widths):
                        col_width = col_widths[col_idx]

                        # Set font style - bold for header row
                        if row_idx == 0:
                            c.setFont("Helvetica-Bold", 9)
                        else:
                            c.setFont("Helvetica", 9)

                        # Draw cell border
                        c.rect(current_x, current_y - row_height + 5, col_width, row_height)

                        # Draw cell text (centered)
                        text_width = c.stringWidth(str(cell), c._fontname, c._fontsize)
                        text_x = current_x + (col_width - text_width) / 2
                        text_y = current_y - row_height + 10

                        c.drawString(text_x, text_y, str(cell))

                        current_x += col_width

        except Exception as e:
            logging.error(f"Error drawing table: {e}")

    def _save_pdf(self, canvas_obj: canvas.Canvas, output_path: Path, overwrite: bool = True):
        """Save PDF file with error handling."""
        try:
            if output_path.exists() and overwrite:
                output_path.unlink()  # Delete existing file

            canvas_obj.save()

        except Exception as e:
            logging.error(f"Error saving PDF to {output_path}: {e}")
            raise


    def generate_manual_phenotype_pdf(self, phenotype_results: Dict[str, Any],
                                      syllable_database_path: str,
                                      overwrite: bool = True,
                                      overwrite_spectrograms: bool = False) -> str:
        """
        Generate PDF for manual phenotype analysis.
        ENHANCED: Now includes all same analyses as automated (transitions, repeats).
        """
        try:
            output_path = self.pdf_output_dir / f'{self.bird_name}_manual_phenotypes.pdf'

            if output_path.exists() and not overwrite:
                logging.info(f"Manual PDF already exists: {output_path}")
                return str(output_path)

            # Load syllable database for duration/gap statistics
            syllable_stats = self._load_syllable_database_stats(syllable_database_path)

            # Find existing phenotype plots (now includes manual plots)
            plot_paths = self._find_phenotype_plots('manual')

            # Generate example spectrograms
            spec_images = self._generate_example_spectrograms(n_specs=8, label_type='manual',
                                                            overwrite_spectrograms=overwrite_spectrograms)

            # Create PDF using ReportLab canvas
            c = canvas.Canvas(str(output_path), pagesize=landscape(letter))
            width, height = landscape(letter)

            # Page 1: Header (no clustering info for manual)
            self._add_phenotype_header(c, width, height, phenotype_results, 'Manual Labels')
            c.showPage()

            # Page 2: Statistics tables (same structure as automated)
            self._add_basic_phenotype_stats(c, width, height, phenotype_results)
            self._add_syllable_statistics_table(c, width, height, phenotype_results, syllable_stats)
            self._add_repeat_statistics_table(c, width, height, phenotype_results)
            c.showPage()

            # Page 3: Analysis images (transitions, repeats, vocabulary comparison)
            self._add_analysis_images_page(c, width, height, plot_paths)
            c.showPage()

            # Page 4: UMAP comparison (manual vs automated)
            self._add_umap_comparison_page(c, width, height, rank=0)
            c.showPage()

            # Page 5: Spectrogram examples
            self._add_spectrogram_examples_page(c, width, height, spec_images, 'manual', phenotype_results)

            # Save PDF
            c.save()

            logging.info(f"Generated manual phenotype PDF: {output_path}")
            return str(output_path)

        except Exception as e:
            logging.error(f"Error generating manual phenotype PDF: {e}")
            return ""


    def generate_automated_phenotype_pdf(self, phenotype_results: Dict[str, Any],
                                         clustering_metadata: Dict[str, Any],
                                         syllable_database_path: str,
                                         rank: int = 0,
                                         overwrite: bool = True,
                                         overwrite_spectrograms: bool = False) -> str:
        """
        Generate PDF for automated phenotype analysis.
        ENHANCED: Clustering info moved to title page, same structure as manual.
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

            # Generate example spectrograms
            spec_images = self._generate_example_spectrograms(n_specs=8, label_type='automated',
                                                              rank=rank, overwrite_spectrograms=overwrite_spectrograms)

            # Create PDF using ReportLab canvas
            c = canvas.Canvas(str(output_path), pagesize=landscape(letter))
            width, height = landscape(letter)

            # Page 1: Header WITH clustering info
            self._add_phenotype_header(c, width, height, phenotype_results,
                                       f'Automated Labels (Rank {rank})', clustering_metadata)
            c.showPage()

            # Page 2: Statistics tables (same structure as manual)
            self._add_basic_phenotype_stats(c, width, height, phenotype_results)
            self._add_syllable_statistics_table(c, width, height, phenotype_results, syllable_stats)
            self._add_repeat_statistics_table(c, width, height, phenotype_results)
            c.showPage()

            # Page 3: Analysis images (transitions, repeats, vocabulary comparison)
            self._add_analysis_images_page(c, width, height, plot_paths)
            c.showPage()

            # Page 4: UMAP comparison (manual vs automated)
            self._add_umap_comparison_page(c, width, height, rank=rank)
            c.showPage()

            # Page 5: Spectrogram examples
            self._add_spectrogram_examples_page(c, width, height, spec_images, 'automated', phenotype_results, rank)

            # Save PDF
            c.save()

            logging.info(f"Generated automated phenotype PDF: {output_path}")
            return str(output_path)

        except Exception as e:
            logging.error(f"Error generating automated phenotype PDF: {e}")
            return ""


    def _add_analysis_images_page(self, c: canvas.Canvas, width: float, height: float,
                                  plot_paths: PhenotypePDFPaths):
        """
        Add analysis images page with improved layout.
        FIXED: Repeat matrix on left, vocabulary comparison on right.
        """
        # Page title
        c.setFont("Helvetica-Bold", 14)
        title_text = "Phenotype Analysis"
        title_width = c.stringWidth(title_text, "Helvetica-Bold", 14)
        c.drawString((width - title_width) / 2, height - 50, title_text)

        # Image dimensions
        large_image_width = 350
        large_image_height = 280
        small_image_width = 300
        small_image_height = 200

        y_start = height - 100

        # Top row: Transition matrices (side by side)
        if plot_paths.transition_counts_img and os.path.exists(plot_paths.transition_counts_img):
            c.drawImage(plot_paths.transition_counts_img,
                        x=50, y=y_start - large_image_height,
                        width=large_image_width, height=large_image_height)

        if plot_paths.transition_matrix_img and os.path.exists(plot_paths.transition_matrix_img):
            c.drawImage(plot_paths.transition_matrix_img,
                        x=50 + large_image_width + 20, y=y_start - large_image_height,
                        width=large_image_width, height=large_image_height)

        # Bottom row: Repeat patterns (left) and vocabulary comparison (right)
        y_bottom = y_start - large_image_height - 50

        if plot_paths.repeat_patterns_img and os.path.exists(plot_paths.repeat_patterns_img):
            c.drawImage(plot_paths.repeat_patterns_img,
                        x=50, y=y_bottom - small_image_height,
                        width=small_image_width, height=small_image_height)

        if plot_paths.vocabulary_comparison_img and os.path.exists(plot_paths.vocabulary_comparison_img):
            c.drawImage(plot_paths.vocabulary_comparison_img,
                        x=50 + small_image_width + 70, y=y_bottom - small_image_height,
                        width=small_image_width, height=small_image_height)


    def _add_umap_comparison_page(self, c: canvas.Canvas, width: float, height: float, rank: int = 0):
        """
        Add UMAP comparison page showing manual vs automated clustering.
        NEW: Shows both manual and automated UMAP plots side by side.
        """
        # Page title
        c.setFont("Helvetica-Bold", 14)
        title_text = f"UMAP Embedding Comparison - Manual vs Automated (Rank {rank})"
        title_width = c.stringWidth(title_text, "Helvetica-Bold", 14)
        c.drawString((width - title_width) / 2, height - 50, title_text)

        # Image dimensions for side-by-side layout
        image_width = 350
        image_height = 300
        y_position = height - 100

        # Manual UMAP (left side)
        manual_umap_path = self.bird_path / 'figures' / 'clusters' / 'manual_labels_umap.jpg'
        if manual_umap_path.exists():
            c.drawImage(str(manual_umap_path),
                        x=50, y=y_position - image_height,
                        width=image_width, height=image_height)

            # Label for manual plot
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50 + image_width // 2 - 30, y_position - image_height - 20, "Manual Labels")

        # Automated UMAP (right side)
        clusters_dir = self.bird_path / 'figures' / 'clusters'

        # Look for automated clustering plot matching the rank
        automated_umap_pattern = f"euclidean_*_clusters.jpg"
        automated_umap_files = list(clusters_dir.glob(automated_umap_pattern))

        if automated_umap_files:
            # Use the first matching file (should correspond to rank 0 parameters)
            automated_umap_path = automated_umap_files[0]
            c.drawImage(str(automated_umap_path),
                        x=50 + image_width + 50, y=y_position - image_height,
                        width=image_width, height=image_height)
            # Label for automated plot
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50 + image_width + 50 + image_width // 2 - 40, y_position - image_height - 20,
                         f"Automated Labels (Rank {rank})")

        # Add descriptive text
        c.setFont("Helvetica", 10)
        description_y = y_position - image_height - 60
        description_text = [
            "Left: UMAP embedding colored by manual syllable labels",
            "Right: UMAP embedding colored by automated clustering labels",
            "Both plots use the same UMAP parameters and feature space for direct comparison"
        ]

        for i, text in enumerate(description_text):
            text_width = c.stringWidth(text, "Helvetica", 10)
            c.drawString((width - text_width) / 2, description_y - (i * 15), text)


    def _add_spectrogram_examples_page(self, c: canvas.Canvas, width: float, height: float,
                                       spec_paths: List[str], label_type: str,
                                       phenotype_results: Dict[str, Any], rank: int = 0):
        """
        Add spectrogram examples page with improved layout and compact legend.
        FIXED: Better spacing, smaller spectrograms, more compact legend.
        """
        if not spec_paths:
            return

        # Page title
        c.setFont("Helvetica-Bold", 14)
        title_text = "Example Spectrograms"
        c.drawString(50, height - 50, title_text)

        # Adjusted dimensions for better fit
        image_width = 300  # Reduced from 320
        image_height = 110  # Reduced from 120
        spacing_x = 15  # Reduced spacing
        spacing_y = 10  # Reduced spacing

        # Legend dimensions - more compact
        legend_width = 120  # Reduced from 150
        legend_x = 50 + (2 * image_width) + (2 * spacing_x) + 15

        # Get all possible labels for legend
        all_labels = self._get_all_possible_labels(phenotype_results, label_type, rank)

        # Add spectrograms in 4x2 grid
        start_y = height - 90  # Moved up slightly

        for row in range(4):
            y_position = start_y - (row * (image_height + spacing_y))

            for col in range(2):
                spec_idx = row * 2 + col
                if spec_idx < len(spec_paths) and os.path.exists(spec_paths[spec_idx]):
                    x_position = 50 + col * (image_width + spacing_x)
                    c.drawImage(spec_paths[spec_idx], x=x_position, y=y_position - image_height,
                                width=image_width, height=image_height)

            # Add compact legend on the first row only
            if row == 0:
                self._add_compact_spectrogram_legend(c, all_labels, legend_x, y_position - image_height, legend_width)


    def _add_compact_spectrogram_legend(self, c: canvas.Canvas, labels: List[str],
                                        legend_x: float, legend_y: float, legend_width: float):
        """
        Add compact vertical legend for dual-labeled spectrograms.
        FIXED: More compact layout, better fit within page boundaries.
        """
        if not labels:
            return

        # Compact legend styling
        box_size = 10  # Reduced from 12
        text_height = 12  # Reduced from 15
        legend_font_size = 9  # Reduced from 10

        # Check if legend will fit on page
        max_labels = min(len(labels), 15)  # Limit labels
        total_legend_height = 80 + (max_labels * text_height)

        if legend_y - total_legend_height < 50:  # If legend would go off page
            max_labels = min(max_labels, int((legend_y - 100) / text_height))

        c.setFont("Helvetica-Bold", legend_font_size)

        # Legend title
        current_y = legend_y + 30  # Reduced spacing
        c.drawString(legend_x, current_y, "Dual Labels:")
        current_y -= 15

        c.setFont("Helvetica", 7)  # Smaller font for descriptions
        c.drawString(legend_x, current_y, "Manual: above")
        current_y -= 10
        c.drawString(legend_x, current_y, "Auto: below")
        current_y -= 15

        c.setFont("Helvetica", legend_font_size)

        # Color palette (same as used in spectrograms)
        colors = plt.cm.Set1(np.linspace(0, 1, 10))

        # Add each label with its color (limited to fit on page)
        for i, label in enumerate(labels[:max_labels]):
            # Calculate color (same logic as in spectrogram creation)
            color_idx = hash(str(label)) % len(colors)
            color_rgb = colors[color_idx][:3]

            # Convert to 0-1 range for ReportLab
            r, g, b = color_rgb[0], color_rgb[1], color_rgb[2]

            # Draw colored box
            c.setFillColorRGB(r, g, b)
            c.rect(legend_x, current_y, box_size, box_size, fill=1, stroke=1)

            # Add label text
            c.setFillColorRGB(0, 0, 0)  # Reset to black text
            c.drawString(legend_x + box_size + 3, current_y + 2, str(label))

            current_y -= text_height

            # Stop if we're approaching page bottom
            if current_y < legend_y - 250:
                break

        # Add "..." if we had to truncate labels
        if len(labels) > max_labels:
            c.setFont("Helvetica", 8)
            c.drawString(legend_x, current_y, f"... and {len(labels) - max_labels} more")


def generate_phenotype_pdfs_from_saved_data(bird_path: str,
                                            overwrite: bool = True,
                                            overwrite_spectrograms: bool = False,
                                            rank: int = 0) -> Dict[str, str]:
    """
    Generate phenotype PDFs from saved detailed phenotype data.
    Simplified to focus on rank 0 by default with option for other ranks.
    """
    try:
        generator = PhenotypePDFGenerator(bird_path)
        generated_pdfs = {}

        # Path to detailed phenotype data
        from song_phenotyping.tools.pipeline_paths import PHENOTYPE_DIR, STAGES_DIR
        detailed_data_dir = os.path.join(bird_path, PHENOTYPE_DIR)
        syllable_db_path = os.path.join(bird_path, STAGES_DIR, 'syllable_database', 'syllable_features.csv')

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
                    manual_results, syllable_db_path, overwrite, overwrite_spectrograms
                )
                if manual_pdf:
                    generated_pdfs['manual'] = manual_pdf

            except Exception as e:
                logging.error(f"Error loading manual phenotype data: {e}")

        # Generate automated PDF for specified rank (default rank 0)
        auto_data_path = os.path.join(detailed_data_dir, f'automated_phenotype_data_rank{rank}.pkl')
        if os.path.exists(auto_data_path):
            try:
                with open(auto_data_path, 'rb') as f:
                    auto_data = pkl.load(f)

                auto_results = auto_data['phenotype_results']
                clustering_metadata = auto_data['clustering_metadata']

                auto_pdf = generator.generate_automated_phenotype_pdf(
                    auto_results, clustering_metadata, syllable_db_path, rank, overwrite, overwrite_spectrograms
                )
                if auto_pdf:
                    generated_pdfs['automated'] = auto_pdf

            except Exception as e:
                logging.error(f"Error loading automated phenotype data: {e}")

        return generated_pdfs

    except Exception as e:
        logging.error(f"Error generating phenotype PDFs from saved data: {e}")
        return {}


def integrate_with_phenotyping_pipeline(bird_path: str, config, rank: int = 0) -> Dict[str, str]:
    """
    Integration function to be called from the main phenotyping pipeline.
    Simplified to focus on best rank (rank 0) by default.

    Args:
        bird_path: Path to bird directory
        config: PhenotypingConfig object
        rank: Clustering rank to generate PDFs for (default 0 = best)

    Returns:
        Dict mapping PDF type to generated file path
    """
    try:
        if not config.generate_plots:
            logging.info(f"Plot generation disabled for {os.path.basename(bird_path)}")
            return {}

        # Generate PDFs from saved detailed data for specified rank
        generated_pdfs = generate_phenotype_pdfs_from_saved_data(
            bird_path, overwrite=True, overwrite_spectrograms=False, rank=rank
        )

        if generated_pdfs:
            bird_name = os.path.basename(bird_path)
            logging.info(f"Generated phenotype PDFs for {bird_name} (rank {rank}): {list(generated_pdfs.keys())}")

        return generated_pdfs

    except Exception as e:
        logging.error(f"Error in phenotype PDF integration: {e}")
        return {}


def batch_generate_phenotype_pdfs_from_saved_data(bird_paths: List[str],
                                                  overwrite: bool = True,
                                                  overwrite_spectrograms: bool = False,
                                                  rank: int = 0) -> Dict[str, Dict[str, str]]:
    """
    Generate phenotype PDFs for multiple birds using saved detailed data.
    Simplified to process single rank (default 0) for efficiency.

    Args:
        bird_paths: List of paths to bird directories
        overwrite: Whether to overwrite existing PDFs
        overwrite_spectrograms: Whether to regenerate existing spectrograms
        rank: Clustering rank to process (default 0 = best)

    Returns:
        Dict mapping bird names to their generated PDF paths
    """
    all_pdfs = {}

    for bird_path in bird_paths:
        bird_name = os.path.basename(bird_path)

        try:
            bird_pdfs = generate_phenotype_pdfs_from_saved_data(
                bird_path, overwrite, overwrite_spectrograms, rank
            )

            if bird_pdfs:
                all_pdfs[bird_name] = bird_pdfs
                logging.info(f"Generated {len(bird_pdfs)} PDFs for {bird_name} (rank {rank})")
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
        from song_phenotyping.tools.pipeline_paths import PHENOTYPE_DIR
        detailed_data_dir = os.path.join(bird_path, PHENOTYPE_DIR)

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


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test on example datasets
    test_paths = [
        # os.path.join('/Volumes', 'Extreme SSD', 'wseg test', 'bu85bu97'),
        # os.path.join('/Volumes', 'Extreme SSD', 'evsong test', 'or18or24')
        Path('E:/') / 'xfosters' / 'bk1bk3'
    ]

    for example_bird_path in test_paths:
        if not os.path.exists(example_bird_path):
            print(f"Path does not exist: {example_bird_path}")
            continue

        try:
            print("Testing Phenotype PDF Generation Pipeline...")

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

            # Test PDF generation from saved data using ReportLab
            print("\n2. Testing PDF generation from saved detailed data...")
            generated_pdfs = generate_phenotype_pdfs_from_saved_data(
                bird_path=example_bird_path,
                overwrite=True,
                overwrite_spectrograms=False  # Don't regenerate spectrograms by default
            )

            if generated_pdfs:
                print(f"✓ Generated PDFs from saved data: {list(generated_pdfs.keys())}")
                for pdf_type, pdf_path in generated_pdfs.items():
                    print(f"  - {pdf_type}: {pdf_path}")
            else:
                print("✗ No PDFs generated from saved data")

        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

