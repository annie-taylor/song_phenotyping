# cluster_pdfs.py

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import pandas as pd
import os
import tables
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import Counter
import random

# Import your existing functions
from song_phenotyping.signal import logger


@dataclass
class ClusterPDFConfig:
    """Configuration for cluster PDF creation."""

    # Sampling settings
    max_samples_per_cluster: int = 100
    min_samples_per_cluster: int = 5
    sampling_strategy: str = 'random'  # 'random', 'spread', 'representative'
    random_seed: int = 42

    # PDF layout
    grid_size: Tuple[int, int] = (5, 5)  # (n_cols, n_rows)
    figure_size: Tuple[float, float] = (20, 20)
    images_per_page: int = 25  # Should match grid_size

    # Appearance
    colormap: str = 'viridis'
    show_hash_ids: bool = True
    show_ground_truth: bool = True
    hash_length: int = 8
    title_fontsize: int = 8

    # Processing options
    skip_noise_cluster: bool = True  # Skip cluster -1 (HDBSCAN noise)
    skip_existing: bool = True
    create_summary_page: bool = True

    # Output options
    separate_pdfs: bool = True  # One PDF per cluster vs. one combined PDF
    include_metadata: bool = True


@dataclass
class ClusterData:
    """Data structure for a single cluster."""
    cluster_id: Any
    spectrograms: List[np.ndarray] = field(default_factory=list)
    hash_ids: List[str] = field(default_factory=list)
    ground_truth_labels: List[str] = field(default_factory=list)
    sample_indices: List[int] = field(default_factory=list)  # Original indices in full dataset


class ClusterPDFCreator:
    """Create PDFs with spectrograms organized by cluster labels."""

    def __init__(self, config: ClusterPDFConfig = None):
        self.config = config or ClusterPDFConfig()

    def create_cluster_pdfs(self, embeddings_path: str, cluster_labels_path: str,
                            bird_path: str, output_dir: str, bird_name: str) -> Dict[str, Any]:
        """
        Create cluster PDFs from embeddings and cluster labels.

        Args:
            embeddings_path: Path to UMAP embeddings HDF5 file
            cluster_labels_path: Path to cluster labels HDF5 file
            bird_path: Path to bird directory (for loading spectrograms)
            output_dir: Directory to save PDFs
            bird_name: Bird identifier for filenames

        Returns:
            Dict with results and metadata
        """

        logger.info(f"Creating cluster PDFs for {bird_name}")

        try:
            # Load all required data
            data = self._load_all_data(embeddings_path, cluster_labels_path, bird_path)
            if not data:
                return {'success': False, 'error': 'Failed to load data'}

            # Organize data by clusters
            cluster_data = self._organize_by_clusters(data)
            if not cluster_data:
                return {'success': False, 'error': 'No valid clusters found'}

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create PDFs
            if self.config.separate_pdfs:
                pdf_paths = self._create_separate_pdfs(cluster_data, output_dir, bird_name)
            else:
                pdf_paths = self._create_combined_pdf(cluster_data, output_dir, bird_name)

            # Create summary
            summary = self._create_summary(cluster_data, bird_name)

            logger.info(f"✅ Created cluster PDFs for {bird_name}: {len(pdf_paths)} files")

            return {
                'success': True,
                'pdf_paths': pdf_paths,
                'summary': summary,
                'n_clusters': len(cluster_data),
                'total_spectrograms': sum(len(cd.spectrograms) for cd in cluster_data.values())
            }

        except Exception as e:
            logger.error(f"Error creating cluster PDFs for {bird_name}: {e}")
            return {'success': False, 'error': str(e)}

    def _load_all_data(self, embeddings_path: str, cluster_labels_path: str,
                       bird_path: str) -> Optional[Dict]:
        """Load embeddings, cluster labels, and spectrograms."""

        try:
            # Load embeddings and ground truth labels
            embeddings, hashes, gt_labels = self._load_embeddings(embeddings_path)
            if embeddings is None:
                return None

            # Load cluster labels
            cluster_labels, cluster_hashes, scores = self._load_cluster_labels(cluster_labels_path)
            if cluster_labels is None:
                return None

            # Verify hash alignment
            if len(hashes) != len(cluster_hashes) or not all(h1 == h2 for h1, h2 in zip(hashes, cluster_hashes)):
                logger.error("Hash mismatch between embeddings and cluster labels")
                return None

            logger.info(f"Loaded {len(hashes)} samples with {len(np.unique(cluster_labels))} unique clusters")

            return {
                'hashes': hashes,
                'ground_truth_labels': gt_labels,
                'cluster_labels': cluster_labels,
                'bird_path': bird_path,
                'cluster_scores': scores
            }

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def _load_embeddings(self, embeddings_path: str) -> Tuple[
        Optional[np.ndarray], Optional[List[str]], Optional[List[str]]]:
        """Load UMAP embeddings."""
        try:
            with tables.open_file(embeddings_path, mode='r') as f:
                embeddings = f.root.embeddings.read()
                hashes = [hash_id.decode('utf-8') for hash_id in f.root.hashes.read()]
                labels = [label.decode('utf-8') for label in f.root.labels.read()]
                return embeddings, hashes, labels
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None, None, None

    def _load_cluster_labels(self, cluster_labels_path: str) -> Tuple[
        Optional[np.ndarray], Optional[List[str]], Optional[Dict]]:
        """Load cluster labels."""
        try:
            with tables.open_file(cluster_labels_path, mode='r') as f:
                labels = f.root.labels.read()
                hashes_raw = f.root.hashes.read()
                hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h) for h in hashes_raw]

                # Load scores
                scores = {}
                for node in f.list_nodes(f.root, classname='Array'):
                    if node._v_name not in ['labels', 'hashes']:
                        score_value = node.read()
                        scores[node._v_name] = float(score_value) if score_value.ndim == 0 else score_value

                return labels, hashes, scores
        except Exception as e:
            logger.error(f"Error loading cluster labels: {e}")
            return None, None, None

    def _organize_by_clusters(self, data: Dict) -> Dict[Any, ClusterData]:
        """Organize data by cluster labels and load spectrograms."""

        cluster_data = {}
        unique_clusters = np.unique(data['cluster_labels'])

        # Skip noise cluster if configured
        if self.config.skip_noise_cluster:
            unique_clusters = unique_clusters[unique_clusters != -1]

        logger.info(f"Processing {len(unique_clusters)} clusters")

        for cluster_id in tqdm(unique_clusters, desc="Loading cluster spectrograms"):
            try:
                # Get indices for this cluster
                cluster_indices = np.where(data['cluster_labels'] == cluster_id)[0]

                if len(cluster_indices) < self.config.min_samples_per_cluster:
                    logger.debug(f"Skipping cluster {cluster_id}: only {len(cluster_indices)} samples")
                    continue

                # Sample indices if too many
                sampled_indices = self._sample_cluster_indices(cluster_indices, cluster_id)

                # Load spectrograms for sampled indices
                spectrograms = []
                valid_hashes = []
                valid_gt_labels = []
                valid_indices = []

                for idx in sampled_indices:
                    hash_id = data['hashes'][idx]
                    spec = self._load_spectrogram_from_hash(hash_id, data['bird_path'])

                    if spec is not None:
                        spectrograms.append(spec)
                        valid_hashes.append(hash_id)
                        valid_gt_labels.append(data['ground_truth_labels'][idx])
                        valid_indices.append(idx)

                if spectrograms:
                    cluster_data[cluster_id] = ClusterData(
                        cluster_id=cluster_id,
                        spectrograms=spectrograms,
                        hash_ids=valid_hashes,
                        ground_truth_labels=valid_gt_labels,
                        sample_indices=valid_indices
                    )

                    logger.debug(f"Cluster {cluster_id}: loaded {len(spectrograms)} spectrograms")
                else:
                    logger.warning(f"No spectrograms loaded for cluster {cluster_id}")

            except Exception as e:
                logger.error(f"Error processing cluster {cluster_id}: {e}")
                continue

        logger.info(f"Successfully organized {len(cluster_data)} clusters")
        return cluster_data

    def _sample_cluster_indices(self, cluster_indices: np.ndarray, cluster_id: Any) -> np.ndarray:
        """Sample indices from a cluster based on configured strategy."""

        if len(cluster_indices) <= self.config.max_samples_per_cluster:
            return cluster_indices

        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        n_samples = self.config.max_samples_per_cluster

        if self.config.sampling_strategy == 'random':
            return np.random.choice(cluster_indices, size=n_samples, replace=False)

        elif self.config.sampling_strategy == 'spread':
            # Evenly spaced sampling
            step = len(cluster_indices) // n_samples
            return cluster_indices[::step][:n_samples]

        elif self.config.sampling_strategy == 'representative':
            # First few, last few, and some from middle
            n_start = n_samples // 3
            n_end = n_samples // 3
            n_middle = n_samples - n_start - n_end

            start_indices = cluster_indices[:n_start]
            end_indices = cluster_indices[-n_end:]

            if n_middle > 0:
                middle_start = len(cluster_indices) // 4
                middle_end = 3 * len(cluster_indices) // 4
                middle_indices = np.random.choice(
                    cluster_indices[middle_start:middle_end],
                    size=n_middle,
                    replace=False
                )
                return np.concatenate([start_indices, middle_indices, end_indices])
            else:
                return np.concatenate([start_indices, end_indices])

        else:
            logger.warning(f"Unknown sampling strategy: {self.config.sampling_strategy}, using random")
            return np.random.choice(cluster_indices, size=n_samples, replace=False)

    def _load_spectrogram_from_hash(self, hash_id: str, bird_path: str) -> Optional[np.ndarray]:
        """Load spectrogram from hash ID."""

        try:
            syllables_dir = os.path.join(bird_path, 'data', 'syllables')

            if not os.path.exists(syllables_dir):
                return None

            syllable_files = [f for f in os.listdir(syllables_dir)
                              if f.endswith('.h5') and f.startswith('syllables_')]

            for filename in syllable_files:
                file_path = os.path.join(syllables_dir, filename)

                try:
                    with tables.open_file(file_path, mode='r') as f:
                        hashes_raw = f.root.hashes.read()
                        hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h)
                                  for h in hashes_raw]

                        if hash_id in hashes:
                            hash_idx = hashes.index(hash_id)
                            return f.root.spectrograms[hash_idx]

                except Exception as e:
                    logger.debug(f"Error reading {filename}: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"Error loading spectrogram for hash {hash_id}: {e}")
            return None

    def _create_separate_pdfs(self, cluster_data: Dict[Any, ClusterData],
                              output_dir: str, bird_name: str) -> List[str]:
        """Create separate PDF for each cluster."""

        pdf_paths = []

        for cluster_id, cd in cluster_data.items():
            try:
                pdf_filename = f'{bird_name}_cluster_{cluster_id}.pdf'
                pdf_path = os.path.join(output_dir, pdf_filename)

                # Skip if exists and configured
                if self.config.skip_existing and os.path.exists(pdf_path):
                    logger.debug(f"Skipping existing PDF: {pdf_filename}")
                    pdf_paths.append(pdf_path)
                    continue

                with pdf_backend.PdfPages(pdf_path) as pdf:
                    # Create summary page if configured
                    if self.config.create_summary_page:
                        if self.config.create_summary_page:
                            summary_fig = self._create_cluster_summary_page(cd, bird_name)
                            if summary_fig:
                                pdf.savefig(summary_fig, bbox_inches='tight')
                                plt.close(summary_fig)

                            # Create spectrogram pages
                        self._create_spectrogram_pages(pdf, cd, bird_name)

                        pdf_paths.append(pdf_path)
                        logger.debug(f"Created PDF for cluster {cluster_id}: {len(cd.spectrograms)} spectrograms")

            except Exception as e:
                logger.error(f"Error creating PDF for cluster {cluster_id}: {e}")
                continue

        return pdf_paths

    def _create_combined_pdf(self, cluster_data: Dict[Any, ClusterData],
                             output_dir: str, bird_name: str) -> List[str]:
        """Create single combined PDF with all clusters."""

        pdf_filename = f'{bird_name}_all_clusters.pdf'
        pdf_path = os.path.join(output_dir, pdf_filename)

        # Skip if exists and configured
        if self.config.skip_existing and os.path.exists(pdf_path):
            logger.debug(f"Skipping existing combined PDF: {pdf_filename}")
            return [pdf_path]

        try:
            with pdf_backend.PdfPages(pdf_path) as pdf:
                # Create overall summary page
                if self.config.create_summary_page:
                    overall_summary_fig = self._create_overall_summary_page(cluster_data, bird_name)
                    if overall_summary_fig:
                        pdf.savefig(overall_summary_fig, bbox_inches='tight')
                        plt.close(overall_summary_fig)

                # Process each cluster
                for cluster_id, cd in cluster_data.items():
                    try:
                        # Individual cluster summary
                        cluster_summary_fig = self._create_cluster_summary_page(cd, bird_name)
                        if cluster_summary_fig:
                            pdf.savefig(cluster_summary_fig, bbox_inches='tight')
                            plt.close(cluster_summary_fig)

                        # Spectrogram pages for this cluster
                        self._create_spectrogram_pages(pdf, cd, bird_name)

                    except Exception as e:
                        logger.error(f"Error adding cluster {cluster_id} to combined PDF: {e}")
                        continue

            logger.info(f"Created combined PDF: {pdf_path}")
            return [pdf_path]

        except Exception as e:
            logger.error(f"Error creating combined PDF: {e}")
            return []

    def _create_cluster_summary_page(self, cluster_data: ClusterData, bird_name: str) -> Optional[
        plt.Figure]:
        """Create summary page for a single cluster."""

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{bird_name} - Cluster {cluster_data.cluster_id} Summary',
                         fontsize=16, fontweight='bold')

            # Ground truth label distribution
            gt_labels = cluster_data.ground_truth_labels
            if gt_labels and any(label != "N/A" for label in gt_labels):
                label_counts = Counter(gt_labels)

                axes[0, 0].bar(range(len(label_counts)), list(label_counts.values()), alpha=0.7)
                axes[0, 0].set_xlabel('Ground Truth Label')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title('Ground Truth Label Distribution')
                axes[0, 0].set_xticks(range(len(label_counts)))
                axes[0, 0].set_xticklabels(list(label_counts.keys()), rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, 'No ground truth labels available',
                                ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Ground Truth Label Distribution')

            # Sample spectrograms
            n_preview = min(6, len(cluster_data.spectrograms))
            preview_indices = np.linspace(0, len(cluster_data.spectrograms) - 1, n_preview, dtype=int)

            axes[0, 1].set_title('Sample Spectrograms')
            axes[0, 1].axis('off')

            for i, idx in enumerate(preview_indices):
                spec = cluster_data.spectrograms[idx]

                # Calculate subplot position
                row = i // 3
                col = i % 3
                left = col * 0.33
                bottom = 0.5 - row * 0.5
                width = 0.3
                height = 0.4

                mini_ax = fig.add_axes(
                    [0.55 + left * 0.33, 0.55 + bottom * 0.4, width * 0.33, height * 0.4])
                mini_ax.imshow(spec, aspect='auto', origin='lower', cmap=self.config.colormap)
                mini_ax.set_xticks([])
                mini_ax.set_yticks([])

                # Add label if available
                if self.config.show_ground_truth and cluster_data.ground_truth_labels:
                    label = cluster_data.ground_truth_labels[idx]
                    mini_ax.set_title(f"{label}", fontsize=6)

            # Statistics table
            axes[1, 0].axis('off')
            stats_data = [
                ['Cluster ID', str(cluster_data.cluster_id)],
                ['Total Spectrograms', str(len(cluster_data.spectrograms))],
                ['Sampling Strategy', self.config.sampling_strategy],
                ['Unique GT Labels', str(len(
                    set(cluster_data.ground_truth_labels))) if cluster_data.ground_truth_labels else 'N/A']
            ]

            table = axes[1, 0].table(cellText=stats_data,
                                     colLabels=['Metric', 'Value'],
                                     cellLoc='center',
                                     loc='center',
                                     bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 0].set_title('Cluster Statistics')

            # Hash ID preview (first few)
            axes[1, 1].axis('off')
            hash_preview = cluster_data.hash_ids[:10]  # First 10 hashes
            hash_text = "Hash ID Preview:\n" + "\n".join(
                f"{i + 1:2d}. {h[:12]}..." for i, h in enumerate(hash_preview))

            axes[1, 1].text(0.05, 0.95, hash_text, transform=axes[1, 1].transAxes,
                            fontsize=8, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Sample Hash IDs')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating cluster summary page: {e}")
            return None

    def _create_overall_summary_page(self, cluster_data: Dict[Any, ClusterData], bird_name: str) -> (
            Optional)[plt.Figure]:
        """Create overall summary page for all clusters."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{bird_name} - All Clusters Summary', fontsize=16, fontweight='bold')

            # Cluster size distribution
            cluster_ids = list(cluster_data.keys())
            cluster_sizes = [len(cd.spectrograms) for cd in cluster_data.values()]

            axes[0, 0].bar(range(len(cluster_ids)), cluster_sizes, alpha=0.7)
            axes[0, 0].set_xlabel('Cluster ID')
            axes[0, 0].set_ylabel('Number of Spectrograms')
            axes[0, 0].set_title('Cluster Size Distribution')
            axes[0, 0].set_xticks(range(len(cluster_ids)))
            axes[0, 0].set_xticklabels([str(cid) for cid in cluster_ids])

            # Ground truth label distribution across all clusters
            all_gt_labels = []
            for cd in cluster_data.values():
                all_gt_labels.extend(cd.ground_truth_labels)

            if all_gt_labels and any(label != "N/A" for label in all_gt_labels):
                gt_counts = Counter(all_gt_labels)

                axes[0, 1].bar(range(len(gt_counts)), list(gt_counts.values()), alpha=0.7, color='orange')
                axes[0, 1].set_xlabel('Ground Truth Label')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Overall Ground Truth Distribution')
                axes[0, 1].set_xticks(range(len(gt_counts)))
                axes[0, 1].set_xticklabels(list(gt_counts.keys()), rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No ground truth labels available',
                                ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Overall Ground Truth Distribution')

            # Overall statistics table
            axes[1, 0].axis('off')
            total_spectrograms = sum(len(cd.spectrograms) for cd in cluster_data.values())
            unique_gt_labels = len(set(all_gt_labels)) if all_gt_labels else 0

            overall_stats = [
                ['Total Clusters', str(len(cluster_data))],
                ['Total Spectrograms', str(total_spectrograms)],
                ['Unique GT Labels', str(unique_gt_labels)],
                ['Avg Spectrograms/Cluster', f"{total_spectrograms / len(cluster_data):.1f}"],
                ['Largest Cluster', str(max(cluster_sizes)) if cluster_sizes else '0'],
                ['Smallest Cluster', str(min(cluster_sizes)) if cluster_sizes else '0']
            ]

            table = axes[1, 0].table(cellText=overall_stats,
                                     colLabels=['Metric', 'Value'],
                                     cellLoc='center',
                                     loc='center',
                                     bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 0].set_title('Overall Statistics')

            # Cluster composition heatmap (GT labels vs clusters)
            if all_gt_labels and any(label != "N/A" for label in all_gt_labels):
                axes[1, 1] = self._create_composition_heatmap(axes[1, 1], cluster_data)
            else:
                axes[1, 1].text(0.5, 0.5, 'No ground truth labels\nfor composition analysis',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Cluster Composition')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating overall summary page: {e}")
            return None


    def _create_composition_heatmap(self, ax, cluster_data: Dict[Any, ClusterData]):
        """Create heatmap showing ground truth label composition of clusters."""

        try:
            # Collect all unique GT labels
            all_gt_labels = []
            for cd in cluster_data.values():
                all_gt_labels.extend(cd.ground_truth_labels)

            unique_gt_labels = sorted(list(set(all_gt_labels)))
            cluster_ids = sorted(list(cluster_data.keys()))

            # Create composition matrix
            composition_matrix = np.zeros((len(unique_gt_labels), len(cluster_ids)))

            for j, cluster_id in enumerate(cluster_ids):
                cd = cluster_data[cluster_id]
                gt_counts = Counter(cd.ground_truth_labels)

                for i, gt_label in enumerate(unique_gt_labels):
                    composition_matrix[i, j] = gt_counts.get(gt_label, 0)

            # Normalize by cluster size to show proportions
            cluster_sizes = np.sum(composition_matrix, axis=0)
            composition_matrix = composition_matrix / cluster_sizes[np.newaxis, :]
            composition_matrix = np.nan_to_num(composition_matrix)  # Handle division by zero

            # Create heatmap
            im = ax.imshow(composition_matrix, cmap='Blues', aspect='auto')

            # Set labels
            ax.set_xticks(range(len(cluster_ids)))
            ax.set_xticklabels([str(cid) for cid in cluster_ids])
            ax.set_yticks(range(len(unique_gt_labels)))
            ax.set_yticklabels(unique_gt_labels)
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Ground Truth Label')
            ax.set_title('Cluster Composition (Proportions)')

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.6)

            return ax

        except Exception as e:
            logger.error(f"Error creating composition heatmap: {e}")
            return ax


    def _create_spectrogram_pages(self, pdf: pdf_backend.PdfPages,
                                  cluster_data: ClusterData, bird_name: str):
        """Create spectrogram grid pages for a cluster."""

        spectrograms = cluster_data.spectrograms
        hash_ids = cluster_data.hash_ids
        gt_labels = cluster_data.ground_truth_labels

        n_cols, n_rows = self.config.grid_size
        images_per_page = self.config.images_per_page

        for page_start in range(0, len(spectrograms), images_per_page):
            try:
                page_specs = spectrograms[page_start:page_start + images_per_page]
                page_hashes = hash_ids[page_start:page_start + images_per_page]
                page_gt_labels = gt_labels[page_start:page_start + images_per_page] if gt_labels else None

                fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figure_size)

                # Handle subplot array shapes
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                # Page title
                page_num = (page_start // images_per_page) + 1
                total_pages = (len(spectrograms) + images_per_page - 1) // images_per_page
                fig.suptitle(f'{bird_name} - Cluster {cluster_data.cluster_id} '
                             f'(Page {page_num}/{total_pages})',
                             fontsize=14, fontweight='bold')

                # Plot spectrograms
                for i, spec in enumerate(page_specs):
                    ax = axes[i]

                    # Plot spectrogram
                    ax.imshow(spec, aspect='auto', origin='lower', cmap=self.config.colormap)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Create title with available information
                    title_parts = []

                    if self.config.show_hash_ids:
                        hash_id = page_hashes[i]
                        title_parts.append(f'{hash_id[:self.config.hash_length]}...')

                    if self.config.show_ground_truth and page_gt_labels:
                        gt_label = page_gt_labels[i]
                        if gt_label != "N/A":
                            title_parts.append(f'GT:{gt_label}')

                    if title_parts:
                        ax.set_title(' | '.join(title_parts), fontsize=self.config.title_fontsize)

                # Hide unused subplots
                for i in range(len(page_specs), len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                logger.error(f"Error creating spectrogram page: {e}")
                continue

    def _create_summary(self, cluster_data: Dict[Any, ClusterData], bird_name: str) -> Dict:
        """Create summary statistics for the clustering results."""

        summary = {
            'bird_name': bird_name,
            'n_clusters': len(cluster_data),
            'cluster_details': {},
            'total_spectrograms': 0,
            'overall_gt_distribution': {}
        }

        all_gt_labels = []

        for cluster_id, cd in cluster_data.items():
            cluster_summary = {
                'cluster_id': cluster_id,
                'n_spectrograms': len(cd.spectrograms),
                'gt_label_distribution': dict(
                    Counter(cd.ground_truth_labels)) if cd.ground_truth_labels else {},
                'most_common_gt_label': Counter(cd.ground_truth_labels).most_common(1)[0][
                    0] if cd.ground_truth_labels else None
            }

            summary['cluster_details'][str(cluster_id)] = cluster_summary
            summary['total_spectrograms'] += len(cd.spectrograms)
            all_gt_labels.extend(cd.ground_truth_labels)

        if all_gt_labels:
            summary['overall_gt_distribution'] = dict(Counter(all_gt_labels))

        return summary

    # Utility functions for easy usage

def quick_cluster_pdfs(embeddings_path: str, cluster_labels_path: str,
                       bird_path: str, bird_name: str,
                       output_dir: str = None) -> Dict[str, Any]:
    """
    Quick function to create cluster PDFs with defaults.

    Args:
        embeddings_path: Path to UMAP embeddings HDF5 file
        cluster_labels_path: Path to cluster labels HDF5 file
        bird_path: Path to bird directory
        bird_name: Bird identifier
        output_dir: Output directory (auto-generated if None)

    Returns:
        Results dictionary
    """

    if output_dir is None:
        output_dir = os.path.join(bird_path, 'figures', 'cluster_spectrograms')

    creator = ClusterPDFCreator()
    return creator.create_cluster_pdfs(
        embeddings_path, cluster_labels_path, bird_path, output_dir, bird_name
    )

def create_cluster_pdfs_from_best_results(bird_path: str, bird_name: str,
                                          output_dir: str = None) -> Dict[str, Any]:
    """
    Create cluster PDFs using best clustering results for a bird.

    Args:
        bird_path: Path to bird directory
        bird_name: Bird identifier
        output_dir: Output directory (auto-generated if None)

    Returns:
        Results dictionary
    """

    try:
        import pandas as pd

        # Load master summary to find best results
        from song_phenotyping.tools.pipeline_paths import RESULTS_DIR
        master_summary_path = os.path.join(bird_path, RESULTS_DIR, 'master_summary.csv')
        if not os.path.exists(master_summary_path):
            return {'success': False, 'error': 'No master summary found'}

        summary_df = pd.read_csv(master_summary_path)
        if summary_df.empty:
            return {'success': False, 'error': 'Empty master summary'}

        # Get the best result
        best_row = summary_df.iloc[0]

        # Extract file paths
        n_neighbors = best_row['n_neighbors']
        min_dist = best_row['min_dist']
        metric = best_row['metric']
        embedding_filename = f'{metric}_{int(n_neighbors)}neighbors_{min_dist}dist.h5'

        embeddings_path = os.path.join(bird_path, 'data', 'embeddings', embedding_filename)
        cluster_filename = os.path.basename(str(best_row['label_path']))

        # Find cluster file path
        data_path = os.path.join(bird_path, 'data')
        labelling_dir = os.path.join(data_path, 'labelling')
        cluster_labels_path = None

        for root, dirs, files in os.walk(labelling_dir):
            if cluster_filename in files:
                cluster_labels_path = os.path.join(root, cluster_filename)
                break

        if cluster_labels_path is None:
            return {'success': False, 'error': f'Could not find cluster file {cluster_filename}'}

        logger.info(f"Using best results for {bird_name}:")
        logger.info(f"  Embedding: {embedding_filename}")
        logger.info(f"  Clusters: {cluster_filename}")

        # Create PDFs
        return quick_cluster_pdfs(embeddings_path, cluster_labels_path, bird_path, bird_name, output_dir)

    except Exception as e:
        logger.error(f"Error creating cluster PDFs from best results: {e}")
        return {'success': False, 'error': str(e)}

def create_custom_cluster_pdfs(embeddings_path: str, cluster_labels_path: str,
                               bird_path: str, bird_name: str, output_dir: str,
                               config: ClusterPDFConfig) -> Dict[str, Any]:
    """
    Create cluster PDFs with custom configuration.

    Args:
        embeddings_path: Path to UMAP embeddings HDF5 file
        cluster_labels_path: Path to cluster labels HDF5 file
        bird_path: Path to bird directory
        bird_name: Bird identifier
        output_dir: Output directory
        config: Custom configuration

    Returns:
        Results dictionary
    """

    creator = ClusterPDFCreator(config)
    return creator.create_cluster_pdfs(
        embeddings_path, cluster_labels_path, bird_path, output_dir, bird_name
    )

def batch_create_cluster_pdfs(project_directory: str, birds: List[str] = None) -> Dict[str, Dict]:
    """
    Create cluster PDFs for multiple birds using their best results.

    Args:
        project_directory: Path to project directory
        birds: List of birds to process (all if None)

    Returns:
        Dict mapping bird names to results
    """

    if birds is None:
        # Discover birds with master summaries
        project_path = Path(project_directory)
        birds = []
        for item in project_path.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data' and
                    (item / 'results' / 'master_summary.csv').exists()):
                birds.append(item.name)

    logger.info(f"Creating cluster PDFs for {len(birds)} birds")

    results = {}
    successful = 0

    for bird in tqdm(birds, desc="Creating cluster PDFs"):
        bird_path = os.path.join(project_directory, bird)

        try:
            result = create_cluster_pdfs_from_best_results(bird_path, bird)
            results[bird] = result

            if result.get('success'):
                successful += 1
                logger.info(f"✅ {bird}: {result.get('n_clusters', 0)} clusters, "
                            f"{result.get('total_spectrograms', 0)} spectrograms")
            else:
                logger.warning(f"❌ {bird}: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"💥 Error processing {bird}: {e}")
            results[bird] = {'success': False, 'error': str(e)}

    logger.info(f"Batch processing complete: {successful}/{len(birds)} birds successful")
    return results

# Example usage and testing
def example_usage():
    """Example of how to use the cluster PDF module."""

    # Example paths (adjust for your data)
    bird_path = "/path/to/bird/directory"
    bird_name = "bu85bu97"

    # Method 1: Quick PDFs from best results
    print("Creating cluster PDFs from best results...")
    result = create_cluster_pdfs_from_best_results(bird_path, bird_name)
    if result['success']:
        print(f"Created {len(result['pdf_paths'])} PDFs for {result['n_clusters']} clusters")

    # Method 2: Quick PDFs with specific files
    embeddings_path = os.path.join(bird_path, 'data', 'embeddings', 'euclidean_20neighbors_0.1dist.h5')
    cluster_path = os.path.join(bird_path, 'data', 'labelling', 'hdbscan_labels.h5')
    output_dir = os.path.join(bird_path, 'figures', 'cluster_spectrograms')

    result = quick_cluster_pdfs(embeddings_path, cluster_path, bird_path, bird_name, output_dir)

    # Method 3: Custom configuration
    custom_config = ClusterPDFConfig(
        max_samples_per_cluster=50,
        grid_size=(6, 4),
        images_per_page=24,
        sampling_strategy='representative',
        separate_pdfs=False,  # Create combined PDF
        create_summary_page=True,
        show_ground_truth=True,
        colormap='plasma'
    )

    result = create_custom_cluster_pdfs(
        embeddings_path, cluster_path, bird_path, bird_name, output_dir, custom_config
    )

    # Method 4: Batch processing
    project_dir = "/path/to/project"
    batch_results = batch_create_cluster_pdfs(project_dir, ['bird1', 'bird2', 'bird3'])

    for bird, result in batch_results.items():
        if result['success']:
            print(f"{bird}: {result['n_clusters']} clusters")
        else:
            print(f"{bird}: Failed - {result['error']}")

if __name__ == '__main__':
    # Test the module
    example_usage()