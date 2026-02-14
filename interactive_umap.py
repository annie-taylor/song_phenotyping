# interactive_umap.py

import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import tables

# Import your existing functions
from tools.song_io import logger


@dataclass
class InteractiveUMAPConfig:
    """Configuration for interactive UMAP plotting."""

    # Plot appearance
    figure_size: Tuple[float, float] = (12, 10)
    point_size: float = 20
    point_alpha: float = 0.7
    colormap: str = 'tab10'

    # Spectrogram display
    spec_figure_size: Tuple[float, float] = (12, 8)
    spec_colormap: str = 'viridis'

    # Interaction
    click_tolerance: float = 0.1  # How close click needs to be to point
    show_click_feedback: bool = True
    auto_close_spec_windows: bool = False

    # Info display
    show_info_panel: bool = True
    info_fontsize: int = 10


@dataclass
class UMAPPoint:
    """Data structure for a single UMAP point."""
    index: int
    x: float
    y: float
    hash_id: str
    ground_truth_label: str = "N/A"
    cluster_label: Any = "N/A"
    metadata: Dict = field(default_factory=dict)


class InteractiveUMAPPlotter:
    """Interactive UMAP plotter with click-to-view functionality."""

    def __init__(self, config: InteractiveUMAPConfig = None):
        self.config = config or InteractiveUMAPConfig()
        self.points: List[UMAPPoint] = []
        self.bird_path: Optional[str] = None
        self.spectrogram_loader: Optional[Callable] = None

        # Plot state
        self.fig = None
        self.ax = None
        self.scatter = None
        self.info_text = None
        self.selected_point = None

        # Open spectrogram windows (for cleanup)
        self.open_spec_windows = []

    def load_data(self, embeddings: np.ndarray, hashes: List[str],
                  ground_truth_labels: List[str] = None,
                  cluster_labels: np.ndarray = None,
                  bird_path: str = None) -> bool:
        """Load UMAP data for interactive plotting."""

        try:
            if len(embeddings) != len(hashes):
                logger.error("Embeddings and hashes must have same length")
                return False

            # Create UMAPPoint objects
            self.points = []
            for i, (x, y) in enumerate(embeddings):
                point = UMAPPoint(
                    index=i,
                    x=float(x),
                    y=float(y),
                    hash_id=hashes[i],
                    ground_truth_label=ground_truth_labels[i] if ground_truth_labels else "N/A",
                    cluster_label=cluster_labels[i] if cluster_labels is not None else "N/A"
                )
                self.points.append(point)

            self.bird_path = bird_path
            logger.info(f"Loaded {len(self.points)} UMAP points for interactive plotting")
            return True

        except Exception as e:
            logger.error(f"Error loading UMAP data: {e}")
            return False

    def set_spectrogram_loader(self, loader_func: Callable[[str, str], np.ndarray]):
        """Set custom spectrogram loading function."""
        self.spectrogram_loader = loader_func

    def plot(self, title: str = "Interactive UMAP", show_legend: bool = True) -> bool:
        """Create the interactive UMAP plot."""

        if not self.points:
            logger.error("No data loaded. Call load_data() first.")
            return False

        try:
            # Setup matplotlib for interactivity
            plt.ion()

            # Create figure
            self.fig, self.ax = plt.subplots(figsize=self.config.figure_size)

            # Prepare data for plotting
            x_coords = [p.x for p in self.points]
            y_coords = [p.y for p in self.points]

            # Determine colors
            colors, color_label = self._prepare_colors()

            # Create scatter plot
            self.scatter = self.ax.scatter(
                x_coords, y_coords,
                c=colors,
                s=self.config.point_size,
                alpha=self.config.point_alpha,
                cmap=self.config.colormap,
                picker=True  # Enable picking
            )

            # Customize plot
            self.ax.set_xlabel('UMAP 1', fontsize=12)
            self.ax.set_ylabel('UMAP 2', fontsize=12)
            self.ax.set_title(f'{title}\nClick points to view spectrograms', fontsize=14)

            # Add colorbar if using numeric colors
            if color_label and isinstance(colors[0], (int, float)):
                cbar = plt.colorbar(self.scatter, ax=self.ax)
                cbar.set_label(color_label, fontsize=10)

            # Add info panel
            if self.config.show_info_panel:
                self._setup_info_panel()

            # Connect event handlers
            self._connect_events()

            # Add instructions
            self._add_instructions()

            plt.tight_layout()
            plt.show()

            logger.info("Interactive UMAP plot created. Click points to view spectrograms.")
            return True

        except Exception as e:
            logger.error(f"Error creating interactive plot: {e}")
            return False

    def _prepare_colors(self) -> Tuple[List, str]:
        """Prepare colors for scatter plot based on available labels."""

        # Priority: cluster labels > ground truth labels > no coloring
        if any(p.cluster_label != "N/A" for p in self.points):
            # Use cluster labels
            cluster_labels = [p.cluster_label for p in self.points]
            unique_labels, color_indices = np.unique(cluster_labels, return_inverse=True)
            return color_indices.tolist(), "Cluster Labels"

        elif any(p.ground_truth_label != "N/A" for p in self.points):
            # Use ground truth labels
            gt_labels = [p.ground_truth_label for p in self.points]
            unique_labels, color_indices = np.unique(gt_labels, return_inverse=True)
            return color_indices.tolist(), "Ground Truth Labels"

        else:
            # No coloring
            return ['blue'] * len(self.points), None

    def _setup_info_panel(self):
        """Setup info panel for displaying point information."""

        # Create text box for info display
        info_text = "Click on points to view details"
        self.info_text = self.ax.text(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            fontsize=self.config.info_fontsize,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='lightyellow',
                alpha=0.8,
                edgecolor='gray'
            )
        )

    def _connect_events(self):
        """Connect matplotlib event handlers."""

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Connect key press events for additional functionality
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Connect close event for cleanup
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _add_instructions(self):
        """Add instruction text to the plot."""

        instructions = [
            "Click: View spectrogram",
            "H: Toggle help",
            "C: Clear selection",
            "Q: Quit"
        ]

        instruction_text = " | ".join(instructions)
        self.ax.text(
            0.5, -0.08, instruction_text,
            transform=self.ax.transAxes,
            ha='center', va='top',
            fontsize=9, style='italic'
        )

    def _on_click(self, event):
        """Handle mouse click events."""

        if event.inaxes != self.ax:
            return

        try:
            # Find closest point
            closest_point = self._find_closest_point(event.xdata, event.ydata)

            if closest_point:
                self._handle_point_selection(closest_point)

        except Exception as e:
            logger.error(f"Error handling click: {e}")
            if self.config.show_click_feedback:
                print(f"Click error: {e}")

    def _find_closest_point(self, x: float, y: float) -> Optional[UMAPPoint]:
        """Find the closest UMAP point to click coordinates."""

        if x is None or y is None:
            return None

        # Calculate distances to all points
        distances = []
        for point in self.points:
            dist = np.sqrt((point.x - x) ** 2 + (point.y - y) ** 2)
            distances.append(dist)

        # Find closest point within tolerance
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist <= self._get_click_tolerance():
            return self.points[min_dist_idx]

        return None

    def _get_click_tolerance(self) -> float:
        """Calculate click tolerance based on plot scale."""

        # Get axis ranges
        x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]

        # Use percentage of range as tolerance
        return max(x_range, y_range) * self.config.click_tolerance / 100

    def _handle_point_selection(self, point: UMAPPoint):
        """Handle selection of a UMAP point."""

        self.selected_point = point

        # Update info panel
        if self.config.show_info_panel and self.info_text:
            self._update_info_panel(point)

        # Highlight selected point
        self._highlight_point(point)

        # Load and display spectrogram
        self._display_spectrogram(point)

        # Print info to console
        if self.config.show_click_feedback:
            print(f"Selected point {point.index}: Hash={point.hash_id[:12]}..., "
                  f"GT_Label={point.ground_truth_label}, Cluster={point.cluster_label}")

    def _update_info_panel(self, point: UMAPPoint):
        """Update the info panel with point details."""

        info_lines = [
            f"Point: {point.index}",
            f"Hash: {point.hash_id[:12]}...",
            f"GT Label: {point.ground_truth_label}",
            f"Cluster: {point.cluster_label}",
            f"Coords: ({point.x:.3f}, {point.y:.3f})"
        ]

        info_text = "\n".join(info_lines)
        self.info_text.set_text(info_text)
        self.fig.canvas.draw_idle()

    def _highlight_point(self, point: UMAPPoint):
        """Visually highlight the selected point."""

        # Remove previous highlight if exists
        if hasattr(self, '_highlight_scatter') and self._highlight_scatter:
            self._highlight_scatter.remove()

        # Add new highlight
        self._highlight_scatter = self.ax.scatter(
            [point.x], [point.y],
            s=self.config.point_size * 3,
            c='red',
            marker='o',
            facecolors='none',
            edgecolors='red',
            linewidths=2
        )

        self.fig.canvas.draw_idle()

    def _display_spectrogram(self, point: UMAPPoint):
        """Load and display spectrogram for the selected point."""

        try:
            # Load spectrogram
            spectrogram = self._load_spectrogram(point.hash_id)

            if spectrogram is not None:
                self._create_spectrogram_window(point, spectrogram)
            else:
                if self.config.show_click_feedback:
                    print(f"Could not load spectrogram for hash: {point.hash_id}")

        except Exception as e:
            logger.error(f"Error displaying spectrogram: {e}")
            if self.config.show_click_feedback:
                print(f"Spectrogram error: {e}")

    def _load_spectrogram(self, hash_id: str) -> Optional[np.ndarray]:
        """Load spectrogram using configured loader or default method."""

        if self.spectrogram_loader:
            # Use custom loader
            return self.spectrogram_loader(hash_id, self.bird_path)
        else:
            # Use default loader
            return self._default_load_spectrogram(hash_id)

    def _default_load_spectrogram(self, hash_id: str) -> Optional[np.ndarray]:
        """Default spectrogram loading from syllable files."""

        if not self.bird_path:
            logger.warning("No bird_path provided for spectrogram loading")
            return None

        try:
            syllables_dir = os.path.join(self.bird_path, 'data', 'syllables')

            if not os.path.exists(syllables_dir):
                logger.warning(f"Syllables directory not found: {syllables_dir}")
                return None

            # Search through syllable files
            syllable_files = [f for f in os.listdir(syllables_dir)
                              if f.endswith('.h5') and f.startswith('syllables_')]

            for filename in syllable_files:
                file_path = os.path.join(syllables_dir, filename)

                try:
                    with tables.open_file(file_path, mode='r') as f:
                        # Read hashes from this file
                        hashes_raw = f.root.hashes.read()
                        hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h)
                                  for h in hashes_raw]

                        # Check if our target hash is in this file
                        if hash_id in hashes:
                            # Find the index
                            hash_idx = hashes.index(hash_id)
                            # Load the spectrogram
                            spectrogram = f.root.spectrograms[hash_idx]
                            return spectrogram

                except Exception as e:
                    logger.debug(f"Error reading {filename}: {e}")
                    continue

                logger.warning(f"Hash {hash_id} not found in any syllable file")
                return None

        except Exception as e:
            logger.error(f"Error loading spectrogram for hash {hash_id}: {e}")
            return None

    def _create_spectrogram_window(self, point: UMAPPoint, spectrogram: np.ndarray):
        """Create new window to display spectrogram."""

        try:
            # Close previous windows if configured
            if self.config.auto_close_spec_windows:
                self._close_spectrogram_windows()

            # Create new figure
            spec_fig, spec_ax = plt.subplots(figsize=self.config.spec_figure_size)

            # Display spectrogram
            im = spec_ax.imshow(
                spectrogram,
                aspect='auto',
                origin='lower',
                cmap=self.config.spec_colormap,
                interpolation='nearest'
            )

            # Add colorbar
            plt.colorbar(im, ax=spec_ax, label='Amplitude')

            # Create title with all available information
            title_parts = [f'Point {point.index}']
            title_parts.append(f'Hash: {point.hash_id[:12]}...')
            if point.ground_truth_label != "N/A":
                title_parts.append(f'GT Label: {point.ground_truth_label}')
            if point.cluster_label != "N/A":
                title_parts.append(f'Cluster: {point.cluster_label}')

            spec_ax.set_title(' | '.join(title_parts), fontsize=12)
            spec_ax.set_xlabel('Time Bins')
            spec_ax.set_ylabel('Frequency Bins')

            plt.tight_layout()

            # Track window for cleanup
            self.open_spec_windows.append(spec_fig)

            plt.show()

        except Exception as e:
            logger.error(f"Error creating spectrogram window: {e}")

    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""

        try:
            if event.key == 'h':
                self._show_help()
            elif event.key == 'c':
                self._clear_selection()
            elif event.key == 'q':
                self._quit()
            elif event.key == 's' and self.selected_point:
                self._save_selection()

        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    def _show_help(self):
        """Display help information."""

        help_text = """
                    Interactive UMAP Controls:

                    Mouse:
                    - Click on points to view spectrograms

                    Keyboard:
                    - H: Show this help
                    - C: Clear current selection
                    - S: Save current selection (if implemented)
                    - Q: Quit interactive mode

                    Features:
                    - Point information shown in yellow box
                    - Selected points highlighted in red
                    - Multiple spectrogram windows can be open
                    """

        # Create help window
        help_fig, help_ax = plt.subplots(figsize=(8, 6))
        help_ax.text(0.05, 0.95, help_text, transform=help_ax.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        help_ax.set_title('Interactive UMAP Help', fontsize=14, fontweight='bold')
        help_ax.axis('off')

        plt.tight_layout()
        plt.show()

    def _clear_selection(self):
        """Clear current point selection."""

        self.selected_point = None

        # Remove highlight
        if hasattr(self, '_highlight_scatter') and self._highlight_scatter:
            self._highlight_scatter.remove()
            self._highlight_scatter = None

        # Reset info panel
        if self.config.show_info_panel and self.info_text:
            self.info_text.set_text("Click on points to view details")

        self.fig.canvas.draw_idle()

        if self.config.show_click_feedback:
            print("Selection cleared")

    def _save_selection(self):
        """Save current selection (placeholder for future functionality)."""

        if not self.selected_point:
            print("No point selected")
            return

        # Placeholder for saving functionality
        print(f"Saving selection: {self.selected_point.hash_id}")
        # Could save to file, add to list, etc.

    def _quit(self):
        """Quit interactive mode."""

        if self.config.show_click_feedback:
            print("Quitting interactive mode...")

        self._close_spectrogram_windows()
        plt.close(self.fig)

    def _close_spectrogram_windows(self):
        """Close all open spectrogram windows."""

        for fig in self.open_spec_windows:
            try:
                plt.close(fig)
            except:
                pass  # Figure might already be closed

        self.open_spec_windows.clear()

    def _on_close(self, event):
        """Handle main window close event."""

        self._close_spectrogram_windows()

class UMAPDataLoader:
    """Helper class for loading UMAP data from various sources."""

    @staticmethod
    def from_embedding_and_cluster_files(embeddings_path: str,
                                         cluster_labels_path: str = None) -> Dict:
        """Load data from embedding and cluster label files."""

        try:
            # Load embeddings
            embeddings, hashes, ground_truth_labels = UMAPDataLoader._load_umap_embeddings(embeddings_path)
            if embeddings is None:
                return None

            data = {
                'embeddings': embeddings,
                'hashes': hashes,
                'ground_truth_labels': ground_truth_labels,
                'cluster_labels': None
            }

            # Load cluster labels if provided
            if cluster_labels_path:
                cluster_labels, cluster_hashes, scores = UMAPDataLoader._load_cluster_labels(
                    cluster_labels_path)
                if cluster_labels is not None:
                    # Verify hash alignment
                    if len(hashes) == len(cluster_hashes) and all(
                            h1 == h2 for h1, h2 in zip(hashes, cluster_hashes)):
                        data['cluster_labels'] = cluster_labels
                    else:
                        logger.warning("Hash mismatch between embeddings and cluster labels")

            return data

        except Exception as e:
            logger.error(f"Error loading UMAP data: {e}")
            return None

    @staticmethod
    def _load_umap_embeddings(embedding_path: str):
        """Load embeddings from HDF5 file."""
        try:
            with tables.open_file(embedding_path, mode='r') as f:
                embeddings = f.root.embeddings.read()
                hashes = [hash_id.decode('utf-8') for hash_id in f.root.hashes.read()]
                labels = [label.decode('utf-8') for label in f.root.labels.read()]
                return embeddings, hashes, labels
        except Exception as e:
            logger.error(f"Error loading embeddings from {embedding_path}: {e}")
            return None, None, None

    @staticmethod
    def _load_cluster_labels(label_path: str):
        """Load cluster labels from HDF5 file."""
        try:
            with tables.open_file(label_path, mode='r') as f:
                labels = f.root.labels.read()
                hashes_raw = f.root.hashes.read()
                hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h) for h in hashes_raw]

                # Load scores (all arrays except labels and hashes)
                scores = {}
                for node in f.list_nodes(f.root, classname='Array'):
                    if node._v_name not in ['labels', 'hashes']:
                        score_value = node.read()
                        scores[node._v_name] = float(score_value) if score_value.ndim == 0 else score_value

                return labels, hashes, scores
        except Exception as e:
            logger.error(f"Error loading labels from {label_path}: {e}")
            return None, None, None

# Utility functions for easy usage

def quick_interactive_umap(embeddings_path: str,
                           cluster_labels_path: str = None,
                           bird_path: str = None,
                           title: str = "Interactive UMAP") -> bool:
    """
    Quick function to create interactive UMAP with defaults.

    Args:
        embeddings_path: Path to UMAP embeddings HDF5 file
        cluster_labels_path: Path to cluster labels HDF5 file (optional)
        bird_path: Path to bird directory for spectrogram loading
        title: Plot title

    Returns:
        True if successful, False otherwise
    """

    try:
        # Load data
        data = UMAPDataLoader.from_embedding_and_cluster_files(
            embeddings_path, cluster_labels_path
        )

        if data is None:
            logger.error("Failed to load UMAP data")
            return False

        # Create plotter
        plotter = InteractiveUMAPPlotter()

        # Load data into plotter
        success = plotter.load_data(
            embeddings=data['embeddings'],
            hashes=data['hashes'],
            ground_truth_labels=data['ground_truth_labels'],
            cluster_labels=data['cluster_labels'],
            bird_path=bird_path
        )

        if not success:
            return False

        # Create plot
        return plotter.plot(title)

    except Exception as e:
        logger.error(f"Error creating interactive UMAP: {e}")
        return False

def interactive_umap_from_best_results(bird_path: str, bird_name: str) -> bool:
    """
    Create interactive UMAP using best clustering results for a bird.

    Args:
        bird_path: Path to bird directory
        bird_name: Bird identifier

    Returns:
        True if successful, False otherwise
    """

    try:
        import pandas as pd

        # Load master summary to find best results
        master_summary_path = os.path.join(bird_path, 'master_summary.csv')
        if not os.path.exists(master_summary_path):
            logger.error(f"No master summary found for {bird_name}")
            return False

        summary_df = pd.read_csv(master_summary_path)
        if summary_df.empty:
            logger.error(f"Empty master summary for {bird_name}")
            return False

        # Get the best result (first row, assuming sorted by performance)
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
            logger.error(f"Could not find cluster file {cluster_filename}")
            return False

        logger.info(f"Using best results for {bird_name}:")
        logger.info(f"  Embedding: {embedding_filename}")
        logger.info(f"  Clusters: {cluster_filename}")
        logger.info(f"  Composite Score: {best_row.get('composite_score', 'N/A')}")

        # Create interactive plot
        title = f"Interactive UMAP - {bird_name} (Best Results)"
        return quick_interactive_umap(embeddings_path, cluster_labels_path, bird_path, title)

    except Exception as e:
        logger.error(f"Error creating interactive UMAP from best results: {e}")
        return False

def create_custom_interactive_umap(embeddings_path: str,
                                   cluster_labels_path: str = None,
                                   bird_path: str = None,
                                   config: InteractiveUMAPConfig = None,
                                   spectrogram_loader: Callable = None) -> InteractiveUMAPPlotter:
    """
    Create custom interactive UMAP with full configuration options.

    Args:
        embeddings_path: Path to UMAP embeddings HDF5 file
        cluster_labels_path: Path to cluster labels HDF5 file (optional)
        bird_path: Path to bird directory for spectrogram loading
        config: Custom configuration
        spectrogram_loader: Custom spectrogram loading function

    Returns:
        InteractiveUMAPPlotter instance (call .plot() to show)
    """

    # Load data
    data = UMAPDataLoader.from_embedding_and_cluster_files(
        embeddings_path, cluster_labels_path
    )

    if data is None:
        raise ValueError("Failed to load UMAP data")

    # Create plotter with custom config
    plotter = InteractiveUMAPPlotter(config)

    # Set custom spectrogram loader if provided
    if spectrogram_loader:
        plotter.set_spectrogram_loader(spectrogram_loader)

    # Load data
    success = plotter.load_data(
        embeddings=data['embeddings'],
        hashes=data['hashes'],
        ground_truth_labels=data['ground_truth_labels'],
        cluster_labels=data['cluster_labels'],
        bird_path=bird_path
    )

    if not success:
        raise ValueError("Failed to load data into plotter")

    return plotter

# Example usage and testing
def example_usage():
    """Example of how to use the interactive UMAP module."""

    # Example paths (adjust for your data)
    bird_path = "/path/to/bird/directory"
    bird_name = "bu85bu97"

    # Method 1: Quick interactive UMAP from best results
    print("Creating interactive UMAP from best results...")
    success = interactive_umap_from_best_results(bird_path, bird_name)
    if success:
        print("Interactive plot created! Click points to view spectrograms.")

    # Method 2: Quick interactive UMAP with specific files
    embeddings_path = os.path.join(bird_path, 'data', 'embeddings', 'euclidean_20neighbors_0.1dist.h5')
    cluster_path = os.path.join(bird_path, 'data', 'labelling', 'hdbscan_labels.h5')

    success = quick_interactive_umap(
        embeddings_path=embeddings_path,
        cluster_labels_path=cluster_path,
        bird_path=bird_path,
        title=f"Interactive UMAP - {bird_name}"
    )

    # Method 3: Custom configuration
    custom_config = InteractiveUMAPConfig(
        figure_size=(15, 12),
        point_size=30,
        point_alpha=0.8,
        spec_figure_size=(14, 10),
        spec_colormap='plasma',
        show_info_panel=True,
        auto_close_spec_windows=True
    )

    plotter = create_custom_interactive_umap(
        embeddings_path=embeddings_path,
        cluster_labels_path=cluster_path,
        bird_path=bird_path,
        config=custom_config
    )

    # Show the plot
    plotter.plot(f"Custom Interactive UMAP - {bird_name}")

    # Method 4: With custom spectrogram loader
    def my_custom_loader(hash_id: str, bird_path: str) -> np.ndarray:
        """Custom spectrogram loading function."""
        # Your custom loading logic here
        # Return spectrogram as numpy array
        pass

    plotter_with_custom_loader = create_custom_interactive_umap(
        embeddings_path=embeddings_path,
        bird_path=bird_path,
        spectrogram_loader=my_custom_loader
    )

    plotter_with_custom_loader.plot("UMAP with Custom Loader")

def batch_interactive_exploration(project_directory: str, birds: List[str] = None):
    """
    Launch interactive plots for multiple birds sequentially.

    Args:
        project_directory: Path to project directory
        birds: List of birds to explore (all if None)
    """

    if birds is None:
        # Discover birds
        project_path = Path(project_directory)
        birds = []
        for item in project_path.iterdir():
            if (item.is_dir() and
                    not item.name.startswith('.') and
                    item.name != 'copied_data' and
                    (item / 'master_summary.csv').exists()):
                birds.append(item.name)

    print(f"Found {len(birds)} birds for interactive exploration: {birds}")

    for i, bird in enumerate(birds):
        bird_path = os.path.join(project_directory, bird)

        print(f"\n{'=' * 50}")
        print(f"Bird {i + 1}/{len(birds)}: {bird}")
        print(f"{'=' * 50}")

        try:
            success = interactive_umap_from_best_results(bird_path, bird)

            if success:
                print(f"Interactive plot launched for {bird}")
                input("Press Enter to continue to next bird (or Ctrl+C to stop)...")
            else:
                print(f"Failed to create interactive plot for {bird}")

        except KeyboardInterrupt:
            print("\nStopping interactive exploration...")
            break
        except Exception as e:
            print(f"Error processing {bird}: {e}")

    print("\nInteractive exploration complete!")

if __name__ == '__main__':
    # Test the module
    example_usage()