# interactive_umap.py

import pandas as pd
import numpy as np
import os
import tables
from song_phenotyping.signal import logger
import matplotlib.pyplot as plt

# Set matplotlib backend at the top
import matplotlib
# Uncomment the one you want to use:
# matplotlib.use('MacOSX')     # macOS
matplotlib.use('TkAgg')      # Cross-platform default
# matplotlib.use('Qt5Agg')     # If you have PyQt5
# matplotlib.use('Agg')        # Headless/remote


def load_spectrogram_for_hash(hash_id: str, bird_path: str) -> np.ndarray:
    """Load spectrogram for a given hash ID."""
    syllables_dir = os.path.join(bird_path, 'stages', '01_specs')

    if not os.path.exists(syllables_dir):
        return None

    syllable_files = [f for f in os.listdir(syllables_dir)
                      if f.endswith('.h5') and f.startswith('syllables_')]

    for filename in syllable_files:
        file_path = os.path.join(syllables_dir, filename)
        try:
            with tables.open_file(file_path, 'r') as f:
                file_hashes = [h.decode('utf-8') if isinstance(h, bytes) else str(h)
                               for h in f.root.hashes.read()]

                if hash_id in file_hashes:
                    hash_idx = file_hashes.index(hash_id)
                    spectrogram = f.root.spectrograms[hash_idx]
                    return spectrogram
        except Exception as e:
            continue

    return None


def create_color_mapping(labels):
    """Create a consistent color mapping for labels."""
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Use a combination of colormaps for more colors
    if n_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_labels]
    elif n_labels <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_labels]
    else:
        # For many labels, use a continuous colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n_labels))

    color_map = dict(zip(unique_labels, colors))
    return color_map, unique_labels


def simple_interactive_umap(bird_path: str, color_by: str = 'auto'):
    """
    Interactive UMAP with flexible coloring options.

    Args:
        bird_path: Path to bird directory
        color_by: 'auto' for automated labels, 'manual' for ground truth labels, or 'both' for side-by-side
    """

    bird_name = os.path.basename(bird_path.rstrip('/'))
    print(f"🎵 Creating interactive UMAP for {bird_name}...")
    print(f"🎨 Coloring by: {color_by}")

    # Load data
    master_summary_path = os.path.join(bird_path, 'results', 'master_summary.csv')
    if not os.path.exists(master_summary_path):
        print(f"❌ No master summary found: {master_summary_path}")
        return False

    summary_df = pd.read_csv(master_summary_path)
    best_row = summary_df.iloc[0]

    n_neighbors = best_row['n_neighbors']
    min_dist = best_row['min_dist']
    metric = best_row['metric']

    embeddings_path = os.path.join(
        bird_path, 'stages', '03_embeddings',
        f'{metric}_{int(n_neighbors)}neighbors_{min_dist}dist.h5'
    )

    with tables.open_file(embeddings_path, 'r') as f:
        embeddings = f.root.embeddings.read()
        hashes = [h.decode('utf-8') for h in f.root.hashes.read()]
        gt_labels = [l.decode('utf-8') for l in f.root.labels.read()]

    # Load cluster labels
    cluster_filename = os.path.basename(str(best_row['label_path']))
    cluster_labels = None

    for root, dirs, files in os.walk(os.path.join(bird_path, 'stages', '04_labels')):
        if cluster_filename in files:
            with tables.open_file(os.path.join(root, cluster_filename), 'r') as f:
                cluster_labels = f.root.labels.read()
            break

    # Determine what to color by
    if color_by == 'both':
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        axes = [ax1, ax2]
        color_options = ['manual', 'auto']
        titles = ['Manual Labels', 'Automated Labels']
    else:
        # Single plot
        fig, ax = plt.subplots(figsize=(12, 8))
        axes = [ax]
        color_options = [color_by]
        titles = [f'{color_by.title()} Labels']

    # Plot for each axis
    for i, (current_ax, color_option, title) in enumerate(zip(axes, color_options, titles)):

        # Choose labels based on color_option
        if color_option == 'auto' and cluster_labels is not None:
            plot_labels = cluster_labels
            label_name = 'Automated'
        elif color_option == 'manual':
            plot_labels = gt_labels
            label_name = 'Manual'
        else:
            # Fallback to manual if auto not available
            plot_labels = gt_labels
            label_name = 'Manual (auto unavailable)'

        # Create color mapping
        color_map, unique_labels = create_color_mapping(plot_labels)

        # Plot each label group separately for proper legend
        for label in unique_labels:
            mask = np.array(plot_labels) == label
            current_ax.scatter(
                embeddings[mask, 0], embeddings[mask, 1],
                c=[color_map[label]], s=30, alpha=0.7,
                label=str(label)  # Just the label itself
            )

        # Add legend
        legend = current_ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=7, markerscale=0.7)
        legend.set_title(f'{label_name}', prop={'size': 6, 'weight': 'bold'})

        # Customize plot
        current_ax.set_xlabel('UMAP 1')
        current_ax.set_ylabel('UMAP 2')
        current_ax.set_title(f'{title}\nClick points to view spectrograms')

    # Click handler
    def on_click(event):
        if event.inaxes not in axes:
            return

        distances = np.sqrt((embeddings[:, 0] - event.xdata) ** 2 +
                            (embeddings[:, 1] - event.ydata) ** 2)
        closest_idx = np.argmin(distances)

        hash_id = hashes[closest_idx]
        spec = load_spectrogram_for_hash(hash_id, bird_path)

        if spec is not None:
            # Create comprehensive title with both labels
            title_parts = [f'Point {closest_idx}']
            #title_parts.append(f'Hash: {hash_id[:12]}...')
            title_parts.append(f'Manual: {gt_labels[closest_idx]}')

            if cluster_labels is not None:
                title_parts.append(f'Auto: {cluster_labels[closest_idx]}')
            else:
                title_parts.append('Auto: N/A')
            spec_title = ' | '.join(title_parts)

            plt.figure(figsize=(12, 6))
            plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            plt.title(spec_title, fontsize=12)
            plt.xlabel('Time Bins')
            plt.ylabel('Frequency Bins')
            plt.colorbar(label='Amplitude')
            plt.tight_layout()
            plt.show(block=False)

            # Print info to console
            print(f"Point {closest_idx}:")
            print(f"  Manual Label: {gt_labels[closest_idx]}")
            if cluster_labels is not None:
                print(f"  Auto Label: {cluster_labels[closest_idx]}")
            else:
                print(f"  Auto Label: N/A")
            print(f"  Hash: {hash_id}")
        else:
            print(f"Could not load spectrogram for hash {hash_id}")

    # Connect click events to all axes
    for current_ax in axes:
        fig.canvas.mpl_connect('button_press_event', on_click)

    print("🎵 Interactive UMAP created!")
    print("Click on any point to view its spectrogram")
    print("Close the window when done")

    plt.tight_layout()
    plt.show(block=True)
    return True


def interactive_umap_with_options(bird_path: str):
    """
    Interactive function to choose coloring options.
    """

    print("🎨 Choose coloring option:")
    print("1. Manual labels only")
    print("2. Automated labels only")
    print("3. Side-by-side comparison")
    print("4. Let me specify...")

    choice = input("Enter choice (1-4): ").strip()

    if choice == '1':
        return simple_interactive_umap(bird_path, color_by='manual')
    elif choice == '2':
        return simple_interactive_umap(bird_path, color_by='auto')
    elif choice == '3':
        return simple_interactive_umap(bird_path, color_by='both')
    elif choice == '4':
        color_by = input("Enter 'manual', 'auto', or 'both': ").strip().lower()
        if color_by in ['manual', 'auto', 'both']:
            return simple_interactive_umap(bird_path, color_by=color_by)
        else:
            print("Invalid option, using manual labels")
            return simple_interactive_umap(bird_path, color_by='manual')
    else:
        print("Invalid choice, using manual labels")
        return simple_interactive_umap(bird_path, color_by='manual')


def quick_comparison_umap(bird_path: str):
    """
    Quick function to create side-by-side comparison.
    """
    return simple_interactive_umap(bird_path, color_by='both')


def analyze_label_distribution(bird_path: str):
    """
    Analyze and print label distribution for both manual and automated labels.
    """

    bird_name = os.path.basename(bird_path.rstrip('/'))
    print(f"📊 Label distribution analysis for {bird_name}")

    # Load data
    master_summary_path = os.path.join(bird_path, 'results', 'master_summary.csv')
    summary_df = pd.read_csv(master_summary_path)
    best_row = summary_df.iloc[0]

    n_neighbors = best_row['n_neighbors']
    min_dist = best_row['min_dist']
    metric = best_row['metric']

    embeddings_path = os.path.join(
        bird_path, 'stages', '03_embeddings',
        f'{metric}_{int(n_neighbors)}neighbors_{min_dist}dist.h5'
    )

    with tables.open_file(embeddings_path, 'r') as f:
        gt_labels = [l.decode('utf-8') for l in f.root.labels.read()]

    # Load cluster labels
    cluster_filename = os.path.basename(str(best_row['label_path']))
    cluster_labels = None

    for root, dirs, files in os.walk(os.path.join(bird_path, 'stages', '04_labels')):
        if cluster_filename in files:
            with tables.open_file(os.path.join(root, cluster_filename), 'r') as f:
                cluster_labels = f.root.labels.read()
            break

    # Analyze manual labels
    manual_counts = pd.Series(gt_labels).value_counts().sort_index()
    print(f"\n📝 Manual Labels ({len(manual_counts)} unique):")
    for label, count in manual_counts.items():
        print(f"  {label}: {count}")

    # Analyze automated labels
    if cluster_labels is not None:
        auto_counts = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"\n🤖 Automated Labels ({len(auto_counts)} unique):")
        for label, count in auto_counts.items():
            print(f"  {label}: {count}")
    else:
        print(f"\n🤖 Automated Labels: Not available")

    return manual_counts, auto_counts if cluster_labels is not None else None

def example_usage():
    """Example usage with different options."""

    from song_phenotyping.tools.project_config import ProjectConfig
    cfg = ProjectConfig.load()
    bird_name = 'bu85bu97'
    bird_path = str(cfg.bird_dir(bird_name, experiment='wseg test'))

    print("🎵 Interactive UMAP Examples")
    print("=" * 50)

    # Option 1: Interactive choice
    print("\n1. Interactive choice:")
    # interactive_umap_with_options(bird_path)

    # Option 2: Direct function calls
    print("\n2. Direct function calls:")

    # Analyze labels first
    print("\nAnalyzing label distribution...")
    analyze_label_distribution(bird_path)

    # Create UMAP with manual labels
    print("\nCreating UMAP with manual labels...")
    # simple_interactive_umap(bird_path, color_by='manual')

    # Create UMAP with automated labels
    print("\nCreating UMAP with automated labels...")
    # simple_interactive_umap(bird_path, color_by='auto')

    # Create side-by-side comparison
    print("\nCreating side-by-side comparison...")
    quick_comparison_umap(bird_path)


if __name__ == '__main__':
    example_usage()