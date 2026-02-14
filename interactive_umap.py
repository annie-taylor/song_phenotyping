# interactive_umap_simple.py

import pandas as pd
import numpy as np
import os
import tables
from tools.song_io import logger

# Set matplotlib backend at the top
import matplotlib

matplotlib.use('MacOSX')  # On macOS
import matplotlib.pyplot as plt


def load_spectrogram_for_hash(hash_id: str, bird_path: str) -> np.ndarray:
    """Load spectrogram for a given hash ID."""
    syllables_dir = os.path.join(bird_path, 'data', 'syllables')

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


def simple_interactive_umap(bird_path: str):
    """
    Simplest possible interactive UMAP - just run this!
    """

    bird_name = os.path.basename(bird_path.rstrip('/'))
    print(f"🎵 Creating interactive UMAP for {bird_name}...")

    # Load data
    master_summary_path = os.path.join(bird_path, 'master_summary.csv')
    if not os.path.exists(master_summary_path):
        print(f"❌ No master summary found: {master_summary_path}")
        return False

    summary_df = pd.read_csv(master_summary_path)
    best_row = summary_df.iloc[0]

    n_neighbors = best_row['n_neighbors']
    min_dist = best_row['min_dist']
    metric = best_row['metric']

    embeddings_path = os.path.join(
        bird_path, 'data', 'embeddings',
        f'{metric}_{int(n_neighbors)}neighbors_{min_dist}dist.h5'
    )

    with tables.open_file(embeddings_path, 'r') as f:
        embeddings = f.root.embeddings.read()
        hashes = [h.decode('utf-8') for h in f.root.hashes.read()]
        manual_labels = [l.decode('utf-8') for l in f.root.labels.read()]

    # Load cluster labels
    cluster_filename = os.path.basename(str(best_row['label_path']))
    cluster_labels = None

    for root, dirs, files in os.walk(os.path.join(bird_path, 'data', 'labelling')):
        if cluster_filename in files:
            with tables.open_file(os.path.join(root, cluster_filename), 'r') as f:
                cluster_labels = f.root.labels.read()
            break

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    if cluster_labels is not None:
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                             c=cluster_labels, s=30, alpha=0.7, cmap='tab10')
        plt.colorbar(scatter, label='Cluster')
    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], s=30, alpha=0.7)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'Interactive UMAP - {bird_name}\nClick points to view spectrograms')

    # Click handler
    def on_click(event):
        if event.inaxes != ax:
            return

        distances = np.sqrt((embeddings[:, 0] - event.xdata) ** 2 +
                            (embeddings[:, 1] - event.ydata) ** 2)
        closest_idx = np.argmin(distances)

        hash_id = hashes[closest_idx]
        spec = load_spectrogram_for_hash(hash_id, bird_path)

        if spec is not None:
            plt.figure(figsize=(10, 6))
            plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f'Point {closest_idx} | Hash: {hash_id[:12]}... | Manual Label: {manual_labels[closest_idx]}')
            plt.xlabel('Time Bins')
            plt.ylabel('Frequency Bins')
            plt.colorbar(label='Amplitude')
            plt.tight_layout()
            plt.show(block=False)
            print(f"Showing spectrogram for point {closest_idx}")
        else:
            print(f"Could not load spectrogram for hash {hash_id}")

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    print("🎵 Interactive UMAP created!")
    print("Click on any point to view its spectrogram")
    print("Close the window when done")

    plt.tight_layout()
    plt.show(block=True)  # This keeps the window open!
    return True


# Example usage
def example_usage():
    """Simplest example usage."""

    project_dir = '/Volumes/Extreme SSD/wseg test'
    bird_name = 'bu85bu97'
    bird_path = os.path.join(project_dir, bird_name)

    # Just call this one function!
    simple_interactive_umap(bird_path)


if __name__ == '__main__':
    example_usage()
