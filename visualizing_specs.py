import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from scipy import signal
import librosa
import os
from tqdm import tqdm
import numpy as np
import logging
import random
import tables
import shutil


def create_song_spectrograms_pdf(project_directory: str,
                                 target_duration: float = 8.0,
                                 sample_rate: int = 32000,
                                 prefer_local: bool = True):
    """
    Create PDF with spectrograms for all songs for each bird.

    Parameters:
    - project_directory: Root project directory
    - target_duration: Target duration in seconds (default 8.0)
    - sample_rate: Expected sample rate
    - prefer_local: If True, prefer local audio files over server files
    """
    # Find all birds in project directory
    birds = [d for d in os.listdir(project_directory)
             if os.path.isdir(os.path.join(project_directory, d)) and d != 'copied_data']

    for bird in birds:
        bird_dir = os.path.join(project_directory, bird)

        # Load audio paths mapping
        try:
            from tools.audio_path_management import load_audio_paths_mapping, get_audio_path
            mapping = load_audio_paths_mapping(bird_dir)

            if not mapping:
                logging.warning(f"No audio paths mapping found for {bird}, skipping")
                continue

        except Exception as e:
            logging.error(f"Could not load audio paths for {bird}: {e}")
            continue

        # Create directories
        pdf_dir = os.path.join(bird_dir, 'pdfs')
        spec_dir = os.path.join(bird_dir, 'spectrograms')
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        # Create PDF
        pdf_path = os.path.join(pdf_dir, f'{bird}_all_song_specs.pdf')

        with pdf_backend.PdfPages(pdf_path) as pdf:
            for filename in tqdm(mapping.keys(), desc=f"Creating spectrograms for {bird}"):
                try:
                    # Get the appropriate file path
                    file_path = get_audio_path(bird_dir, filename, prefer_local)

                    # Load audio
                    audio, sr = librosa.load(file_path, sr=sample_rate)

                    # Pad or trim to target duration
                    target_samples = int(target_duration * sr)
                    if len(audio) < target_samples:
                        audio = np.pad(audio, (0, target_samples - len(audio)))
                    else:
                        audio = audio[:target_samples]

                    # Create spectrogram
                    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024, noverlap=512)

                    # Create plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set_xlabel('Time (s)')
                    ax.set_title(f'Spectrogram: {filename}')
                    plt.colorbar(im, ax=ax, label='Power (dB)')

                    # Add file path as text (show which path was actually used)
                    path_type = "Local" if prefer_local and mapping[filename]['local'] else "Server"
                    display_text = f"{path_type}: {file_path}"
                    ax.text(0.02, 0.98, display_text, transform=ax.transAxes,
                            fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Save PNG
                    png_filename = f"{os.path.splitext(filename)[0]}.png"
                    png_path = os.path.join(spec_dir, png_filename)
                    plt.savefig(png_path, dpi=150, bbox_inches='tight')

                    # Add to PDF
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

                except Exception as e:
                    logging.error(f"Failed to create spectrogram for {filename}: {e}")

        logging.info(f"Created spectrogram PDF for {bird}: {pdf_path}")


def create_syllable_sample_pdfs(project_directory: str,
                                syllables_per_bird: int = 40):
    """
    Create PDFs with sampled syllable spectrograms for each bird.

    Parameters:
    - project_directory: Root project directory
    - syllables_per_bird: Number of syllables to sample per bird
    """

    # Find all birds with syllable data
    birds = [d for d in os.listdir(project_directory)
             if os.path.isdir(os.path.join(project_directory, d, 'data', 'syllables'))]

    for bird in birds:
        syllables_dir = os.path.join(project_directory, bird, 'data', 'syllables')
        pdf_dir = os.path.join(project_directory, bird, 'pdfs')
        os.makedirs(pdf_dir, exist_ok=True)

        # Find all syllable files
        syllable_files = [f for f in os.listdir(syllables_dir)
                          if f.endswith('.h5') and 'syllables' in f]

        if not syllable_files:
            logging.warning(f"No syllable files found for {bird}")
            continue

        # Collect all syllables
        all_syllables = []
        for syl_file in syllable_files:
            try:
                file_path = os.path.join(syllables_dir, syl_file)
                with tables.open_file(file_path, mode='r') as f:
                    specs = f.root.spectrograms.read()
                    labels = f.root.manual[:]
                    for i, (spec, label) in enumerate(zip(specs, labels)):
                        all_syllables.append({
                            'spectrogram': spec,
                            'label': label.decode('utf-8') if isinstance(label, bytes) else str(label),
                            'file': syl_file,
                            'index': i
                        })
            except Exception as e:
                logging.error(f"Failed to load syllables from {file_path}: {e}")

        if not all_syllables:
            logging.warning(f"No syllables found for {bird}")
            continue

            # Sample syllables
        n_to_sample = min(syllables_per_bird, len(all_syllables))
        sampled_syllables = random.sample(all_syllables, n_to_sample)

        # Create PDF
        pdf_path = os.path.join(pdf_dir, f'{bird}_example_syllables_sampled.pdf')

        with pdf_backend.PdfPages(pdf_path) as pdf:
            # Create plots in grid format (4x2 = 8 per page)
            n_cols, n_rows = 4, 2
            syllables_per_page = n_cols * n_rows

            for page_start in range(0, len(sampled_syllables), syllables_per_page):
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
                fig.suptitle(f'{bird} - Sampled Syllables (Page {page_start // syllables_per_page + 1})',
                             fontsize=16)

                # Flatten axes for easier indexing
                if n_rows == 1:
                    axes = [axes]
                elif n_cols == 1:
                    axes = [[ax] for ax in axes]
                axes_flat = [ax for row in axes for ax in row]

                page_syllables = sampled_syllables[page_start:page_start + syllables_per_page]

                for i, syl_data in enumerate(page_syllables):
                    ax = axes_flat[i]
                    spec = syl_data['spectrogram']

                    # Plot spectrogram
                    im = ax.imshow(spec, aspect='auto', origin='lower',
                                   cmap='viridis', interpolation='nearest')

                    # Add title with label info
                    title = f"Label: {syl_data['label']}\n{syl_data['file']}[{syl_data['index']}]"
                    ax.set_title(title, fontsize=8)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Frequency')

                # Hide unused subplots
                for i in range(len(page_syllables), len(axes_flat)):
                    axes_flat[i].set_visible(False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        logging.info(f"Created syllable sample PDF for {bird}: {pdf_path} ({n_to_sample} syllables)")


def main():
    evsong_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
    wseg_test_directory = os.path.join('/Volumes', 'Extreme SSD', 'wseg test')

    # Create visualization PDFs using local files
    create_song_spectrograms_pdf(evsong_test_directory, prefer_local=True)
    create_song_spectrograms_pdf(wseg_test_directory, prefer_local=True)

    # Create syllable sample PDFs (no changes needed)
    create_syllable_sample_pdfs(evsong_test_directory)
    create_syllable_sample_pdfs(wseg_test_directory)

if __name__ == '__main__':
    main()