"""
Full A→E pipeline runner for visual inspection / real output.

Edit the SOURCE and SAVE_PATH variables below, then run:

    python run_pipeline.py

Outputs land under SAVE_PATH/<bird>/ on the SSD.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Where to write pipeline outputs (one subdirectory per bird)
SAVE_PATH = "/Volumes/Extreme SSD/pipeline_runs"

# evsonganaly bird
EVSONG_SOURCE  = "/Volumes/Extreme SSD/smoke_test_birds/or18or24/source"
EVSONG_BIRD    = "or18or24"

# wseg bird (set WSEG_BIRD = None to skip)
WSEG_METADATA  = "/Volumes/Extreme SSD/smoke_test_birds/bu78bu77/source_metadata"
WSEG_BIRD      = "bu78bu77"

# How many songs per bird to process (None = all)
SONGS_PER_BIRD = None

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_evsonganaly(save_path: str, source_dir: str, bird: str, songs_per_bird):
    from A_spec_saving import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
    from B_flattening import flatten_bird_spectrograms
    from C_embedding import explore_embedding_parameters_robust
    from D_labelling import label_bird, DEFAULT_HDBSCAN_GRID
    from E_phenotyping import phenotype_bird, PhenotypingConfig
    from tools.spectrogram_configs import SpectrogramParams

    print(f"\n{'='*60}")
    print(f"  Running pipeline for {bird} (evsonganaly)")
    print(f"  Source : {source_dir}")
    print(f"  Output : {save_path}/{bird}")
    print(f"{'='*60}\n")

    print("[ A ] Saving spectrograms...")
    meta, audio = filepaths_from_evsonganaly(
        wav_directory=source_dir, bird_subset=[bird]
    )
    save_specs_for_evsonganaly_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=SpectrogramParams(),
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(
        save_path=save_path,
        bird=bird,
    )

    print("[ D ] Clustering / labelling...")
    label_bird(
        save_path=save_path,
        bird=bird,
        metrics=['silhouette', 'dbi', 'ch'],
        hdbscan_params=[p.to_dict() for p in DEFAULT_HDBSCAN_GRID],
    )

    print("[ E ] Phenotyping...")
    bird_path = str(Path(save_path) / bird)
    phenotype_bird(bird_path=bird_path, config=PhenotypingConfig())

    print(f"\n✓ Done. Outputs at: {bird_path}\n")


def run_wseg(save_path: str, metadata_dir: str, bird: str, songs_per_bird):
    from A_spec_saving import filepaths_from_wseg, save_specs_for_wseg_birds
    from B_flattening import flatten_bird_spectrograms
    from C_embedding import explore_embedding_parameters_robust
    from D_labelling import label_bird, DEFAULT_HDBSCAN_GRID
    from E_phenotyping import phenotype_bird, PhenotypingConfig
    from tools.spectrogram_configs import SpectrogramParams

    print(f"\n{'='*60}")
    print(f"  Running pipeline for {bird} (wseg)")
    print(f"  Metadata: {metadata_dir}")
    print(f"  Output  : {save_path}/{bird}")
    print(f"{'='*60}\n")

    print("[ A ] Saving spectrograms...")
    meta, audio = filepaths_from_wseg(
        seg_directory=metadata_dir, bird_subset=[bird]
    )
    save_specs_for_wseg_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=SpectrogramParams(),
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(
        save_path=save_path,
        bird=bird,
    )

    print("[ D ] Clustering / labelling...")
    label_bird(
        save_path=save_path,
        bird=bird,
        metrics=['silhouette', 'dbi', 'ch'],
        hdbscan_params=[p.to_dict() for p in DEFAULT_HDBSCAN_GRID],
    )

    print("[ E ] Phenotyping...")
    bird_path = str(Path(save_path) / bird)
    phenotype_bird(bird_path=bird_path, config=PhenotypingConfig())

    print(f"\n✓ Done. Outputs at: {bird_path}\n")


if __name__ == "__main__":
    run_evsonganaly(SAVE_PATH, EVSONG_SOURCE, EVSONG_BIRD, SONGS_PER_BIRD)

    if WSEG_BIRD is not None:
        run_wseg(SAVE_PATH, WSEG_METADATA, WSEG_BIRD, SONGS_PER_BIRD)
