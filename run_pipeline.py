"""
Full A→E pipeline runner.

Configuration is read from ``config.yaml`` (see ``config.yaml.example``).
The ALL-CAPS constants below act as *overrides*: leave them as ``None`` to
use the values in config.yaml, or set them to a non-None value for a
quick one-off run without editing the config file.

Typical usage
-------------
1. Copy ``config.yaml.example`` → ``config.yaml`` and fill in your paths.
2. Run:  python run_pipeline.py

To process a single bird for a quick test::

    BIRDS = ['or18or24']          # edit here, or set in config.yaml

To process an entire library::

    BIRDS = None                  # auto-discovers all birds under EVSONG_SOURCE
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Optional in-script overrides — set to None to use config.yaml values
# ---------------------------------------------------------------------------

SAVE_PATH      = None    # e.g. "E:/pipeline_runs" or "/Volumes/Extreme SSD/pipeline_runs"
EVSONG_SOURCE  = None    # parent dir containing evsonganaly bird subdirs
WSEG_METADATA  = None    # wseg metadata dir; None = skip wseg
BIRDS          = None    # None = all discovered; or e.g. ['or18or24', 'bu78bu77']
SONGS_PER_BIRD = None    # None = use config.yaml value (or all songs if unset there)

# Feature flags (these are not in config.yaml — override here if needed)
SAVE_INST_FREQ          = True   # append instantaneous-frequency channel
SAVE_GROUP_DELAY        = True   # append group-delay channel
DURATION_FEATURE_WEIGHT = 1.0   # 0 = disabled; ~1.0 weights duration as one time bin

# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _load_pipeline_cfg():
    """Load config.yaml and merge with any in-script overrides."""
    from song_phenotyping.tools.project_config import ProjectConfig
    cfg = ProjectConfig.load()
    p   = cfg.pipeline

    save_path      = SAVE_PATH      or (str(p.save_path)     if p.save_path     else None)
    evsong_source  = EVSONG_SOURCE  or (str(p.evsong_source) if p.evsong_source else None)
    wseg_metadata  = WSEG_METADATA  or (str(p.wseg_metadata) if p.wseg_metadata else None)
    birds          = BIRDS          if BIRDS is not None else p.birds
    songs_per_bird = SONGS_PER_BIRD if SONGS_PER_BIRD is not None else p.songs_per_bird

    if save_path is None:
        raise ValueError(
            "save_path is not set. Add 'pipeline.save_path' to config.yaml "
            "or set SAVE_PATH at the top of run_pipeline.py."
        )

    return dict(
        save_path      = save_path,
        evsong_source  = evsong_source,
        wseg_metadata  = wseg_metadata,
        birds          = birds,
        songs_per_bird = songs_per_bird,
    )

# ---------------------------------------------------------------------------
# Single-bird pipeline runners
# ---------------------------------------------------------------------------

def run_evsonganaly(save_path: str, source_dir: str, bird: str, songs_per_bird,
                    save_inst_freq=False, save_group_delay=False,
                    duration_feature_weight=0.0):
    """Run stages A–E for a single evsonganaly bird."""
    from song_phenotyping.ingestion import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
    from song_phenotyping.flattening import flatten_bird_spectrograms
    from song_phenotyping.embedding import explore_embedding_parameters_robust
    from song_phenotyping.labelling import label_bird, DEFAULT_HDBSCAN_GRID
    from song_phenotyping.phenotyping import phenotype_bird, PhenotypingConfig
    from song_phenotyping.tools.spectrogram_configs import SpectrogramParams

    print(f"\n{'='*60}")
    print(f"  Running pipeline for {bird} (evsonganaly)")
    print(f"  Source : {source_dir}")
    print(f"  Output : {save_path}/{bird}")
    print(f"{'='*60}\n")

    print("[ A ] Saving spectrograms...")
    meta, audio = filepaths_from_evsonganaly(
        wav_directory=source_dir, bird_subset=[bird]
    )
    spec_params = SpectrogramParams(
        songs_per_bird=songs_per_bird or 9999,
        save_inst_freq=save_inst_freq,
        save_group_delay=save_group_delay,
        duration_feature_weight=duration_feature_weight,
    )
    save_specs_for_evsonganaly_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=spec_params,
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird, params=spec_params)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(save_path=save_path, bird=bird)

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


def run_wseg(save_path: str, metadata_dir: str, bird: str, songs_per_bird,
             save_inst_freq=False, save_group_delay=False,
             duration_feature_weight=0.0):
    """Run stages A–E for a single wseg bird."""
    from song_phenotyping.ingestion import filepaths_from_wseg, save_specs_for_wseg_birds
    from song_phenotyping.flattening import flatten_bird_spectrograms
    from song_phenotyping.embedding import explore_embedding_parameters_robust
    from song_phenotyping.labelling import label_bird, DEFAULT_HDBSCAN_GRID
    from song_phenotyping.phenotyping import phenotype_bird, PhenotypingConfig
    from song_phenotyping.tools.spectrogram_configs import SpectrogramParams

    print(f"\n{'='*60}")
    print(f"  Running pipeline for {bird} (wseg)")
    print(f"  Metadata: {metadata_dir}")
    print(f"  Output  : {save_path}/{bird}")
    print(f"{'='*60}\n")

    print("[ A ] Saving spectrograms...")
    meta, audio = filepaths_from_wseg(
        seg_directory=metadata_dir, bird_subset=[bird]
    )
    spec_params = SpectrogramParams(
        songs_per_bird=songs_per_bird or 9999,
        save_inst_freq=save_inst_freq,
        save_group_delay=save_group_delay,
        duration_feature_weight=duration_feature_weight,
    )
    save_specs_for_wseg_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=spec_params,
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird, params=spec_params)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(save_path=save_path, bird=bird)

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

# ---------------------------------------------------------------------------
# Cohort runners
# ---------------------------------------------------------------------------

def run_evsonganaly_cohort(save_path, evsong_source, birds, songs_per_bird):
    """Run the pipeline for all (or a filtered subset of) evsonganaly birds.

    Parameters
    ----------
    save_path:
        Output root; one subdirectory is created per bird.
    evsong_source:
        Parent directory containing evsonganaly bird folders.
    birds:
        ``None`` to auto-discover all birds, or a list of bird ID strings
        to restrict processing to a subset.
    songs_per_bird:
        Maximum number of songs to process per bird; ``None`` = all.
    """
    from song_phenotyping.ingestion import filepaths_from_evsonganaly

    meta, audio = filepaths_from_evsonganaly(
        wav_directory=evsong_source,
        bird_subset=birds,  # None → discover all
    )

    discovered = sorted(meta.keys())
    if not discovered:
        print(f"No evsonganaly birds found under: {evsong_source}")
        return

    print(f"\nFound {len(discovered)} evsonganaly bird(s): {discovered}")
    for bird in discovered:
        run_evsonganaly(
            save_path=save_path,
            source_dir=evsong_source,
            bird=bird,
            songs_per_bird=songs_per_bird,
            save_inst_freq=SAVE_INST_FREQ,
            save_group_delay=SAVE_GROUP_DELAY,
            duration_feature_weight=DURATION_FEATURE_WEIGHT,
        )


def run_wseg_cohort(save_path, wseg_metadata, birds, songs_per_bird):
    """Run the pipeline for all (or a filtered subset of) wseg birds."""
    from song_phenotyping.ingestion import filepaths_from_wseg

    meta, audio = filepaths_from_wseg(
        seg_directory=wseg_metadata,
        bird_subset=birds,
    )

    discovered = sorted(meta.keys())
    if not discovered:
        print(f"No wseg birds found under: {wseg_metadata}")
        return

    print(f"\nFound {len(discovered)} wseg bird(s): {discovered}")
    for bird in discovered:
        run_wseg(
            save_path=save_path,
            metadata_dir=wseg_metadata,
            bird=bird,
            songs_per_bird=songs_per_bird,
            save_inst_freq=SAVE_INST_FREQ,
            save_group_delay=SAVE_GROUP_DELAY,
            duration_feature_weight=DURATION_FEATURE_WEIGHT,
        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_pipeline_cfg()

    if cfg['evsong_source']:
        run_evsonganaly_cohort(
            save_path      = cfg['save_path'],
            evsong_source  = cfg['evsong_source'],
            birds          = cfg['birds'],
            songs_per_bird = cfg['songs_per_bird'],
        )

    if cfg['wseg_metadata']:
        run_wseg_cohort(
            save_path      = cfg['save_path'],
            wseg_metadata  = cfg['wseg_metadata'],
            birds          = cfg['birds'],
            songs_per_bird = cfg['songs_per_bird'],
        )
