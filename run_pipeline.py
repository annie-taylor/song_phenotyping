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
# Run identity
# ---------------------------------------------------------------------------

RUN_NAME       = None    # None = auto-derive from params hash; or e.g. "baseline"
OVERWRITE_MODE = "skip"  # "skip" | "overwrite" | "archive"

# ---------------------------------------------------------------------------
# Spectrogram parameters (Stage A)
# ---------------------------------------------------------------------------

NFFT            = 1024
HOP             = 1
MIN_FREQ        = 200.0
MAX_FREQ        = 15000.0
MAX_DUR         = 0.080
FS              = 32000.0
TARGET_SHAPE    = None       # None → (NFFT//2+1, 300)
PADDING         = 0.0
USE_WARPING     = False
DOWNSAMPLE      = False

# Phase / duration features
SAVE_INST_FREQ          = False
SAVE_GROUP_DELAY        = False
DURATION_FEATURE_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# UMAP parameters (Stage C)
# ---------------------------------------------------------------------------

UMAP_MIN_DISTS   = None  # None → default grid [0.01, 0.05, 0.1, 0.3, 0.5]
UMAP_N_NEIGHBORS = None  # None → default grid [5, 10, 20, 50, 100]
UMAP_MAX_SAMPLES = None  # None → use all syllables

# ---------------------------------------------------------------------------
# HDBSCAN parameters (Stage D)
# ---------------------------------------------------------------------------

HDBSCAN_GRID = None  # None → DEFAULT_HDBSCAN_GRID from labelling.py

# ---------------------------------------------------------------------------
# Phenotyping parameters (Stage E)
# ---------------------------------------------------------------------------

PHENO_MIN_SYLLABLE_PROPORTION = 0.02
PHENO_GENERATE_PLOTS          = True

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
# Run management helper
# ---------------------------------------------------------------------------

def _prepare_run(bird_root, run_name: str, mode: str) -> None:
    """Handle existing run directory according to OVERWRITE_MODE."""
    import shutil
    from datetime import datetime
    from pathlib import Path as _Path
    run_dir = _Path(bird_root) / "runs" / run_name
    if not run_dir.exists():
        return
    if mode == "overwrite":
        shutil.rmtree(run_dir)
        print(f"  Deleted existing run: {run_dir}")
    elif mode == "archive":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = run_dir.parent / f"{run_name}_archived_{ts}"
        run_dir.rename(archived)
        print(f"  Archived existing run to: {archived}")
    # "skip": do nothing


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
        nfft=NFFT,
        hop=HOP,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        max_dur=MAX_DUR,
        fs=FS,
        target_shape=TARGET_SHAPE or (NFFT // 2 + 1, 300),
        padding=PADDING,
        use_warping=USE_WARPING,
        downsample=DOWNSAMPLE,
        songs_per_bird=songs_per_bird or 9999,
        save_inst_freq=SAVE_INST_FREQ,
        save_group_delay=SAVE_GROUP_DELAY,
        duration_feature_weight=DURATION_FEATURE_WEIGHT,
    )
    run_name = RUN_NAME or spec_params.run_hash()
    _prepare_run(Path(save_path) / bird, run_name, OVERWRITE_MODE)
    print(f"  Run name: {run_name}")

    save_specs_for_evsonganaly_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=spec_params,
        run_name=run_name,
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird, params=spec_params, run_name=run_name)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(
        save_path=save_path,
        bird=bird,
        min_dists=UMAP_MIN_DISTS,
        n_neighbors_list=UMAP_N_NEIGHBORS,
        max_samples=UMAP_MAX_SAMPLES,
        run_name=run_name,
    )

    print("[ D ] Clustering / labelling...")
    hdbscan_params = [p.to_dict() for p in HDBSCAN_GRID] if HDBSCAN_GRID is not None else [p.to_dict() for p in DEFAULT_HDBSCAN_GRID]
    label_bird(
        save_path=save_path,
        bird=bird,
        metrics=['silhouette', 'dbi', 'ch'],
        hdbscan_params=hdbscan_params,
        run_name=run_name,
    )

    print("[ E ] Phenotyping...")
    bird_path = str(Path(save_path) / bird)
    phenotype_bird(
        bird_path=bird_path,
        config=PhenotypingConfig(
            min_syllable_proportion=PHENO_MIN_SYLLABLE_PROPORTION,
            generate_plots=PHENO_GENERATE_PLOTS,
        ),
        run_name=run_name,
    )

    print(f"\n Done. Outputs at: {bird_path}/runs/{run_name}\n")


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
        nfft=NFFT,
        hop=HOP,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        max_dur=MAX_DUR,
        fs=FS,
        target_shape=TARGET_SHAPE or (NFFT // 2 + 1, 300),
        padding=PADDING,
        use_warping=USE_WARPING,
        downsample=DOWNSAMPLE,
        songs_per_bird=songs_per_bird or 9999,
        save_inst_freq=SAVE_INST_FREQ,
        save_group_delay=SAVE_GROUP_DELAY,
        duration_feature_weight=DURATION_FEATURE_WEIGHT,
    )
    run_name = RUN_NAME or spec_params.run_hash()
    _prepare_run(Path(save_path) / bird, run_name, OVERWRITE_MODE)
    print(f"  Run name: {run_name}")

    save_specs_for_wseg_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=spec_params,
        run_name=run_name,
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird, params=spec_params, run_name=run_name)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(
        save_path=save_path,
        bird=bird,
        min_dists=UMAP_MIN_DISTS,
        n_neighbors_list=UMAP_N_NEIGHBORS,
        max_samples=UMAP_MAX_SAMPLES,
        run_name=run_name,
    )

    print("[ D ] Clustering / labelling...")
    hdbscan_params = [p.to_dict() for p in HDBSCAN_GRID] if HDBSCAN_GRID is not None else [p.to_dict() for p in DEFAULT_HDBSCAN_GRID]
    label_bird(
        save_path=save_path,
        bird=bird,
        metrics=['silhouette', 'dbi', 'ch'],
        hdbscan_params=hdbscan_params,
        run_name=run_name,
    )

    print("[ E ] Phenotyping...")
    bird_path = str(Path(save_path) / bird)
    phenotype_bird(
        bird_path=bird_path,
        config=PhenotypingConfig(
            min_syllable_proportion=PHENO_MIN_SYLLABLE_PROPORTION,
            generate_plots=PHENO_GENERATE_PLOTS,
        ),
        run_name=run_name,
    )

    print(f"\n Done. Outputs at: {bird_path}/runs/{run_name}\n")

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
