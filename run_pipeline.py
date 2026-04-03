"""
Full A→E pipeline runner — with stage re-entry support.

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

To re-run labelling from existing embeddings with different metrics::

    if __name__ == '__main__':
        cfg = _load_pipeline_cfg()
        run_from_labelling(
            save_path=cfg['save_path'],
            birds=cfg['birds'],
            metrics=['silhouette', 'dbi'],   # removed ch
            replace_labels=True,
        )

See SETUP.md for a full stage re-entry guide.
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
SONGS_SEED     = None    # None = non-deterministic; int for reproducible subset

# Spectrogram flags (Stage A)
SAVE_INST_FREQ          = None   # None → use config.yaml; True/False to override
SAVE_GROUP_DELAY        = None
DURATION_FEATURE_WEIGHT = None   # 0 = disabled; ~1.0 weights duration as one time bin

# Labelling / clustering (Stage D)
METRICS              = None   # e.g. ['silhouette', 'dbi']   (default removes 'ch')
METRIC_WEIGHTS       = None   # e.g. {'silhouette': 2.0, 'dbi': 1.0}  (None = equal)
REPLACE_LABELS       = None   # None → False; True = delete existing labels + re-cluster
HDBSCAN_PARAMS       = None   # None = use grid from config.yaml (or built-in default)
GENERATE_CLUSTER_PDF = None   # None → False; True = write cluster summary PDF

# Phenotyping / output (Stage E + catalog)
GENERATE_CATALOG     = None   # None → True; False = skip HTML catalog generation
GENERATE_PLOTS       = None   # None → True (phenotype summary figures)

# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------

_DEFAULT_METRICS = ['silhouette', 'dbi']   # ch removed from default


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
    songs_seed     = SONGS_SEED     if SONGS_SEED is not None else p.songs_seed

    # Spectrogram overrides
    spec_cfg = dict(p.spectrogram_params or {})
    if SAVE_INST_FREQ is not None:
        spec_cfg['save_inst_freq'] = SAVE_INST_FREQ
    if SAVE_GROUP_DELAY is not None:
        spec_cfg['save_group_delay'] = SAVE_GROUP_DELAY
    if DURATION_FEATURE_WEIGHT is not None:
        spec_cfg['duration_feature_weight'] = DURATION_FEATURE_WEIGHT

    # Labelling overrides
    lab_cfg = dict(p.labelling_params or {})
    if METRICS is not None:
        lab_cfg['metrics'] = METRICS
    if METRIC_WEIGHTS is not None:
        lab_cfg['metric_weights'] = METRIC_WEIGHTS
    if REPLACE_LABELS is not None:
        lab_cfg['replace_labels'] = REPLACE_LABELS
    if HDBSCAN_PARAMS is not None:
        lab_cfg['hdbscan_grid'] = HDBSCAN_PARAMS
    if GENERATE_CLUSTER_PDF is not None:
        lab_cfg['generate_cluster_pdf'] = GENERATE_CLUSTER_PDF

    # Phenotyping / catalog overrides
    pheno_cfg = dict(p.phenotyping_params or {})
    if GENERATE_PLOTS is not None:
        pheno_cfg['generate_plots'] = GENERATE_PLOTS

    generate_catalog = (
        GENERATE_CATALOG if GENERATE_CATALOG is not None else p.generate_catalog
    )

    if save_path is None:
        raise ValueError(
            "save_path is not set. Add 'pipeline.save_path' to config.yaml "
            "or set SAVE_PATH at the top of run_pipeline.py."
        )

    return dict(
        save_path        = save_path,
        evsong_source    = evsong_source,
        wseg_metadata    = wseg_metadata,
        birds            = birds,
        songs_per_bird   = songs_per_bird,
        songs_seed       = songs_seed,
        spec_cfg         = spec_cfg,
        lab_cfg          = lab_cfg,
        pheno_cfg        = pheno_cfg,
        generate_catalog = generate_catalog,
    )


def _build_spec_params(songs_per_bird, spec_cfg: dict):
    """Build a SpectrogramParams from config values."""
    from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(SpectrogramParams)}
    filtered = {k: v for k, v in spec_cfg.items() if k in valid_fields}
    return SpectrogramParams(
        songs_per_bird=songs_per_bird or 9999,
        **filtered,
    )


def _build_hdbscan_grid(lab_cfg: dict):
    """Build a list of HDBSCAN param dicts from config or fall back to DEFAULT_HDBSCAN_GRID."""
    from song_phenotyping.labelling import DEFAULT_HDBSCAN_GRID, HDBSCANParams
    grid_cfg = lab_cfg.get('hdbscan_grid')
    if grid_cfg is None:
        return [p.to_dict() for p in DEFAULT_HDBSCAN_GRID]
    # grid_cfg is expected to be a dict with keys min_cluster_size and min_samples
    # each being a list of values
    mcs_list = grid_cfg.get('min_cluster_size', [5, 20, 60])
    ms_list  = grid_cfg.get('min_samples', [5, 15])
    return [
        HDBSCANParams(min_cluster_size=mcs, min_samples=ms).to_dict()
        for mcs in mcs_list
        for ms in ms_list
    ]


def _build_phenotype_config(pheno_cfg: dict):
    """Build a PhenotypingConfig from config values."""
    from song_phenotyping.phenotyping import PhenotypingConfig
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(PhenotypingConfig)}
    filtered = {k: v for k, v in pheno_cfg.items() if k in valid_fields}
    return PhenotypingConfig(**filtered)


def _save_run_config(bird_path: str, bird: str, spec_params, lab_cfg: dict,
                     pheno_cfg: dict, generate_catalog: bool):
    """Write a run_config.json to <bird_path>/results/ capturing all params."""
    import os
    from song_phenotyping.tools.run_config import RunConfig
    from song_phenotyping.tools.pipeline_paths import RESULTS_DIR

    results_dir = os.path.join(bird_path, RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'run_config.json')

    # Load existing config if present (to preserve upstream stage params on re-entry)
    existing = {}
    if os.path.exists(json_path):
        try:
            existing = RunConfig.load(json_path).to_dict()
        except Exception:
            pass

    # Build new config; allow existing spec/umap params to survive a labelling re-run
    rc = RunConfig.create(
        spec_mode='syllable',
        bird_ids=[bird],
        spec_params=spec_params,
        phenotype_params=_build_phenotype_config(pheno_cfg),
    )

    # Merge: keep spec/umap from existing if this is a downstream re-entry
    if existing:
        rc_dict = rc.to_dict()
        rc_dict['spec_params'] = existing.get('spec_params', rc_dict['spec_params'])
        rc_dict['umap_params'] = existing.get('umap_params', rc_dict['umap_params'])
        # Annotate labelling params in notes
        rc_dict['notes'] = (
            f"metrics={lab_cfg.get('metrics', _DEFAULT_METRICS)} "
            f"weights={lab_cfg.get('metric_weights')} "
            f"replace={lab_cfg.get('replace_labels', False)} "
            f"catalog={generate_catalog}"
        )
        from song_phenotyping.tools.run_config import RunConfig as _RC
        rc = _RC(**rc_dict)

    rc.save(json_path)


def _run_catalog(bird_path: str, generate_catalog: bool):
    """Call generate_all_catalogs() if requested."""
    if not generate_catalog:
        return
    try:
        from song_phenotyping.catalog import generate_all_catalogs
        print("[ Catalog ] Generating HTML catalogs...")
        generate_all_catalogs(bird_path=str(bird_path))
        print(f"[ Catalog ] HTML catalogs written to {bird_path}/results/catalog/")
    except Exception as e:
        print(f"[ Catalog ] Warning: catalog generation failed ({e}); pipeline output is still complete.")


# ---------------------------------------------------------------------------
# Single-bird pipeline runners
# ---------------------------------------------------------------------------

def run_evsonganaly(save_path: str, source_dir: str, bird: str, songs_per_bird,
                    songs_seed=None, spec_cfg=None, lab_cfg=None, pheno_cfg=None,
                    generate_catalog=True):
    """Run stages A–E for a single evsonganaly bird."""
    from song_phenotyping.ingestion import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
    from song_phenotyping.flattening import flatten_bird_spectrograms
    from song_phenotyping.embedding import explore_embedding_parameters_robust
    from song_phenotyping.labelling import label_bird

    spec_cfg   = spec_cfg   or {}
    lab_cfg    = lab_cfg    or {}
    pheno_cfg  = pheno_cfg  or {}

    print(f"\n{'='*60}")
    print(f"  Running pipeline for {bird} (evsonganaly)")
    print(f"  Source : {source_dir}")
    print(f"  Output : {save_path}/{bird}")
    print(f"{'='*60}\n")

    spec_params = _build_spec_params(songs_per_bird, spec_cfg)
    bird_path = str(Path(save_path) / bird)

    print("[ A ] Saving spectrograms...")
    meta, audio = filepaths_from_evsonganaly(
        wav_directory=source_dir, bird_subset=[bird]
    )
    save_specs_for_evsonganaly_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=spec_params,
        songs_seed=songs_seed,
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird, params=spec_params)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(save_path=save_path, bird=bird)

    print("[ D ] Clustering / labelling...")
    label_bird(
        save_path=save_path,
        bird=bird,
        metrics=lab_cfg.get('metrics', _DEFAULT_METRICS),
        metric_weights=lab_cfg.get('metric_weights'),
        replace_labels=lab_cfg.get('replace_labels', False),
        hdbscan_params=_build_hdbscan_grid(lab_cfg),
        generate_cluster_pdf=lab_cfg.get('generate_cluster_pdf', False),
    )

    print("[ E ] Phenotyping...")
    from song_phenotyping.phenotyping import phenotype_bird
    phenotype_bird(bird_path=bird_path, config=_build_phenotype_config(pheno_cfg))

    _run_catalog(bird_path, generate_catalog)
    _save_run_config(bird_path, bird, spec_params, lab_cfg, pheno_cfg, generate_catalog)

    print(f"\n[OK] Done. Outputs at: {bird_path}\n")


def run_wseg(save_path: str, metadata_dir: str, bird: str, songs_per_bird,
             songs_seed=None, spec_cfg=None, lab_cfg=None, pheno_cfg=None,
             generate_catalog=True):
    """Run stages A–E for a single wseg bird."""
    from song_phenotyping.ingestion import filepaths_from_wseg, save_specs_for_wseg_birds
    from song_phenotyping.flattening import flatten_bird_spectrograms
    from song_phenotyping.embedding import explore_embedding_parameters_robust
    from song_phenotyping.labelling import label_bird
    from song_phenotyping.phenotyping import phenotype_bird

    spec_cfg   = spec_cfg   or {}
    lab_cfg    = lab_cfg    or {}
    pheno_cfg  = pheno_cfg  or {}

    print(f"\n{'='*60}")
    print(f"  Running pipeline for {bird} (wseg)")
    print(f"  Metadata: {metadata_dir}")
    print(f"  Output  : {save_path}/{bird}")
    print(f"{'='*60}\n")

    spec_params = _build_spec_params(songs_per_bird, spec_cfg)
    bird_path = str(Path(save_path) / bird)

    print("[ A ] Saving spectrograms...")
    meta, audio = filepaths_from_wseg(
        seg_directory=metadata_dir, bird_subset=[bird]
    )
    save_specs_for_wseg_birds(
        metadata_file_paths=meta,
        audio_file_paths=audio,
        save_path=save_path,
        songs_per_bird=songs_per_bird,
        params=spec_params,
        songs_seed=songs_seed,
    )

    print("[ B ] Flattening spectrograms...")
    flatten_bird_spectrograms(directory=save_path, bird=bird, params=spec_params)

    print("[ C ] Computing UMAP embeddings...")
    explore_embedding_parameters_robust(save_path=save_path, bird=bird)

    print("[ D ] Clustering / labelling...")
    label_bird(
        save_path=save_path,
        bird=bird,
        metrics=lab_cfg.get('metrics', _DEFAULT_METRICS),
        metric_weights=lab_cfg.get('metric_weights'),
        replace_labels=lab_cfg.get('replace_labels', False),
        hdbscan_params=_build_hdbscan_grid(lab_cfg),
        generate_cluster_pdf=lab_cfg.get('generate_cluster_pdf', False),
    )

    print("[ E ] Phenotyping...")
    phenotype_bird(bird_path=bird_path, config=_build_phenotype_config(pheno_cfg))

    _run_catalog(bird_path, generate_catalog)
    _save_run_config(bird_path, bird, spec_params, lab_cfg, pheno_cfg, generate_catalog)

    print(f"\n[OK] Done. Outputs at: {bird_path}\n")


# ---------------------------------------------------------------------------
# Stage re-entry runners
# ---------------------------------------------------------------------------

def run_from_embedding(save_path: str, birds, lab_cfg=None, pheno_cfg=None,
                       generate_catalog=True):
    """Re-run Stages C→E from existing flattened features (stages/02_features/).

    Use when UMAP parameters have changed. Stages A and B are skipped.

    Requires: ``stages/02_features/`` HDF5 files for each bird.

    Parameters
    ----------
    save_path : str
        Pipeline output root (same value as the original full run).
    birds : list of str or None
        Birds to process.  ``None`` is not valid here — provide an explicit list.
    lab_cfg : dict, optional
        Labelling overrides (metrics, weights, HDBSCAN grid, etc.).
    pheno_cfg : dict, optional
        Phenotyping overrides.
    generate_catalog : bool, optional
        Whether to regenerate HTML catalogs after Stage E.  Default ``True``.
    """
    from song_phenotyping.embedding import explore_embedding_parameters_robust
    from song_phenotyping.labelling import label_bird
    from song_phenotyping.phenotyping import phenotype_bird

    lab_cfg   = lab_cfg   or {}
    pheno_cfg = pheno_cfg or {}

    if not birds:
        raise ValueError("run_from_embedding() requires an explicit birds list.")

    for bird in birds:
        bird_path = str(Path(save_path) / bird)
        print(f"\n[C→E] Re-running from embeddings for {bird}")

        print("[ C ] Computing UMAP embeddings...")
        explore_embedding_parameters_robust(save_path=save_path, bird=bird)

        print("[ D ] Clustering / labelling...")
        label_bird(
            save_path=save_path,
            bird=bird,
            metrics=lab_cfg.get('metrics', _DEFAULT_METRICS),
            metric_weights=lab_cfg.get('metric_weights'),
            replace_labels=lab_cfg.get('replace_labels', True),
            hdbscan_params=_build_hdbscan_grid(lab_cfg),
            generate_cluster_pdf=lab_cfg.get('generate_cluster_pdf', False),
        )

        print("[ E ] Phenotyping...")
        phenotype_bird(bird_path=bird_path, config=_build_phenotype_config(pheno_cfg))

        _run_catalog(bird_path, generate_catalog)
        print(f"[OK] Done. Outputs at: {bird_path}")


def run_from_labelling(save_path: str, birds, metrics=None, metric_weights=None,
                       replace_labels=True, hdbscan_params=None,
                       generate_cluster_pdf=False, pheno_cfg=None,
                       generate_catalog=True):
    """Re-run Stages D→E from existing UMAP embeddings (stages/03_embeddings/).

    Use when clustering metrics or HDBSCAN parameters have changed.
    Stages A, B, and C are skipped.  ``replace_labels=True`` by default so
    old label files and master_summary.csv are cleared before re-clustering.

    Requires: ``stages/03_embeddings/`` HDF5 files for each bird.

    Parameters
    ----------
    save_path : str
        Pipeline output root.
    birds : list of str
        Birds to re-label.  Provide an explicit list.
    metrics : list of str, optional
        Evaluation metrics.  Defaults to ``['silhouette', 'dbi']``.
    metric_weights : dict or None, optional
        Per-metric weights, e.g. ``{'silhouette': 2.0, 'dbi': 1.0}``.
    replace_labels : bool, optional
        Delete existing label files before re-clustering.  Default ``True``.
    hdbscan_params : dict or None, optional
        Custom HDBSCAN grid dict (same format as ``config.yaml labelling.hdbscan_grid``).
        ``None`` uses the built-in default grid.
    generate_cluster_pdf : bool, optional
        Write a cluster summary PDF.  Default ``False``.
    pheno_cfg : dict, optional
        Phenotyping overrides.
    generate_catalog : bool, optional
        Whether to regenerate HTML catalogs after Stage E.  Default ``True``.

    Examples
    --------
    Re-run with silhouette + dbi only (removing ch)::

        run_from_labelling(
            save_path='E:/pipeline_runs',
            birds=['or18or24'],
            metrics=['silhouette', 'dbi'],
        )
    """
    from song_phenotyping.labelling import label_bird
    from song_phenotyping.phenotyping import phenotype_bird

    if metrics is None:
        metrics = _DEFAULT_METRICS
    pheno_cfg = pheno_cfg or {}
    lab_cfg   = dict(
        metrics=metrics,
        metric_weights=metric_weights,
        replace_labels=replace_labels,
        hdbscan_grid=hdbscan_params,
        generate_cluster_pdf=generate_cluster_pdf,
    )

    if not birds:
        raise ValueError("run_from_labelling() requires an explicit birds list.")

    for bird in birds:
        bird_path = str(Path(save_path) / bird)
        print(f"\n[D→E] Re-running labelling + phenotyping for {bird}")
        print(f"      metrics={metrics}  weights={metric_weights}  replace={replace_labels}")

        print("[ D ] Clustering / labelling...")
        label_bird(
            save_path=save_path,
            bird=bird,
            metrics=metrics,
            metric_weights=metric_weights,
            replace_labels=replace_labels,
            hdbscan_params=_build_hdbscan_grid(lab_cfg),
            generate_cluster_pdf=generate_cluster_pdf,
        )

        print("[ E ] Phenotyping...")
        phenotype_bird(bird_path=bird_path, config=_build_phenotype_config(pheno_cfg))

        _run_catalog(bird_path, generate_catalog)
        print(f"[OK] Done. Outputs at: {bird_path}")


def run_from_phenotyping(save_path: str, birds, pheno_cfg=None, generate_catalog=True):
    """Re-run Stage E from existing cluster labels (stages/04_labels/).

    Use when phenotyping thresholds or plot settings have changed.
    Stages A through D are skipped.

    Requires: ``stages/04_labels/`` and ``results/master_summary.csv`` for each bird.

    Parameters
    ----------
    save_path : str
        Pipeline output root.
    birds : list of str
        Birds to re-phenotype.
    pheno_cfg : dict, optional
        Phenotyping overrides (min_syllable_proportion, dyad_threshold, etc.).
    generate_catalog : bool, optional
        Whether to regenerate HTML catalogs after Stage E.  Default ``True``.
    """
    from song_phenotyping.phenotyping import phenotype_bird

    pheno_cfg = pheno_cfg or {}

    if not birds:
        raise ValueError("run_from_phenotyping() requires an explicit birds list.")

    for bird in birds:
        bird_path = str(Path(save_path) / bird)
        print(f"\n[E] Re-running phenotyping for {bird}")

        print("[ E ] Phenotyping...")
        phenotype_bird(bird_path=bird_path, config=_build_phenotype_config(pheno_cfg))

        _run_catalog(bird_path, generate_catalog)
        print(f"[OK] Done. Outputs at: {bird_path}")


# ---------------------------------------------------------------------------
# Cohort runners
# ---------------------------------------------------------------------------

def run_evsonganaly_cohort(save_path, evsong_source, birds, songs_per_bird,
                           songs_seed=None, spec_cfg=None, lab_cfg=None,
                           pheno_cfg=None, generate_catalog=True):
    """Run the full pipeline for all (or a filtered subset of) evsonganaly birds."""
    from song_phenotyping.ingestion import filepaths_from_evsonganaly

    meta, audio = filepaths_from_evsonganaly(
        wav_directory=evsong_source,
        bird_subset=birds,
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
            songs_seed=songs_seed,
            spec_cfg=spec_cfg,
            lab_cfg=lab_cfg,
            pheno_cfg=pheno_cfg,
            generate_catalog=generate_catalog,
        )


def run_wseg_cohort(save_path, wseg_metadata, birds, songs_per_bird,
                    songs_seed=None, spec_cfg=None, lab_cfg=None,
                    pheno_cfg=None, generate_catalog=True):
    """Run the full pipeline for all (or a filtered subset of) wseg birds."""
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
            songs_seed=songs_seed,
            spec_cfg=spec_cfg,
            lab_cfg=lab_cfg,
            pheno_cfg=pheno_cfg,
            generate_catalog=generate_catalog,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_pipeline_cfg()

    # -------------------------------------------------------------------
    # Full pipeline (default)
    # -------------------------------------------------------------------
    # To re-run only later stages, replace the block below with one of:
    #
    #   run_from_labelling(cfg['save_path'], birds=['or18or24'],
    #                      metrics=['silhouette', 'dbi'])
    #
    #   run_from_phenotyping(cfg['save_path'], birds=['or18or24'])
    #
    # See SETUP.md → "Re-entering the pipeline" for full examples.
    # -------------------------------------------------------------------

    if cfg['evsong_source']:
        run_evsonganaly_cohort(
            save_path        = cfg['save_path'],
            evsong_source    = cfg['evsong_source'],
            birds            = cfg['birds'],
            songs_per_bird   = cfg['songs_per_bird'],
            songs_seed       = cfg['songs_seed'],
            spec_cfg         = cfg['spec_cfg'],
            lab_cfg          = cfg['lab_cfg'],
            pheno_cfg        = cfg['pheno_cfg'],
            generate_catalog = cfg['generate_catalog'],
        )

    if cfg['wseg_metadata']:
        run_wseg_cohort(
            save_path        = cfg['save_path'],
            wseg_metadata    = cfg['wseg_metadata'],
            birds            = cfg['birds'],
            songs_per_bird   = cfg['songs_per_bird'],
            songs_seed       = cfg['songs_seed'],
            spec_cfg         = cfg['spec_cfg'],
            lab_cfg          = cfg['lab_cfg'],
            pheno_cfg        = cfg['pheno_cfg'],
            generate_catalog = cfg['generate_catalog'],
        )
