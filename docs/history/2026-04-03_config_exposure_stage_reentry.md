## State as of 2026-04-03

**What was built / changed**

- `config.yaml.example`: Added `pipeline.spectrograms`, `pipeline.embedding`, `pipeline.labelling`, `pipeline.phenotyping` sub-sections covering all tunable params; added `songs_seed` and `generate_catalog` keys. This is the canonical documentation of every supported config value.
- `song_phenotyping/tools/project_config.py`: Extended `PipelineConfig` dataclass with `songs_seed`, `spectrogram_params`, `embedding_params`, `labelling_params`, `phenotyping_params`, `generate_catalog`. All sub-sections stored as plain `dict`, merged into typed dataclasses at call sites.
- `song_phenotyping/ingestion.py`: Added `seed` parameter to `select_new_file_pairs()` using `random.Random(seed)` (instance-local, no global state side-effects); candidates sorted before sampling so selection is fully deterministic when seed is set. `songs_seed` threaded through both `save_specs_for_evsonganaly_birds()` and `save_specs_for_wseg_birds()`.
- `song_phenotyping/labelling.py`: `label_bird()` gains `generate_cluster_pdf=False` (PDF now opt-in, saves matplotlib overhead on every run) and `metric_weights=None` (passed to `compute_composite_score()`).
- `run_pipeline.py`: Full rewrite. Added ALL-CAPS overrides for every param category; `_build_spec_params()`, `_build_hdbscan_grid()`, `_build_phenotype_config()` helpers merge config dicts into typed objects; `run_from_embedding()`, `run_from_labelling()`, `run_from_phenotyping()` stage re-entry functions; `_run_catalog()` calls `generate_all_catalogs()` after Stage E; `_save_run_config()` writes `results/run_config.json` per bird; default metrics changed to `['silhouette', 'dbi']` (ch removed).
- `SETUP.md`: Added sections 5–7: full config reference table, stage re-entry guide with copy-paste examples, reproducibility section.
- `docs/PROJECT_REFERENCE.md`, `docs/HANDOFF.md`, `docs/history/`: Created permanent reference docs and full historical handoff series.

**Key decisions made**

- **`random.Random(seed)` instance, not global `random.seed()`**: Avoids side-effects on any other code using the stdlib random module.
- **Candidates sorted before sampling**: Dict iteration order is not stable; sort is required for full determinism when seed is set.
- **`generate_cluster_pdf` defaults to False**: matplotlib PdfPages overhead on every run; opt-in preserves the capability without the cost.
- **`metric_weights` commented-out in config.yaml**: Advanced knob; shown as a comment rather than active default to avoid cluttering typical workflow.
- **`_save_run_config()` merges upstream params on re-entry**: When `run_from_labelling()` runs, existing `run_config.json` spec/umap entries are preserved so the JSON reflects full provenance.
- **`generate_all_catalogs()` wrapped in try/except**: Catalog failure must never abort a run that successfully produced HDF5 labels and CSVs.

**Known issues / open questions**

- `select_wseg_file_pairs_from_metadata()` (the `copy_locally=False` wseg path) does NOT receive `songs_seed` — seed only applies to `copy_locally=True` wseg and evsonganaly paths.
- `_save_run_config()` stores `UMAPParams()` defaults for `umap_params`, not the actual grid-search winner; fix requires reading `processing_metadata` from the winning embedding HDF5.
- `generate_all_catalogs()` silently produces empty/partial HTML if Stage D labels don't exist; not handled explicitly.
- Claude usage optimization advice (requested this session) was never addressed.

**Do not touch**

- `song_phenotyping/catalog.py`: Stable; wired into pipeline. Don't add PDF generation back.
- `scripts/phenotype_pdfs.py`, `scripts/song_catalog_pdf.py`: reportlab optional scripts; PDF dependency reduction deferred.
- `song_phenotyping/tools/run_config.py`: Stable SQLite schema; column renames break existing databases.

**Next steps**

1. Test full pipeline with new config on PC; verify `results/catalog/` HTML opens correctly in browser.
2. Test `run_from_labelling()` end-to-end; confirm labels refreshed, `run_config.json` updated, catalog regenerated.
3. Fix wseg seed gap: thread `songs_seed` into `select_wseg_file_pairs_from_metadata()`.
4. Wire actual umap_params into `_save_run_config()` by reading from winning embedding HDF5 metadata.
5. Open and review HTML catalogs: confirm song catalog labels correct above/below spectrograms, syllable type grids look right.
