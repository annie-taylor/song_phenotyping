## State as of 2026-04-02

**What was built / changed**

- `song_phenotyping/tools/pipeline_paths.py` created: single source of truth for all output directory name constants (`SPECS_DIR`, `FEATURES_DIR`, `EMBEDDINGS_DIR`, `LABELS_DIR`, `PHENOTYPE_DIR`, `CATALOG_DIR`, `PLOTS_DIR`, `RESULTS_DIR`, `STAGES_DIR`). All modules import from here.
- Output directory structure renamed from opaque `syllable_data/*` to self-documenting `stages/01_specs/` through `stages/05_phenotype/` (internal) and `results/` (human-facing CSVs, HTML, PDFs).
- `song_phenotyping/tools/project_config.py`: `ProjectConfig` and `PipelineConfig` dataclasses added; `config.yaml.example` committed as template; `config.yaml` gitignored. Reads `pipeline.save_path`, `pipeline.evsong_source`, `pipeline.birds`, `pipeline.songs_per_bird`.
- `run_pipeline.py` rewritten: ALL-CAPS override constants at top (default `None` = use config.yaml); `_load_pipeline_cfg()` merges overrides with config; `run_evsonganaly_cohort()` and `run_wseg_cohort()` loop all discovered birds.
- `song_phenotyping/ingestion.py`: flat-directory fallback added to `filepaths_from_evsonganaly()` — activates when no `batch.txt.keep` files found; scans bird subdirectories for co-located `.wav`/`.wav.not.mat` pairs. Confirmed working on real PC data (52 pairs found for or18or24).
- All core modules (`signal.py`, `flattening.py`, `embedding.py`, `labelling.py`, `phenotyping.py`, `catalog.py`) updated to use `pipeline_paths.py` constants instead of inline strings.
- `scripts/` reorganized: root-level analysis scripts moved to `scripts/`; all updated to use new directory names.
- `tests/test_smoke.py` updated: all path assertions updated to new `stages/01_specs/` etc. names.
- `SETUP.md` created: machine setup guide covering clone/install, config.yaml creation, Windows path formatting (forward slashes), and git workflow.

**Key decisions made**

- **`stages/` (internal HDF5) vs `results/` (human-facing)**: Separates numbered pipeline artifacts from outputs users actually inspect. Rejected: flat `syllable_data/` (opaque, mixes concerns).
- **Single `main` branch + gitignored `config.yaml`**: Each machine has its own config; no permanent machine branches. Rejected: separate Mac/PC branches (merge overhead, no benefit).
- **`pipeline_paths.py` as central constant module**: All path strings defined once; import everywhere. Rejected: inline string literals (every rename required synchronized edits across 6+ files).
- **Flat-directory fallback (no batch files)**: Both PC source datasets (`ssharma_RNA_seq`, `xfosters`) had no `batch.txt.keep` files. Fallback activates automatically when no batch files found; co-located `.wav`/`.wav.not.mat` pairs are used directly.
- **Auto-discover + filter list for cohort mode**: `birds=None` discovers all birds; `birds=['or18or24']` restricts. Rejected: explicit list only (too rigid), single-bird only (too slow for large datasets).

**Known issues at this point**

- All pipeline parameters (spectrogram, UMAP, HDBSCAN, phenotyping) still hardcoded in `run_pipeline.py` — not in config.yaml.
- No stage re-entry: can't skip A–C and re-run D+E with different metrics.
- `generate_all_catalogs()` in `catalog.py` ready but not called from `run_pipeline.py`.
- Cluster summary PDF runs unconditionally inside `label_bird()`.
- `ch` metric included in default metrics despite being noisy.

**Next steps (as of this date)**

- Expose all stage params (spectrogram, UMAP, labelling, phenotyping) in config.yaml.
- Add stage re-entry functions (`run_from_labelling()` etc.).
- Wire `generate_all_catalogs()` into pipeline.
- Make cluster PDF opt-in.
- Remove ch from default metrics.
