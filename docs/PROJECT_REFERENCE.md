## Project: Song Phenotyping Pipeline
Automated 5-stage pipeline that extracts, embeds, clusters, and phenotypes Bengalese finch syllables from raw WAV recordings; used by Annie Taylor's research group.

## Stack
- Python 3.10, numpy, scipy, matplotlib, pandas, scikit-learn
- `tables` (PyTables) — all intermediate data stored as HDF5 via `tables.open_file()`
- `umap-learn` — Stage C dimensionality reduction
- `hdbscan` — Stage D density-based clustering
- `librosa`, `pyfftw` — audio processing and spectrogram computation
- `reportlab` — PDF generation in `scripts/` only (not in core package)
- `pyyaml` — config loading
- `pytest` — tests under `tests/`
- JupyterLab — interactive exploration (separate from pipeline runner)

## Structure
```
song_phenotyping/          # installable package (pip install -e .)
  signal.py                # Stage A: spectrogram extraction
  flattening.py            # Stage B: feature flattening
  embedding.py             # Stage C: UMAP grid search
  labelling.py             # Stage D: HDBSCAN clustering + metric scoring
  phenotyping.py           # Stage E: song-level phenotype computation
  catalog.py               # HTML catalog generation (called after Stage E)
  ingestion.py             # evsonganaly + wseg file discovery and Stage A dispatch
  tools/
    pipeline_paths.py      # SINGLE SOURCE OF TRUTH for all output directory names
    project_config.py      # ProjectConfig + PipelineConfig loaded from config.yaml
    spectrogram_configs.py # SpectrogramParams dataclass
    run_config.py          # RunConfig (provenance JSON) + RunRegistry (SQLite)
scripts/                   # standalone analysis scripts, not part of core pipeline
  phenotype_pdfs.py        # reportlab PDF generation (optional)
  song_catalog_pdf.py      # reportlab PDF catalog (optional)
  interactive_umap.py      # Dash app for exploring embeddings
  syllable_database.py     # syllable feature CSV builder
run_pipeline.py            # ENTRY POINT: full A→E run + stage re-entry functions
config.yaml.example        # committed template; copy to config.yaml (gitignored)
SETUP.md                   # machine setup + config reference + re-entry guide
tests/test_smoke.py        # end-to-end smoke test with a synthetic bird
docs/
  PROJECT_REFERENCE.md     # this file
  HANDOFF.md               # current session handoff (update at end of each session)
  history/                 # archived handoffs, one per major session
```

### Per-bird output layout
```
<save_path>/<bird>/
  stages/
    01_specs/          Stage A HDF5 spectrograms
    02_features/       Stage B flattened features
    03_embeddings/     Stage C UMAP HDF5 + pkl models (one file per grid combo)
    04_labels/         Stage D cluster label HDF5 (subdirs per umap_id)
    05_phenotype/      Stage E detailed phenotype pkls
  results/
    master_summary.csv      all UMAP × HDBSCAN combos + composite scores
    phenotype_results.csv   per-bird phenotype summary
    run_config.json         full parameter provenance
    catalog/                HTML song + syllable type catalogs
    plots/                  PDFs / figures (opt-in)
```

## Architecture decisions

- **5-stage linear pipeline (A→B→C→D→E)**: Each stage writes HDF5/CSV; the next reads them. Enables re-entry at any stage without reprocessing earlier ones. Rejected: monolithic in-memory pipeline (can't resume, too much RAM for large datasets).
- **`stages/` vs `results/` separation**: `stages/` = numbered internal HDF5/pkl; `results/` = human-inspectable CSV/HTML/JSON. Rejected: flat `syllable_data/` directory (opaque, mixes pipeline internals with user outputs).
- **Centralized path constants in `pipeline_paths.py`**: All modules import `SPECS_DIR`, `LABELS_DIR`, etc. from one file. Rejected: inline string literals (every directory rename required synchronized edits across 6+ files).
- **`config.yaml` gitignored, `config.yaml.example` committed**: Each machine keeps its own paths; no machine branches needed. Override constants at top of `run_pipeline.py` allow per-run tweaks without editing config.
- **UMAP grid search stores all results**: Stage C tries all `n_neighbors × min_dist` combinations and keeps all embedding files. Stage D scores all; composite score selects winner. `stages/03_embeddings/` may contain many files.
- **Composite score**: Weighted average of normalized metrics (silhouette + dbi by default). `ch` (Calinski-Harabasz) removed — found to add noise without benefit. Higher silhouette = better; dbi is inverted.
- **HTML-only catalogs in core pipeline**: `catalog.py` renders matplotlib → base64 PNG → HTML. reportlab confined to `scripts/` for optional separate use. Rejected: PDF in core pipeline (heavy dependency, slow, less browser-friendly).
- **Stage re-entry via explicit entry-point functions**: `run_from_labelling()`, `run_from_embedding()`, `run_from_phenotyping()` in `run_pipeline.py`. Rejected: re-running the full pipeline with skip flags (harder to reason about which stages actually ran).

## Conventions (non-obvious only)

- **Forward slashes on Windows**: Python accepts `E:/pipeline_runs`; backslash sequences like `\x` or `\n` in path strings are dangerous escape sequences that cause silent failures.
- **`select_new_file_pairs()` sorts before sampling**: Dict iteration order is not guaranteed stable; sort is required for `songs_seed` to produce the same subset across Python versions.
- **Stage re-entry defaults `replace_labels=True`**: `run_from_labelling()` and `run_from_embedding()` assume clean re-run; pass `replace_labels=False` explicitly to append.
- **`generate_cluster_pdf=False` by default in `label_bird()`**: Opt-in; matplotlib PdfPages adds overhead on every run even when the PDF isn't needed.
- **`songs_per_bird=9999` means "all songs"**: `SpectrogramParams.songs_per_bird` is an int, not Optional; 9999 is the sentinel for no limit at call sites in `run_pipeline.py`.
- **evsonganaly flat-directory fallback**: `filepaths_from_evsonganaly()` falls back to scanning subdirectories for co-located `.wav`/`.wav.not.mat` pairs when no `batch.txt.keep` files exist. Both PC source datasets use this path.
- **wseg `fname` field**: `.wav.not.mat` files contain an `fname` pointing to the actual audio (may be on Macaw server); `prefer_local=True` resolves from local cache first.
- **`random.Random(seed)` not `random.seed()`**: Song selection uses a local `Random` instance so the seed doesn't affect any other code using the stdlib random module.

## Do not touch

- `song_phenotyping/catalog.py`: Stable HTML generation; no PDF code. Don't add PDF generation back here.
- `song_phenotyping/tools/run_config.py` / `RunRegistry`: Stable SQLite schema; column renames break existing databases.
- `scripts/phenotype_pdfs.py`, `scripts/song_catalog_pdf.py`: reportlab optional scripts; dependency reduction deferred.

## External quirks

- **Macaw server (SMB mount)**: Auto-detected by `system_utils.check_sys_for_macaw_root()`; path differs by OS (macOS: `/Volumes/users`, Windows: `Z:\`). Set `paths.macaw_root` in config.yaml to override.
- **PyTables HDF5**: Opened with `mode='a'` for append. Never open the same file from two processes — no file locking.
- **HDBSCAN memory**: >50k syllables can exceed RAM during grid search. Stage C has `subsample_data()` fallback gated by memory threshold; seed is `embedding.subsample_seed` in config (default 42).

## Session workflow

**Start of session** — paste into your first message:
> "Read `docs/HANDOFF.md` and `docs/PROJECT_REFERENCE.md` before we begin."

**End of session** — four steps:
1. Write the session summary into the blank `docs/HANDOFF.md`.
2. Copy it to `docs/history/YYYY-MM-DD_short_description.md`.
3. Reset `docs/HANDOFF.md` to the blank template (see below).
4. Update `docs/PROJECT_REFERENCE.md` if any architecture decisions changed.
5. Commit all changes.

`docs/HANDOFF.md` should always be the blank template between sessions — a filled-in file is a signal the end-of-session steps weren't completed.

**Blank template for `docs/HANDOFF.md`:**
```markdown
## State as of [date]

_Fill in at the end of the session._

**What was built / changed**
-

**Key decisions made**
-

**Known issues / open questions**
-

**Do not touch**
-

**Next steps**
1.
```

## Run
```bash
pytest tests/test_smoke.py          # smoke test
python run_pipeline.py              # full pipeline
python scripts/interactive_umap.py  # Dash embedding explorer
```
