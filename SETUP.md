# Machine Setup Guide

Quick reference for getting the pipeline running on a new machine (Mac, PC, or Linux).

## 1. Clone and install

```bash
git clone <repo-url>
cd song_phenotyping
conda env create -f environment.yml
conda activate song_phenotyping
pip install -e .
```

## 2. Create your local config

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` — it is **gitignored** and never committed, so each machine keeps its own copy:

```yaml
paths:
  # macOS
  local_cache: /Volumes/Extreme SSD
  # Windows (use forward slashes)
  # local_cache: E:/

  macaw_root: null   # auto-detect, or set explicitly
  run_registry: db.sqlite3

pipeline:
  # macOS
  save_path: /Volumes/Extreme SSD/pipeline_runs
  evsong_source: /Volumes/Extreme SSD/birds/evsong
  # Windows
  # save_path: E:/pipeline_runs
  # evsong_source: E:/birds/evsong

  wseg_metadata: null     # set to wseg metadata dir, or leave null to skip
  birds: null             # null = all discovered birds
  songs_per_bird: 30      # reduce for quick smoke tests
```

> **Windows note:** Forward slashes work fine on Windows in Python (`E:/pipeline_runs`).
> You do NOT need backslashes.

## 3. Run the pipeline

```bash
python run_pipeline.py
```

To quickly test a single bird without editing `config.yaml`, set the override
constants at the top of `run_pipeline.py`:

```python
EVSONG_SOURCE  = "E:/smoke_test_birds"
BIRDS          = ["or18or24"]
SONGS_PER_BIRD = 5
```

Leave them as `None` to fall back to `config.yaml` values.

## 4. Output structure

```
pipeline_runs/
└── or18or24/
    ├── stages/              ← internal pipeline artifacts (not for everyday browsing)
    │   ├── 01_specs/        Stage A  HDF5 spectrograms
    │   ├── 02_features/     Stage B  HDF5 flattened features
    │   ├── 03_embeddings/   Stage C  UMAP HDF5 + pkl models
    │   ├── 04_labels/       Stage D  cluster label HDF5 files
    │   └── 05_phenotype/    Stage E  detailed phenotype pkl files
    └── results/             ← human-facing outputs
        ├── master_summary.csv
        ├── phenotype_results.csv
        ├── catalog/         HTML song catalogs
        └── plots/           PDFs / images
```

## 5. Config reference

The `pipeline:` section of `config.yaml` controls all pipeline stages.
Every key has a default; omit it to accept the default.

| Key | Stage | Default | Purpose |
|-----|-------|---------|---------|
| `save_path` | all | — | Root output directory |
| `evsong_source` | A | null | evsonganaly bird folder parent |
| `wseg_metadata` | A | null | WhisperSeg metadata dir |
| `birds` | all | null (all) | Bird IDs to process |
| `songs_per_bird` | A | null (all) | Max songs per bird |
| `songs_seed` | A | null | Seed for song subset selection |
| `generate_catalog` | catalog | true | Write HTML catalogs after Stage E |
| `spectrograms.*` | A | see below | Spectrogram computation params |
| `embedding.*` | C | see below | UMAP params + grid |
| `labelling.*` | D | see below | Metrics, HDBSCAN grid, flags |
| `phenotyping.*` | E | see below | Phenotyping thresholds and flags |

### `spectrograms:` sub-section (Stage A)

| Key | Default | Notes |
|-----|---------|-------|
| `nfft` | 1024 | FFT window size |
| `hop` | 1 | Hop size in samples |
| `target_shape` | [513, 300] | [freq_bins, time_bins] per syllable |
| `min_freq` / `max_freq` | 200 / 15000 | Hz |
| `max_dur` | 0.080 | Max syllable duration (s) |
| `fs` | 32000 | Expected sample rate |
| `save_inst_freq` | false | Append instantaneous-frequency channel |
| `save_group_delay` | false | Append group-delay channel |
| `duration_feature_weight` | 0.0 | 0 = off; ~1.0 = weight duration |
| `use_warping` | false | Frequency-warping augmentation |

### `embedding:` sub-section (Stage C)

| Key | Default | Notes |
|-----|---------|-------|
| `n_neighbors` | 20 | UMAP neighbourhood size |
| `min_dist` | 0.5 | UMAP minimum distance |
| `metric` | euclidean | Distance metric |
| `n_components` | 2 | Embedding dimensions |
| `subsample_seed` | 42 | Seed for syllable subsampling |
| `n_neighbors_grid` | [5,10,20,50,100] | Grid values tried in parameter search |
| `min_dist_grid` | [0.01,0.05,0.1,0.3,0.5] | Grid values tried in parameter search |

### `labelling:` sub-section (Stage D)

| Key | Default | Notes |
|-----|---------|-------|
| `metrics` | [silhouette, dbi] | Evaluation metrics (ch excluded by default) |
| `metric_weights` | null (equal) | Dict of per-metric weights |
| `replace_labels` | false | Delete existing labels before re-clustering |
| `hdbscan_grid` | null (built-in) | Custom grid: `{min_cluster_size: [...], min_samples: [...]}` |
| `generate_cluster_pdf` | false | Write PDF of top clusterings |

### `phenotyping:` sub-section (Stage E)

| Key | Default | Notes |
|-----|---------|-------|
| `min_syllable_proportion` | 0.02 | Min fraction of songs for a syllable type |
| `repeat_significance_threshold` | 0.2 | |
| `dyad_threshold` | 0.7 | |
| `use_top_n_clusterings` | 5 | |
| `generate_plots` | true | Write phenotype summary figures |

---

## 6. Re-entering the pipeline

The pipeline exposes entry-point functions for Stages C, D, and E so you
can re-run from any point without reprocessing earlier stages.

### When to use which entry point

| Scenario | Function | Skips |
|----------|----------|-------|
| Source data changed | `run_evsonganaly_cohort()` (full run) | — |
| UMAP parameters changed | `run_from_embedding()` | A, B |
| Metrics / HDBSCAN changed | `run_from_labelling()` | A, B, C |
| Phenotype thresholds changed | `run_from_phenotyping()` | A, B, C, D |

### Option A — edit `run_pipeline.py` `__main__` block

Replace the `run_evsonganaly_cohort(...)` call at the bottom of
`run_pipeline.py` with the appropriate entry point:

```python
# Re-run labelling with silhouette + dbi only (removed ch):
if __name__ == '__main__':
    cfg = _load_pipeline_cfg()
    run_from_labelling(
        save_path=cfg['save_path'],
        birds=['or18or24'],           # explicit list required
        metrics=['silhouette', 'dbi'],
    )
```

### Option B — Python session / script

```python
from run_pipeline import run_from_labelling, run_from_phenotyping

# Re-run D + E with new metrics
run_from_labelling(
    save_path='E:/pipeline_runs',
    birds=['or18or24'],
    metrics=['silhouette', 'dbi'],
    metric_weights={'silhouette': 2.0, 'dbi': 1.0},
)

# Re-run only E (phenotype thresholds changed)
run_from_phenotyping(
    save_path='E:/pipeline_runs',
    birds=['or18or24'],
    pheno_cfg={'min_syllable_proportion': 0.03},
)
```

### What each stage needs to exist

| Stage | Entry function | Requires |
|-------|---------------|----------|
| A — specs | `run_evsonganaly_cohort()` | Raw WAV source |
| B — features | (called internally) | `stages/01_specs/` |
| C — embeddings | `run_from_embedding()` | `stages/02_features/` |
| D — labels | `run_from_labelling()` | `stages/03_embeddings/` |
| E — phenotype | `run_from_phenotyping()` | `stages/04_labels/` + `results/master_summary.csv` |
| Catalog | (called after E) | `stages/01_specs/`, `stages/04_labels/` |

---

## 7. Reproducibility

When `songs_per_bird` is less than the total available songs, Stage A
selects a random subset.  To make this selection reproducible:

```yaml
# config.yaml
pipeline:
  songs_per_bird: 30
  songs_seed: 42   # same 30 songs every run
```

Or override in `run_pipeline.py`:

```python
SONGS_SEED = 42
```

The UMAP subsampling seed is fixed at 42 by default
(`pipeline.embedding.subsample_seed`).  Together, these two seeds fully
specify which data produced a given result, assuming the source file list
is stable.

---

## 8. Git workflow (Mac ↔ PC)


**TL;DR:** `config.yaml` is gitignored — it stays on each machine. Push code
changes to `main`; pull on the other machine to stay in sync.

```
Mac                              PC
───                              ──
edit code                        git pull origin main
git add / commit / push     →    run pipeline on large datasets
git pull                    ←    (no code changes on PC)
```

- **Feature work:** Create a short-lived feature branch (`git checkout -b feat/my-feature`),
  do your work, then merge to `main` and delete the branch.
- **No permanent machine branches** — `config.yaml` already handles the only
  real machine difference (paths). Separate branches create merge overhead
  without benefit.
- **Push frequently** to avoid drift. Prefer small, focused commits so that
  `git pull` on the other machine is always clean.
- If you need to run a quick experiment with different parameters, use the
  override constants in `run_pipeline.py` (they are never committed since you
  shouldn't commit the override values).
