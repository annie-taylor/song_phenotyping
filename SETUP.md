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

## 5. Git workflow (Mac ↔ PC)

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
