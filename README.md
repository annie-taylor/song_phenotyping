# song_phenotyping

Automated pipeline for phenotyping Bengalese finch song.

`song_phenotyping` segments, embeds, clusters, and characterises syllable repertoires from raw audio recordings. It supports two annotation formats (evsonganaly and wseg) and produces standardised phenotype tables suitable for statistical comparison across birds and conditions.

## Installation

**Requirements:** Python 3.9+, conda recommended.

```bash
git clone https://github.com/annie-taylor/song_phenotyping.git
cd song_phenotyping
conda env create -f environment.yml
conda activate song_phenotyping
pip install -e .
```

**Optional extras:**

```bash
pip install -e ".[docs]"   # build docs locally
pip install -e ".[dev]"    # pytest
```

## Configuration

Copy the example config and edit paths for your machine:

```bash
cp config.yaml.example config.yaml
```

Minimum required fields:

```yaml
paths:
  local_cache: /Volumes/Extreme SSD
  run_registry: db.sqlite3

pipeline:
  save_path: /Volumes/Extreme SSD/pipeline_runs
  evsong_source: /Volumes/Extreme SSD/birds/evsong
```

`config.yaml` is gitignored — each machine keeps its own copy.

## Running the pipeline

```bash
python run_pipeline.py
```

Outputs land under `<save_path>/<bird>/`:

```
pipeline_runs/
└── or18or24/
    ├── stages/
    │   ├── 01_specs/        # Stage A  HDF5 spectrograms
    │   ├── 02_features/     # Stage B  HDF5 flattened features
    │   ├── 03_embeddings/   # Stage C  UMAP embeddings
    │   ├── 04_labels/       # Stage D  cluster labels
    │   └── 05_phenotype/    # Stage E  phenotype data
    └── results/
        ├── master_summary.csv
        ├── phenotype_results.csv
        ├── catalog/         # HTML song catalogs
        └── plots/
```

## Re-entering the pipeline

You can re-run from any stage without reprocessing earlier ones:

```python
from run_pipeline import run_from_embedding, run_from_labelling, run_from_phenotyping

# Re-run from Stage C (UMAP params changed)
run_from_embedding(save_path="...", birds=["or18or24"])

# Re-run from Stage D (clustering metrics changed)
run_from_labelling(
    save_path="...",
    birds=["or18or24"],
    metrics=["silhouette", "dbi"],
    metric_weights={"silhouette": 2.0, "dbi": 1.0},
)

# Re-run only Stage E (phenotype thresholds changed)
run_from_phenotyping(
    save_path="...",
    birds=["or18or24"],
    pheno_cfg={"min_syllable_proportion": 0.03},
)
```

## API

The pipeline stages are available as individual Python functions.

### Stage A — Spectrogram ingestion

```python
from song_phenotyping.ingestion import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds

meta, audio = filepaths_from_evsonganaly(wav_directory="/data/raw/evsonganaly", bird_subset=["or18or24"])
save_specs_for_evsonganaly_birds(metadata_file_paths=meta, audio_file_paths=audio, save_path="/data/pipeline_runs")
```

wseg format is also supported via `filepaths_from_wseg` / `save_specs_for_wseg_birds`.

### Stage B — Flattening

```python
from song_phenotyping.flattening import flatten_bird_spectrograms

flatten_bird_spectrograms(directory="/data/pipeline_runs", bird="or18or24")
```

### Stage C — UMAP embedding

```python
from song_phenotyping.embedding import explore_embedding_parameters_robust

explore_embedding_parameters_robust(
    save_path="/data/pipeline_runs",
    bird="or18or24",
    n_neighbors_list=[10, 30, 50],
    min_dists=[0.1, 0.3],
    metrics=["euclidean", "cosine"],
)
```

### Stage D — Clustering and labelling

```python
from song_phenotyping.labelling import label_bird, DEFAULT_HDBSCAN_GRID

label_bird(
    save_path="/data/pipeline_runs",
    bird="or18or24",
    metrics=["silhouette", "dbi"],
    hdbscan_params=[p.to_dict() for p in DEFAULT_HDBSCAN_GRID],
)
```

### Stage E — Phenotyping

```python
from song_phenotyping.phenotyping import phenotype_bird, PhenotypingConfig

phenotype_bird(
    bird_path="/data/pipeline_runs/or18or24",
    config=PhenotypingConfig(use_top_n_clusterings=5, generate_plots=True),
)
```

### HTML catalogs

```python
from song_phenotyping.catalog import generate_all_catalogs

results = generate_all_catalogs(bird_path="/data/pipeline_runs/or18or24", rank=0)
```

## Documentation

Full documentation (installation, usage guide, and API reference) is in [`docs/`](docs/). To build locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

For machine setup, config reference, and git workflow see [`SETUP.md`](SETUP.md).
