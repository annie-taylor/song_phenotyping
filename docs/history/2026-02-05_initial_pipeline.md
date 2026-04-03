## State as of 2026-02-05 to 2026-02-10

**What was built / changed**

- First commit: flat-file scripts `A_spec_saving.py`, `B_flattening.py`, `C_embedding.py`, `D_labelling.py`, `E_phenotyping.py` at project root alongside `tools/` utilities.
- UMAP embedding module added (`12fc4f8`); functional embeddings achieved (`0a118ba`).
- Cluster penalty explored then removed (`385bd14`); multi-bird testing began (`ee4a750`).
- Path bug in spec saving fixed (`bf95cae`).
- Phenotyping script revised into working state (`f2fd1cb`); `gitignore` updated; pipeline described as "functional" (`a8c1f5f`).

**Key decisions made**

- **Pipeline as flat Python scripts A→E**: Simple sequential structure; each script reads the previous stage's output directory. No packaging yet.
- **HDF5 (PyTables) for all intermediate data**: Syllable spectrograms, flattened features, embeddings, and labels all stored as HDF5 files rather than numpy arrays or pickles. Enables partial reads on large datasets.
- **HDBSCAN for clustering**: Density-based; no need to specify number of clusters in advance. Cluster penalty explored to penalize over-segmentation but removed as unhelpful.

**Known issues at this point**

- Scripts referenced each other via local imports; circular import issues present.
- All paths hardcoded as strings inside scripts.
- No tests.

**Next steps (as of this date)**

- Add syllable database and analysis tools.
- Build visualization and PDF reporting.
