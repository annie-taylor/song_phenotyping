## State as of 2026-04-01

**What was built / changed**

- `pyproject.toml` and `song_phenotyping/` package skeleton created (`11446ca`, Step 0): `pip install -e .` now works; Sphinx docs scaffold added.
- `tools/` migrated to `song_phenotyping/tools/` with scipy-style docstrings (`cf01e41`, Step 1).
- `A_spec_saving.py` → `song_phenotyping/ingestion.py` (`b4eb857`, Step 2).
- `B_flattening.py` → `song_phenotyping/flattening.py` (`ee2b07a`, Step 3).
- `C_embedding.py` → `song_phenotyping/embedding.py` (`cfbaa41`, Step 4).
- `D_labelling.py` → `song_phenotyping/labelling.py` (`c37e937`, Step 5).
- `E_phenotyping.py` → `song_phenotyping/phenotyping.py` (`6fbbd08`, Step 6).
- Catalog and slicing modules migrated into package (`2586f13`, Step 7).
- `run_pipeline.py` and `tests/test_smoke.py` updated to import from package (`41a23d5`, Step 8).
- Shims (compatibility re-exports from old names) deleted; usage guide added; Sphinx build cleaned (`731c542`, Step 9).
- Pre-migration reorganization: tools, config stub, smoke test skeleton added (`3d059d8`).

**Key decisions made**

- **Scipy-style docstrings throughout**: Consistent with scientific Python conventions; supports Sphinx autodoc.
- **Shims for migration, then deleted**: Temporary `A_spec_saving.py` stubs re-exporting from package allowed incremental migration without breaking in-progress runs. Deleted once migration confirmed working.
- **`pip install -e .`**: Editable install means any script or notebook can `import song_phenotyping` without path manipulation.
- **`run_pipeline.py` stays at project root**: Not inside the package — it's the user-facing entry point, not a library module.

**Known issues at this point**

- Output directory names still use old `syllable_data/specs/` etc. scheme.
- All paths still hardcoded in `run_pipeline.py`; no config file yet.
- No cohort mode (one bird at a time).
- Mac-only paths; PC requires manual edits.

**Next steps (as of this date)**

- Add `config.yaml` system for cross-machine paths.
- Add cohort mode.
- Restructure output directories to `stages/` + `results/`.
- Add flat-directory fallback for evsonganaly data without batch files.
