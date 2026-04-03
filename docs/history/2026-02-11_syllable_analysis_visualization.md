## State as of 2026-02-11 to 2026-02-13

**What was built / changed**

- Syllable database module incorporated (`ec4a712`): builds a per-syllable feature CSV alongside the HDF5 files, enabling spreadsheet-level inspection of clustering results.
- Syllable feature analysis working end-to-end (`2c708e5`): computes acoustic features per cluster type.
- PDF report generation revised iteratively (`4a16fd0` through `4ca208b`): cluster summary PDFs using reportlab and matplotlib.
- `interactive_umap.py` completed (`2cbf3aa`): Dash app for exploring UMAP embeddings interactively in browser.
- New visualization modules added (`9352907`): song visualization tools; old `song_visualization.py` removed.
- Slices (fixed-window spectrogram segments) added alongside syllable-based spectrograms (`8996f8f`).

**Key decisions made**

- **reportlab for PDF reports**: Chosen for precise layout control over cluster grids and summary tables. Used in both cluster summary PDFs and song catalog PDFs.
- **Dash for interactive UMAP explorer**: Allows brushing/selecting clusters in the embedding and seeing the corresponding spectrograms without running a full Jupyter session.
- **Syllable database as a CSV sidecar**: Human-readable complement to the HDF5 label files; enables quick filtering and inspection without loading HDF5.

**Known issues at this point**

- PDF generation is slow and always runs even when not needed.
- Paths still hardcoded; no config file.
- All code still as flat scripts; no package structure.

**Next steps (as of this date)**

- Extend to wseg (WhisperSeg) segmentation format.
- Add support for xfoster cohort data.
- Improve path handling for cross-OS use.
