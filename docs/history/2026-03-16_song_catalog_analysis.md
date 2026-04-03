## State as of 2026-03-16 to 2026-03-28

**What was built / changed**

- Song catalog largely completed (`178aac7`, `8b9660a`): HTML + PDF catalog of songs with spectrograms and cluster labels, written to `syllable_data/html/`. Generates per-song continuous spectrogram views with manual labels above and auto labels below.
- Remaining pipeline items run on new wseg data (`c65108c`, `af481e9`).
- Song age analysis added (`1e37e39`, `871bd20`): measures acoustic changes over developmental time.
- xfoster cohort search code reorganized (`54a0877`): cleaner separation of file discovery from pipeline dispatch.
- Notebooks converted to scripts (`db45d68`): analysis workflows moved out of Jupyter into standalone `.py` files.
- `gen_path` added to CSV summaries (`95ee382`).
- cbin → WAV conversion utility added (`54558bd`): converts `.cbin` recordings to WAV before spectrogram extraction.
- Windows-friendly YAML path handling started (`f4fa18e`).

**Key decisions made**

- **Song catalog as HTML + PDF**: HTML for quick browser review; PDF for sharing. Both generated from the same spectrogram data.
- **Manual labels above, auto labels below** in song catalog: Preserves ground-truth context when comparing automated clustering to hand labels.

**Known issues at this point**

- Output directory names (`syllable_data/specs/`, `syllable_data/labelling/`, etc.) are opaque and inconsistently named across modules.
- All code still as flat root-level scripts; no installable package.
- No automated tests; path handling still partly hardcoded.
- Cannot easily process multiple birds in sequence (no cohort mode).
- Mac paths hardcoded; PC requires manual edits.

**Next steps (as of this date)**

- Package the code properly (Steps 0-9 migration).
- Add cross-platform config system.
- Add cohort mode.
- Restructure output directories.
