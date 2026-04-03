## State as of 2026-03-04 to 2026-03-16

**What was built / changed**

- wseg (WhisperSeg) pipeline support added: `filepaths_from_wseg()` reads `.wav.not.mat` metadata files where `fname` field points to actual audio (may be on Macaw server). Audio resolution distinguishes server vs local cache.
- Audio file reading from Macaw server debugged (`d481ed5`); batch file path reconstruction fixed (`ec1c6b1`).
- Embedding pipeline updated for new UMAP structure (`a4aa49c`, `17a6718`): labelling function updated to accommodate new embedding format.
- Memory management bugs in embedding fixed (`7c952d5`); embedding parallelization made adaptive based on available workers (`16bd087`).
- Parallelized with `versa` library (`e3a5a54`).
- xfoster cohort filepath search reorganized (`54a0877`): `filepaths_from_evsonganaly()` extended to handle xfoster directory layout.
- OS-independent path handling added for screening directories (`a137c04`).
- cbin (Canary Bird INterface) WAV format support added (`a16cbb5`, `54558bd`).
- Audio analysis enhancements for xfoster data (`fad32c9` through `a2eafda`); serialization bug fixed (`4bf472f`).
- Manual label debugging (`bb3dceb`); automated phenotyping debugged (`301acfb`).
- Logging updated (`4d8e65e`); deduplicated data saved (`8b12011`).

**Key decisions made**

- **wseg metadata via `fname` field**: The `.wav.not.mat` file is the segmentation metadata; the audio path it references may differ from the metadata file location (important for Macaw server data).
- **`prefer_local=True` default**: Audio is resolved from local cache before Macaw server, reducing network load.
- **Adaptive parallelization**: Number of UMAP workers scales with available memory/CPUs rather than being hardcoded.

**Known issues at this point**

- Song catalog not yet built.
- Song age analysis not yet present.
- Still flat scripts; no package; no tests.
- Path handling partially OS-independent but still fragile.

**Next steps (as of this date)**

- Build song catalog HTML/PDF.
- Add song age analysis.
- Convert notebooks to scripts.
