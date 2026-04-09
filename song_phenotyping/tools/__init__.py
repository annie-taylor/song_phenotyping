"""
song_phenotyping.tools
======================

Shared utilities used across pipeline stages.

Modules
-------
:mod:`~song_phenotyping.tools.label_handler`
    Unified label type handling for manual and HDBSCAN labels.
:mod:`~song_phenotyping.tools.spectrogram_configs`
    Parameter dataclasses for spectrogram computation.
:mod:`~song_phenotyping.tools.run_config`
    RunConfig and RunRegistry for experiment provenance tracking.
:mod:`~song_phenotyping.tools.project_config`
    Machine-local path configuration loaded from ``config.yaml``.
:mod:`~song_phenotyping.tools.evfuncs`
    Python implementations of EvTAF/evsonganaly functions (readrecf, load_cbin).
:mod:`~song_phenotyping.tools.signal_utils`
    Signal processing utilities (bandpass filter, smoothing, RMS norm).
:mod:`~song_phenotyping.tools.system_utils`
    OS/platform utilities and PyTables network-tuning helpers.
:mod:`~song_phenotyping.tools.logging_utils`
    UTF-8-safe logger setup with cross-platform emoji handling.
:mod:`~song_phenotyping.tools.audio_path_management`
    Local/server audio path resolution via ``audio_paths.txt``.
:mod:`~song_phenotyping.tools.audio_utils`
    Audio file reading (wav, cbin) and spectrogram downsampling helpers.
:mod:`~song_phenotyping.tools.filerecords`
    FileRecord dataclass and helpers for building per-bird file records.
:mod:`~song_phenotyping.tools.dbquery`
    SQLite database queries for bird band/genetic/expression data.
:mod:`~song_phenotyping.tools.bird_name`
    Bird name normalization: expand color abbreviations to canonical 2-letter codes.
"""
