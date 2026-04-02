# Compatibility shim — all symbols now live in song_phenotyping.slicing
from song_phenotyping.slicing import (  # noqa: F401
    slice_syllable_files_from_evsonganaly,
    slice_syllable_files_from_wseg,
    process_slicing_pipeline,
)
