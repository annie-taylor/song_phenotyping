# Compatibility shim — import from the new package location.
# Will be removed once all pipeline modules are migrated to song_phenotyping/.
from song_phenotyping.flattening import (  # noqa: F401
    flatten_bird_spectrograms,
    flatten_spectrograms,
    load_syllable_data,
    save_flattened_data,
    find_syllable_files,
    process_single_syllable_file,
    extract_song_id,
    create_flattened_output_path,
)
