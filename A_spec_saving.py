# Compatibility shim — import from the new package location.
# Will be removed once all pipeline modules are migrated to song_phenotyping/.
from song_phenotyping.ingestion import (  # noqa: F401
    filepaths_from_evsonganaly,
    filepaths_from_wseg,
    save_specs_for_evsonganaly_birds,
    save_specs_for_wseg_birds,
    copy_audio_and_partner_rec,
    select_new_file_pairs,
    select_new_files,
    standardize_bird_band,
    resolve_audio_file_path,
    reconstruct_server_path,
    create_segmented_audio_data,
    create_empty_segmented_data,
    process_and_save_audio,
    filepaths_from_local_cache,
    select_wseg_file_pairs_from_metadata,
    process_single_file,
    save_data_specs,
    process_pipeline,
)
