import os
from sys import platform
from random import sample
import gc
from tqdm import tqdm

# Import consolidated functions from song_io
from tools.song_io import (
    setup_logging,
    get_memory_usage,
    parse_audio_filename,
    load_and_validate_metadata,
    create_output_paths,
    save_spec_slices,  # This is the main function you'll call
    logger  # Use the centralized logger
)
from tools.spectrogram_configs import SpectrogramParams
from tools.system_utils import check_sys_for_macaw_root
from A_spec_saving import filepaths_from_evsonganaly, filepaths_from_wseg, select_new_file_pairs


def slice_syllable_files_from_evsonganaly(wav_directory: str = None, save_path: str = None,
                                          batch_file_naming: str = 'batch.txt.labeled',
                                          params: 'SpectrogramParams' = None, verbose: bool = False,
                                          slice_length: float = None,
                                          overwrite: bool = False, songs_per_bird: int = 20,
                                          bird_subset: list = None, copy_locally: bool = False,
                                          prefer_local: bool = True, preferred_subdirs: list = None):
    """
    Updated to use efficient file discovery from the first module
    """
    if params is None:
        params = SpectrogramParams()

    # Set configuration from arguments
    if slice_length is not None:
        params.slice_length = slice_length
    params.songs_per_bird = songs_per_bird
    params.overwrite_existing = overwrite

    logger.info(f"🔍 Starting evsonganaly slicing with slice_length={slice_length}ms")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    # Use the efficient file discovery from first module
    metadata_file_paths, audio_file_paths = filepaths_from_evsonganaly(
        wav_directory=wav_directory,
        save_path=save_path,
        batch_file_naming=batch_file_naming,
        bird_subset=bird_subset,
        copy_locally=copy_locally,
        prefer_local=prefer_local,
        preferred_subdirs=preferred_subdirs
    )

    birds = list(metadata_file_paths.keys())
    logger.info(f"🐦 Found {len(birds)} birds to process")

    successful_birds = []
    failed_birds = []

    # Process each bird using consolidated utilities
    for bird_idx, bird in enumerate(birds, 1):
        logger.info(f"🔄 Processing bird {bird_idx}/{len(birds)}: {bird}")
        logger.info(f"📊 Memory usage before {bird}: {get_memory_usage():.1f} MB")

        try:
            # Use consolidated path creation
            paths = create_output_paths(save_path, bird)
            slices_dir = paths['slice_specs_dir']  # Use slices_dir instead of syllables_dir
            already_saved_files = os.listdir(slices_dir) if os.path.isdir(slices_dir) else []

            needed_count = params.songs_per_bird - len(already_saved_files)
            logger.info(f"  📁 Found {len(already_saved_files)} existing files, need {needed_count} more")

            if needed_count <= 0:
                logger.info(f'  ⏭️ {bird} already processed, skipping...')
                successful_birds.append(bird)
                continue

            # Use the efficient file selection from your first module
            logger.info(f"  🔍 Selecting files from {len(metadata_file_paths[bird])} available files")
            candidate_file_pairs = select_new_file_pairs(
                metadata_file_paths[bird],
                audio_file_paths[bird],  # Now we have this!
                already_saved_files,
                needed_count
            )

            if candidate_file_pairs:  # Note: renamed from candidate_file_paths
                logger.info(f"  🎵 Processing {len(candidate_file_pairs)} files for {bird}")

                # Process files using the consolidated function
                save_spec_slices(
                    candidate_files=candidate_file_pairs,
                    save_path=save_path,
                    params=params,
                    slice_length=params.slice_length,
                    verbose=verbose,
                    read_songpath_from_metadata=False,
                    prefer_local=prefer_local
                )

                logger.info(f"  ✅ {bird} slicing complete")
                successful_birds.append(bird)
            else:
                logger.warning(f"  ❌ No new files found for {bird}")
                failed_birds.append(bird)

        except Exception as e:
            logger.error(f"  💥 Error processing bird {bird}: {e}")
            failed_birds.append(bird)

        logger.info(f"📊 Memory usage after {bird}: {get_memory_usage():.1f} MB")
        gc.collect()  # Force cleanup between birds

    # Final summary
    logger.info(f"🎯 Evsonganaly slicing complete! Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
    if failed_birds:
        logger.warning(f"❌ Failed birds: {failed_birds}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")


def slice_syllable_files_from_wseg(seg_directory: str = None, save_path: str = None,
                                   file_ext: str = '.wav.not.mat',
                                   params: 'SpectrogramParams' = None, verbose: bool = False,
                                   songs_per_bird: int = 20, bird_subset: list = None,
                                   song_or_call: str = 'song', copy_locally: bool = False,
                                   prefer_local: bool = True):
    """
    Updated to use efficient file discovery from the first module
    """
    song_or_call = song_or_call.lower()
    if song_or_call not in ['song', 'call']:
        logger.warning('🔊 Song/call argument not supported... assuming song.')
        song_or_call = 'song'

    assert seg_directory is not None, "seg_directory cannot be None"
    assert save_path is not None, "save_path cannot be None"

    if params is None:
        params = SpectrogramParams()

    # Set configuration using consolidated approach
    if not hasattr(params, 'slice_length') or params.slice_length is None:
        logger.warning("⚠️ No slice_length specified in params, using default 50ms")
        params.slice_length = 50

    params.songs_per_bird = songs_per_bird
    # Update max_dur to match slice_length
    params.max_dur = params.slice_length / 1000

    logger.info(f"🔍 Starting wseg slicing with slice_length={params.slice_length}ms")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    # Use the efficient file discovery from your first module
    metadata_file_paths, audio_file_paths = filepaths_from_wseg(
        seg_directory=seg_directory,
        save_path=save_path,
        song_or_call=song_or_call,
        file_ext=file_ext,
        bird_subset=bird_subset,
        copy_locally=copy_locally,
        prefer_local=prefer_local
    )

    birds = list(metadata_file_paths.keys())
    logger.info(f"🐦 Found {len(birds)} birds to process")

    successful_birds = []
    failed_birds = []

    # Process each bird using consolidated utilities
    for bird_idx, bird in enumerate(birds, 1):
        logger.info(f"🔄 Processing wseg bird {bird_idx}/{len(birds)}: {bird}")
        logger.info(f"📊 Memory usage before {bird}: {get_memory_usage():.1f} MB")

        try:
            # Use consolidated path creation
            paths = create_output_paths(save_path, bird)
            slices_dir = paths['slice_specs_dir']  # Use slices_dir instead of syllables_dir
            already_saved_files = os.listdir(slices_dir) if os.path.isdir(slices_dir) else []

            needed_count = params.songs_per_bird - len(already_saved_files)
            logger.info(f"  📁 Found {len(already_saved_files)} existing files, need {needed_count} more")

            if needed_count <= 0:
                logger.info(f'  ⏭️ {bird} already processed, skipping...')
                successful_birds.append(bird)
                continue

            # Use the efficient file selection from your first module
            logger.info(f"  🔍 Selecting files from {len(metadata_file_paths[bird])} available files")
            candidate_file_pairs = select_new_file_pairs(
                metadata_file_paths[bird],
                audio_file_paths[bird],
                already_saved_files,
                needed_count
            )

            if candidate_file_pairs:
                logger.info(f"  🎵 Processing {len(candidate_file_pairs)} wseg files for {bird}")

                # Process files using the updated consolidated function
                save_spec_slices(
                    candidate_files=candidate_file_pairs,  # Now passing file pairs
                    save_path=save_path,
                    params=params,
                    slice_length=params.slice_length,
                    verbose=verbose,
                    read_songpath_from_metadata=True,  # wseg reads from metadata
                    prefer_local=prefer_local  # Add this parameter
                )

                logger.info(f"  ✅ {bird} slicing complete")
                successful_birds.append(bird)
            else:
                logger.warning(f"  ❌ No new files found for {bird}")
                failed_birds.append(bird)

        except Exception as e:
            logger.error(f"  💥 Error processing wseg bird {bird}: {e}")
            failed_birds.append(bird)

        logger.info(f"📊 Memory usage after {bird}: {get_memory_usage():.1f} MB")
        gc.collect()  # Force cleanup between birds

    # Final summary
    logger.info(f"🎯 Wseg slicing complete! Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
    if failed_birds:
        logger.warning(f"❌ Failed birds: {failed_birds}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")


def process_slicing_pipeline(pipeline_name: str, settings: dict):
    """Process a single slicing pipeline (evsonganaly or wseg)."""
    logger.info("=" * 60)
    logger.info(f"✂️ {pipeline_name.upper()} SLICING")
    logger.info("=" * 60)

    try:
        if pipeline_name == 'evsonganaly':
            slice_syllable_files_from_evsonganaly(
                wav_directory=settings['source_dir'],
                save_path=settings['save_dir'],
                batch_file_naming=settings['batch_file_naming'],
                params=settings['params'],
                verbose=settings.get('verbose', False),
                slice_length=settings['params'].slice_length,  # Read from params
                overwrite=settings['params'].overwrite_existing,
                songs_per_bird=settings['params'].songs_per_bird,
                bird_subset=settings['bird_subset'],
                copy_locally=settings['copy_locally'],
                prefer_local=settings.get('prefer_local', True),  # Add prefer_local
                preferred_subdirs=settings.get('preferred_subdirs', None)  # Add preferred_subdirs
            )

        elif pipeline_name == 'wseg':
            slice_syllable_files_from_wseg(
                seg_directory=settings['source_dir'],
                save_path=settings['save_dir'],
                file_ext=settings.get('file_ext', '.wav.not.mat'),
                params=settings['params'],  # slice_length now comes from params
                verbose=settings.get('verbose', False),
                songs_per_bird=settings['params'].songs_per_bird,
                bird_subset=settings['bird_subset'],
                song_or_call=settings.get('song_or_call', 'song'),
                copy_locally=settings['copy_locally'],
                prefer_local=settings.get('prefer_local', True)  # Add prefer_local
            )

        logger.info(f"✅ {pipeline_name.capitalize()} slicing complete!")

    except Exception as e:
        logger.error(f"💥 Error in {pipeline_name} slicing: {e}")
        raise


def main():
    """Main slicing pipeline with configurable parameters."""
    import time
    start_time = time.time()
    logger.info("🚀 Starting slice processing pipeline")

    # Setup
    path_to_macaw = check_sys_for_macaw_root()

    config = {
        'evsonganaly': {
            'enabled': True,
            'source_dir': os.path.join(path_to_macaw, 'ssharma', 'RNA_seq', 'family_analysis_labeled', 'or-or'),
            'save_dir': os.path.join('/Volumes', 'Extreme SSD', 'evsong slice test'),
            'batch_file_naming': 'batch.txt.labeled',
            'bird_subset': ['or18or24'],
            'copy_locally': True,
            'prefer_local': False,  # Add prefer_local setting
            'preferred_subdirs': ['labeled_song_final'],  # Add preferred_subdirs
            'verbose': False,
            'params': SpectrogramParams(
                nfft=1024,
                hop=1,
                max_dur=0.050,  # 50ms slices
                slice_length=50,  # 50ms slices
                songs_per_bird=2,
                overwrite_existing=True
            )
        },
        'wseg': {
            'enabled': True,
            'source_dir': os.path.join(path_to_macaw, 'annietaylor', 'bubu-rdyw', 'metadata'),
            'save_dir': os.path.join('/Volumes', 'Extreme SSD', 'wseg slice test'),
            'bird_subset': ['bu10wh86'],
            'copy_locally': True,
            'prefer_local': False,  # Add prefer_local setting
            'verbose': False,
            'file_ext': '.wav.not.mat',
            'song_or_call': 'song',
            'params': SpectrogramParams(
                nfft=1024,
                hop=1,
                max_dur=0.050,  # 50ms slices
                slice_length=50,  # 50ms slices
                songs_per_bird=2,
                overwrite_existing=True
            )
        }
    }

    # Process each pipeline
    for pipeline_name, settings in config.items():
        if not settings['enabled']:
            logger.info(f"⏭️ Skipping {pipeline_name.upper()} slicing (disabled)")
            continue

        process_slicing_pipeline(pipeline_name, settings)

    total_time = time.time() - start_time
    logger.info(f"🎯 All slicing complete! Total time: {total_time:.1f} seconds")


# Updated main section for slicing module
if __name__ == "__main__":
    # Setup logging using consolidated function
    logger = setup_logging()

    logger.info("🚀 Starting slice processing pipeline")
    main()