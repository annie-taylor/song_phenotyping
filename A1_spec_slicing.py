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


def slice_syllable_files_from_evsonganaly(wav_directory: str = None, save_path: str = None,
                                          file_ext: str = '.wav.not.mat',
                                          params: 'SpectrogramParams' = None, verbose: bool = False,
                                          slice_mode: bool = False, slice_length: float = None,
                                          overwrite: bool = False, songs_per_bird: int = 20,
                                          read_from_batch_file: bool = False, birds_keep: list = None):
    """
    Updated to use consolidated utilities from song_io
    """
    if params is None:
        params = SpectrogramParams()

    # Set configuration from arguments
    if slice_length is not None:
        params.slice_length = slice_length
    params.songs_per_bird = songs_per_bird
    params.overwrite_existing = overwrite

    logger.info(f"🔍 Starting evsonganaly processing with slice_length={slice_length}ms")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    birds = []
    metadata_file_paths = {}

    # File discovery logic (keep your existing logic)
    if read_from_batch_file:
        for root, dirs, files in os.walk(wav_directory, topdown=False):
            for file in files:
                if 'batch.txt.labeled' in file and 'labeled_songs_final' in root:
                    songs = []
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            song_name = line.replace('\n', '.not.mat')
                            songs.append(os.path.join(root, song_name))

                    try:
                        [bird, _, _] = line.split('_')[0:3]
                    except ValueError:
                        try:
                            [bird, _] = line.split('_')[0:2]
                        except ValueError:
                            [bird, _] = line.split('.')[0:2]

                    bird_folder = os.path.join(save_path, bird)
                    if not os.path.isdir(bird_folder):
                        os.mkdir(bird_folder)

                    if bird not in birds:
                        birds.append(bird)
                        metadata_file_paths[bird] = songs
                    else:
                        for song in songs:
                            metadata_file_paths[bird].append(song)

    if birds_keep is not None:
        birds = [bird for bird in birds if bird in birds_keep]

    # Process each bird using consolidated utilities
    for bird in birds:
        logger.info(f"🐦 Processing bird: {bird}")

        # Use consolidated path creation
        paths = create_output_paths(save_path, bird)
        slices_dir = paths['slices_dir']  # Use slices_dir instead of syllables_dir
        already_saved_files = os.listdir(slices_dir) if os.path.isdir(slices_dir) else []

        if len(already_saved_files) < params.songs_per_bird:
            needed_files = params.songs_per_bird - len(already_saved_files)

            if len(metadata_file_paths[bird]) > needed_files:
                candidate_file_paths = sample(metadata_file_paths[bird], k=needed_files)
            else:
                candidate_file_paths = metadata_file_paths[bird]

            # Remove already processed files using consolidated filename parsing
            remaining_candidates = []
            for path in candidate_file_paths:
                file_info = parse_audio_filename(path)
                if file_info['success']:
                    # Check if this file was already processed
                    expected_output = f"syllables_{file_info['bird']}_{file_info['day']}_{file_info['time']}.h5"
                    if expected_output not in already_saved_files:
                        remaining_candidates.append(path)

            # Process files using the consolidated function
            if remaining_candidates:
                logger.info(f"🎵 Processing {len(remaining_candidates)} files for {bird}")
                save_spec_slices(
                    metadata_file_paths=remaining_candidates,
                    save_path=save_path,  # save_spec_slices will use create_output_paths internally
                    params=params,
                    slice_length=params.slice_length,
                    verbose=verbose,
                    read_songpath_from_metadata=False
                )
            else:
                logger.info(f'⏭️ {bird} already processed, skipping...')


def slice_syllable_files_from_wseg(seg_directory: str = None, save_path: str = None,
                                   file_ext: str = '.wav.not.mat',
                                   params: 'SpectrogramParams' = None, verbose: bool = False,
                                   songs_per_bird: int = 20, bird_subset: list = None,
                                   song_or_call: str = 'song'):
    """
    Updated to use consolidated utilities from song_io
    """
    song_or_call = song_or_call.lower()
    if song_or_call not in ['song', 'call']:
        logger.warning('🔊 Song/call argument not supported... assuming song.')
        song_or_call = 'song'

    slice_length = 50  # in milliseconds

    assert seg_directory is not None, "seg_directory cannot be None"
    assert save_path is not None, "save_path cannot be None"

    if params is None:
        params = SpectrogramParams(max_dur=(slice_length / 1000))

    # Set configuration using consolidated approach
    params.slice_length = slice_length
    params.songs_per_bird = songs_per_bird

    logger.info(f"🔍 Starting wseg processing with slice_length={slice_length}ms")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    song_file_paths = {}
    birds = []

    # Walk through the directory structure (keep existing logic)
    for root, dirs, files in os.walk(seg_directory, topdown=False):
        for file in files:
            if file.endswith(file_ext) and (song_or_call in root):
                path_to_file = os.path.join(root, file)

                # Use consolidated filename parsing
                file_info = parse_audio_filename(path_to_file)
                if not file_info['success']:
                    logger.warning(f"⚠️ Could not parse filename: {file}")
                    continue

                bird = file_info['bird']

                if (bird_subset is None) or (bird in bird_subset):
                    # Use consolidated path creation
                    paths = create_output_paths(save_path, bird)

                    if bird not in birds:
                        birds.append(bird)
                        song_file_paths[bird] = [path_to_file]
                    else:
                        song_file_paths[bird].append(path_to_file)

    # Process each bird using consolidated utilities
    for bird in birds:
        logger.info(f"🐦 Processing wseg bird: {bird}")

        # Use consolidated path creation
        paths = create_output_paths(save_path, bird)
        slices_dir = paths['slices_dir']  # Use slices_dir instead of syllables_dir
        already_saved_files = os.listdir(slices_dir) if os.path.isdir(slices_dir) else []

        if len(already_saved_files) < params.songs_per_bird:
            needed_files = params.songs_per_bird - len(already_saved_files)

            if len(song_file_paths[bird]) > needed_files:
                metadata_file_paths = sample(song_file_paths[bird], k=needed_files)
            else:
                metadata_file_paths = song_file_paths[bird]

            # Remove already processed files using consolidated filename parsing
            remaining_candidates = []
            for path in metadata_file_paths:
                file_info = parse_audio_filename(path)
                if file_info['success']:
                    # Check if this file was already processed - UPDATE FILENAME PREFIX
                    expected_output = f"slices_{file_info['bird']}_{file_info['day']}_{file_info['time']}.h5"
                    if expected_output not in already_saved_files:
                        remaining_candidates.append(path)

            # Process files using the consolidated function
            if remaining_candidates:
                logger.info(f"🎵 Processing {len(remaining_candidates)} wseg files for {bird}")
                save_spec_slices(
                    metadata_file_paths=remaining_candidates,
                    save_path=save_path,  # save_spec_slices will use create_output_paths internally
                    params=params,
                    slice_length=slice_length,
                    verbose=verbose,
                    read_songpath_from_metadata=True
                )
            else:
                logger.info(f'⏭️ No new files found for {bird}')
        else:
            logger.info(f'⏭️ {bird} already processed, skipping...')


# Updated main section for slicing module
if __name__ == "__main__":
    # Setup logging using consolidated function
    logger = setup_logging()

    from tools.system_utils import check_sys_for_macaw_root

    path_to_macaw = check_sys_for_macaw_root()

    evsong_save_directory = os.path.join('/Volumes', 'Extreme SSD', 'evsong test')
    evsong_directory = os.path.join(evsong_save_directory, 'evsong test', 'copied data')

    wseg_save_directory = os.path.join('/Volumes', 'Extreme SSD', 'wseg test')
    wseg_directory = os.path.join(path_to_macaw, 'annietaylor', 'bubu-rdyw', 'metadata')

    songs_per_bird = 5
    bird_subset = ['bu10wh86']

    logger.info("🚀 Starting slice processing pipeline")

    slice_syllable_files_from_wseg(
        seg_directory=wseg_directory,
        file_ext='.wav.not.mat',
        save_path=wseg_save_directory,
        verbose=False,
        songs_per_bird=songs_per_bird,
        song_or_call='song',
        bird_subset=bird_subset
    )

    slice_syllable_files_from_evsonganaly(wav_directory=evsong_directory,
                                          save_path=evsong_save_directory)

    logger.info("✅ Slice processing complete!")
