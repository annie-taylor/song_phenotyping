import os
import logging
import re
from sys import platform
import time
from scipy.io import loadmat
import numpy as np
import hashlib
from typing import Any, Dict, List
from random import sample
import gc
import shutil
from tqdm import tqdm

# Import consolidated functions from song_io
from tools.song_io import (
    setup_logging,
    get_memory_usage,
    parse_audio_filename,
    load_and_validate_metadata,
    pad_waveforms_to_same_length,
    generate_syllable_hashes,
    split_long_syllables_with_mapping,
    create_output_paths,
    save_segmented_audio_data,
    get_song_specs,
    logger  # Use the centralized logger
)
from tools.system_utils import check_sys_for_macaw_root, optimize_pytables_for_network
from tools.spectrogram_configs import SpectrogramParams
from tools.audio_path_management import *


def filepaths_from_wseg(seg_directory: str, save_path: str = None,
                        song_or_call: str = 'song',
                        file_ext: str = '.wav.not.mat',
                        bird_subset: None | list = None,
                        copy_locally: bool = False) -> Dict[str, List[str]]:
    """
    Extract WhisperSeg metadata file paths organized by bird and optionally copy audio files.
    """
    logger.info(f"🔍 Scanning wseg directory: {seg_directory}")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    metadata_file_paths = {}
    processed_files = 0
    failed_files = 0

    for root, dirs, files in os.walk(seg_directory):
        if song_or_call.lower() not in root.lower():
            continue

        matching_files = [f for f in files if f.endswith(file_ext)]
        if matching_files:
            logger.info(f"  📁 Found {len(matching_files)} files in {root}")

        for file in matching_files:
            try:
                file_path = os.path.join(root, file)
                path_parts = root.split(os.sep)
                bird = path_parts[-2]

                # Apply bird subset filter early
                if bird_subset is not None and bird not in bird_subset:
                    continue

                if bird not in metadata_file_paths:
                    metadata_file_paths[bird] = []
                    logger.info(f"  🐦 Started processing bird: {bird}")

                metadata_file_paths[bird].append(file_path)

                # Handle copying and mapping for wseg files
                if save_path and copy_locally:
                    bird_folder = os.path.join(save_path, bird)
                    os.makedirs(bird_folder, exist_ok=True)

                    # For wseg, we need to load metadata to get audio path
                    try:
                        metadata_matfile = loadmat(file_path, squeeze_me=True)
                        audio_path, _ = resolve_audio_file_path(
                            file_path, metadata_matfile, read_songpath_from_metadata=True
                        )

                        if audio_path and os.path.exists(audio_path):
                            filename = os.path.basename(audio_path)

                            # Copy file locally
                            copied_data_dir = os.path.join(save_path, 'copied_data', bird)
                            os.makedirs(copied_data_dir, exist_ok=True)
                            local_path = os.path.join(copied_data_dir, filename)

                            if not os.path.exists(local_path):
                                try:
                                    shutil.copy2(audio_path, local_path)
                                    logger.debug(f"  📋 Copied {filename}")
                                except Exception as e:
                                    logger.error(f"  ❌ Failed to copy {audio_path}: {e}")
                                    local_path = None

                            # Update mapping with both paths
                            update_audio_paths_file(bird_folder, filename,
                                                    local_path=local_path,
                                                    server_path=audio_path)
                        else:
                            logger.warning(f"  ⚠️ Could not resolve audio path for {file_path}")

                        # Clean up metadata after processing
                        del metadata_matfile

                    except Exception as e:
                        logger.error(f"  💥 Failed to process wseg metadata {file_path}: {e}")
                        failed_files += 1

                elif save_path:
                    # Just create bird folders without copying
                    bird_folder = os.path.join(save_path, bird)
                    os.makedirs(bird_folder, exist_ok=True)

                processed_files += 1

                # Periodic progress updates for large datasets
                if processed_files % 100 == 0:
                    logger.info(f"  📊 Processed {processed_files} files, memory: {get_memory_usage():.1f} MB")
                    gc.collect()  # Periodic cleanup

            except Exception as e:
                logger.error(f"  💥 Error processing file {file}: {e}")
                failed_files += 1

    # Final summary
    total_birds = len(metadata_file_paths)
    total_files = sum(len(files) for files in metadata_file_paths.values())

    logger.info(f"🎯 Wseg scanning complete:")
    logger.info(f"  🐦 Found {total_birds} birds")
    logger.info(f"  📄 Found {total_files} total metadata files")
    logger.info(f"  ✅ Successfully processed: {processed_files}")
    logger.info(f"  ❌ Failed: {failed_files}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")

    for bird, files in metadata_file_paths.items():
        logger.info(f"  {bird}: {len(files)} files")

    return metadata_file_paths


def filepaths_from_evsonganaly(wav_directory: str = None, save_path: str = None,
                               batch_file_naming: str = 'batch.txt.keep',
                               bird_subset: None | list = None,
                               copy_locally: bool = False) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Extract file paths from evsonganaly batch files and optionally copy files locally.
    """
    logger.info(f"🔍 Scanning evsonganaly directory: {wav_directory}")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    birds = []
    metadata_file_paths = {}
    wav_file_paths = {}
    processed_files = 0
    failed_files = 0
    batch_files_found = 0

    for root, dirs, files in os.walk(wav_directory, topdown=False):
        batch_files = [f for f in files if batch_file_naming in f]

        if batch_files:
            batch_files_found += len(batch_files)
            logger.info(f"  📁 Found {len(batch_files)} batch files in {root}")

        for file in batch_files:
            try:
                song_metadata = []
                song_audio = []
                batch_file_path = os.path.join(root, file)

                logger.info(f"  📄 Processing batch file: {file}")

                with open(batch_file_path, 'r') as f:
                    lines = f.readlines()
                    logger.info(f"    📝 Found {len(lines)} entries in batch file")

                    for line_idx, line in enumerate(lines):
                        try:
                            song_name = line.replace('\n', '.not.mat')
                            song_metadata_path = os.path.join(root, song_name)

                            if os.path.isfile(song_metadata_path):
                                audio_path = os.path.join(root, line.replace('\n', ''))
                                song_audio.append(audio_path)
                                song_metadata.append(song_metadata_path)

                                # Extract bird name
                                try:
                                    [bird, _, _] = line.split('_')[0:3]
                                except ValueError:
                                    logger.debug(f"    ⚠️ Trouble reading birdname from {line}, trying fallback")
                                    try:
                                        [bird, _] = line.split('_')[0:2]
                                    except ValueError:
                                        logger.debug(f"    ⚠️ Using final fallback for birdname from {line}")
                                        [bird, _] = line.split('.')[0:2]

                                # Apply bird subset filter
                                if bird_subset is not None and bird not in bird_subset:
                                    continue

                                # Create bird folder and handle copying/mapping
                                if save_path:
                                    bird_folder = os.path.join(save_path, bird)
                                    os.makedirs(bird_folder, exist_ok=True)

                                    filename = os.path.basename(audio_path)

                                    if copy_locally:
                                        # Copy file locally
                                        copied_data_dir = os.path.join(save_path, 'copied_data', bird)
                                        os.makedirs(copied_data_dir, exist_ok=True)
                                        local_path = os.path.join(copied_data_dir, filename)

                                        if not os.path.exists(local_path):
                                            try:
                                                shutil.copy2(audio_path, local_path)
                                                logger.debug(f"      📋 Copied {filename}")
                                            except Exception as e:
                                                logger.error(f"      ❌ Failed to copy {audio_path}: {e}")
                                                local_path = None

                                        # Update mapping with both paths
                                        update_audio_paths_file(bird_folder, filename,
                                                              local_path=local_path,
                                                              server_path=audio_path)
                                    else:
                                        # Just update with server path
                                        update_audio_paths_file(bird_folder, filename,
                                                              server_path=audio_path)

                                # Collect paths by bird
                                if bird not in birds:
                                    birds.append(bird)
                                    metadata_file_paths[bird] = song_metadata.copy()
                                    wav_file_paths[bird] = song_audio.copy()
                                    logger.info(f"    🐦 Started processing bird: {bird}")
                                else:
                                    metadata_file_paths[bird].extend(song_metadata)
                                    wav_file_paths[bird].extend(song_audio)

                                processed_files += 1
                            else:
                                logger.debug(f"    ⚠️ Metadata file not found: {song_metadata_path}")

                        except Exception as e:
                            logger.error(f"    💥 Error processing line {line_idx} in {file}: {e}")
                            failed_files += 1

                        # Periodic progress updates for large batch files
                        if (line_idx + 1) % 50 == 0:
                            logger.info(f"    📊 Processed {line_idx + 1}/{len(lines)} lines, "
                                       f"memory: {get_memory_usage():.1f} MB")

                logger.info(f"  ✅ Completed batch file {file}: {len(song_metadata)} valid entries")

            except Exception as e:
                logger.error(f"  💥 Error processing batch file {file}: {e}")
                failed_files += 1

    # Apply bird subset filter to final results
    if bird_subset is not None:
        metadata_file_paths = {bird: paths for bird, paths in metadata_file_paths.items()
                               if bird in bird_subset}
        wav_file_paths = {bird: paths for bird, paths in wav_file_paths.items()
                          if bird in bird_subset}

    # Final summary
    total_birds = len(metadata_file_paths)
    total_metadata_files = sum(len(files) for files in metadata_file_paths.values())
    total_wav_files = sum(len(files) for files in wav_file_paths.values())

    logger.info(f"🎯 Evsonganaly scanning complete:")
    logger.info(f"  📄 Found {batch_files_found} batch files")
    logger.info(f"  🐦 Found {total_birds} birds")
    logger.info(f"  📄 Found {total_metadata_files} metadata files")
    logger.info(f"  🎵 Found {total_wav_files} wav files")
    logger.info(f"  ✅ Successfully processed: {processed_files}")
    logger.info(f"  ❌ Failed: {failed_files}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")

    for bird in metadata_file_paths:
        logger.info(f"  {bird}: {len(metadata_file_paths[bird])} files")

    return metadata_file_paths, wav_file_paths


def select_new_files(available_files: list[str], already_saved_files: list[str], needed_count: int) -> list[str]:
    """
    Select files that haven't been processed yet.
    """
    if needed_count <= 0:
        return []

    # Get base names of already processed files
    processed_bases = set()
    for saved_file in already_saved_files:
        # Use the consolidated filename parsing
        file_info = parse_audio_filename(saved_file)
        if file_info['success']:
            base_name = f"{file_info['bird']}_{file_info['day']}_{file_info['time']}"
            processed_bases.add(base_name)

    # Find unprocessed files
    unprocessed_files = []
    for file_path in available_files:
        file_info = parse_audio_filename(file_path)
        if file_info['success']:
            base_name = f"{file_info['bird']}_{file_info['day']}_{file_info['time']}"
            if base_name not in processed_bases:
                unprocessed_files.append(file_path)
        else:
            logger.warning(f"Skipping file with unparseable name: {file_path}")

    # Return up to needed_count files
    if len(unprocessed_files) <= needed_count:
        return unprocessed_files
    else:
        return sample(unprocessed_files, needed_count)


def standardize_bird_band(band_string):
    """
    Convert bird band strings from various formats to standardized 'co#co#' format.

    Args:
        band_string (str): Bird band identifier in various formats

    Returns:
        str: Standardized band string in 'co#co#' format, or None if invalid
    """
    if not band_string or not isinstance(band_string, str):
        return None

    # Remove whitespace and convert to lowercase
    band_string = band_string.strip().lower()

    # Color abbreviation mapping (single char -> two char)
    color_map = {
        'b': 'bk',  # black (not blue)
        'w': 'wh',  # white
        'g': 'gr',  # green
        'y': 'ye',  # yellow
        'r': 'rd',  # red
        'o': 'or',  # orange
        # purple, pink, and blue have two char instances and full name instances, but not in original dbquery function
    }

    # Full color name to two-char abbreviation
    full_color_map = {
        'black': 'bk',
        'white': 'wh',
        'green': 'gr',
        'yellow': 'ye',
        'red': 'rd',
        'pink': 'pk',
        'orange': 'or',
        'purple': 'pu',
        'brown': 'br',
        'blue': 'bl',
        'noband': 'nb'
    }

    def normalize_color(color_str):
        """Convert color string to standardized two-character format"""
        if not color_str:
            return ''

        color_str = color_str.lower().strip()

        # If it's already a two-character code, check if it's valid
        if len(color_str) == 2:
            # Check if it's already in the correct format
            valid_two_char = set(color_map.values()) | set(full_color_map.values())
            if color_str in valid_two_char:
                return color_str

        # If it's a single character, map it
        if len(color_str) == 1 and color_str in color_map:
            return color_map[color_str]

        # If it's a full color name, map it
        if color_str in full_color_map:
            return full_color_map[color_str]

        # If it's an abbreviation we don't recognize, try to guess
        # This handles cases like "pk" for pink, "rd" for red, etc.
        for full_name, abbrev in full_color_map.items():
            if color_str == abbrev or full_name.startswith(color_str):
                return abbrev

        # Return as-is if we can't map it (might need manual review)
        return color_str

    # Parse the band string using regex to separate colors and numbers
    # This pattern captures sequences of letters and sequences of digits
    parts = re.findall(r'([a-zA-Z]+)|(\d+)', band_string)

    if not parts:
        return None

    # Extract components
    colors = []
    numbers = []

    for letter_part, digit_part in parts:
        if letter_part:
            colors.append(normalize_color(letter_part))
        if digit_part:
            numbers.append(digit_part)

    # Build standardized string based on what we found
    if len(colors) == 1 and len(numbers) == 1:
        # Single band: co#
        return f"{colors[0]}{numbers[0]}"
    elif len(colors) == 2 and len(numbers) == 2:
        # Two bands: co#co#
        return f"{colors[0]}{numbers[0]}{colors[1]}{numbers[1]}"
    elif len(colors) == 2 and len(numbers) == 1:
        # Two colors, one number - might be co#co format
        return f"{colors[0]}{numbers[0]}{colors[1]}"
    elif len(colors) == 1 and len(numbers) == 2:
        # One color, two numbers - unusual but handle it
        return f"{colors[0]}{numbers[0]}{numbers[1]}"
    else:
        # Return None for unrecognized patterns
        return None


def resolve_audio_file_path(metadata_file_path: str, metadata_matfile: dict,
                            read_songpath_from_metadata: bool,
                            bird_folder: str = None, prefer_local: bool = True) -> tuple[str, float]:
    """
    Resolve the path to the audio file and return offset.

    Args:
        bird_folder: Path to bird folder for audio path mapping (optional)
        prefer_local: If True and bird_folder provided, prefer local files

    Returns:
        tuple: (audio_file_path, wseg_offset) or (None, offset) if file not found
    """
    if not read_songpath_from_metadata:
        # Simple case: same path as metadata but with .wav extension
        audio_file_path = metadata_file_path.replace('.wav.not.mat', '.wav')
        wseg_offset = 0.0

        # Try to use path mapping if available
        if bird_folder:
            try:
                audio_file_path = get_audio_path(bird_folder, audio_file_path, prefer_local)
            except FileNotFoundError:
                # Fall back to original logic
                pass
    else:
        # Complex case: read path from metadata and reconstruct with server root
        try:
            fs = metadata_matfile.get('Fs', 32000.0)
            wseg_offset = (256 * 1000) / fs

            # Try to get filename from metadata (handle both field names)
            fname = None
            for key in ['fname', 'fnamecell']:
                if key in metadata_matfile:
                    fname = metadata_matfile[key]
                    break

            if fname is None:
                raise KeyError("No filename found in metadata (tried 'fname' and 'fnamecell')")

            # Try path mapping first if available
            if bird_folder:
                try:
                    audio_file_path = get_audio_path(bird_folder, fname, prefer_local)
                except FileNotFoundError:
                    # Fall back to server reconstruction
                    audio_file_path = reconstruct_server_path(fname)
            else:
                # Reconstruct path using your server root function
                audio_file_path = reconstruct_server_path(fname)

        except KeyError as e:
            logger.error(f"Could not extract filename from metadata {metadata_file_path}: {e}")
            return None, 0.0

    # Verify file exists
    if not os.path.exists(audio_file_path):
        logger.warning(f"Audio file not found: {audio_file_path}")
        return None, wseg_offset

    return audio_file_path, wseg_offset


def reconstruct_server_path(stored_path: str) -> str:
    """
    Reconstruct full server path from stored relative path using current platform.
    """
    # Get the appropriate server root for current platform
    server_root = check_sys_for_macaw_root()

    if not server_root:
        raise RuntimeError("Could not determine server root for current platform")

    # Clean up the stored path and extract relative part
    # Handle mixed separators in stored path
    clean_path = stored_path.replace('\\', '/').replace('//', '/')

    # Remove any existing server root prefixes and get relative path
    path_parts = [part for part in clean_path.split('/') if part]
    if len(path_parts) > 0:
        # Skip the first part which is usually the old server root
        relative_path = '/'.join(path_parts[1:])

        # Combine with current platform's server root
        if platform == "win32":
            return os.path.join(server_root, relative_path.replace('/', '\\'))
        else:
            return os.path.join(server_root, relative_path)

    return stored_path  # Fallback if parsing fails


def create_segmented_audio_data(specs: List[np.ndarray], wavs: List[np.ndarray],
                                ts: List[np.ndarray], onsets: np.ndarray,
                                offsets: np.ndarray, labels: np.ndarray, mean_top_3: np.float64, low_f_mean: np.float64,
                                mean_all: np.float64,
                                valid_indices: List[int], file_identifier: str) -> Dict[str, Any]:
    """
    Create organized segmented audio data structure from processing results.

    Args:
        specs: List of spectrogram arrays
        wavs: List of waveform arrays
        ts: List of time reference arrays
        onsets: Array of onset times
        offsets: Array of offset times
        labels: Array of syllable labels
        mean_top_3: Mean of the three highest area frequency peaks (representing tempo)
        low_f_mean: Mean of frequency peaks below cutoff of 3Hz (possibly rep. motif structure)
        mean_all: Mean of all frequency peaks, scaled by area
        valid_indices: Indices of successfully processed syllables
        file_identifier: Base string for generating unique hashes

    Returns:
        Dictionary with organized segmented audio data
    """
    if not valid_indices:
        return create_empty_segmented_data()

    # Validate input lengths
    expected_length = len(valid_indices)
    if not (len(specs) == len(wavs) == len(ts) == expected_length):
        raise ValueError(
            f"Inconsistent array lengths: specs={len(specs)}, wavs={len(wavs)}, ts={len(ts)}, valid_indices={expected_length}")

    # Filter arrays to valid indices only
    valid_onsets = [onsets[i] for i in valid_indices]
    valid_offsets = [offsets[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    # Generate unique hashes for each syllable
    hashes = generate_syllable_hashes(file_identifier, valid_indices)

    # Pad waveforms and time arrays to same length
    padded_wavs, padded_ts = pad_waveforms_to_same_length(wavs, ts)

    # Build the data structure
    segmented_data = {
        'spectrograms': specs,
        'waveforms': padded_wavs,
        'spec_t': padded_ts,
        'manual': valid_labels,
        'onsets': valid_onsets,
        'offsets': valid_offsets,
        'mean_top_3': mean_top_3,
        'low_f_mean': low_f_mean,
        'mean_all': mean_all,
        'position_idxs': valid_indices,
        'hashes': hashes
    }

    return segmented_data


def create_empty_segmented_data() -> Dict[str, Any]:
    """Create empty segmented data structure."""
    return {
        'spectrograms': [],
        'waveforms': [],
        'spec_t': [],
        'manual': [],
        'onsets': [],
        'offsets': [],
        'position_idxs': [],
        'hashes': []
    }


def process_and_save_audio(audio_file_path: str, output_path: str, metadata: Dict[str, Any],
                           params: SpectrogramParams, split_syllables: bool = False, verbose: bool = False) -> bool:
    """
    Process audio file and save segmented data with progress tracking.
    Updated to use consolidated ProcessingResult.
    """
    try:
        logger.debug(f" 🎵 Processing audio: {os.path.basename(audio_file_path)}")

        # Use the consolidated get_song_specs function that returns ProcessingResult
        result = get_song_specs(
            audio_file_path, metadata['onsets'], metadata['offsets'],
            params=params, split_syllables=split_syllables, tempo=True
        )

        # Extract from ProcessingResult
        specs = result.specs
        wavs = result.waveforms
        ts = result.spec_times
        valid_inds = result.valid_indices
        tempos = result.tempos or (np.nan, np.nan, np.nan)

        if not valid_inds:
            logger.warning(f" ⚠️ No valid spectrograms generated for {audio_file_path}")
            return False

        logger.debug(f" ✅ Generated {len(valid_inds)} valid spectrograms")
        (mean_top_3, low_f_mean, mean_all) = tempos

        # Create organized data structure using original onset/offset/label arrays
        segmented_audio_data = create_segmented_audio_data(
            specs=specs,
            wavs=wavs,
            ts=ts,
            onsets=metadata['onsets'],  # Original arrays
            offsets=metadata['offsets'],
            labels=metadata['labels'],
            mean_top_3=mean_top_3,
            low_f_mean=low_f_mean,
            mean_all=mean_all,
            valid_indices=valid_inds,  # These now properly map to original indices
            file_identifier=output_path
        )

        # Save to HDF5 file
        logger.debug(f" 💾 Saving to {os.path.basename(output_path)}")
        save_segmented_audio_data(output_path, audio_file_path, segmented_audio_data)

        # Clean up large data structures
        del segmented_audio_data, specs, wavs, ts
        gc.collect()

        if verbose:
            logger.info(f" ✅ Saved {len(valid_inds)} syllables to {output_path}")

        return True

    except Exception as e:
        logger.error(f" 💥 Error processing audio {audio_file_path}: {e}")
        return False


def process_single_metadata_file(metadata_file_path: str, save_path: str, params: SpectrogramParams,
                                 read_songpath_from_metadata: bool, verbose: bool,
                                 prefer_local: bool = True) -> Dict[str, str]:
    """
    Process a single metadata file and save spectrograms if conditions are met.

    Args:
        prefer_local: If True, prefer local audio files over server files
    """
    # Check if metadata file exists
    if not os.path.exists(metadata_file_path):
        return {'status': 'failed', 'reason': 'Metadata file not found'}

    # Load and validate metadata
    metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)

    # Determine bird folder for path mapping
    filename_info = parse_audio_filename(metadata_file_path)
    bird_folder = None
    if filename_info['success']:
        bird_folder = os.path.join(save_path, filename_info['bird'])

    # Resolve audio file path
    audio_file_path, wseg_offset = resolve_audio_file_path(
        metadata_file_path, metadata_matfile, read_songpath_from_metadata,
        bird_folder, prefer_local
    )
    del metadata_matfile
    gc.collect()

    if audio_file_path is None:
        return {'status': 'failed', 'reason': 'Audio file not found'}

    # Update audio paths mapping if we have a bird folder and server path
    if bird_folder and not prefer_local:
        try:
            filename = os.path.basename(audio_file_path)
            update_audio_paths_file(bird_folder, filename, server_path=audio_file_path)
        except Exception as e:
            logger.debug(f"Could not update audio paths file: {e}")

    # Load and validate metadata arrays
    metadata = load_and_validate_metadata(metadata_file_path, wseg_offset)
    if metadata['error']:
        return {'status': 'failed', 'reason': metadata['error']}

    if not metadata['is_valid_song']:
        return {'status': 'skipped', 'reason': 'Single syllable file'}

    # Parse filename for output path
    filename_info = parse_audio_filename(audio_file_path)
    if not filename_info['success']:
        return {'status': 'failed', 'reason': 'Could not parse filename'}

    # Check song duration meets minimum threshold
    total_duration_ms = metadata['offsets'][-1] - metadata['onsets'][0]
    if total_duration_ms <= 2000:  # 2 seconds in milliseconds
        return {'status': 'skipped', 'reason': 'Song too short (< 2 seconds)'}

    # Create output path and check if already exists
    paths = create_output_paths(save_path, filename_info['bird'])
    output_path = os.path.join(
        paths['syllables_dir'],
        f"syllables_{filename_info['bird']}_{filename_info['day']}_{filename_info['time']}.h5"
    )
    if os.path.exists(output_path):
        return {'status': 'skipped', 'reason': 'Output file already exists'}

    # Process the audio
    try:
        success = process_and_save_audio(
            audio_file_path, output_path, metadata, params, verbose
        )

        if success:
            return {'status': 'processed', 'reason': 'Successfully saved'}
        else:
            return {'status': 'failed', 'reason': 'Audio processing failed'}

    except Exception as e:
        logger.error(f"Error processing audio for {metadata_file_path}: {e}")
        return {'status': 'failed', 'reason': f'Processing error: {str(e)}'}


def save_data_specs(metadata_file_paths: List[str], save_path: str, params: SpectrogramParams,
                    verbose: bool = False, read_songpath_from_metadata: bool = True,
                    prefer_local: bool = True) -> Dict[str, List[str]]:
    """
    Process metadata files and save spectrograms to HDF5 files with detailed progress tracking.
    """
    results = {
        'processed': [],
        'skipped': [],
        'failed': []
    }

    logger.info(f"🎵 Starting spectrogram processing for {len(metadata_file_paths)} files")
    logger.info(f"📊 Memory usage at start: {get_memory_usage():.1f} MB")

    for idx, metadata_file_path in enumerate(tqdm(metadata_file_paths, desc="Processing audio files"), 1):
        try:
            logger.debug(
                f"  🔄 Processing file {idx}/{len(metadata_file_paths)}: {os.path.basename(metadata_file_path)}")

            result = process_single_metadata_file(
                metadata_file_path, save_path, params,
                read_songpath_from_metadata, verbose, prefer_local
            )

            results[result['status']].append(metadata_file_path)

            # Log individual results
            if result['status'] == 'processed':
                logger.debug(f"    ✅ Success: {result['reason']}")
            elif result['status'] == 'skipped':
                logger.debug(f"    ⏭️ Skipped: {result['reason']}")
            else:
                logger.warning(f"    ❌ Failed: {result['reason']}")

            # Periodic memory reporting
            if idx % 10 == 0:
                logger.info(f"  📊 Progress {idx}/{len(metadata_file_paths)}, "
                             f"memory: {get_memory_usage():.1f} MB")
                gc.collect()  # Periodic cleanup

        except Exception as e:
            logger.error(f"  💥 Unexpected error processing {metadata_file_path}: {e}")
            results['failed'].append(metadata_file_path)

    # Final summary
    logger.info(f"🎯 Spectrogram processing complete:")
    logger.info(f"  ✅ Processed: {len(results['processed'])}")
    logger.info(f"  ⏭️ Skipped: {len(results['skipped'])}")
    logger.info(f"  ❌ Failed: {len(results['failed'])}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")

    return results


def save_specs_for_evsonganaly_birds(metadata_file_paths: dict, save_path: str = None,
                                     songs_per_bird: int = 5, params: 'SpectrogramParams' = None,
                                     verbose: bool = False, prefer_local: bool = True):
    if params is None:
        params = SpectrogramParams()

    if save_path is None:
        raise ValueError("save_path cannot be None")

    birds = list(metadata_file_paths.keys())
    logger.info(f"🐦 Starting processing for {len(birds)} birds: {birds}")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    successful_birds = []
    failed_birds = []

    for bird_idx, bird in enumerate(birds, 1):
        logger.info(f"🔄 Processing bird {bird_idx}/{len(birds)}: {bird}")
        logger.info(f"📊 Memory usage before {bird}: {get_memory_usage():.1f} MB")

        try:
            syllables_dir = os.path.join(save_path, bird, 'data', 'syllables')

            if os.path.isdir(syllables_dir):
                already_saved_files = os.listdir(syllables_dir)
            else:
                already_saved_files = []

            needed_count = songs_per_bird - len(already_saved_files)
            logger.info(f"  📁 Found {len(already_saved_files)} existing files, need {needed_count} more")

            if needed_count <= 0:
                logger.info(f'  ⏭️ {bird} already processed, skipping...')
                successful_birds.append(bird)
                continue

            logger.info(f"  🔍 Selecting files from {len(metadata_file_paths[bird])} available files")
            candidate_file_paths = select_new_files(
                metadata_file_paths[bird],
                already_saved_files,
                needed_count
            )

            if candidate_file_paths:
                logger.info(f"  🎵 Processing {len(candidate_file_paths)} audio files for {bird}")

                results = save_data_specs(
                    metadata_file_paths=candidate_file_paths,
                    save_path=save_path,
                    params=params,
                    verbose=verbose,
                    read_songpath_from_metadata=False,
                    prefer_local=prefer_local
                )

                # Report detailed results
                logger.info(f"  ✅ {bird} complete: {len(results['processed'])} processed, "
                             f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")

                if results['processed']:
                    successful_birds.append(bird)
                else:
                    failed_birds.append(bird)
                    logger.warning(f"  ⚠️ No files successfully processed for {bird}")
            else:
                logger.warning(f"  ❌ No new files found for {bird}")
                failed_birds.append(bird)

        except Exception as e:
            logger.error(f"  💥 Error processing bird {bird}: {e}")
            failed_birds.append(bird)

        logger.info(f"📊 Memory usage after {bird}: {get_memory_usage():.1f} MB")

        # Force cleanup between birds
        gc.collect()

    # Final summary
    logger.info(f"🎯 Processing complete! Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
    if failed_birds:
        logger.warning(f"❌ Failed birds: {failed_birds}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")


def save_specs_for_wseg_birds(metadata_file_paths: Dict[str, List[str]],
                              save_path: str,
                              songs_per_bird: int = 20,
                              params: SpectrogramParams = None,
                              verbose: bool = False, prefer_local: bool = True):
    """
    Process WhisperSeg files for multiple birds using the new modular pipeline.
    """
    if params is None:
        params = SpectrogramParams()

    birds = list(metadata_file_paths.keys())
    logger.info(f"🐦 Starting wseg processing for {len(birds)} birds: {birds}")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    successful_birds = []
    failed_birds = []

    for bird_idx, bird in enumerate(birds, 1):
        logger.info(f"🔄 Processing wseg bird {bird_idx}/{len(birds)}: {bird}")
        logger.info(f"📊 Memory usage before {bird}: {get_memory_usage():.1f} MB")

        try:
            syllables_dir = os.path.join(save_path, bird, 'data', 'syllables')

            if os.path.isdir(syllables_dir):
                already_saved_files = os.listdir(syllables_dir)
            else:
                already_saved_files = []

            needed_count = songs_per_bird - len(already_saved_files)
            logger.info(f"  📁 Found {len(already_saved_files)} existing files, need {needed_count} more")

            if needed_count <= 0:
                logger.info(f'  ⏭️ {bird} already processed, skipping...')
                successful_birds.append(bird)
                continue

            logger.info(f"  🔍 Selecting files from {len(metadata_file_paths[bird])} available files")
            candidate_file_paths = select_new_files(
                metadata_file_paths[bird],
                already_saved_files,
                needed_count
            )

            if candidate_file_paths:
                logger.info(f"  🎵 Processing {len(candidate_file_paths)} wseg files for {bird}")

                results = save_data_specs(
                    metadata_file_paths=candidate_file_paths,
                    save_path=save_path,
                    params=params,
                    verbose=verbose,
                    read_songpath_from_metadata=True,
                    prefer_local=prefer_local
                )

                # Report detailed results
                logger.info(f"  ✅ {bird} complete: {len(results['processed'])} processed, "
                             f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")

                if results['processed']:
                    successful_birds.append(bird)
                else:
                    failed_birds.append(bird)
                    logger.warning(f"  ⚠️ No files successfully processed for {bird}")
            else:
                logger.warning(f"  ❌ No new files found for {bird}")
                failed_birds.append(bird)

        except Exception as e:
            logger.error(f"  💥 Error processing wseg bird {bird}: {e}")
            failed_birds.append(bird)

        logger.info(f"📊 Memory usage after {bird}: {get_memory_usage():.1f} MB")

        # Force cleanup between birds
        gc.collect()

    # Final summary
    logger.info(f"🎯 Wseg processing complete! Success: {len(successful_birds)}, Failed: {len(failed_birds)}")
    if failed_birds:
        logger.warning(f"❌ Failed birds: {failed_birds}")
    logger.info(f"📊 Final memory usage: {get_memory_usage():.1f} MB")


def process_pipeline(pipeline_name: str, settings: dict):
    """Process a single pipeline (evsonganaly or wseg)."""
    logger.info("=" * 60)
    logger.info(f"🎵 {pipeline_name.upper()} PROCESSING")
    logger.info("=" * 60)

    try:
        if pipeline_name == 'evsonganaly':
            # Get file paths and copy locally
            metadata_file_paths, wav_file_paths = filepaths_from_evsonganaly(
                wav_directory=settings['source_dir'],
                save_path=settings['save_dir'],
                batch_file_naming=settings['batch_file_naming'],
                bird_subset=settings['bird_subset'],
                copy_locally=settings['copy_locally']
            )

            # Process spectrograms - songs_per_bird comes from params now
            save_specs_for_evsonganaly_birds(
                metadata_file_paths=metadata_file_paths,
                save_path=settings['save_dir'],
                songs_per_bird=settings['params'].songs_per_bird,  # Extract from params
                params=settings['params'],
                prefer_local=settings['copy_locally']
            )

        elif pipeline_name == 'wseg':
            # Get wseg paths and copy locally
            wseg_metadata_paths = filepaths_from_wseg(
                seg_directory=settings['source_dir'],
                save_path=settings['save_dir'],
                song_or_call='song',
                bird_subset=settings['bird_subset'],
                copy_locally=settings['copy_locally']
            )

            # Process spectrograms - songs_per_bird comes from params now
            save_specs_for_wseg_birds(
                metadata_file_paths=wseg_metadata_paths,
                save_path=settings['save_dir'],
                songs_per_bird=settings['params'].songs_per_bird,  # Extract from params
                params=settings['params'],
                prefer_local=settings['copy_locally']
            )

        logger.info(f"✅ {pipeline_name.capitalize()} processing complete!")

    except Exception as e:
        logger.error(f"💥 Error in {pipeline_name} processing: {e}")
        raise


def main():
    """Main processing pipeline with configurable parameters."""
    start_time = time.time()
    logger.info("🚀 Starting spectrogram processing pipeline")

    # Setup
    path_to_macaw = check_sys_for_macaw_root()
    optimize_pytables_for_network()

    config = {
        'evsonganaly': {
            'enabled': True,
            'source_dir': os.path.join(path_to_macaw, 'ssharma', 'RNA_seq', 'family_analysis_labeled', 'or-or'),
            'save_dir': os.path.join('/Volumes', 'Extreme SSD', 'evsong test'),
            'batch_file_naming': 'batch.txt.labeled',
            'bird_subset': ['or18or24'],
            'copy_locally': True,
            'params': SpectrogramParams(
                nfft=1024,
                hop=128,
                max_dur=0.150,
                songs_per_bird=5,
                overwrite_existing=True
            )
        },
        'wseg': {
            'enabled': True,
            'source_dir': os.path.join(path_to_macaw, 'annietaylor', 'bubu-rdyw', 'metadata'),
            'save_dir': os.path.join('/Volumes', 'Extreme SSD', 'wseg test'),
            'bird_subset': ['bu85bu97'],
            'copy_locally': True,
            'params': SpectrogramParams(
                nfft=1024,
                hop=128,
                max_dur=0.150,
                songs_per_bird=5,
                overwrite_existing=True
            )
        }
    }

    # Process each pipeline
    for pipeline_name, settings in config.items():
        if not settings['enabled']:
            logger.info(f"⏭️ Skipping {pipeline_name.upper()} processing (disabled)")
            continue

        process_pipeline(pipeline_name, settings)

    total_time = time.time() - start_time
    logger.info(f"🎯 All processing complete! Total time: {total_time:.1f} seconds")


if __name__ == '__main__':
    # Setup logging using consolidated function (consistent with slicing module)
    logger = setup_logging()

    logger.info("🚀 Starting spectrogram processing pipeline")
    main()
