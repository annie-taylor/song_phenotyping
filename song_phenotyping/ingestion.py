"""Syllable spectrogram extraction from raw audio and segmentation files (Stage A).

This module handles the first stage of the song phenotyping pipeline: reading
segmentation metadata (evsonganaly ``.wav.not.mat`` batch files or WhisperSeg
``.wav.not.mat`` metadata files), locating the corresponding audio, computing
short-time Fourier transform spectrograms for each labelled syllable, and
saving the results to HDF5 files.

Two segmentation formats are supported:

evsonganaly
    Produced by the EvSongAnaly MATLAB package. Audio and metadata
    (``.wav.not.mat``) files are co-located under dated subdirectories;
    a ``batch.txt.keep`` file lists the valid recordings.

wseg / WhisperSeg
    Metadata (``.wav.not.mat``) files live in a ``<bird>/song/`` hierarchy
    separate from the audio. The ``fname`` field inside each metadata file
    points to the original audio path.

Public API
----------
- :func:`filepaths_from_evsonganaly` — discover file paths for evsonganaly birds
- :func:`filepaths_from_wseg` — discover file paths for WhisperSeg birds
- :func:`save_specs_for_evsonganaly_birds` — run Stage A for evsonganaly data
- :func:`save_specs_for_wseg_birds` — run Stage A for WhisperSeg data
"""

import os
import logging
import re
from sys import platform
import time
from scipy.io import loadmat
import numpy as np
import hashlib
from typing import Any, Dict, List, Tuple
from random import sample
import gc
import shutil
from tqdm import tqdm
from pathlib import Path

from song_phenotyping.signal import (
    setup_logging,
    get_memory_usage,
    parse_audio_filename,
    load_and_validate_metadata,
    pad_waveforms_to_same_length,
    generate_syllable_hashes,
    create_output_paths,
    save_segmented_audio_data,
    get_song_specs,
)
from song_phenotyping.tools.system_utils import check_sys_for_macaw_root, optimize_pytables_for_network
from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
from song_phenotyping.tools.audio_path_management import *
from song_phenotyping.tools.filerecords import *
from song_phenotyping.tools.logging_utils import setup_logger

logger = setup_logger(__name__, 'spectrogram_saving.log')


def copy_audio_and_partner_rec(audio_path: str, copied_data_dir: str) -> tuple[str | None, str | None]:
    """
    Copy the audio file to `copied_data_dir` (if not already copied) and also try to
    find & copy the matching .rec file (same recording base). Returns (local_audio_path, local_rec_path).
    If the .rec file is not found, local_rec_path is None.
    """
    p = Path(audio_path)
    copied_data_dir = Path(copied_data_dir)
    copied_data_dir.mkdir(parents=True, exist_ok=True)

    filename = p.name
    local_audio_path = copied_data_dir / filename  # Keep as Path object

    # Copy audio if not already present
    if not local_audio_path.exists():  # Use Path methods directly
        try:
            shutil.copy2(str(p), str(local_audio_path))  # Convert to str only for shutil
            logger.debug(" 📋 Copied audio: %s -> %s", p, local_audio_path)
        except Exception as e:
            logger.error(" ❌ Failed to copy audio %s: %s", p, e)
            return None, None

    # Only attempt to find/copy .rec if the audio extension looks like .cbin (or similar)
    # We allow .cbin, .Cbin, or other case variants.
    if p.suffix.lower() != ".cbin":
        return str(local_audio_path), None  # FIX: Convert to string

    # Candidate .rec file search:
    parent = p.parent

    candidates = [
        parent / (p.with_suffix(".rec").name),                # replace final suffix -> name.rec
        parent / (str(p).rsplit(".", 1)[0] + ".rec"),         # rsplit on full path (fallback)
        parent / (p.name.split(".", 1)[0] + ".rec"),         # first-dot prefix.rec
        parent / (p.stem + ".rec"),                          # Path.stem + .rec (handles name.21.cbin -> name.21.rec)
    ]

    # If file has at least two dots, also try removing the penultimate numeric suffix:
    parts = p.name.split(".")
    if len(parts) >= 3:
        base_no_last_two = ".".join(parts[:-2])  # e.g. name.21.cbin -> name
        candidates.append(parent / (base_no_last_two + ".rec"))

    # Check explicit candidates first
    tried = []
    for cand in candidates:
        tried.append(str(cand))
        if cand.exists():
            try:
                local_rec_path = copied_data_dir / cand.name
                if not local_rec_path.exists():
                    shutil.copy2(str(cand), str(local_rec_path))
                    logger.debug("  📋 Copied rec: %s -> %s", cand, local_rec_path)
                return str(local_audio_path), str(local_rec_path)
            except Exception as e:
                logger.error("  ❌ Failed to copy rec %s: %s", cand, e)
                return str(local_audio_path), None

    # Fallback: glob for any *.rec in same directory that starts with the same prefix before first dot or stem
    glob_prefixes = [p.name.split(".", 1)[0], p.stem, ".".join(parts[:-1])]
    for prefix in dict.fromkeys(glob_prefixes):  # preserve order, avoid duplicates
        if not prefix:
            continue
        for g in parent.glob(f"{prefix}*.rec"):
            tried.append(str(g))
            if g.exists():
                try:
                    local_rec_path = copied_data_dir / g.name
                    if not local_rec_path.exists():
                        shutil.copy2(str(g), str(local_rec_path))
                        logger.debug("  📋 Copied rec by glob: %s -> %s", g, local_rec_path)
                    return str(local_audio_path), str(local_rec_path)
                except Exception as e:
                    logger.error("  ❌ Failed to copy rec %s: %s", g, e)
                    return str(local_audio_path), None

    logger.debug("No .rec found for %s. Tried: %s", p, tried)
    return str(local_audio_path), None

def filepaths_from_wseg(seg_directory: str, save_path: str = None,
                        song_or_call: str = 'song',
                        file_ext: str = '.wav.not.mat',
                        bird_subset: None | list = None,
                        copy_locally: bool = False,
                        prefer_local: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Discover WhisperSeg metadata file paths organised by bird ID.

    Walks *seg_directory* recursively, collecting ``.wav.not.mat`` metadata
    files from subdirectories whose path contains *song_or_call*. Bird IDs
    are inferred from the directory two levels above the ``song/`` folder
    (i.e. the structure ``<seg_directory>/<bird>/song/*.wav.not.mat``).

    Parameters
    ----------
    seg_directory : str
        Root directory to scan (e.g. ``metadata/``). Must follow the
        layout ``<seg_directory>/<bird>/song/``.
    save_path : str, optional
        If provided, bird subdirectories are created here. Required when
        *copy_locally* is ``True``.
    song_or_call : str, optional
        Subdirectory name to match — ``'song'`` (default) or ``'call'``.
    file_ext : str, optional
        Metadata file extension. Default is ``'.wav.not.mat'``.
    bird_subset : list of str, optional
        Restrict discovery to these bird IDs. ``None`` returns all birds.
    copy_locally : bool, optional
        If ``True``, copy audio and metadata files to *save_path*.
        Requires Macaw to be mounted. Default is ``False``.
    prefer_local : bool, optional
        If ``True``, attempt to read from the local cache in *save_path*
        before scanning *seg_directory*. Default is ``False``.

    Returns
    -------
    metadata_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_metadata_file, ...]}``.
    audio_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_audio_file, ...]}``. Populated only when
        *copy_locally* is ``True``; otherwise an empty list per bird.

    See Also
    --------
    save_specs_for_wseg_birds : Run Stage A using paths returned by this function.
    """

    # If prefer_local is True, try local cache first (avoids server scanning)
    if prefer_local and save_path and os.path.exists(save_path):
        logger.info("🔄 prefer_local=True, using local cache (no server access)")

        metadata_file_paths, audio_file_paths = filepaths_from_local_cache(save_path, bird_subset)

        if metadata_file_paths:
            logger.info(f"✅ Using local cache with {len(metadata_file_paths)} birds")
            return metadata_file_paths, audio_file_paths
        else:
            logger.warning("⚠️ No local cache found but prefer_local=True. Set prefer_local=False to scan server.")
            return {}

    logger.info(f"🔍 Scanning wseg directory: {seg_directory}")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    metadata_file_paths = {}
    audio_file_paths = {}
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
                    audio_file_paths[bird] = []
                    logger.info(f"  🐦 Started processing bird: {bird}")

                metadata_file_paths[bird].append(file_path)

                # Handle copying and mapping for wseg files
                if save_path and copy_locally:
                    bird_folder = Path(save_path) / bird  # Use Path operations
                    bird_folder.mkdir(parents=True, exist_ok=True)

                    # For wseg, we need to load metadata to get audio path
                    try:
                        metadata_matfile = loadmat(file_path, squeeze_me=True)
                        audio_path, _ = resolve_audio_file_path(
                            file_path, metadata_matfile, read_songpath_from_metadata=True
                        )

                        if audio_path and os.path.exists(audio_path):
                            filename = os.path.basename(audio_path)

                            # Copy audio file locally - use Path operations
                            copied_data_dir = Path(save_path) / 'copied_data' / bird
                            copied_data_dir.mkdir(parents=True, exist_ok=True)

                            # copy audio and attempt to copy partner .rec if audio is .cbin
                            local_audio_path, local_rec_path = copy_audio_and_partner_rec(audio_path,
                                                                                          str(copied_data_dir))

                            # audio_file_paths[bird] should append the local_audio_path
                            audio_file_paths[bird].append(local_audio_path)

                            # ALSO COPY THE METADATA FILE - use Path operations
                            metadata_filename = os.path.basename(file_path)
                            local_metadata_path = copied_data_dir / metadata_filename

                            if not local_metadata_path.exists():
                                try:
                                    shutil.copy2(file_path, str(local_metadata_path))
                                    logger.debug(f" 📋 Copied metadata: {metadata_filename}")
                                except Exception as e:
                                    logger.error(f" ❌ Failed to copy metadata {file_path}: {e}")

                            # Update mapping with both paths - convert to strings for the function
                            update_paths_file(str(bird_folder), filename,
                                              local_path=local_audio_path,
                                              server_path=audio_path)

                            # Also update metadata mapping
                            update_paths_file(str(bird_folder), metadata_filename,
                                              local_path=str(local_metadata_path),
                                              server_path=file_path)
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

    return metadata_file_paths, audio_file_paths


def filepaths_from_evsonganaly(wav_directory: str = None, save_path: str = None,
                               batch_file_naming: str = 'batch.txt.keep',
                               bird_subset: None | list = None,
                               copy_locally: bool = False,
                               prefer_local: bool = False,
                               preferred_subdirs: list = None) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Discover file paths from evsonganaly ``batch.txt.keep`` files.

    Walks *wav_directory* recursively, finds ``batch.txt.keep`` files, and
    extracts paired metadata (``.wav.not.mat``) and audio (``.wav``) paths
    for each bird. Bird IDs are detected via a letter–digit pattern applied
    to the directory path components.

    Parameters
    ----------
    wav_directory : str
        Root directory containing dated subdirectories with audio and
        ``.wav.not.mat`` files (e.g. ``or18or24/18-08-2023/``).
    save_path : str, optional
        If provided, bird output subdirectories are created here.
    batch_file_naming : str, optional
        Name (or substring) of the batch file. Default is
        ``'batch.txt.keep'``.
    bird_subset : list of str, optional
        Restrict discovery to these bird IDs. ``None`` returns all birds.
    copy_locally : bool, optional
        Copy audio and metadata files to *save_path*. Default is ``False``.
    prefer_local : bool, optional
        Attempt to read from the local cache in *save_path* before
        scanning *wav_directory*. Default is ``False``.
    preferred_subdirs : list of str, optional
        If given, only scan directories whose name matches one of these
        values. ``None`` scans all subdirectories.

    Returns
    -------
    metadata_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_not_mat_file, ...]}``.
    audio_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_wav_file, ...]}``.

    See Also
    --------
    save_specs_for_evsonganaly_birds : Run Stage A using paths returned by this function.
    """

    # If prefer_local is True, try local cache first (avoids server scanning)
    if prefer_local and save_path and os.path.exists(save_path):
        logger.info("🔄 prefer_local=True, using local cache (no server access)")

        metadata_file_paths, audio_file_paths = filepaths_from_local_cache(save_path, bird_subset)

        if metadata_file_paths:
            logger.info(f"✅ Using local cache with {len(metadata_file_paths)} birds")
            # For evsonganaly, derive audio_file_paths from metadata paths
            return metadata_file_paths, audio_file_paths
        else:
            logger.warning("⚠️ No local cache found but prefer_local=True. Set prefer_local=False to scan server.")
            return {}, {}

    logger.info(f"🔍 Scanning evsonganaly directory: {wav_directory}")
    logger.info(f"📊 Initial memory usage: {get_memory_usage():.1f} MB")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # if preferred_subdirs is None:
    #     preferred_subdirs = ['labeled_song_final']  # , 'screening', 'baseline', 'baseline_for_tempo']

    batch_files_found = 0
    batch_file_candidates = []
    processed_directories = set()

    bird_pattern = re.compile(r'\b[a-zA-Z]+\d+[a-zA-Z]*\d*\b')

    for root, dirs, files in os.walk(wav_directory, topdown=False):
        batch_files = [f for f in files if batch_file_naming in f]

        if batch_files:
            batch_files_found += len(batch_files)  # Fix: increment counter

            # Extract bird and date from path
            path_parts = root.split(os.sep)
            if len(path_parts) >= 2:
                date = path_parts[-1]

            # Apply bird subset filter
            if bird_subset is not None and not any(bird in path_parts for bird in bird_subset):
                continue

            # Find bird name using pattern matching
            bird = None
            for part in path_parts:
                if bird_pattern.match(part):
                    bird = part
                    break

            # Check if this is a preferred directory type
            if preferred_subdirs is None:
                is_preferred = True
            else:
                is_preferred = any(preferred in path_parts for preferred in preferred_subdirs)

            if is_preferred:
                bird_date_key = f"{bird}_{date}"

                if bird_date_key not in processed_directories:
                    processed_directories.add(bird_date_key)

                    for file in batch_files:
                        batch_file_candidates.append({
                            'path': os.path.join(root, file),
                            'root': root,
                            'file': file,
                            'bird': bird,
                            'date': date,
                        })

                    logger.info(f"  📁 Selected {len(batch_files)} batch files from {bird}/{date}")
                else:
                    logger.debug(
                        f"  ⏭️ Skipping {bird}/{date} - already processed preferred directory")

    logger.info(f"🎯 Selected {len(batch_file_candidates)} unique batch files to process")

    birds = []
    metadata_file_paths = {}
    audio_file_paths = {}
    processed_files = 0
    failed_files = 0

    for batch_info in batch_file_candidates:
        try:
            song_metadata = []
            song_audio = []

            logger.info(f"  📄 Processing batch file: {batch_info['file']}")

            with open(batch_info['path'], 'r') as f:
                lines = f.readlines()
                logger.info(f"    📝 Found {len(lines)} entries in batch file")

                for line_idx, line in enumerate(lines):
                    try:
                        song_name = line.replace('\n', '.not.mat')
                        song_metadata_path = os.path.join(batch_info['root'], song_name)  # Fix: use batch_info['root']

                        if os.path.isfile(song_metadata_path):
                            audio_path = os.path.join(batch_info['root'],
                                                      line.replace('\n', ''))  # Fix: use batch_info['root']
                            song_audio.append(audio_path)
                            song_metadata.append(song_metadata_path)

                            # Extract bird name (though we already know it from batch_info)
                            try:
                                [bird, _, _] = line.split('_')[0:3]
                            except ValueError:
                                logger.debug(f"    ⚠️ Trouble reading birdname from {line}, trying fallback")
                                try:
                                    [bird, _] = line.split('_')[0:2]
                                except ValueError:
                                    logger.debug(f"    ⚠️ Using final fallback for birdname from {line}")
                                    [bird, _] = line.split('.')[0:2]

                            # Note: No need to filter by bird_subset again - already done above

                            # Create bird folder and handle copying/mapping
                            if save_path:
                                bird_folder = os.path.join(save_path, bird)
                                os.makedirs(bird_folder, exist_ok=True)

                                filename = os.path.basename(audio_path)

                                if copy_locally:
                                    # Copy audio file locally - use Path operations
                                    copied_data_dir = Path(save_path) / 'copied_data' / bird
                                    copied_data_dir.mkdir(parents=True, exist_ok=True)

                                    local_audio_path, local_rec_path = copy_audio_and_partner_rec(audio_path,
                                                                                                  str(copied_data_dir))

                                    # ALSO COPY THE METADATA FILE - use Path operations
                                    metadata_filename = os.path.basename(song_metadata_path)
                                    local_metadata_path = copied_data_dir / metadata_filename

                                    if not local_metadata_path.exists():
                                        try:
                                            shutil.copy2(song_metadata_path, str(local_metadata_path))
                                            logger.debug(f" 📋 Copied metadata: {metadata_filename}")
                                        except Exception as e:
                                            logger.error(f" ❌ Failed to copy metadata {song_metadata_path}: {e}")

                                    # Update mapping with both paths (audio and metadata)
                                    update_paths_file(bird_folder, filename,
                                                      local_path=local_audio_path,
                                                      server_path=audio_path)

                                    # Also update metadata mapping
                                    update_paths_file(bird_folder, metadata_filename,
                                                      local_path=local_metadata_path,
                                                      server_path=song_metadata_path)
                                else:
                                    # Just update with server path
                                    update_paths_file(bird_folder, filename,
                                                      server_path=audio_path)
                                    # Also update metadata mapping
                                    update_paths_file(bird_folder, metadata_filename,
                                                      local_path=local_metadata_path,
                                                      server_path=song_metadata_path)

                            processed_files += 1
                        else:
                            logger.debug(f"    ⚠️ Metadata file not found: {song_metadata_path}")

                    except Exception as e:
                        logger.error(f"    💥 Error processing line {line_idx} in {batch_info['file']}: {e}")
                        failed_files += 1

                    # Periodic progress updates for large batch files
                    if (line_idx + 1) % 50 == 0:
                        logger.info(f"    📊 Processed {line_idx + 1}/{len(lines)} lines, "
                                    f"memory: {get_memory_usage():.1f} MB")

            # Fix: Collect by bird AFTER processing the entire batch file
            if song_metadata:  # Only if we found valid files
                bird = batch_info['bird']  # Use the bird from batch_info

                if bird not in birds:
                    birds.append(bird)
                    metadata_file_paths[bird] = []
                    audio_file_paths[bird] = []
                    logger.info(f"    🐦 Started processing bird: {bird}")

                # Add all files from this batch to the bird's collection
                metadata_file_paths[bird].extend(song_metadata)
                audio_file_paths[bird].extend(song_audio)

            logger.info(f"  ✅ Completed batch file {batch_info['file']}: {len(song_metadata)} valid entries")

        except Exception as e:
            logger.error(f"  💥 Error processing batch file {batch_info['file']}: {e}")
            failed_files += 1

    # Remove redundant bird subset filter - already applied above
    # Final summary
    total_birds = len(metadata_file_paths)
    total_metadata_files = sum(len(files) for files in metadata_file_paths.values())
    total_wav_files = sum(len(files) for files in audio_file_paths.values())

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

    return metadata_file_paths, audio_file_paths

def _file_base_remove_after_first_dot(path: str) -> str:
    """
    #TODO will cause errors for date formate bird.date-idx
    Return the filename without directories and without anything after the FIRST dot.
    Examples:
      '/a/b/foo.bar.wav'   -> 'foo'
      '/a/b/foo.bar.json'  -> 'foo'
      'recording.mp3.md'   -> 'recording'
      '/path/.env'         -> '.env'   # keep whole name for single-leading-dot files
      '/path/.hidden.txt'  -> ''       # leading '.' then another dot -> base is '' (left of first dot)
    """
    name = Path(path).name
    # If it's a single leading-dot file like ".env" (no other dots), keep the full name
    if name.startswith('.') and '.' not in name[1:]:
        return name
    # Otherwise split on the first dot and keep the left side
    return name.split('.', 1)[0]

def select_new_file_pairs(available_metadata_files: list[str], available_audio_files: list[str],
                          already_saved_files: list[str], needed_count: int,) -> list[tuple[str, str]]:
    """
    Return up to `needed_count` (metadata_path, audio_path) pairs whose base names
    are not present in `already_saved_files`. Matching is done by filename stem
    (filename without extension). If a metadata file has no matching audio file,
    it is skipped and a warning is logged.
    """
    if needed_count <= 0:
        return []

    processed_bases = {_file_base_remove_after_first_dot(p) for p in already_saved_files}

    # Index available files by base (prefix before first dot)
    meta_by_base = {}
    for p in available_metadata_files:
        base = _file_base_remove_after_first_dot(p)
        meta_by_base[base] = p

    audio_by_base = {}
    for p in available_audio_files:
        base = _file_base_remove_after_first_dot(p)
        audio_by_base[base] = p

    # Build candidate pairs
    candidates: List[Tuple[str, str]] = []
    for base, meta_path in meta_by_base.items():
        if base in processed_bases:
            continue
        audio_path = audio_by_base.get(base)
        if audio_path:
            candidates.append((meta_path, audio_path))
        else:
            logger.warning("No audio file found for metadata '%s' (base=%s)", meta_path, base)

    # If fewer candidates than needed_count, return all. Otherwise sample.
    if len(candidates) <= needed_count:
        return candidates
    return sample(candidates, needed_count)


def select_new_files(available_metadata_files: list[str],
                     already_saved_files: list[str], needed_count: int) -> list[str]:
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
    for metadata_file_path in available_metadata_files:
        file_info = parse_audio_filename(metadata_file_path)
        if file_info['success']:
            base_name = f"{file_info['bird']}_{file_info['day']}_{file_info['time']}"
            if base_name not in processed_bases:
                unprocessed_files.append(metadata_file_path)
        else:
            logger.warning(f"Skipping file with unparseable name: {metadata_file_path}")

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
        if '.wav' in metadata_file_path:
            audio_file_path = metadata_file_path.replace('.wav.not.mat', '.wav')
        elif '.cbin' in metadata_file_path:
            audio_file_path = metadata_file_path.replace('.cbin.not.mat', '.cbin')
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
            if 'users/' in relative_path: relative_path = relative_path.replace('users/','')
            return os.path.join(server_root, relative_path.replace('/', '\\'))
        else:
            return os.path.join(server_root, relative_path)

    return stored_path  # Fallback if parsing fails


def create_segmented_audio_data(specs: List[np.ndarray], wavs: List[np.ndarray],
                                ts: List[np.ndarray], onsets: np.ndarray,
                                offsets: np.ndarray, labels: np.ndarray, mean_top_3: np.float64, low_f_mean: np.float64,
                                mean_all: np.float64,
                                valid_indices: List[int], file_identifier: str,
                                inst_freq_list: List[np.ndarray] = None,
                                group_delay_list: List[np.ndarray] = None) -> Dict[str, Any]:
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

    # Syllable durations in seconds (always computed from onset/offset)
    durations = np.array([(off - on) / 1000.0 for on, off in zip(valid_onsets, valid_offsets)],
                         dtype=np.float64)

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
        'hashes': hashes,
        'durations': durations,
        'inst_freq': inst_freq_list,
        'group_delay': group_delay_list,
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
        'hashes': [],
        'durations': np.array([], dtype=np.float64),
        'inst_freq': None,
        'group_delay': None,
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
        result = get_song_specs(audio_file_path, metadata['onsets'], metadata['offsets'], params=params,
                               split_syllables=split_syllables, tempo=False)

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

        # Unpack phase features from ProcessingResult (None when flags are off)
        pf = result.phase_features  # list of dicts or None
        inst_freq_list = [d.get('inst_freq') for d in pf] if pf else None
        group_delay_list = [d.get('group_delay') for d in pf] if pf else None

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
            valid_indices=valid_inds,
            file_identifier=output_path,
            inst_freq_list=inst_freq_list,
            group_delay_list=group_delay_list,
        )

        # Save to HDF5 file
        logger.debug(f" 💾 Saving to {os.path.basename(output_path)}")
        save_segmented_audio_data(output_path, audio_file_path, segmented_audio_data)

        # Clean up large data structures
        del segmented_audio_data, specs, wavs, ts
        gc.collect()

        if verbose:
            logger.info(f" ✅ Saved {len(valid_inds)} syllables/slices to {output_path}")

        return True

    except Exception as e:
        logger.error(f" 💥 Error processing audio {audio_file_path}: {e}")
        return False


def filepaths_from_local_cache(save_path: str, bird_subset: list = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Discover metadata file paths from local cache by reading audio_paths.txt files.
    Returns LOCAL metadata paths for truly offline operation.
    """
    logger.info(f"🔍 Discovering files from local cache: {save_path}")

    metadata_file_paths = {}
    audio_file_paths = {}

    if not os.path.exists(save_path):
        logger.warning(f"Save path does not exist: {save_path}")
        return metadata_file_paths

    # Scan for bird directories
    for item in os.listdir(save_path):
        bird_folder = os.path.join(save_path, item)

        # Skip non-directories and special directories
        if not os.path.isdir(bird_folder) or item in ['copied_data']:
            continue

        # Apply bird subset filter
        if bird_subset is not None and item not in bird_subset:
            continue

        # Check if we have cached files for this bird
        audio_paths_file = os.path.join(bird_folder, 'audio_paths.txt')
        if not os.path.exists(audio_paths_file):
            logger.debug(f"  ⚠️ No audio_paths.txt found for {item}")
            continue

        logger.info(f"  🐦 Reading cached files for bird: {item}")

        # Read audio_paths.txt to get LOCAL metadata paths
        bird_metadata_files = []
        bird_audio_files = []
        try:
            with open(audio_paths_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue

                    parts = line.split('|')
                    if len(parts) >= 3:
                        # TODO potential issue here with assuming directory structure in filepath!!!
                        bird_id, local_path, server_path = parts[0], parts[1], parts[2]

                        # NORMALIZE paths using Path - this handles E: vs E:\ automatically
                        local_path = str(Path(local_path))
                        server_path = str(Path(server_path))

                        # Look for metadata files (not audio files)
                        if local_path.endswith('.not.mat'):
                            # Check if LOCAL metadata file exists
                            if os.path.exists(local_path):
                                bird_metadata_files.append(local_path)  # Use LOCAL path!
                            else:
                                logger.debug(f"    ⚠️ Local metadata not found: {os.path.basename(local_path)}")
                        elif local_path.endswith(('.wav', '.cbin')):
                            if os.path.exists(local_path):
                                bird_audio_files.append(local_path)
                            else:
                                logger.debug(f"    ⚠️ Local audio data not found: {os.path.basename(local_path)}")


        except Exception as e:
            logger.warning(f"  ⚠️ Error reading audio_paths.txt for {item}: {e}")
            continue

        if bird_metadata_files:
            metadata_file_paths[item] = bird_metadata_files
            audio_file_paths[item] = bird_audio_files
            logger.info(f"  📄 Found {len(bird_metadata_files)} local cached files for {item}")

    total_birds = len(metadata_file_paths)
    total_files = sum(len(files) for files in metadata_file_paths.values())

    logger.info(f"🎯 Local cache discovery complete:")
    logger.info(f"  🐦 Found {total_birds} birds")
    logger.info(f"  📄 Found {total_files} total LOCAL files")

    return metadata_file_paths, audio_file_paths


def select_wseg_file_pairs_from_metadata(metadata_files: List[str],
                                         already_saved_files: List[str],
                                         needed_count: int) -> List[Tuple[str, str]]:
    """
    For wseg files, extract audio paths from metadata and create pairs.
    """
    if needed_count <= 0:
        return []

    from scipy.io import loadmat

    processed_bases = {_file_base_remove_after_first_dot(p) for p in already_saved_files}

    candidates = []
    for metadata_path in metadata_files:
        base = _file_base_remove_after_first_dot(metadata_path)
        if base in processed_bases:
            continue

        try:
            # Load metadata to get audio path
            metadata_matfile = loadmat(metadata_path, squeeze_me=True)

            # Try to get audio path from metadata
            audio_path, _ = resolve_audio_file_path(
                metadata_path, metadata_matfile,
                read_songpath_from_metadata=True
            )

            if audio_path and os.path.exists(audio_path):
                candidates.append((metadata_path, audio_path))
            else:
                logger.warning(f"Audio file not found for {metadata_path}: {audio_path}")

            del metadata_matfile

        except Exception as e:
            logger.warning(f"Could not extract audio path from {metadata_path}: {e}")
            continue

    # Return up to needed_count pairs
    if len(candidates) <= needed_count:
        return candidates
    return sample(candidates, needed_count)

def process_single_file(metadata_file_path: str, audio_file_path: str, save_path: str, params: SpectrogramParams,
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

    ## Load and validate metadata
    #metadata_matfile = loadmat(metadata_file_path, squeeze_me=True)

    # Determine bird folder for path mapping
    filename_info = parse_audio_filename(metadata_file_path)
    bird_folder = None
    if filename_info['success']:
        bird_folder = os.path.join(save_path, filename_info['bird'])

    # # Resolve audio file path
    # audio_file_path, wseg_offset = resolve_audio_file_path(
    #     metadata_file_path, metadata_matfile, read_songpath_from_metadata,
    #     bird_folder, prefer_local
    # )
    #del metadata_matfile
    #gc.collect()

    if audio_file_path is None:
        return {'status': 'failed', 'reason': 'Audio file not found'}

    # Update audio paths mapping if we have a bird folder and server path
    if bird_folder and not prefer_local:
        try:
            filename = os.path.basename(audio_file_path)
            update_paths_file(bird_folder, filename, server_path=audio_file_path)
        except Exception as e:
            logger.debug(f"Could not update audio paths file: {e}")

    # Load and validate metadata arrays
    wseg_offset = 0.0
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
        paths['syllable_specs_dir'],  # Always syllable_data/specs
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


def save_data_specs(candidate_files: List[str], save_path: str,
                    params: SpectrogramParams, verbose: bool = False, read_songpath_from_metadata: bool = True,
                    prefer_local: bool = True) -> Dict[str, List[str]]:
    """
    Process metadata files and save spectrograms to HDF5 files with detailed progress tracking.
    """
    results = {
        'processed': [],
        'skipped': [],
        'failed': []
    }

    logger.info(f"🎵 Starting spectrogram processing for {len(candidate_files)} files")
    logger.info(f"📊 Memory usage at start: {get_memory_usage():.1f} MB")

    for idx, (metadata_file_path, audio_file_path) in enumerate(tqdm(candidate_files, desc="Processing audio files"), 1):
        try:
            logger.debug(
                f"  🔄 Processing file {idx}/{len(candidate_files)}: {os.path.basename(metadata_file_path)}")

            result = process_single_file(
                metadata_file_path, audio_file_path, save_path, params,
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
                logger.info(f"  📊 Progress {idx}/{len(candidate_files)}, "
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


def save_specs_for_evsonganaly_birds(metadata_file_paths: dict, audio_file_paths: dict | None, save_path: str = None,
                                     songs_per_bird: int = 5, params: 'SpectrogramParams' = None,
                                     verbose: bool = False, prefer_local: bool = True):
    """Run Stage A for evsonganaly birds: extract and save syllable spectrograms.

    For each bird in *metadata_file_paths*, selects up to *songs_per_bird*
    unprocessed recordings, computes syllable spectrograms, and saves them
    as HDF5 files under ``<save_path>/<bird>/syllable_data/specs/``.

    Parameters
    ----------
    metadata_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_not_mat_file, ...]}``, as returned by
        :func:`filepaths_from_evsonganaly`.
    audio_file_paths : dict mapping str to list of str or None
        ``{bird_id: [path_to_wav_file, ...]}``. If ``None``, audio paths
        are resolved from the metadata files directly.
    save_path : str
        Root output directory. Bird subdirectories are created automatically.
    songs_per_bird : int or None, optional
        Maximum number of songs to process per bird. ``None`` processes all
        available recordings. Default is 5.
    params : SpectrogramParams, optional
        Spectrogram computation parameters. Defaults to
        :class:`~song_phenotyping.tools.spectrogram_configs.SpectrogramParams`.
    verbose : bool, optional
        Enable verbose per-file logging. Default is ``False``.
    prefer_local : bool, optional
        Try to resolve audio from the local cache before the server.
        Default is ``True``.

    Notes
    -----
    Already-processed songs are detected by counting files in the output
    ``specs/`` directory; only the remaining quota is processed. Re-running
    is therefore safe and incremental.

    See Also
    --------
    filepaths_from_evsonganaly : Discover input file paths.
    save_specs_for_wseg_birds : Equivalent function for WhisperSeg data.
    """
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
            if params.slice_length:
                syllables_dir = os.path.join(save_path, bird, 'slice_data', 'specs')
            else:
                syllables_dir = os.path.join(save_path, bird, 'syllable_data', 'specs')

            if os.path.isdir(syllables_dir):
                already_saved_files = os.listdir(syllables_dir)
            else:
                already_saved_files = []

            limit = songs_per_bird if songs_per_bird is not None else len(metadata_file_paths[bird])
            needed_count = limit - len(already_saved_files)
            logger.info(f"  📁 Found {len(already_saved_files)} existing files, need {needed_count} more")

            if needed_count <= 0:
                logger.info(f'  ⏭️ {bird} already processed, skipping...')
                successful_birds.append(bird)
                continue

            logger.info(f"  🔍 Selecting files from {len(metadata_file_paths[bird])} available files")
            candidate_files = select_new_file_pairs(
                metadata_file_paths[bird],
                audio_file_paths[bird],
                already_saved_files,
                needed_count
            )

            if candidate_files:
                logger.info(f"  🎵 Processing {len(candidate_files)} audio files for {bird}")

                results = save_data_specs(
                    candidate_files=candidate_files,
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
                              audio_file_paths: Dict[str, List[str]],
                              save_path: str,
                              songs_per_bird: int = 20,
                              params: SpectrogramParams = None,
                              verbose: bool = False, prefer_local: bool = True, copy_locally: bool = False):
    """Run Stage A for WhisperSeg birds: extract and save syllable spectrograms.

    For each bird in *metadata_file_paths*, resolves audio paths from the
    embedded ``fname`` field in each ``.wav.not.mat`` file, selects up to
    *songs_per_bird* unprocessed recordings, computes syllable spectrograms,
    and saves them as HDF5 files under
    ``<save_path>/<bird>/syllable_data/specs/``.

    Parameters
    ----------
    metadata_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_not_mat_file, ...]}``, as returned by
        :func:`filepaths_from_wseg`.
    audio_file_paths : dict mapping str to list of str
        ``{bird_id: [path_to_wav_file, ...]}``. Populated when
        *copy_locally* was ``True`` in :func:`filepaths_from_wseg`;
        otherwise pass an empty-list dict and audio is resolved from
        metadata.
    save_path : str
        Root output directory. Bird subdirectories are created automatically.
    songs_per_bird : int or None, optional
        Maximum number of songs to process per bird. ``None`` processes all
        available recordings. Default is 20.
    params : SpectrogramParams, optional
        Spectrogram computation parameters. Defaults to
        :class:`~song_phenotyping.tools.spectrogram_configs.SpectrogramParams`.
    verbose : bool, optional
        Enable verbose per-file logging. Default is ``False``.
    prefer_local : bool, optional
        Try to resolve audio from the local cache before the server.
        Default is ``True``.
    copy_locally : bool, optional
        Copy audio to *save_path* before processing. Default is ``False``.

    Notes
    -----
    Already-processed songs are detected by counting files in the output
    ``specs/`` directory; only the remaining quota is processed.

    See Also
    --------
    filepaths_from_wseg : Discover input file paths.
    save_specs_for_evsonganaly_birds : Equivalent function for evsonganaly data.
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
            if params.slice_length:
                syllables_dir = os.path.join(save_path, bird, 'slice_data')
            else:
                syllables_dir = os.path.join(save_path, bird, 'syllable_data')

            if os.path.isdir(syllables_dir):
                already_saved_files = os.listdir(syllables_dir)
            else:
                already_saved_files = []

            limit = songs_per_bird if songs_per_bird is not None else len(metadata_file_paths[bird])
            needed_count = limit - len(already_saved_files)
            logger.info(f"  📁 Found {len(already_saved_files)} existing files, need {needed_count} more")

            if needed_count <= 0:
                logger.info(f'  ⏭️ {bird} already processed, skipping...')
                successful_birds.append(bird)
                continue

            logger.info(f"  🔍 Selecting files from {len(metadata_file_paths[bird])} available files")
            if copy_locally:
                # Use existing logic with audio_file_paths matching
                candidate_files = select_new_file_pairs(
                    metadata_file_paths[bird],
                    audio_file_paths[bird],
                    already_saved_files,
                    needed_count
                )
            else:
                # For server processing, create pairs by reading metadata
                candidate_files = select_wseg_file_pairs_from_metadata(
                    metadata_file_paths[bird],
                    already_saved_files,
                    needed_count
                )

            if candidate_files:
                logger.info(f"  🎵 Processing {len(candidate_files)} wseg files for {bird}")

                results = save_data_specs(candidate_files=candidate_files,
                    save_path=save_path,
                    params=params,
                    verbose=verbose,
                    read_songpath_from_metadata=True,
                    prefer_local=prefer_local,
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
            metadata_file_paths, audio_file_paths = filepaths_from_evsonganaly(
                wav_directory=settings['source_dir'],
                save_path=settings['save_dir'],
                batch_file_naming=settings['batch_file_naming'],
                bird_subset=settings['bird_subset'],
                copy_locally=settings['copy_locally'],
                prefer_local=settings['prefer_local'],
                preferred_subdirs = settings['preferred_subdirs']
            )

            # Process spectrograms
            save_specs_for_evsonganaly_birds(
                metadata_file_paths=metadata_file_paths,
                audio_file_paths=audio_file_paths,
                save_path=settings['save_dir'],
                songs_per_bird=settings['params'].songs_per_bird,  # Extract from params
                params=settings['params'],
                prefer_local = settings['prefer_local'],
            )

        elif pipeline_name == 'wseg':
            # Get wseg paths and copy locally
            metadata_file_paths, audio_file_paths = filepaths_from_wseg(
                seg_directory=settings['source_dir'],
                save_path=settings['save_dir'],
                song_or_call='song',
                bird_subset=settings['bird_subset'],
                copy_locally=settings['copy_locally'],
                prefer_local=settings['prefer_local'],
                #preferred_subdirs=settings['preferred_subdirs']
            )

            # Process spectrograms - songs_per_bird comes from params now
            save_specs_for_wseg_birds(
                metadata_file_paths=metadata_file_paths,
                audio_file_paths=audio_file_paths,
                save_path=settings['save_dir'],
                songs_per_bird=settings['params'].songs_per_bird,  # Extract from params
                params=settings['params'],
                prefer_local=settings['prefer_local'],
                copy_locally=settings['copy_locally'],
            )

        logger.info(f"✅ {pipeline_name.capitalize()} processing complete!")

    except Exception as e:
        logger.error(f"💥 Error in {pipeline_name} processing: {e}")
        raise


def main():
    """Main processing pipeline with configurable parameters."""
    start_time = time.time()
    logger = setup_logging()
    logger.info("🚀 Starting spectrogram processing pipeline")

    # Setup
    path_to_macaw = check_sys_for_macaw_root()
    optimize_pytables_for_network()

    config = {
        # 'evsonganaly': {
        #     'enabled': True,
        #     'source_dir': os.path.join(path_to_macaw, 'ssharma', 'RNA_seq', 'family_analysis_labeled'),
        #     'save_dir': os.path.join('E:', 'ssharma_RNA_seq'),
        #     'batch_file_naming': 'batch.txt.labeled',
        #     'bird_subset': None,
        #     'copy_locally': True,  # to write/overwrite audio and metadata files to
        #     'prefer_local': False,  # to use local file where audio/metadata files are saved
        #                             # (TODO double check whether this and bool above are mutually exclusive)
        #     'preferred_subdirs': ['labeled_song_final'],
        #     'params': SpectrogramParams(
        #         nfft=1024,
        #         hop=1,
        #         max_dur=0.150,
        #         songs_per_bird=30,
        #         overwrite_existing=True,
        #         use_warping=True,
        #         downsample=True
        #     )
        # },
        'wseg': {
            'enabled': True,
            'source_dir': os.path.join(path_to_macaw, 'annietaylor', 'x-foster'),
            'save_dir': str(Path('E:/') / 'xfosters'),
            'bird_subset': ['bk1bk3'],
            'copy_locally': False,
            'prefer_local': False,
            'params': SpectrogramParams(
                nfft=1024,
                hop=1,
                max_dur=0.070,
                songs_per_bird=30,
                overwrite_existing=True,
                use_warping=True,
                downsample=True
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
    main()
