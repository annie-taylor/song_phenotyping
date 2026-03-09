import os
import json
import logging
import re
import sqlite3
import signal
import sys
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
from functools import lru_cache

# Configuration
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
FILE_BATCH_SIZE = 500
DB_BATCH_SIZE = 100


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Thread-safe operations
_file_lock = threading.Lock()
_bird_name_cache = {}


def save_data_atomic(data: Dict[str, Any], save_file: str) -> None:
    """Unified thread-safe atomic save with datetime handling."""
    with _file_lock:
        try:
            temp_file = save_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
            shutil.move(temp_file, save_file)
            logging.info(f"Saved data to {save_file}")
        except Exception as e:
            logging.error(f"Error saving data to {save_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise


def load_data(save_file: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    if os.path.exists(save_file):
        try:
            with open(save_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Loaded existing data from {save_file}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {save_file}: {e}")
            return {}
    return {}


# Consolidated save/load functions (backwards compatibility)
save_bird_audio_data = save_data_atomic
save_bird_audio_data_with_datetime = save_data_atomic
load_bird_audio_data = load_data


@lru_cache(maxsize=1000)
def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """Cached datetime extraction from birdsong filename formats."""
    if not filename:
        return None

    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')

    # Try birdsong format: bird_id_DDMMYY_HHMMSS
    if len(parts) >= 3:
        try:
            date_part = parts[-2]
            time_part = parts[-1]

            if len(date_part) == 6 and len(time_part) == 6:
                day, month, year = int(date_part[:2]), int(date_part[2:4]), int(date_part[4:6])
                year = year + 2000 if year <= 30 else year + 1900
                hour, minute, second = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6])

                if (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2030 and
                        0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    return datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            pass

    # Fallback patterns
    patterns = [
        (r'(\d{4})(\d{2})(\d{2})_?(\d{2})(\d{2})(\d{2})', 'full'),
        (r'(\d{4})-(\d{2})-(\d{2})[_T](\d{2})-(\d{2})-(\d{2})', 'dash'),
        (r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', 'underscore'),
        (r'(\d{4})(\d{2})(\d{2})(?![\d])', 'date_only'),
    ]

    for pattern, pattern_type in patterns:
        match = re.search(pattern, name_without_ext)
        if match:
            try:
                groups = match.groups()
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])

                if pattern_type == 'date_only':
                    hour = minute = second = 0
                else:
                    hour = int(groups[3]) if len(groups) > 3 else 0
                    minute = int(groups[4]) if len(groups) > 4 else 0
                    second = int(groups[5]) if len(groups) > 5 else 0

                if (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2030 and
                        0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    return datetime(year, month, day, hour, minute, second)
            except (ValueError, IndexError):
                continue

    return None


def analyze_file_info(filepath: str) -> Dict[str, Any]:
    """Extract comprehensive file information."""
    filename = os.path.basename(filepath)
    directory_depth = len(os.path.dirname(filepath).split(os.sep))
    datetime_obj = extract_datetime_from_filename(filename)

    return {
        'filepath': filepath,
        'filename': filename,
        'directory_depth': directory_depth,
        'datetime': datetime_obj,
        'datetime_str': datetime_obj.isoformat() if datetime_obj else None
    }


# Backwards compatibility
get_filename_from_path = os.path.basename


def get_directory_depth(filepath: str) -> int:
    return len(os.path.dirname(filepath).split(os.sep))


def process_file_batch(file_batch: List[str]) -> List[Dict[str, Any]]:
    """Process a batch of files in parallel."""
    if not file_batch:
        return []

    with ThreadPoolExecutor(max_workers=min(8, len(file_batch))) as executor:
        future_to_file = {executor.submit(analyze_file_info, filepath): filepath
                          for filepath in file_batch}

        results = []
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                filepath = future_to_file[future]
                logging.error(f"Error analyzing file {filepath}: {e}")

        return results


def deduplicate_file_list_parallel(file_list: List[str]) -> Tuple[List[str], List[Dict], int, List[Dict]]:
    """Parallel version of file deduplication with datetime analysis."""
    if not file_list:
        return [], [], 0, []

    # Process files in batches
    all_file_info = []
    for i in range(0, len(file_list), FILE_BATCH_SIZE):
        batch = file_list[i:i + FILE_BATCH_SIZE]
        batch_results = process_file_batch(batch)
        all_file_info.extend(batch_results)

    # Group by filename for deduplication
    filename_groups = defaultdict(list)
    file_analyses = {}

    for file_info in all_file_info:
        filepath = file_info['filepath']
        filename = file_info['filename']
        file_analyses[filepath] = file_info
        filename_groups[filename].append(filepath)

    deduplicated_files = []
    file_info_list = []
    duplicates_removed = 0
    duplicate_details = []

    for filename, paths in filename_groups.items():
        if len(paths) == 1:
            chosen_path = paths[0]
            deduplicated_files.append(chosen_path)
            file_info_list.append(file_analyses[chosen_path])
        else:
            # Keep file with shortest path (least directory depth)
            shortest_path = min(paths, key=lambda p: file_analyses[p]['directory_depth'])
            deduplicated_files.append(shortest_path)
            file_info_list.append(file_analyses[shortest_path])
            duplicates_removed += len(paths) - 1

            # Collect duplicate information
            duplicate_info = {
                'filename': filename,
                'kept_path': shortest_path,
                'removed_paths': [p for p in paths if p != shortest_path],
                'datetime': file_analyses[shortest_path]['datetime_str'],
                'all_paths_info': [file_analyses[p] for p in paths]
            }
            duplicate_details.append(duplicate_info)

            logging.info(f"Found {len(paths)} copies of '{filename}', keeping: {shortest_path}")

    return deduplicated_files, file_info_list, duplicates_removed, duplicate_details


def deduplicate_file_list_with_datetime(file_list: List[str]) -> Tuple[List[str], List[Dict], int, List[Dict]]:
    """Backwards compatibility wrapper."""
    return deduplicate_file_list_parallel(file_list)


def deduplicate_file_list(file_list: List[str]) -> Tuple[List[str], int]:
    """Simplified deduplication for backwards compatibility."""
    deduplicated_files, _, duplicates_removed, _ = deduplicate_file_list_parallel(file_list)
    return deduplicated_files, duplicates_removed


def analyze_datetime_distribution(file_info_list: List[Dict]) -> Dict[str, Any]:
    """Analyze temporal distribution of recordings."""
    datetimes = [info['datetime'] for info in file_info_list if info['datetime'] is not None]

    if not datetimes:
        return {
            'total_files': len(file_info_list),
            'files_with_datetime': 0,
            'files_without_datetime': len(file_info_list),
            'datetime_coverage': 0.0
        }

    datetimes.sort()

    # Basic statistics
    stats = {
        'total_files': len(file_info_list),
        'files_with_datetime': len(datetimes),
        'files_without_datetime': len(file_info_list) - len(datetimes),
        'datetime_coverage': len(datetimes) / len(file_info_list) if file_info_list else 0,
        'earliest_recording': datetimes[0].isoformat(),
        'latest_recording': datetimes[-1].isoformat(),
        'date_range_days': (datetimes[-1] - datetimes[0]).days if len(datetimes) > 1 else 0
    }

    # Group by date for distribution analysis
    daily_counts = defaultdict(int)
    hourly_counts = defaultdict(int)
    monthly_counts = defaultdict(int)

    for dt in datetimes:
        daily_counts[dt.date().isoformat()] += 1
        hourly_counts[dt.hour] += 1
        monthly_counts[f"{dt.year}-{dt.month:02d}"] += 1

    stats.update({
        'recordings_per_day': dict(daily_counts),
        'recordings_per_hour': dict(hourly_counts),
        'recordings_per_month': dict(monthly_counts),
        'most_active_day': max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None,
        'most_active_hour': max(hourly_counts.items(), key=lambda x: x[1]) if hourly_counts else None,
        'unique_recording_days': len(daily_counts),
        'average_recordings_per_day': len(datetimes) / len(daily_counts) if daily_counts else 0
    })

    return stats


def process_bird_deduplication(bird_data: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Process deduplication for a single bird - designed for parallel execution."""
    bird_id, data = bird_data

    file_list_fields = ['audio_files', 'wav_files', 'cbin_files', 'batch_files', 'audio_from_batch']
    count_fields = {
        'audio_files': None,
        'wav_files': 'wav_count',
        'cbin_files': 'cbin_count',
        'batch_files': 'batch_file_count',
        'audio_from_batch': 'batch_audio_count'
    }

    new_bird_data = data.copy()
    bird_stats = {
        'deduplication': {},
        'datetime_analysis': {},
        'duplicate_details': []
    }

    for field in file_list_fields:
        if field in data and isinstance(data[field], list):
            original_count = len(data[field])

            # Enhanced deduplication with datetime analysis
            deduplicated_list, file_info_list, duplicates_removed, duplicate_details = \
                deduplicate_file_list_parallel(data[field])

            # Update the data
            new_bird_data[field] = deduplicated_list
            new_bird_data[f'{field}_info'] = file_info_list

            # Update corresponding count field
            count_field = count_fields.get(field)
            if count_field and count_field in new_bird_data:
                new_bird_data[count_field] = len(deduplicated_list)

            # Analyze datetime distribution
            datetime_stats = analyze_datetime_distribution(file_info_list)

            # Track statistics
            bird_stats['deduplication'][field] = {
                'original_count': original_count,
                'final_count': len(deduplicated_list),
                'duplicates_removed': duplicates_removed
            }
            bird_stats['datetime_analysis'][field] = datetime_stats
            bird_stats['duplicate_details'].extend(duplicate_details)

    # Update combined counts
    if 'wav_and_cbin_count' in new_bird_data:
        wav_count = new_bird_data.get('wav_count', 0)
        cbin_count = new_bird_data.get('cbin_count', 0)
        new_bird_data['wav_and_cbin_count'] = wav_count + cbin_count

    # Overall bird datetime analysis
    all_files_info = []
    for field in file_list_fields:
        if f'{field}_info' in new_bird_data:
            all_files_info.extend(new_bird_data[f'{field}_info'])

    bird_stats['overall_datetime_analysis'] = analyze_datetime_distribution(all_files_info)

    return bird_id, new_bird_data, bird_stats


def deduplicate_bird_audio_data_enhanced(bird_audio_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parallel enhanced deduplication with datetime analysis."""
    if not bird_audio_data:
        return {}, {}

    deduplicated_data = {}
    comprehensive_stats = {}
    total_duplicates_removed = 0
    all_duplicate_details = []

    # Process birds in parallel
    bird_items = list(bird_audio_data.items())

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(bird_items))) as executor:
        future_to_bird = {executor.submit(process_bird_deduplication, bird_item): bird_item[0]
                          for bird_item in bird_items}

        for future in as_completed(future_to_bird):
            bird_id = future_to_bird[future]
            try:
                bird_id_result, new_bird_data, bird_stats = future.result()

                deduplicated_data[bird_id_result] = new_bird_data
                comprehensive_stats[bird_id_result] = bird_stats

                # Accumulate statistics
                for field_stats in bird_stats['deduplication'].values():
                    total_duplicates_removed += field_stats.get('duplicates_removed', 0)

                all_duplicate_details.extend(bird_stats.get('duplicate_details', []))

                # Log progress
                duplicates_for_bird = sum(field_stats.get('duplicates_removed', 0)
                                          for field_stats in bird_stats['deduplication'].values())
                if duplicates_for_bird > 0:
                    logging.info(f"Processed {bird_id}: {duplicates_for_bird} duplicates removed")

            except Exception as e:
                logging.error(f"Error processing bird {bird_id}: {e}")

    # Add overall summary statistics
    comprehensive_stats['_summary'] = {
        'total_duplicates_removed': total_duplicates_removed,
        'birds_processed': len(bird_audio_data),
        'all_duplicate_details': all_duplicate_details,
        'processing_timestamp': datetime.now().isoformat()
    }

    logging.info(f"Enhanced deduplication complete. Total duplicates removed: {total_duplicates_removed}")
    return deduplicated_data, comprehensive_stats


def deduplicate_bird_audio_data(bird_audio_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Simplified deduplication for backwards compatibility."""
    deduplicated_data, comprehensive_stats = deduplicate_bird_audio_data_enhanced(bird_audio_data)

    # Convert to simple stats format for backwards compatibility
    duplicate_stats = {}
    for bird_id, stats in comprehensive_stats.items():
        if bird_id == '_summary':
            duplicate_stats[bird_id] = stats
        else:
            duplicate_stats[bird_id] = stats.get('deduplication', {})

    return deduplicated_data, duplicate_stats


@lru_cache(maxsize=1000)
def generate_bird_name_variations(bird_name: str) -> frozenset:
    """Cached bird name variation generation."""
    if not bird_name or not isinstance(bird_name, str):
        return frozenset([bird_name])

    # Color mappings
    color_variations = {
        'bk': ['bk', 'b', 'k', 'black'],
        'wh': ['wh', 'w', 'white'],
        'gr': ['gr', 'g', 'green'],
        'ye': ['ye', 'y', 'yw', 'yellow'],
        'rd': ['rd', 'r', 'red'],
        'pk': ['pk', 'pink'],
        'or': ['or', 'o', 'orange'],
        'pu': ['pu', 'purple'],
        'br': ['br', 'brown'],
        'bl': ['bl', 'bu', 'blue'],
        'nb': ['nb', 'noband']
    }

    # Reverse mapping for standardization
    standardization_map = {}
    for standard, variations in color_variations.items():
        for variation in variations:
            standardization_map[variation.lower()] = standard

    def standardize_color(color_str):  # Fixed: added proper indentation
        if not color_str:
            return ''
        color_str = color_str.lower().strip()
        return standardization_map.get(color_str, color_str)

    # Parse the input bird name
    parts = re.findall(r'([a-zA-Z]+)|(\d+)', bird_name.lower())

    if not parts:
        return frozenset([bird_name])

    standardized_colors = []
    numbers = []

    for letter_part, digit_part in parts:
        if letter_part:
            standardized_colors.append(standardize_color(letter_part))
        if digit_part:
            numbers.append(digit_part)

    variations = set()

    # Generate all combinations based on structure
    if len(standardized_colors) == 1 and len(numbers) == 1:
        # Single band: co#
        color_vars = color_variations.get(standardized_colors[0], [standardized_colors[0]])
        for color_var in color_vars:
            variations.add(f"{color_var}{numbers[0]}")
    elif len(standardized_colors) == 2 and len(numbers) == 2:
        # Two bands: co#co#
        color1_vars = color_variations.get(standardized_colors[0], [standardized_colors[0]])
        color2_vars = color_variations.get(standardized_colors[1], [standardized_colors[1]])
        for color1_var in color1_vars:
            for color2_var in color2_vars:
                variations.add(f"{color1_var}{numbers[0]}{color2_var}{numbers[1]}")
    elif len(standardized_colors) == 2 and len(numbers) == 1:
        # Two colors, one number: co#co
        color1_vars = color_variations.get(standardized_colors[0], [standardized_colors[0]])
        color2_vars = color_variations.get(standardized_colors[1], [standardized_colors[1]])
        for color1_var in color1_vars:
            for color2_var in color2_vars:
                variations.add(f"{color1_var}{numbers[0]}{color2_var}")
    elif len(standardized_colors) == 1 and len(numbers) == 2:
        # One color, two numbers: co##
        color_vars = color_variations.get(standardized_colors[0], [standardized_colors[0]])
        for color_var in color_vars:
            variations.add(f"{color_var}{numbers[0]}{numbers[1]}")

    # Add original input
    variations.add(bird_name.lower())
    return frozenset(variations)

def scan_directory_parallel(directory_info: Tuple[str, str, set]) -> Tuple[
    str, List[str], List[str], List[str], List[str]]:
    """Scan a single directory for audio files - designed for parallel execution."""
    bird_name, bird_dir, bird_name_variations = directory_info

    if not os.path.exists(bird_dir):
        logging.warning(f"Directory not found: {bird_dir}")
        return bird_name, [], [], [], []

    def is_date_file(filename, variations):
        """Check if file matches birdname.date.* pattern."""
        file_parts = filename.split('.')
        if len(file_parts) < 3:
            return False
        if not (len(file_parts[1]) == 8 and file_parts[1].isdigit()):
            return False
        return file_parts[0].lower() in {var.lower() for var in variations}

    wav_files = []
    cbin_files = []
    batch_files = []
    audio_from_batch_files = []

    try:
        for file in os.listdir(bird_dir):
            try:
                full_path = os.path.join(bird_dir, file)
                if not os.path.isfile(full_path):
                    continue

                # Check for audio files
                if file.lower().endswith('.wav') or is_date_file(file, bird_name_variations):
                    wav_files.append(full_path)
                elif file.lower().endswith('.cbin'):
                    cbin_files.append(full_path)
                elif file.lower().endswith('.keep'):
                    batch_files.append(full_path)
                    # Process batch file contents
                    try:
                        with open(full_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    audio_from_batch_files.append(os.path.join(bird_dir, line))
                    except Exception as e:
                        logging.error(f"Error reading batch file {full_path}: {e}")

            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    except Exception as e:
        logging.error(f"Error accessing directory {bird_dir}: {e}")

    return bird_name, wav_files, cbin_files, batch_files, audio_from_batch_files


def scan_directory_parallel(directory_info: Tuple[str, str, frozenset]) -> Tuple[
    str, List[str], List[str], List[str], List[str]]:
    """Scan a single directory for audio files - designed for parallel execution."""
    bird_name, bird_dir, bird_name_variations = directory_info

    if not os.path.exists(bird_dir):
        logging.warning(f"Directory not found: {bird_dir}")
        return bird_name, [], [], [], []

    def is_date_file(filename, variations):
        """Check if file matches birdname.date.* pattern."""
        file_parts = filename.split('.')
        if len(file_parts) < 3:
            return False
        if not (len(file_parts[1]) == 8 and file_parts[1].isdigit()):
            return False
        return file_parts[0].lower() in {var.lower() for var in variations}

    wav_files = []
    cbin_files = []
    batch_files = []
    audio_from_batch_files = []

    try:
        for file in os.listdir(bird_dir):
            try:
                full_path = os.path.join(bird_dir, file)
                if not os.path.isfile(full_path):
                    continue

                # Check for audio files
                if file.lower().endswith('.wav') or is_date_file(file, bird_name_variations):
                    wav_files.append(full_path)
                elif file.lower().endswith('.cbin'):
                    cbin_files.append(full_path)
                elif file.lower().endswith('.keep'):
                    batch_files.append(full_path)
                    # Process batch file contents
                    try:
                        with open(full_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    audio_from_batch_files.append(os.path.join(bird_dir, line))
                    except Exception as e:
                        logging.error(f"Error reading batch file {full_path}: {e}")

            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

    except Exception as e:
        logging.error(f"Error accessing directory {bird_dir}: {e}")

    return bird_name, wav_files, cbin_files, batch_files, audio_from_batch_files


def get_all_audio_files_for_birds_parallel(bird_file_locations: Dict[str, List[str]],
                                           root_directory: str,
                                           save_file: str = "bird_audio_data.json") -> Dict[str, Any]:
    """Parallel version of audio file search with progress saving."""

    # Load existing data if available
    bird_audio_files = load_data(save_file)

    # Set up signal handler for graceful interruption
    def signal_handler(sig, frame):
        logging.info(f"\nInterrupted! Saving current progress to {save_file}...")
        save_data_atomic(bird_audio_files, save_file)
        logging.info("Progress saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Get birds to process (skip completed ones)
    birds_completed = [bird for bird, data in bird_audio_files.items()
                       if data and isinstance(data, dict) and data.get('audio_files')]
    birds_to_process = [bird for bird in bird_file_locations.keys()
                        if bird not in birds_completed]

    logging.info(f"Processing {len(birds_to_process)} birds ({len(birds_completed)} already completed)")

    # Check screening directories
    from tools.system_utils import check_sys_for_macaw_root
    screening_directories = [
        os.path.join(check_sys_for_macaw_root(), 'public', 'screening'),
        os.path.join(check_sys_for_macaw_root(), 'public', 'adult_screening'),
        os.path.join(check_sys_for_macaw_root(), 'public', 'from_egret', 'egret', 'screening'),
        os.path.join(check_sys_for_macaw_root(), 'public', 'from_stork', 'stork', 'screening'),
    ]

    try:
        # Process birds in batches to avoid overwhelming the system
        for batch_start in range(0, len(birds_to_process), DB_BATCH_SIZE):
            batch_birds = birds_to_process[batch_start:batch_start + DB_BATCH_SIZE]

            # Prepare directory scan tasks
            scan_tasks = []
            for bird_name in batch_birds:
                bird_directories = set()
                bird_name_variations = generate_bird_name_variations(bird_name)

                # Add screening directories that match bird name
                for screen_dir in screening_directories:
                    if os.path.exists(screen_dir):
                        try:
                            for sub_dir in os.listdir(screen_dir):
                                if any(var in sub_dir.lower() for var in bird_name_variations):
                                    bird_directories.add(os.path.join(screen_dir, sub_dir))
                        except Exception as e:
                            logging.error(f"Error scanning screening directory {screen_dir}: {e}")

                # Add file path directories
                for file_path in bird_file_locations[bird_name]:
                    try:
                        full_file_path = os.path.join(root_directory, file_path)
                        normalized_path = str(Path(full_file_path).resolve())
                        bird_directories.add(str(Path(normalized_path).parent))
                    except Exception as e:
                        logging.error(f"Error processing file path {file_path}: {e}")

                # Create scan tasks for each directory
                for bird_dir in bird_directories:
                    scan_tasks.append((bird_name, bird_dir, bird_name_variations))

            # Execute directory scans in parallel
            batch_results = {}
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {executor.submit(scan_directory_parallel, task): task
                                  for task in scan_tasks}

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    bird_name = task[0]
                    try:
                        _, wav_files, cbin_files, batch_files, audio_from_batch = future.result()

                        # Accumulate results for this bird
                        if bird_name not in batch_results:
                            batch_results[bird_name] = {
                                'directories_searched': set(),
                                'wav_files': [],
                                'cbin_files': [],
                                'batch_files': [],
                                'audio_from_batch': []
                            }

                        batch_results[bird_name]['directories_searched'].add(task[1])
                        batch_results[bird_name]['wav_files'].extend(wav_files)
                        batch_results[bird_name]['cbin_files'].extend(cbin_files)
                        batch_results[bird_name]['batch_files'].extend(batch_files)
                        batch_results[bird_name]['audio_from_batch'].extend(audio_from_batch)

                    except Exception as e:
                        logging.error(f"Error scanning directory for {bird_name}: {e}")

            # Process batch results
            for bird_name, results in batch_results.items():
                # Remove duplicates and sort
                all_audio_files = sorted(list(set(results['wav_files'] + results['cbin_files'])))
                wav_files = sorted(list(set(results['wav_files'])))
                cbin_files = sorted(list(set(results['cbin_files'])))
                batch_files = sorted(list(set(results['batch_files'])))
                audio_from_batch_files = sorted(list(set(results['audio_from_batch'])))

                # Store results
                bird_audio_files[bird_name] = {
                    'directories_searched': list(results['directories_searched']),
                    'audio_files': all_audio_files,
                    'wav_files': wav_files,
                    'cbin_files': cbin_files,
                    'batch_files': batch_files,
                    'audio_from_batch': audio_from_batch_files,
                    'wav_count': len(wav_files),
                    'cbin_count': len(cbin_files),
                    'wav_and_cbin_count': len(all_audio_files),
                    'batch_file_count': len(batch_files),
                    'batch_audio_count': len(audio_from_batch_files),
                }

                logging.info(f"Processed {bird_name}: {len(all_audio_files)} audio files found")

            # Save progress after each batch
            try:
                save_data_atomic(bird_audio_files, save_file)
            except Exception as e:
                logging.warning(f"Warning: Could not save progress: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.info("Saving current progress...")
        save_data_atomic(bird_audio_files, save_file)
        raise

    logging.info(f"Completed processing all birds! Final data saved to {save_file}")
    return bird_audio_files


# Backwards compatibility wrapper
def get_all_audio_files_for_birds(bird_file_locations: Dict[str, List[str]],
                                  root_directory: str,
                                  save_file: str = "bird_audio_data.json") -> Dict[str, Any]:
    """Backwards compatibility wrapper for parallel audio file search."""
    return get_all_audio_files_for_birds_parallel(bird_file_locations, root_directory, save_file)


def create_datetime_summary_report(comprehensive_stats: Dict[str, Any],
                                   output_file: str = "datetime_analysis_report.txt"):
    """Create a human-readable report of the datetime analysis."""
    with open(output_file, 'w') as f:
        f.write("=== BIRDSONG RECORDING DATETIME ANALYSIS REPORT ===\n\n")
        f.write(f"Generated: {comprehensive_stats['_summary']['processing_timestamp']}\n")
        f.write(f"Total duplicates removed: {comprehensive_stats['_summary']['total_duplicates_removed']}\n")
        f.write(f"Birds processed: {comprehensive_stats['_summary']['birds_processed']}\n\n")

        # Per-bird analysis
        for bird_id, stats in comprehensive_stats.items():
            if bird_id == '_summary':
                continue

            f.write(f"\n{'=' * 50}\n")
            f.write(f"BIRD: {bird_id}\n")
            f.write(f"{'=' * 50}\n")

            # Overall datetime analysis for this bird
            overall = stats.get('overall_datetime_analysis', {})
            if overall:
                f.write(f"\nOVERALL RECORDING SUMMARY:\n")
                f.write(f"  Total files: {overall['total_files']}\n")
                f.write(f"  Files with datetime: {overall['files_with_datetime']}\n")
                f.write(f"  Datetime coverage: {overall['datetime_coverage']:.1%}\n")

                if overall.get('earliest_recording'):
                    f.write(f"  Recording period: {overall['earliest_recording']} to {overall['latest_recording']}\n")
                    f.write(f"  Date range: {overall['date_range_days']} days\n")
                    f.write(f"  Unique recording days: {overall['unique_recording_days']}\n")
                    f.write(f"  Average recordings per day: {overall['average_recordings_per_day']:.1f}\n")

                    if overall.get('most_active_day'):
                        f.write(
                            f"  Most active day: {overall['most_active_day'][0]} ({overall['most_active_day'][1]} recordings)\n")
                    if overall.get('most_active_hour'):
                        f.write(
                            f"  Most active hour: {overall['most_active_hour'][0]}:00 ({overall['most_active_hour'][1]} recordings)\n")

            # Per-file-type analysis
            f.write(f"\nPER-FILE-TYPE ANALYSIS:\n")
            datetime_analysis = stats.get('datetime_analysis', {})
            dedup_analysis = stats.get('deduplication', {})

            for field in ['wav_files', 'cbin_files', 'batch_files', 'audio_from_batch']:
                if field in datetime_analysis:
                    dt_stats = datetime_analysis[field]
                    dedup_stats = dedup_analysis.get(field, {})

                    f.write(f"\n  {field.upper()}:\n")
                    f.write(f"    Original count: {dedup_stats.get('original_count', 'N/A')}\n")
                    f.write(f"    After deduplication: {dedup_stats.get('final_count', 'N/A')}\n")
                    f.write(f"    Duplicates removed: {dedup_stats.get('duplicates_removed', 0)}\n")
                    f.write(f"    Datetime coverage: {dt_stats['datetime_coverage']:.1%}\n")

                    if dt_stats.get('earliest_recording'):
                        f.write(f"    Date range: {dt_stats['earliest_recording']} to {dt_stats['latest_recording']}\n")
                        f.write(f"    Span: {dt_stats['date_range_days']} days\n")

            # Duplicate details
            duplicate_details = stats.get('duplicate_details', [])
            if duplicate_details:
                f.write(f"\nDUPLICATE FILES FOUND:\n")
                for detail in duplicate_details[:10]:  # Show first 10 duplicates
                    f.write(f"  Filename: {detail['filename']}\n")
                    f.write(f"    Kept: {detail['kept_path']}\n")
                    f.write(f"    Removed: {len(detail['removed_paths'])} copies\n")
                    if detail['datetime']:
                        f.write(f"    Recording time: {detail['datetime']}\n")

                if len(duplicate_details) > 10:
                    f.write(f"  ... and {len(duplicate_details) - 10} more duplicate sets\n")

    logging.info(f"Datetime analysis report saved to: {output_file}")


def main_enhanced_deduplication(input_file="cross_fostered_bird_audio_data.json",
                                output_file="cross_fostered_bird_audio_data_deduplicated_enhanced.json",
                                report_file="datetime_analysis_report.txt",
                                stats_file="comprehensive_deduplication_stats.json"):
    """Main function to run the enhanced deduplication with datetime analysis"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_deduplication.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Load original data
    logging.info(f"Loading data from {input_file}")
    bird_audio_data = load_data(input_file)

    if not bird_audio_data:
        logging.error("No data loaded. Exiting.")
        return

    logging.info(f"Loaded data for {len(bird_audio_data)} birds")

    # Perform enhanced deduplication with datetime analysis
    logging.info("Starting enhanced deduplication with datetime analysis...")
    deduplicated_data, comprehensive_stats = deduplicate_bird_audio_data_enhanced(bird_audio_data)

    # Save deduplicated data with error handling
    try:
        save_data_atomic(deduplicated_data, output_file)
        logging.info(f"Successfully saved deduplicated data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving deduplicated data: {e}")
        return

    # Save comprehensive statistics with error handling
    try:
        save_data_atomic(comprehensive_stats, stats_file)
        logging.info(f"Successfully saved stats to {stats_file}")
    except Exception as e:
        logging.error(f"Error saving stats: {e}")

    # Create human-readable report
    try:
        create_datetime_summary_report(comprehensive_stats, report_file)
    except Exception as e:
        logging.error(f"Error creating report: {e}")

    # Print summary
    total_removed = comprehensive_stats['_summary']['total_duplicates_removed']
    print(f"\n=== ENHANCED DEDUPLICATION SUMMARY ===")
    print(f"Total duplicate files removed: {total_removed}")
    print(f"Birds processed: {len(bird_audio_data)}")
    print(f"Deduplicated data saved to: {output_file}")
    print(f"Comprehensive statistics saved to: {stats_file}")
    print(f"Human-readable report saved to: {report_file}")

    # Show quick datetime coverage summary
    print(f"\n=== DATETIME EXTRACTION SUMMARY ===")
    total_files = 0
    total_with_datetime = 0

    for bird_id, stats in comprehensive_stats.items():
        if bird_id == '_summary':
            continue

        overall = stats.get('overall_datetime_analysis', {})
        if overall:
            bird_total = overall.get('total_files', 0)
            bird_with_dt = overall.get('files_with_datetime', 0)
            total_files += bird_total
            total_with_datetime += bird_with_dt

            coverage = overall.get('datetime_coverage', 0)
            print(f"{bird_id}: {bird_with_dt}/{bird_total} files ({coverage:.1%} coverage)")

    overall_coverage = total_with_datetime / total_files if total_files > 0 else 0
    print(f"\nOVERALL: {total_with_datetime}/{total_files} files ({overall_coverage:.1%} coverage)")

    # Show some example datetime patterns found
    print(f"\n=== EXAMPLE DATETIME PATTERNS DETECTED ===")
    example_count = 0
    for bird_id, stats in comprehensive_stats.items():
        if bird_id == '_summary' or example_count >= 5:
            continue

        duplicate_details = stats.get('duplicate_details', [])
        for detail in duplicate_details:
            if detail.get('datetime') and example_count < 5:
                print(f"Filename: {detail['filename']} -> {detail['datetime']}")
                example_count += 1

def main_deduplication():
    """Backwards compatibility wrapper for simple deduplication."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deduplication.log'),
            logging.StreamHandler()
        ]
    )

    # Load original data
    input_file = "cross_fostered_bird_audio_data.json"
    bird_audio_data = load_data(input_file)

    if not bird_audio_data:
        logging.error("No data loaded. Exiting.")
        return

    logging.info(f"Loaded data for {len(bird_audio_data)} birds")

    # Perform deduplication
    deduplicated_data, duplicate_stats = deduplicate_bird_audio_data(bird_audio_data)

    # Save deduplicated data
    output_file = "cross_fostered_bird_audio_data_deduplicated.json"
    save_data_atomic(deduplicated_data, output_file)

    # Save deduplication statistics
    stats_file = "deduplication_stats.json"
    save_data_atomic(duplicate_stats, stats_file)

    # Print summary
    total_removed = duplicate_stats['_summary']['total_duplicates_removed']
    print(f"\n=== DEDUPLICATION SUMMARY ===")
    print(f"Total duplicate files removed: {total_removed}")
    print(f"Birds processed: {len(bird_audio_data)}")
    print(f"Deduplicated data saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")

    # Show per-bird summary
    print(f"\nPer-bird duplicate removal:")
    for bird_id, stats in duplicate_stats.items():
        if bird_id == '_summary':
            continue

        bird_total = sum(field_stats.get('duplicates_removed', 0) for field_stats in stats.values())
        if bird_total > 0:
            print(f"  {bird_id}: {bird_total} duplicates removed")
            for field, field_stats in stats.items():
                if field_stats.get('duplicates_removed', 0) > 0:
                    print(f"    {field}: {field_stats['original_count']} → {field_stats['final_count']}")

# Import and compatibility functions from your original code
def get_file_locations_for_birds(cross_fostered_birds, txt_file_path="MacawAllDirsByBird.txt",
                                db_path='2026-01-15-db.sqlite3'):
    """Parse the text file and return a dictionary mapping bird names to their file locations"""
    bird_files = {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Import database functions
    from tools.dbquery import getUUIDfromBands

    # First, get the bird names from cross_fostered_birds and filter by sex
    target_birds = {}
    target_bird_variations = {}  # Map variations back to original names

    for bird_name in cross_fostered_birds:
        try:
            uuid, exists = getUUIDfromBands(cur, bird_name)
            if exists:
                sex_query = "SELECT sex FROM birds_animal WHERE uuid=?"
                sex_result = cur.execute(sex_query, (uuid,))
                sex_row = sex_result.fetchone()
                sex = sex_row[0] if sex_row and sex_row[0] else 'U'

                # Only include males (M) or unknown (U) sex birds
                if sex in ['M', 'U']:
                    target_birds[bird_name] = sex
                    # Generate all variations for this bird name and map them back
                    variations = generate_bird_name_variations(bird_name)
                    for variation in variations:
                        target_bird_variations[variation.lower()] = bird_name
                else:
                    logging.info(f"Skipping {bird_name} - gender: {sex}")
            else:
                # If bird not found in DB, assume unknown and include
                target_birds[bird_name] = 'U'
                variations = generate_bird_name_variations(bird_name)
                for variation in variations:
                    target_bird_variations[variation.lower()] = bird_name

        except Exception as e:
            logging.error(f"Error checking gender for {bird_name}: {e}")
            # If error, assume unknown and include
            target_birds[bird_name] = 'U'
            variations = generate_bird_name_variations(bird_name)
            for variation in variations:
                target_bird_variations[variation.lower()] = bird_name

    con.close()

    logging.info(f"Filtering to {len(target_birds)} male/unknown birds from {len(cross_fostered_birds)} total")
    logging.info(f"Generated {len(target_bird_variations)} name variations for matching")

    try:
        with open(txt_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    # Split on comma to get bird name (first element)
                    parts = line.split(',')
                    if len(parts) >= 4:  # Ensure we have all expected parts
                        txt_file_bird_name = parts[0].strip()

                        # Generate all variations of the bird name from the text file
                        txt_variations = generate_bird_name_variations(txt_file_bird_name)

                        # Check if any variation of the txt file bird name matches any target bird
                        matched_target_bird = None
                        for txt_variation in txt_variations:
                            if txt_variation.lower() in target_bird_variations:
                                matched_target_bird = target_bird_variations[txt_variation.lower()]
                                break

                        if matched_target_bird:
                            # Extract the file path string (last element, remove brackets)
                            file_path_string = parts[3].strip('[]')

                            # Split on '//' to get individual file paths
                            file_paths = [path for path in file_path_string.split('//') if path.strip()]

                            # Add to dictionary using the original target bird name as key
                            if matched_target_bird in bird_files:
                                bird_files[matched_target_bird].extend(file_paths)
                            else:
                                bird_files[matched_target_bird] = file_paths

                            logging.info(
                                f"  Matched '{txt_file_bird_name}' from file to target bird '{matched_target_bird}'")
    except FileNotFoundError:
        logging.error(f"File {txt_file_path} not found!")
        return {}

    # Remove duplicates from file paths for each bird
    for bird_name in bird_files:
        bird_files[bird_name] = list(set(bird_files[bird_name]))

    logging.info(f"Found file locations for {len(bird_files)} birds")
    return bird_files


def main():
    """Main function that combines file search with enhanced deduplication."""
    from tools.system_utils import check_sys_for_macaw_root

    root_dir = check_sys_for_macaw_root()
    save_file = "xfoster_audio_paths.json"

    birds = ['wh15wh94', 'wh14wh93', 'wh50wh90', 'wh49wh89', 'wh47wh87', 'wh48wh88',
             'bu26wh39', 'bu25wh38', 'bu82wh76', 'rd39wh3', 'rd40wh4', 'bu9wh89',
             'bu10wh90', 'bu7wh87', 'rd33wh42', 'rd25wh57', 'gr6gr5', 'ye30ye82',
             'ye23ye43', 'rd42pu47', 'rd43pu46', 'rd50gr1', 'rd39gr23', 'bk45wh59',
             'wh86or19', 'wh85or81', 'wh82or15', 'wh83or16', 'rd77wh77', 'bk24wh25',
             'bk100wh89', 'bk17wh56', 'bk81wh20', 'bk79wh14', 'bk38bk39', 'bk4bk47',
             'bk36bk37', 'bk83bk73', 'bk94bk87', 'bk67bk44', 'bk76bk63', 'bk75bk62',
             'bk43bk70', 'bk13bk12', 'bk41bk14', 'pk61bk8', 'pk15bk43', 'pu57wh52',
             'pu55wh32', 'ye92br4', 'ye93br5', 'ye91br6', 'bk34bk51', 'bk73bk71',
             'bk1bk3', 'bk91wh31', 'bk61wh42', 'bk63wh43', 'pu1wh51', 'wh93pk62',
             'wh91pk61', 'pk6bk65', 'pk37bk19', 'pk46bk46', 'bk40wh47', 'pu31br476',
             'pu33br479', 'pk5bk39', 'pk2bk37', 'pu58br33', 'pu55br34', 'pu22wh17',
             'pk1bk31', 'pk43bk33', 'pu51wh43', 'pu71wh42', 'pu39wh79', 'bk13wh63',
             'bk72wh64', 'bk74wh76', 'bu34or18', 'pk24bu3', 'pk72bk90', 'pk85gr19',
             'pk97rd22', 'pu11bk85', 'pu42wh35', 'pu91wh67', 'rd75wh72', 'wh71br49',
             'wh88br85', 'ye1tut0', 'ye81br444', 'bk1bk3', 'bk34bk51', 'bk37wh86',
             'bu10wh86', 'bu24wh84', 'bu7wh67', 'gr73gr72', 'gr99or87', 'or10bk88',
             'pk100bk68', 'pk24bu3', 'pu23wh38', 'pu35wh17', 'rd81wh45', 'ye25']

    bird_file_locations = get_file_locations_for_birds(birds,
                                                       txt_file_path='../refs/MacawAllDirsByBird.txt',
                                                       db_path='../refs/2026-01-15-db.sqlite3')

    # Use parallel audio file search
    get_all_audio_files_for_birds_parallel(bird_file_locations, root_directory=root_dir, save_file=save_file)

    # Run enhanced deduplication
    main_enhanced_deduplication(input_file="xfoster_audio_paths(new).json",
                                output_file="xfoster_audio_data_deduplicated(new).json",
                                report_file="xfoster_datetime_analysis_report(new).txt",
                                stats_file="comprehensive_dedup_stats(new).json")


# Additional imports and functions from your original code
import csv


def setup_logging():
    """Set up logging configuration for this run"""
    from datetime import datetime

    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
    LOG_TO_CONSOLE = True

    # Create logs directory if it doesn't exist
    log_dir = "../logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create log filename with timestamp
        log_filename = os.path.join(log_dir, f"cross_foster_analysis_{RUN_TIMESTAMP}.log")

        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up file handler
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Set up console handler if requested
        handlers = [file_handler]
        if LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(console_handler)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=handlers,
            format=log_format
        )

        logger = logging.getLogger(__name__)
        logger.info(f"=== Cross-Foster Analysis Run Started: {RUN_TIMESTAMP} ===")
        logger.info(f"Log file: {log_filename}")

        return logger

    def create_backup_directory():
        """Create backups directory if it doesn't exist"""
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            logging.info(f"Created backup directory: {backup_dir}")
        return backup_dir

    def backup_file_if_exists(filepath, backup_dir):
        """Create a backup of a file if it exists"""
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
            backup_filename = f"{name}.backup_{RUN_TIMESTAMP}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)

            try:
                shutil.copy2(filepath, backup_path)
                logging.info(f"Backed up {filepath} to {backup_path}")
                return True
            except Exception as e:
                logging.error(f"Failed to backup {filepath}: {e}")
                return False
        return False

    def get_birds_with_genetic_parents(db_path='2026-01-15-db.sqlite3'):
        """Get all birds that have genetic parent data."""
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        query = """
                SELECT DISTINCT ba.uuid,
                                bc1.abbrv || ba.band_number ||
                                CASE
                                    WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                        THEN bc2.abbrv || ba.band_number2
                                    ELSE ''
                                    END as bird_name
                FROM birds_animal ba
                         INNER JOIN birds_geneticparent bgp ON ba.uuid = bgp.genchild_id
                         LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                         LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                WHERE bgp.genparent_id IS NOT NULL
                  AND bgp.genparent_id != 'NULL';
                """

        result = cur.execute(query)
        birds_with_genetic_parents = result.fetchall()
        con.close()

        return birds_with_genetic_parents

    def find_cross_fostered_birds_father_only(bird_list, db_path='2026-01-15-db.sqlite3'):
        """Find birds where genetic father differs from nest father."""
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        cross_fostered_birds = []

        for uuid, bird_name in bird_list:
            # Get genetic parents
            genetic_query = """
                            SELECT DISTINCT bgp.genparent_id,
                                            bc1.abbrv || ba.band_number ||
                                            CASE
                                                WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                                    THEN bc2.abbrv || ba.band_number2
                                                ELSE ''
                                                END as parent_name,
                                            ba.sex
                            FROM birds_geneticparent bgp
                                     LEFT JOIN birds_animal ba ON bgp.genparent_id = ba.uuid
                                     LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                                     LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                            WHERE bgp.genchild_id = ?
                              AND bgp.genparent_id IS NOT NULL
                              AND bgp.genparent_id != 'NULL' \
                            """

            # Get nest parents
            nest_query = """
                         SELECT DISTINCT bp.parent_id,
                                         bc1.abbrv || ba.band_number ||
                                         CASE
                                             WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                                 THEN bc2.abbrv || ba.band_number2
                                             ELSE ''
                                             END as parent_name,
                                         ba.sex
                         FROM birds_parent bp
                                  LEFT JOIN birds_animal ba ON bp.parent_id = ba.uuid
                                  LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                                  LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                         WHERE bp.child_id = ?
                           AND bp.parent_id IS NOT NULL
                           AND bp.parent_id != 'NULL' \
                         """

            genetic_result = cur.execute(genetic_query, (uuid,))
            genetic_parents = genetic_result.fetchall()

            nest_result = cur.execute(nest_query, (uuid,))
            nest_parents = nest_result.fetchall()

            # Extract fathers only
            genetic_father = next((p for p in genetic_parents if p[2] == 'M'), None)
            nest_father = next((p for p in nest_parents if p[2] == 'M'), None)

            # Only proceed if we have both genetic and nest father data
            if genetic_father and nest_father:
                # Check if fathers are different (cross-fostered)
                if genetic_father[0] != nest_father[0]:  # Compare UUIDs
                    # Also get mothers for context
                    genetic_mother = next((p for p in genetic_parents if p[2] == 'F'), None)
                    nest_mother = next((p for p in nest_parents if p[2] == 'F'), None)

                    cross_fostered_birds.append({
                        'child_uuid': uuid,
                        'child_name': bird_name,
                        'genetic_father': genetic_father[1],
                        'nest_father': nest_father[1],
                        'genetic_mother': genetic_mother[1] if genetic_mother else 'Unknown',
                        'nest_mother': nest_mother[1] if nest_mother else 'Unknown'
                    })

        con.close()
        return cross_fostered_birds

    def create_cross_fostered_birds_csv(cross_fostered_fathers, bird_audio_data,
                                        output_file="cross_fostered_birds_summary.csv",
                                        db_path='2026-01-15-db.sqlite3'):
        """Create a CSV file with cross-fostered bird information"""
        from tools.dbquery import getUUIDfromBands

        # Backup existing file if it exists
        backup_dir = create_backup_directory()
        backup_file_if_exists(output_file, backup_dir)

        con = sqlite3.connect(db_path)
        cur = con.cursor()

        headers = [
            'Bird Name',
            'Sex',
            'DOB',
            'Genetic Father',
            'Genetic Mother',
            'Nest Father',
            'Nest Mother',
            'Total Songs',
            'Directories Searched'
        ]

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)

                for bird in cross_fostered_fathers:
                    bird_name = bird['child_name']

                    try:
                        uuid, exists = getUUIDfromBands(cur, bird_name)
                        if exists:
                            # Get sex
                            sex_query = "SELECT sex FROM birds_animal WHERE uuid=?"
                            sex_result = cur.execute(sex_query, (uuid,))
                            sex_row = sex_result.fetchone()
                            sex = sex_row[0] if sex_row and sex_row[0] else 'Unknown'

                            # Get birth date
                            birth_query = "SELECT hatch_date FROM birds_animal WHERE uuid=?"
                            birth_result = cur.execute(birth_query, (uuid,))
                            birth_row = birth_result.fetchone()
                            birth_date = birth_row[0] if birth_row and birth_row[0] else 'Unknown'
                        else:
                            sex = 'Unknown'
                            birth_date = 'Unknown'
                    except Exception as e:
                        logging.error(f"Error getting sex/birth date for {bird_name}: {e}")
                        sex = 'Error'
                        birth_date = 'Error'

                    # Get audio data for this bird (if available)
                    if bird_name in bird_audio_data:
                        audio_data = bird_audio_data[bird_name]
                        total_songs = audio_data.get('wav_and_cbin_count', 0)
                        directories = '; '.join(audio_data.get('directories_searched', []))
                    else:
                        total_songs = 0
                        directories = 'No audio data found'

                    row = [
                        bird_name,
                        sex,
                        birth_date,
                        bird['genetic_father'],
                        bird['genetic_mother'],
                        bird['nest_father'],
                        bird['nest_mother'],
                        total_songs,
                        directories
                    ]

                    writer.writerow(row)

            con.close()
            logging.info(f"CSV file created successfully: {output_file}")
            logging.info(f"Exported data for {len(cross_fostered_fathers)} cross-fostered birds")

        except Exception as e:
            con.close()
            logging.error(f"Error creating CSV file: {e}")


def main_xfoster_analysis():
    """
    Cross-foster analysis pipeline with your specific birds and file paths.
    Steps: File location mapping -> Audio search -> Deduplication
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('xfoster_analysis.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Get root directory
    from tools.system_utils import check_sys_for_macaw_root
    root_dir = check_sys_for_macaw_root()

    # Your specific save file (fixed to use (new) consistently)
    save_file = "xfoster_audio_paths(new).json"

    # Your specific bird list
    birds = ['wh15wh94', 'wh14wh93', 'wh50wh90',
             'wh49wh89',
             'wh47wh87',
             'wh48wh88',
             'bu26wh39',
             'bu25wh38',
             'bu82wh76',
             'rd39wh3',
             'rd40wh4',
             'bu9wh89',
             'bu10wh90',
             'bu7wh87',
             'rd33wh42',
             'rd25wh57',
             'gr6gr5',
             'ye30ye82',
             'ye23ye43',
             'rd42pu47',
             'rd43pu46',
             'rd50gr1',
             'rd39gr23',
             'bk45wh59',
             'wh86or19',
             'wh85or81',
             'wh82or15',
             'wh83or16',
             'rd77wh77',
             'bk24wh25',
             'bk100wh89',
             'bk17wh56',
             'bk81wh20',
             'bk79wh14',
             'bk38bk39',
             'bk4bk47',
             'bk36bk37',
             'bk83bk73',
             'bk94bk87',
             'bk67bk44',
             'bk76bk63',
             'bk75bk62',
             'bk43bk70',
             'bk13bk12',
             'bk41bk14',
             'pk61bk8',
             'pk15bk43',
             'pu57wh52',
             'pu55wh32',
             'ye92br4',
             'ye93br5',
             'ye91br6',
             'bk34bk51',
             'bk73bk71',
             'bk1bk3',
             'bk91wh31',
             'bk61wh42',
             'bk63wh43',
             'pu1wh51',
             'wh93pk62',
             'wh91pk61',
             'pk6bk65',
             'pk37bk19',
             'pk46bk46',
             'bk40wh47',
             'pu31br476',
             'pu33br479',
             'pk5bk39',
             'pk2bk37',
             'pu58br33',
             'pu55br34',
             'pu22wh17',
             'pk1bk31',
             'pk43bk33',
             'pu51wh43',
             'pu71wh42',
             'pu39wh79',
             'bk13wh63',
             'bk72wh64',
             'bk74wh76',
             'bu34or18',
             'pk24bu3',
             'pk72bk90',
             'pk85gr19',
             'pk97rd22',
             'pu11bk85',
             'pu42wh35',
             'pu91wh67',
             'rd75wh72',
             'wh71br49',
             'wh88br85',
             'ye1tut0',
             'ye81br444',
             'bk1bk3',
             'bk34bk51',
             'bk37wh86',
             'bu10wh86',
             'bu24wh84',
             'bu7wh67',
             'gr73gr72',
             'gr99or87',
             'or10bk88',
             'pk100bk68',
             'pk24bu3',
             'pu23wh38',
             'pu35wh17',
             'rd81wh45',
             'ye25']

    # STEP 2: File location mapping
    logging.info("Step 2: Getting file locations from MacawAllDirsByBird.txt...")
    bird_file_locations = get_file_locations_for_birds(
        birds,
        txt_file_path='../refs/MacawAllDirsByBird.txt',
        db_path='../refs/2026-01-15-db.sqlite3'
    )

    logging.info(f"Found file locations for {len(bird_file_locations)} birds")

    # STEP 4: Audio file search (using parallel version)
    logging.info("Step 4: Searching for audio files...")
    bird_audio_data = get_all_audio_files_for_birds_parallel(
        bird_file_locations,
        root_directory=root_dir,
        save_file=save_file
    )

    # Show search results summary
    total_files = sum(data.get('wav_and_cbin_count', 0) for data in bird_audio_data.values())
    logging.info(f"Audio search complete: {total_files} total audio files found across {len(bird_audio_data)} birds")

    # STEP 5: Enhanced deduplication (now using consistent filename)
    logging.info("Step 5: Running enhanced deduplication with datetime analysis...")
    main_enhanced_deduplication(
        input_file=save_file,  # Now uses the same file: "xfoster_audio_paths(new).json"
        output_file="xfoster_audio_data_deduplicated(new).json",
        report_file="xfoster_datetime_analysis_report(new).txt",
        stats_file="comprehensive_dedup_stats(new).json"
    )

    logging.info("=== XFOSTER ANALYSIS COMPLETED SUCCESSFULLY ===")
    logging.info(f"Raw audio data: {save_file}")
    logging.info(f"Deduplicated data: xfoster_audio_data_deduplicated(new).json")
    logging.info(f"Analysis report: xfoster_datetime_analysis_report(new).txt")
    logging.info(f"Statistics: comprehensive_dedup_stats(new).json")

    # Print quick summary
    print(f"\n=== QUICK SUMMARY ===")
    print(f"Total birds processed: {len(birds)}")
    print(f"Birds with file locations found: {len(bird_file_locations)}")
    print(f"Birds with audio files found: {len(bird_audio_data)}")
    print(f"Total audio files found: {total_files}")

    return bird_audio_data


if __name__ == "__main__":
    main_xfoster_analysis()

#
# def main():
#     root_dir = check_sys_for_macaw_root()
#     save_file = "xfoster_audio_paths.json"
#
#     birds = ['wh15wh94', 'wh14wh93', 'wh50wh90',
#          'wh49wh89',
#          'wh47wh87',
#          'wh48wh88',
#          'bu26wh39',
#          'bu25wh38',
#          'bu82wh76',
#          'rd39wh3',
#          'rd40wh4',
#          'bu9wh89',
#          'bu10wh90',
#          'bu7wh87',
#          'rd33wh42',
#          'rd25wh57',
#          'gr6gr5',
#          'ye30ye82',
#          'ye23ye43',
#          'rd42pu47',
#          'rd43pu46',
#          'rd50gr1',
#          'rd39gr23',
#          'bk45wh59',
#          'wh86or19',
#          'wh85or81',
#          'wh82or15',
#          'wh83or16',
#          'rd77wh77',
#          'bk24wh25',
#          'bk100wh89',
#          'bk17wh56',
#          'bk81wh20',
#          'bk79wh14',
#          'bk38bk39',
#          'bk4bk47',
#          'bk36bk37',
#          'bk83bk73',
#          'bk94bk87',
#          'bk67bk44',
#          'bk76bk63',
#          'bk75bk62',
#          'bk43bk70',
#          'bk13bk12',
#          'bk41bk14',
#          'pk61bk8',
#          'pk15bk43',
#          'pu57wh52',
#          'pu55wh32',
#          'ye92br4',
#          'ye93br5',
#          'ye91br6',
#          'bk34bk51',
#          'bk73bk71',
#          'bk1bk3',
#          'bk91wh31',
#          'bk61wh42',
#          'bk63wh43',
#          'pu1wh51',
#          'wh93pk62',
#          'wh91pk61',
#          'pk6bk65',
#          'pk37bk19',
#          'pk46bk46',
#          'bk40wh47',
#          'pu31br476',
#          'pu33br479',
#          'pk5bk39',
#          'pk2bk37',
#          'pu58br33',
#          'pu55br34',
#          'pu22wh17',
#          'pk1bk31',
#          'pk43bk33',
#          'pu51wh43',
#          'pu71wh42',
#          'pu39wh79',
#          'bk13wh63',
#          'bk72wh64',
#          'bk74wh76',
#          'bu34or18',
#          'pk24bu3',
#          'pk72bk90',
#          'pk85gr19',
#          'pk97rd22',
#          'pu11bk85',
#          'pu42wh35',
#          'pu91wh67',
#          'rd75wh72',
#          'wh71br49',
#          'wh88br85',
#          'ye1tut0',
#          'ye81br444',
#          'bk1bk3',
#          'bk34bk51',
#          'bk37wh86',
#          'bu10wh86',
#          'bu24wh84',
#          'bu7wh67',
#          'gr73gr72',
#          'gr99or87',
#          'or10bk88',
#          'pk100bk68',
#          'pk24bu3',
#          'pu23wh38',
#          'pu35wh17',
#          'rd81wh45',
#          'ye25']
#
#     bird_file_locations = get_file_locations_for_birds(birds, txt_file_path='../refs/MacawAllDirsByBird.txt',
#                                                        db_path='../refs/2026-01-15-db.sqlite3')
#     get_all_audio_files_for_birds(bird_file_locations, root_directory=root_dir, save_file=save_file)
#     main_enhanced_deduplication(input_file="xfoster_audio_paths(new).json",
#                                 output_file="xfoster_audio_data_deduplicated(new).json",
#                                 report_file="xfoster_datetime_analysis_report(new).txt",
#                                 stats_file="comprehensive_dedup_stats(new).json")
#
#
# if __name__ == "__main__":
#     main()