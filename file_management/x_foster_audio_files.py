import os
import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def save_bird_audio_data_with_datetime(data: Dict[str, Any], save_file: str) -> None:
    """Save bird audio data to JSON file with datetime handling."""
    try:
        # Save to temporary file first
        temp_file = save_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder)

        # If successful, rename to final file
        import shutil
        shutil.move(temp_file, save_file)
        logging.info(f"Saved data to {save_file}")

    except Exception as e:
        logging.error(f"Error saving data to {save_file}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)

# You'll also need the load_bird_audio_data function from earlier:
def load_bird_audio_data(save_file):
    """Load previously saved bird audio data from JSON file"""
    if os.path.exists(save_file):
        try:
            with open(save_file, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded existing data from {save_file}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {save_file}: {e}")
            return {}
    else:
        logging.info(f"No existing save file found at {save_file}")
        return {}


def save_bird_audio_data(data: Dict[str, Any], save_file: str) -> None:
    """Save bird audio data to JSON file"""
    try:
        with open(save_file, 'w') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder)  # ← ADDED cls=DateTimeEncoder
        logging.info(f"Saved data to {save_file}")
    except Exception as e:
        logging.error(f"Error saving data to {save_file}: {e}")


def get_filename_from_path(filepath: str) -> str:
    """Extract just the filename from a full path"""
    return os.path.basename(filepath)


def get_directory_depth(filepath: str) -> int:
    """Count the number of directory levels in a path"""
    return len(os.path.dirname(filepath).split(os.sep))


def deduplicate_file_list(file_list: List[str]) -> Tuple[List[str], int]:
    """
    Remove duplicate files based on filename, keeping the one with shortest path.

    Args:
        file_list: List of file paths

    Returns:
        Tuple of (deduplicated_list, number_of_duplicates_removed)
    """
    if not file_list:
        return [], 0

    # Group files by filename
    filename_groups = defaultdict(list)
    for filepath in file_list:
        filename = get_filename_from_path(filepath)
        filename_groups[filename].append(filepath)

    deduplicated_files = []
    duplicates_removed = 0

    for filename, paths in filename_groups.items():
        if len(paths) == 1:
            # No duplicates for this filename
            deduplicated_files.append(paths[0])
        else:
            # Multiple files with same name - keep the one with shortest path
            shortest_path = min(paths, key=get_directory_depth)
            deduplicated_files.append(shortest_path)
            duplicates_removed += len(paths) - 1

            # Log the duplicates being removed
            logging.info(f"Found {len(paths)} copies of '{filename}', keeping: {shortest_path}")
            for path in paths:
                if path != shortest_path:
                    logging.debug(f"  Removing duplicate: {path}")

    return deduplicated_files, duplicates_removed


def deduplicate_bird_audio_data(bird_audio_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Deduplicate all file lists in the bird audio data structure.

    Args:
        bird_audio_data: Original data dictionary

    Returns:
        Tuple of (deduplicated_data, duplicate_stats)
    """
    # Fields that contain file lists to deduplicate
    file_list_fields = ['audio_files', 'wav_files', 'cbin_files', 'batch_files', 'audio_from_batch']

    # Fields that contain counts to update
    count_fields = {
        'audio_files': None,  # No direct count field
        'wav_files': 'wav_count',
        'cbin_files': 'cbin_count',
        'batch_files': 'batch_file_count',
        'audio_from_batch': 'batch_audio_count'
    }

    deduplicated_data = {}
    duplicate_stats = {}
    total_duplicates_removed = 0

    for bird_id, bird_data in bird_audio_data.items():
        logging.info(f"Processing bird: {bird_id}")

        # Copy the original data
        new_bird_data = bird_data.copy()
        bird_duplicates = {}

        # Process each file list field
        for field in file_list_fields:
            if field in bird_data and isinstance(bird_data[field], list):
                original_count = len(bird_data[field])

                # Deduplicate this field
                deduplicated_list, duplicates_removed = deduplicate_file_list(bird_data[field])

                # Update the data
                new_bird_data[field] = deduplicated_list

                # Update corresponding count field if it exists
                count_field = count_fields.get(field)
                if count_field and count_field in new_bird_data:
                    new_bird_data[count_field] = len(deduplicated_list)

                # Track statistics
                bird_duplicates[field] = {
                    'original_count': original_count,
                    'final_count': len(deduplicated_list),
                    'duplicates_removed': duplicates_removed
                }

                total_duplicates_removed += duplicates_removed

                if duplicates_removed > 0:
                    logging.info(f"  {field}: {original_count} → {len(deduplicated_list)} "
                                 f"({duplicates_removed} duplicates removed)")

        # Update combined counts that might be affected
        if 'wav_and_cbin_count' in new_bird_data:
            wav_count = new_bird_data.get('wav_count', 0)
            cbin_count = new_bird_data.get('cbin_count', 0)
            new_bird_data['wav_and_cbin_count'] = wav_count + cbin_count

        deduplicated_data[bird_id] = new_bird_data
        duplicate_stats[bird_id] = bird_duplicates

    # Add overall statistics
    duplicate_stats['_summary'] = {
        'total_duplicates_removed': total_duplicates_removed,
        'birds_processed': len(bird_audio_data)
    }

    logging.info(f"Deduplication complete. Total duplicates removed: {total_duplicates_removed}")

    return deduplicated_data, duplicate_stats

def main_deduplication():
    """Main function to run the deduplication process"""

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
    bird_audio_data = load_bird_audio_data(input_file)

    if not bird_audio_data:
        logging.error("No data loaded. Exiting.")
        return

    logging.info(f"Loaded data for {len(bird_audio_data)} birds")

    # Perform deduplication
    deduplicated_data, duplicate_stats = deduplicate_bird_audio_data(bird_audio_data)

    # Save deduplicated data
    output_file = "cross_fostered_bird_audio_data_deduplicated.json"
    save_bird_audio_data(deduplicated_data, output_file)

    # Save deduplication statistics
    stats_file = "deduplication_stats.json"
    save_bird_audio_data(duplicate_stats, stats_file)

    # Print summary
    total_removed = duplicate_stats['_summary']['total_duplicates_removed']
    print(f"\n=== DEDUPLICATION SUMMARY ===")
    print(f"Total duplicate files removed: {total_removed}")
    print(f"Birds processed: {len(bird_audio_data)}")
    print(f"Deduplicated data saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")

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

        bird_total = sum(field_stats['duplicates_removed'] for field_stats in stats.values())
        if bird_total > 0:
            print(f"  {bird_id}: {bird_total} duplicates removed")
            for field, field_stats in stats.items():
                if field_stats['duplicates_removed'] > 0:
                    print(f"    {field}: {field_stats['original_count']} → {field_stats['final_count']}")


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract datetime from birdsong filename formats.

    Common birdsong formats:
    - bird_id_DDMMYY_HHMMSS.wav (e.g., w26pk33_170114_103345.wav)
    - bird_id_DDMMYY_HHMMSS (without extension)
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]

    # Split by underscore to get components
    parts = name_without_ext.split('_')

    # Try the common birdsong format: bird_id_DDMMYY_HHMMSS
    if len(parts) >= 3:
        try:
            # Get the last two parts (should be date and time)
            date_part = parts[-2]  # DDMMYY
            time_part = parts[-1]  # HHMMSS

            # Check if they have the right length
            if len(date_part) == 6 and len(time_part) == 6:
                # Parse DDMMYY format
                day = int(date_part[:2])
                month = int(date_part[2:4])
                year = int(date_part[4:6])

                # Convert 2-digit year to 4-digit (assume 2000s for 00-30, 1900s for 31-99)
                if year <= 30:
                    year += 2000
                else:
                    year += 1900

                # Parse HHMMSS format
                hour = int(time_part[:2])
                minute = int(time_part[2:4])
                second = int(time_part[4:6])

                # Validate ranges
                if (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2030 and
                        0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    return datetime(year, month, day, hour, minute, second)

        except (ValueError, IndexError):
            pass

    # Fallback to original regex patterns for other formats
    patterns = [
        # YYYYMMDD_HHMMSS or YYYYMMDDHHMMSS
        r'(\d{4})(\d{2})(\d{2})_?(\d{2})(\d{2})(\d{2})',
        # YYYY-MM-DD_HH-MM-SS
        r'(\d{4})-(\d{2})-(\d{2})[_T](\d{2})-(\d{2})-(\d{2})',
        # YYYY_MM_DD_HH_MM_SS
        r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})',
        # YYYYMMDD (date only, assume midnight)
        r'(\d{4})(\d{2})(\d{2})(?![\d])',
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, name_without_ext)
        if match:
            try:
                groups = match.groups()

                if i == 0 or i == 2:  # YYYYMMDD_HHMMSS or YYYY_MM_DD_HH_MM_SS
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    hour = int(groups[3]) if len(groups) > 3 else 0
                    minute = int(groups[4]) if len(groups) > 4 else 0
                    second = int(groups[5]) if len(groups) > 5 else 0

                elif i == 1:  # YYYY-MM-DD_HH-MM-SS
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    hour, minute, second = int(groups[3]), int(groups[4]), int(groups[5])

                elif i == 3:  # YYYYMMDD (date only)
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    hour = minute = second = 0

                # Validate date components
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2030:
                    if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                        return datetime(year, month, day, hour, minute, second)

            except (ValueError, IndexError):
                continue

    # If no pattern matches, log it for debugging
    logging.debug(f"Could not extract datetime from filename: {filename}")
    return None

def get_filename_from_path(filepath: str) -> str:
    """Extract just the filename from a full path"""
    return os.path.basename(filepath)


def get_directory_depth(filepath: str) -> int:
    """Count the number of directory levels in a path"""
    return len(os.path.dirname(filepath).split(os.sep))


def analyze_file_info(filepath: str) -> Dict[str, Any]:
    """
    Extract comprehensive information about a file.

    Returns:
        Dictionary with filename, directory_depth, datetime, etc.
    """
    filename = get_filename_from_path(filepath)
    return {
        'filepath': filepath,
        'filename': filename,
        'directory_depth': get_directory_depth(filepath),
        'datetime': extract_datetime_from_filename(filename),
        'datetime_str': extract_datetime_from_filename(filename).isoformat() if extract_datetime_from_filename(
            filename) else None
    }


def deduplicate_file_list_with_datetime(file_list: List[str]) -> Tuple[List[str], List[Dict], int, List[Dict]]:
    """
    Remove duplicate files and extract datetime info.

    Args:
        file_list: List of file paths

    Returns:
        Tuple of (deduplicated_list, file_info_list, duplicates_removed, duplicate_details)
    """
    if not file_list:
        return [], [], 0, []  # Return 4 values including empty duplicate_details

    # Group files by filename and analyze each
    filename_groups = defaultdict(list)
    file_analyses = {}

    for filepath in file_list:
        filename = get_filename_from_path(filepath)
        analysis = analyze_file_info(filepath)
        file_analyses[filepath] = analysis
        filename_groups[filename].append(filepath)

    deduplicated_files = []
    file_info_list = []
    duplicates_removed = 0
    duplicate_details = []  # Make sure this is initialized

    for filename, paths in filename_groups.items():
        if len(paths) == 1:
            # No duplicates for this filename
            chosen_path = paths[0]
            deduplicated_files.append(chosen_path)
            file_info_list.append(file_analyses[chosen_path])
        else:
            # Multiple files with same name - keep the one with shortest path
            shortest_path = min(paths, key=get_directory_depth)
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

            # Log the duplicates being removed
            logging.info(f"Found {len(paths)} copies of '{filename}', keeping: {shortest_path}")
            for path in paths:
                if path != shortest_path:
                    logging.debug(f"  Removing duplicate: {path}")

    return deduplicated_files, file_info_list, duplicates_removed, duplicate_details


def analyze_datetime_distribution(file_info_list: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the temporal distribution of recordings.

    Args:
        file_info_list: List of file info dictionaries

    Returns:
        Dictionary with datetime analysis statistics
    """
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

    # Group by date for daily distribution
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


def deduplicate_bird_audio_data_enhanced(bird_audio_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Enhanced deduplication with datetime analysis.

    Args:
        bird_audio_data: Original data dictionary

    Returns:
        Tuple of (deduplicated_data, comprehensive_stats)
    """
    # Fields that contain file lists to deduplicate
    file_list_fields = ['audio_files', 'wav_files', 'cbin_files', 'batch_files', 'audio_from_batch']

    # Fields that contain counts to update
    count_fields = {
        'audio_files': None,
        'wav_files': 'wav_count',
        'cbin_files': 'cbin_count',
        'batch_files': 'batch_file_count',
        'audio_from_batch': 'batch_audio_count'
    }

    deduplicated_data = {}
    comprehensive_stats = {}
    total_duplicates_removed = 0
    all_duplicate_details = []

    for bird_id, bird_data in bird_audio_data.items():
        logging.info(f"Processing bird: {bird_id}")

        # Copy the original data
        new_bird_data = bird_data.copy()
        bird_stats = {
            'deduplication': {},
            'datetime_analysis': {},
            'duplicate_details': []
        }

        # Process each file list field
        for field in file_list_fields:
            if field in bird_data and isinstance(bird_data[field], list):
                original_count = len(bird_data[field])

                # Enhanced deduplication with datetime analysis
                deduplicated_list, file_info_list, duplicates_removed, duplicate_details = \
                    deduplicate_file_list_with_datetime(bird_data[field])

                # Update the data
                new_bird_data[field] = deduplicated_list

                # Add datetime information as a new field
                new_bird_data[f'{field}_info'] = file_info_list

                # Update corresponding count field if it exists
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

                total_duplicates_removed += duplicates_removed
                all_duplicate_details.extend(duplicate_details)

                if duplicates_removed > 0:
                    logging.info(f"  {field}: {original_count} → {len(deduplicated_list)} "
                                 f"({duplicates_removed} duplicates removed)")

                # Log datetime coverage
                coverage = datetime_stats['datetime_coverage']
                logging.info(f"  {field} datetime coverage: {coverage:.1%} "
                             f"({datetime_stats['files_with_datetime']}/{datetime_stats['total_files']})")

        # Update combined counts that might be affected
        if 'wav_and_cbin_count' in new_bird_data:
            wav_count = new_bird_data.get('wav_count', 0)
            cbin_count = new_bird_data.get('cbin_count', 0)
            new_bird_data['wav_and_cbin_count'] = wav_count + cbin_count

        # Add overall bird datetime analysis
        all_files_info = []
        for field in file_list_fields:
            if f'{field}_info' in new_bird_data:
                all_files_info.extend(new_bird_data[f'{field}_info'])

        bird_stats['overall_datetime_analysis'] = analyze_datetime_distribution(all_files_info)

        deduplicated_data[bird_id] = new_bird_data
        comprehensive_stats[bird_id] = bird_stats

    # Add overall summary statistics
    comprehensive_stats['_summary'] = {
        'total_duplicates_removed': total_duplicates_removed,
        'birds_processed': len(bird_audio_data),
        'all_duplicate_details': all_duplicate_details,
        'processing_timestamp': datetime.now().isoformat()
    }

    logging.info(f"Enhanced deduplication complete. Total duplicates removed: {total_duplicates_removed}")

    return deduplicated_data, comprehensive_stats


def create_datetime_summary_report(comprehensive_stats: Dict[str, Any],
                                   output_file: str = "datetime_analysis_report.txt"):
    """
    Create a human-readable report of the datetime analysis.
    """
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


def main_enhanced_deduplication():
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
    input_file = "cross_fostered_bird_audio_data.json"
    logging.info(f"Loading data from {input_file}")

    bird_audio_data = load_bird_audio_data(input_file)

    if not bird_audio_data:
        logging.error("No data loaded. Exiting.")
        return

    logging.info(f"Loaded data for {len(bird_audio_data)} birds")

    # Perform enhanced deduplication with datetime analysis
    logging.info("Starting enhanced deduplication with datetime analysis...")
    deduplicated_data, comprehensive_stats = deduplicate_bird_audio_data_enhanced(bird_audio_data)

    # Save with better error handling
    output_file = "cross_fostered_bird_audio_data_deduplicated_enhanced.json"

    try:
        # Save to temporary file first
        temp_file = output_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_data, f, indent=2, cls=DateTimeEncoder)

        # If successful, rename to final file
        import shutil
        shutil.move(temp_file, output_file)
        logging.info(f"Successfully saved deduplicated data to {output_file}")

    except Exception as e:
        logging.error(f"Error saving deduplicated data: {e}")
        return

    # Save comprehensive statistics with error handling
    stats_file = "comprehensive_deduplication_stats.json"
    try:
        temp_stats_file = stats_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_data, f, indent=2, cls=DateTimeEncoder)

        shutil.move(temp_stats_file, stats_file)
        logging.info(f"Successfully saved stats to {stats_file}")

    except Exception as e:
        logging.error(f"Error saving stats: {e}")

    # Create human-readable report
    report_file = "datetime_analysis_report.txt"
    create_datetime_summary_report(comprehensive_stats, report_file)

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


if __name__ == "__main__":
    main_enhanced_deduplication()

