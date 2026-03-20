"""
After pulling all filepaths and deduplicating, you can use this process to get a list of the root directories
(that contain audio files) for each bird.
"""

import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os


def extract_parent_directories(bird_data):
    """Extract unique parent directories for a single bird."""
    bird_id, data = bird_data
    parent_dirs = set()

    # Get kept paths from duplicate details
    for duplicate in data.get('duplicate_details', []):
        kept_path = duplicate.get('kept_path', '')
        if kept_path:
            parent_dir = str(Path(kept_path).parent)
            parent_dirs.add(parent_dir)

    # Also check all paths in duplicate details for completeness
    for duplicate in data.get('duplicate_details', []):
        for path_info in duplicate.get('all_paths_info', []):
            filepath = path_info.get('filepath', '')
            if filepath:
                parent_dir = str(Path(filepath).parent)
                parent_dirs.add(parent_dir)

    return bird_id, sorted(list(parent_dirs))


def process_bird_data(json_file_path, output_file='bird_directories.json'):
    """Process bird data and extract parent directories for each bird."""

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Prepare data for parallel processing
    bird_items = list(data.items())

    # Process in parallel
    result_dict = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(extract_parent_directories, bird_items)

        for bird_id, directories in results:
            result_dict[bird_id] = directories

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)

    return result_dict


# Usage
if __name__ == "__main__":
    result = process_bird_data('comprehensive_dedup_stats(new).json')

    # Print results
    for bird_id, directories in result.items():
        print(f"{bird_id}: {len(directories)} unique directories")
        for directory in directories:
            print(f"  - {directory}")