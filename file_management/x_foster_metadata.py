import os
import sys
import signal
import json
import csv
import re
import logging
import shutil
from datetime import datetime
from pathlib import Path

from tools.system_utils import check_sys_for_macaw_root
from tools.dbquery import *


# Configuration flags
REUSE_AUDIO_DATA = False  # Set to True to reuse existing cross_fostered_bird_audio_data.json, False to replace it
LOG_TO_CONSOLE = True  # Set to True to also print to console, False for log file only

# Generate timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# Cache for bird name variations
_bird_name_variation_cache = {}


def setup_logging():
    """Set up logging configuration for this run"""

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
    logger.info(f"Reuse audio data: {REUSE_AUDIO_DATA}")

    return logger


def create_backup_directory():
    """Create backups directory if it doesn't exist"""
    backup_dir = "backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        logging.info(f"Created backup directory: {backup_dir}")
    return backup_dir


def backup_file_if_exists(filepath, backup_dir):
    """
    Create a backup of a file if it exists

    Args:
        filepath: Path to the file to backup
        backup_dir: Directory to store backups

    Returns:
        bool: True if backup was created, False if file didn't exist
    """
    if os.path.exists(filepath):
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
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


def generate_bird_name_variations_cached(bird_name):
    """Cached version of generate_bird_name_variations"""
    if bird_name not in _bird_name_variation_cache:
        _bird_name_variation_cache[bird_name] = generate_bird_name_variations(bird_name)
    return _bird_name_variation_cache[bird_name]


def get_birds_with_genetic_parents(db_path='2026-01-15-db.sqlite3'):
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


def get_genetic_parents_for_birds(bird_list, db_path='2026-01-15-db.sqlite3'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    genetic_parents_data = []

    for uuid, bird_name in bird_list:
        # Get all genetic parents for this bird
        query = """
                SELECT bgp.genparent_id,
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
                  AND bgp.genparent_id != 'NULL' 
                """

        result = cur.execute(query, (uuid,))
        parents = result.fetchall()

        genetic_parents_data.append({
            'child_uuid': uuid,
            'child_name': bird_name,
            'parents': parents
        })

    con.close()
    return genetic_parents_data


def find_cross_fostered_birds(bird_list, db_path='2026-01-15-db.sqlite3'):
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
                          AND bgp.genparent_id != 'NULL' 
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
                       AND bp.parent_id != 'NULL' 
                     """

        genetic_result = cur.execute(genetic_query, (uuid,))
        genetic_parents = genetic_result.fetchall()

        nest_result = cur.execute(nest_query, (uuid,))
        nest_parents = nest_result.fetchall()

        # Only proceed if we have both genetic and nest parent data
        if genetic_parents and nest_parents:
            # Get parent UUIDs for comparison
            genetic_parent_uuids = set(p[0] for p in genetic_parents)
            nest_parent_uuids = set(p[0] for p in nest_parents)

            # Check if parents are different (cross-fostered)
            if genetic_parent_uuids != nest_parent_uuids:
                # Organize by sex for clearer output
                genetic_father = next((p for p in genetic_parents if p[2] == 'M'), None)
                genetic_mother = next((p for p in genetic_parents if p[2] == 'F'), None)
                nest_father = next((p for p in nest_parents if p[2] == 'M'), None)
                nest_mother = next((p for p in nest_parents if p[2] == 'F'), None)

                cross_fostered_birds.append({
                    'child_uuid': uuid,
                    'child_name': bird_name,
                    'genetic_father': genetic_father[1] if genetic_father else 'Unknown',
                    'genetic_mother': genetic_mother[1] if genetic_mother else 'Unknown',
                    'nest_father': nest_father[1] if nest_father else 'Unknown',
                    'nest_mother': nest_mother[1] if nest_mother else 'Unknown',
                    'genetic_parents': genetic_parents,
                    'nest_parents': nest_parents
                })

    con.close()
    return cross_fostered_birds


def find_cross_fostered_birds_father_only(bird_list, db_path='2026-01-15-db.sqlite3'):
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
                          AND bgp.genparent_id != 'NULL' 
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
                       AND bp.parent_id != 'NULL' 
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


def get_file_locations_for_birds(cross_fostered_birds, txt_file_path="MacawAllDirsByBird.txt",
                                 db_path='2026-01-15-db.sqlite3'):
    """
    Parse the text file and return a dictionary mapping bird names to their file locations
    Only include male (M) or unknown (U) sex birds
    Now handles multiple variations of bird name formats for matching
    """
    bird_files = {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # First, get the bird names from cross_fostered_birds and filter by sex
    target_birds = {}
    target_bird_variations = {}  # Map variations back to original names

    for bird in cross_fostered_birds:
        bird_name = bird['child_name']
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
                # Generate all variations for this bird name and map them back
                variations = generate_bird_name_variations(bird_name)
                for variation in variations:
                    target_bird_variations[variation.lower()] = bird_name

        except Exception as e:
            logging.error(f"Error checking gender for {bird_name}: {e}")
            # If error, assume unknown and include
            target_birds[bird_name] = 'U'
            # Generate all variations for this bird name and map them back
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
                            # Filter out empty strings that might result from splitting
                            file_paths = [path for path in file_path_string.split('//') if path.strip()]

                            # Add to dictionary using the original target bird name as key
                            # If we already have entries for this bird, extend the list
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


def generate_bird_name_variations(bird_name):
    """
    Generate all possible variations of a bird name for matching against directory names.
    Input can be in any format - this function will standardize it first, then generate all variations.

    Args:
        bird_name (str): Bird name in any format (e.g., 'orange85red60', 'o85rd60', 'or85r60')

    Returns:
        set: All possible variations of the bird name
    """
    if not bird_name or not isinstance(bird_name, str):
        return {bird_name}

    # Color mappings - each standardized color maps to all its possible representations
    color_variations = {
        'bk': ['bk', 'b', 'k', 'black'],
        'wh': ['wh', 'w', 'white'],
        'gr': ['gr', 'g', 'green'],
        'ye': ['ye', 'y', 'yw', 'yellow'],  # Include both 'ye' and 'yw' for yellow
        'rd': ['rd', 'r', 'red'],
        'pk': ['pk', 'pink'],
        'or': ['or', 'o', 'orange'],
        'pu': ['pu', 'purple'],
        'br': ['br', 'brown'],
        'bl': ['bl', 'bu', 'blue'],
        'nb': ['nb', 'noband']
    }

    # Reverse mapping for standardization (all variations -> standardized form)
    standardization_map = {}
    for standard, variations in color_variations.items():
        for variation in variations:
            standardization_map[variation.lower()] = standard

    def standardize_color(color_str):
        """Convert any color string to standardized two-character format"""
        if not color_str:
            return ''

        color_str = color_str.lower().strip()

        # Direct lookup in standardization map
        if color_str in standardization_map:
            return standardization_map[color_str]

        # If not found, try partial matching for full color names
        for full_name, standard in standardization_map.items():
            if len(full_name) > 2 and full_name.startswith(color_str):
                return standard

        # Return as-is if we can't map it
        return color_str

    # Parse the input bird name using regex to separate colors and numbers
    parts = re.findall(r'([a-zA-Z]+)|(\d+)', bird_name.lower())

    if not parts:
        return {bird_name}

    # Extract and standardize components
    standardized_colors = []
    numbers = []

    for letter_part, digit_part in parts:
        if letter_part:
            standardized_colors.append(standardize_color(letter_part))
        if digit_part:
            numbers.append(digit_part)

    variations = set()

    # Generate all combinations based on the structure
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
        # This should never happen...
        # One color, two numbers: co##
        color_vars = color_variations.get(standardized_colors[0], [standardized_colors[0]])
        for color_var in color_vars:
            variations.add(f"{color_var}{numbers[0]}{numbers[1]}")

    # Also add the original input name in case it doesn't match our patterns
    variations.add(bird_name.lower())

    return variations


def bird_name_matches_directory(bird_name, directory_part):
    """
    Check if any variation of the bird name matches the directory part.

    Args:
        bird_name (str): Standardized bird name
        directory_part (str): Directory name part to check against

    Returns:
        bool: True if any variation matches
    """
    variations = generate_bird_name_variations(bird_name)
    directory_lower = directory_part.lower()

    return any(variation.lower() in directory_lower for variation in variations)


def normalize_path(path_str):
    """Normalize path for cross-platform compatibility"""
    # First, replace all backslashes with forward slashes
    cleaned = path_str.replace('\\', '/')
    # Then let pathlib handle OS-appropriate conversion
    return str(Path(cleaned).resolve())


def get_all_audio_files_for_birds(bird_file_locations, root_directory, save_file="bird_audio_data.json"):
    """
    For each bird, find audio files in the immediate directories containing their file paths.
    Saves progress after each bird is processed and handles interruptions gracefully.
    """

    # Load existing data if available and indicated
    if REUSE_AUDIO_DATA:
        bird_audio_files = load_bird_audio_data(save_file)
    else:
        bird_audio_files = {}

    # Set up signal handler for graceful interruption
    def signal_handler(sig, frame):
        logging.info(f"\nInterrupted! Saving current progress to {save_file}...")
        save_bird_audio_data(bird_audio_files, save_file)
        logging.info("Progress saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Get list of birds to process (skip already completed ones, i.e. birds in dictionary with non-empty lists)
    bird_audio_files_not_empty = [bird for bird, list in bird_audio_files.items() if list is not []]
    birds_to_process = [bird for bird in bird_file_locations.keys() if bird not in bird_audio_files_not_empty]
    birds_completed = len(bird_file_locations) - len(birds_to_process)

    logging.info(f"Processing {len(birds_to_process)} birds ({birds_completed} already completed)")

    def is_date_file(filename, bird_name):
        """Check if file matches any variation of birdname.date.* pattern"""
        file_parts = filename.split('.')
        if len(file_parts) < 3:  # Need at least birdname.date.something
            return False

        # Check if second part is a date (8 digits)
        if not (len(file_parts[1]) == 8 and file_parts[1].isdigit()):
            return False

        # Check if first part matches any variation of the bird name
        variations = generate_bird_name_variations(bird_name)
        return file_parts[0].lower() in {var.lower() for var in variations}

    screening_directories = ['/Volumes/users/public/screening/', '/Volumes/users/public/adult_screening/',
                             '/Volumes/users/public/from_egret/egret/screening/',
                             '/Volumes/users/public/from_stork/stork/screening/']

    try:
        for i, bird_name in enumerate(birds_to_process):

            # Find unique directories that contain the files
            bird_directories = set()

            logging.info(f"Searching {bird_name} in screening files...")

            for dir in screening_directories:
                sub_dirs = os.listdir(dir)
                for sub_dir in sub_dirs:
                    if bird_name in sub_dir:
                        bird_directories.add(str(Path(sub_dir)))

            file_paths = bird_file_locations[bird_name]
            logging.info(f"Processing {bird_name} ({i + 1}/{len(birds_to_process)})...")

            for file_path in file_paths:
                try:
                    # Join with root directory and get the directory containing the file
                    full_file_path = os.path.join(root_directory, file_path)
                    normalized_path = normalize_path(full_file_path)
                    bird_directories.add(str(Path(normalized_path).parent))
                except Exception as e:
                    logging.error(f"  Error processing file path {file_path}: {e}")
                    continue

            # For each unique directory, list files in that directory only (no subdirectories)
            all_audio_files = []
            wav_files = []
            cbin_files = []
            batch_files = []
            audio_from_batch_files = []

            for bird_dir in bird_directories:
                try:
                    if os.path.exists(bird_dir):
                        logging.info(f"  Searching directory: {bird_dir}")
                        # Only list files in the immediate directory
                        for file in os.listdir(bird_dir):
                            try:
                                full_path = os.path.join(bird_dir, file)
                                # Skip directories
                                if os.path.isfile(full_path):
                                    # Check for .wav files (including date files which are wav without extension)
                                    if file.lower().endswith('.wav') or is_date_file(file, bird_name):
                                        wav_files.append(full_path)
                                        all_audio_files.append(full_path)
                                    elif file.lower().endswith('.cbin'):
                                        cbin_files.append(full_path)
                                        all_audio_files.append(full_path)

                                    if file.lower().endswith('.keep'):
                                        batch_file = str(Path(full_path))
                                        batch_files.append(batch_file)  # keep track of batch files
                                        with open(batch_file, 'r') as f:
                                            lines = f.readlines()
                                            for line in lines:
                                                audio_from_batch_files.append(str(Path(line)))  # seperately track audio in batch files
                                            f.close()

                            except Exception as e:
                                logging.error(f"    Error processing file {file}: {e}")
                                continue
                    else:
                        logging.warning(f"  Directory not found: {bird_dir}")
                except Exception as e:
                    logging.error(f"  Error accessing {bird_dir}: {e}")
                    continue

            # Remove duplicates and sort
            all_audio_files = sorted(list(set(all_audio_files)))
            wav_files = sorted(list(set(wav_files)))
            cbin_files = sorted(list(set(cbin_files)))
            batch_files = sorted(list(set(batch_files)))
            audio_from_batch_files = sorted(list(set(audio_from_batch_files)))

            # Store results for this bird
            bird_audio_files[bird_name] = {
                'directories_searched': list(bird_directories),
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

            logging.info(f"  Found {len(all_audio_files)} audio files:")
            logging.info(f"    {len(wav_files)} .wav files (including date files)")
            logging.info(f"    {len(cbin_files)} .cbin files")
            logging.info(f"    {len(batch_files)} batch file, listing {len(audio_from_batch_files)} song files")

            # Save progress after each bird
            try:
                save_bird_audio_data(bird_audio_files, save_file)
            except Exception as e:
                logging.warning(f"  Warning: Could not save progress: {e}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.info("Saving current progress...")
        save_bird_audio_data(bird_audio_files, save_file)
        raise

    logging.info(f"\nCompleted processing all birds! Final data saved to {save_file}")
    return bird_audio_files


def save_bird_audio_data(bird_audio_data, save_file):
    """Save the bird audio data to a JSON file"""
    try:
        with open(save_file, 'w') as f:
            json.dump(bird_audio_data, f, indent=2)
        logging.info(f"Data saved to {save_file}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


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


def run_audio_file_search(bird_file_locations, root_directory, save_file="bird_audio_data.json"):
    """
    Convenience function to run the audio file search with proper error handling
    """
    try:
        bird_audio_data = get_all_audio_files_for_birds(
            bird_file_locations,
            root_directory,
            save_file
        )

        # Display summary
        logging.info("\n" + "=" * 70)
        logging.info("SUMMARY OF AUDIO FILES FOR CROSS-FOSTERED BIRDS")
        logging.info("=" * 70)

        total_wav = 0
        total_cbin = 0
        total_files = 0

        for bird_name, data in bird_audio_data.items():
            logging.info(
                f"{bird_name}: {data['total_count']} files ({data['wav_count']} .wav, {data['cbin_count']} .cbin)"
            )
            total_wav += data['wav_count']
            total_cbin += data['cbin_count']
            total_files += data['total_count']

        logging.info(f"\nGRAND TOTAL: {total_files} files")
        logging.info(f"  {total_wav} .wav files (including date files)")
        logging.info(f"  {total_cbin} .cbin files")

        return bird_audio_data

    except KeyboardInterrupt:
        logging.info("\nProcess interrupted by user")
        return None
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return None


def resume_from_bird(bird_file_locations, root_directory, start_bird, save_file="bird_audio_data.json"):
    """
    Resume processing from a specific bird (useful for manual restarts)
    """
    # Load existing data
    bird_audio_data = load_bird_audio_data(save_file)

    # Create a new bird_file_locations dict starting from the specified bird
    bird_names = list(bird_file_locations.keys())
    if start_bird in bird_names:
        start_index = bird_names.index(start_bird)
        remaining_birds = {name: bird_file_locations[name] for name in bird_names[start_index:]}
        logging.info(f"Resuming from bird {start_bird} ({len(remaining_birds)} birds remaining)")
        return run_audio_file_search(remaining_birds, root_directory, save_file)
    else:
        logging.error(f"Bird {start_bird} not found in bird_file_locations")
        return bird_audio_data


def create_cross_fostered_birds_csv(cross_fostered_fathers, bird_audio_data,
                                    output_file="cross_fostered_birds_summary.csv", db_path='2026-01-15-db.sqlite3'):
    """
    Create a CSV file with cross-fostered bird information
    """
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
                    total_songs = audio_data['total_count']
                    directories = '; '.join(audio_data['directories_searched'])
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
        print(f"Error creating CSV file: {e}")


def create_cross_fostered_notes_csv(cross_fostered_fathers, output_file="cross_fostered_birds_notes.csv",
                                    db_path='2026-01-15-db.sqlite3'):
    """
    Create a CSV file with notes information for cross-fostered birds
    """
    # Backup existing file if it exists
    backup_dir = create_backup_directory()
    backup_file_if_exists(output_file, backup_dir)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    headers = ['Bird Name', 'Sex', 'DOB', 'Has Notes', 'Notes']

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for bird in cross_fostered_fathers:
                bird_name = bird['child_name']

                try:
                    uuid, exists = getUUIDfromBands(cur, bird_name)

                    if exists:
                        # Get gender
                        gender_query = "SELECT sex FROM birds_animal WHERE uuid=?"
                        gender_result = cur.execute(gender_query, (uuid,))
                        gender_row = gender_result.fetchone()
                        gender = gender_row[0] if gender_row and gender_row[0] else 'Unknown'

                        # Get birth date
                        birth_query = "SELECT hatch_date FROM birds_animal WHERE uuid=?"
                        birth_result = cur.execute(birth_query, (uuid,))
                        birth_row = birth_result.fetchone()
                        birth_date = birth_row[0] if birth_row and birth_row[0] else 'Unknown'

                        # Check if bird has notes
                        query = "SELECT notes FROM birds_animal WHERE uuid=?"
                        res = cur.execute(query, (uuid,))
                        notes_result = res.fetchone()

                        if notes_result and notes_result[0] and notes_result[0].strip():
                            has_notes = True
                            notes_text = notes_result[0].strip()
                        else:
                            has_notes = False
                            notes_text = ""
                    else:
                        gender = 'Unknown'
                        birth_date = 'Unknown'
                        has_notes = False
                        notes_text = "Bird not found in database"

                except Exception as e:
                    logging.error(f"Error processing {bird_name}: {e}")
                    gender = 'Error'
                    birth_date = 'Error'
                    has_notes = False
                    notes_text = f"Error: {e}"

                row = [bird_name, gender, birth_date, has_notes, notes_text]
                writer.writerow(row)

        con.close()
        logging.info(f"Notes CSV file created successfully: {output_file}")

    except Exception as e:
        con.close()
        logging.error(f"Error creating notes CSV file: {e}")


def get_home_reared_offspring_for_parents(cross_fostered_fathers, db_path='2026-01-15-db.sqlite3'):
    """
    Get all home-reared (non-cross-fostered) offspring for parents of cross-fostered birds
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Collect all unique parent UUIDs from cross-fostered birds
    parent_uuids = set()

    for bird in cross_fostered_fathers:
        bird_name = bird['child_name']

        try:
            uuid, exists = getUUIDfromBands(cur, bird_name)
            if exists:
                # Get genetic parents
                genetic_query = """
                                SELECT DISTINCT bgp.genparent_id
                                FROM birds_geneticparent bgp
                                WHERE bgp.genchild_id = ?
                                  AND bgp.genparent_id IS NOT NULL
                                  AND bgp.genparent_id != 'NULL' 
                                """

                # Get nest parents
                nest_query = """
                             SELECT DISTINCT bp.parent_id
                             FROM birds_parent bp
                             WHERE bp.child_id = ?
                               AND bp.parent_id IS NOT NULL
                               AND bp.parent_id != 'NULL' 
                             """

                genetic_result = cur.execute(genetic_query, (uuid,))
                genetic_parents = genetic_result.fetchall()

                nest_result = cur.execute(nest_query, (uuid,))
                nest_parents = nest_result.fetchall()

                # Add all parent UUIDs to our set
                for parent in genetic_parents:
                    parent_uuids.add(parent[0])
                for parent in nest_parents:
                    parent_uuids.add(parent[0])

        except Exception as e:
            logging.error(f"Error processing {bird_name}: {e}")
            continue

    logging.info(f"Found {len(parent_uuids)} unique parent UUIDs")

    # For each parent, get all their offspring
    home_reared_offspring = []

    for parent_uuid in parent_uuids:
        try:
            # Get all genetic offspring of this parent
            genetic_offspring_query = """
                                      SELECT DISTINCT bgp.genchild_id,
                                                      bc1.abbrv || ba.band_number ||
                                                      CASE
                                                          WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                                              THEN bc2.abbrv || ba.band_number2
                                                          ELSE ''
                                                          END as child_name
                                      FROM birds_geneticparent bgp
                                               LEFT JOIN birds_animal ba ON bgp.genchild_id = ba.uuid
                                               LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                                               LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                                      WHERE bgp.genparent_id = ?
                                        AND bgp.genchild_id IS NOT NULL
                                        AND bgp.genchild_id != 'NULL' 
                                      """

            result = cur.execute(genetic_offspring_query, (parent_uuid,))
            genetic_offspring = result.fetchall()

            # Get all nest offspring of this parent
            nest_offspring_query = """
                                   SELECT DISTINCT bp.child_id,
                                                   bc1.abbrv || ba.band_number ||
                                                   CASE
                                                       WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                                           THEN bc2.abbrv || ba.band_number2
                                                       ELSE ''
                                                       END as child_name
                                   FROM birds_parent bp
                                            LEFT JOIN birds_animal ba ON bp.child_id = ba.uuid
                                            LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                                            LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                                   WHERE bp.parent_id = ?
                                     AND bp.child_id IS NOT NULL
                                     AND bp.child_id != 'NULL' 
                                   """

            result = cur.execute(nest_offspring_query, (parent_uuid,))
            nest_offspring = result.fetchall()

            # Combine all offspring for this parent
            all_offspring = set()
            for child_uuid, child_name in genetic_offspring:
                all_offspring.add((child_uuid, child_name))
            for child_uuid, child_name in nest_offspring:
                all_offspring.add((child_uuid, child_name))

            # For each offspring, check if they are cross-fostered
            for child_uuid, child_name in all_offspring:
                try:
                    # Get genetic parents for this child
                    genetic_parents_query = """
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
                                              AND bgp.genparent_id != 'NULL' 
                                            """

                    # Get nest parents for this child
                    nest_parents_query = """
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
                                           AND bp.parent_id != 'NULL' 
                                         """

                    genetic_result = cur.execute(genetic_parents_query, (child_uuid,))
                    genetic_parents = genetic_result.fetchall()

                    nest_result = cur.execute(nest_parents_query, (child_uuid,))
                    nest_parents = nest_result.fetchall()

                    # Extract fathers for comparison
                    genetic_father = next((p for p in genetic_parents if p[2] == 'M'), None)
                    nest_father = next((p for p in nest_parents if p[2] == 'M'), None)

                    # Check if this child is NOT cross-fostered (genetic father == nest father)
                    is_home_reared = False

                    if genetic_father and nest_father:
                        # Both fathers exist - check if they're the same
                        if genetic_father[0] == nest_father[0]:  # Compare UUIDs
                            is_home_reared = True
                    elif not genetic_father and not nest_father:
                        # No father data for either - could be home-reared but unclear
                        is_home_reared = True
                    elif genetic_father and not nest_father:
                        # Has genetic father but no nest father - ambiguous case
                        is_home_reared = False
                    elif not genetic_father and nest_father:
                        # Has nest father but no genetic father - ambiguous case
                        is_home_reared = False

                    # Only include if home-reared
                    if is_home_reared:
                        # Get all parent information for output
                        genetic_father_name = genetic_father[1] if genetic_father else 'Unknown'
                        genetic_mother = next((p for p in genetic_parents if p[2] == 'F'), None)
                        genetic_mother_name = genetic_mother[1] if genetic_mother else 'Unknown'

                        nest_father_name = nest_father[1] if nest_father else 'Unknown'
                        nest_mother = next((p for p in nest_parents if p[2] == 'F'), None)
                        nest_mother_name = nest_mother[1] if nest_mother else 'Unknown'

                        home_reared_offspring.append({
                            'child_uuid': child_uuid,
                            'child_name': child_name,
                            'genetic_father': genetic_father_name,
                            'genetic_mother': genetic_mother_name,
                            'nest_father': nest_father_name,
                            'nest_mother': nest_mother_name
                        })

                except Exception as e:
                    logging.error(f"Error processing offspring {child_name}: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error processing parent {parent_uuid}: {e}")
            continue

    con.close()

    # Remove duplicates based on child_uuid
    seen_uuids = set()
    unique_home_reared = []
    for offspring in home_reared_offspring:
        if offspring['child_uuid'] not in seen_uuids:
            seen_uuids.add(offspring['child_uuid'])
            unique_home_reared.append(offspring)

    logging.info(f"Found {len(unique_home_reared)} home-reared offspring")
    return unique_home_reared


def create_home_reared_offspring_csv(cross_fostered_fathers, output_file="home_reared_offspring.csv",
                                     db_path='2026-01-15-db.sqlite3'):
    """
    Create a CSV file with home-reared offspring information
    """
    # Backup existing file if it exists
    backup_dir = create_backup_directory()
    backup_file_if_exists(output_file, backup_dir)

    # Get home-reared offspring data
    home_reared_offspring = get_home_reared_offspring_for_parents(cross_fostered_fathers, db_path)

    if not home_reared_offspring:
        logging.info("No home-reared offspring found")
        return

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    headers = ['Bird Name', 'Sex', 'DOB', 'Genetic Father', 'Genetic Mother', 'Nest Father', 'Nest Mother']

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for offspring in home_reared_offspring:
                bird_name = offspring['child_name']

                try:
                    uuid, exists = getUUIDfromBands(cur, bird_name)
                    if exists:
                        # Get gender
                        gender_query = "SELECT sex FROM birds_animal WHERE uuid=?"
                        gender_result = cur.execute(gender_query, (uuid,))
                        gender_row = gender_result.fetchone()
                        gender = gender_row[0] if gender_row and gender_row[0] else 'Unknown'

                        # Get birth date
                        birth_query = "SELECT hatch_date FROM birds_animal WHERE uuid=?"
                        birth_result = cur.execute(birth_query, (uuid,))
                        birth_row = birth_result.fetchone()
                        birth_date = birth_row[0] if birth_row and birth_row[0] else 'Unknown'
                    else:
                        gender = 'Unknown'
                        birth_date = 'Unknown'
                except Exception as e:
                    logging.error(f"Error getting gender/birth date for {bird_name}: {e}")
                    gender = 'Error'
                    birth_date = 'Error'

                row = [
                    bird_name,
                    gender,
                    birth_date,
                    offspring['genetic_father'],
                    offspring['genetic_mother'],
                    offspring['nest_father'],
                    offspring['nest_mother']
                ]
                writer.writerow(row)

        con.close()
        logging.info(f"Home-reared offspring CSV created successfully: {output_file}")
        logging.info(f"Exported data for {len(home_reared_offspring)} home-reared birds")

    except Exception as e:
        con.close()
        logging.error(f"Error creating home-reared offspring CSV: {e}")


# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set up logging for this run
    logger = setup_logging()

    # Create backup directory
    backup_dir = create_backup_directory()

    try:
        # Get birds with genetic parents
        logger.info("Getting birds with genetic parents...")
        xfoster_birds = get_birds_with_genetic_parents()

        # Get genetic parent information
        logger.info("Getting genetic parent information...")
        genetic_parent_info = get_genetic_parents_for_birds(xfoster_birds)

        # Find cross-fostered birds (father only)
        logger.info("Finding cross-fostered birds (father comparison only)...")
        cross_fostered_fathers = find_cross_fostered_birds_father_only(xfoster_birds)

        logger.info(f"Found {len(cross_fostered_fathers)} birds with different genetic vs nest fathers:")
        logger.info("=" * 60)

        for bird in cross_fostered_fathers:
            logger.info(f"Bird: {bird['child_name']}")
            logger.info(f"  Genetic father: {bird['genetic_father']}")
            logger.info(f"  Nest father:    {bird['nest_father']}")
            logger.info(f"  Genetic mother: {bird['genetic_mother']}")
            logger.info(f"  Nest mother:    {bird['nest_mother']}")

        if cross_fostered_fathers:
            logger.info(f"Summary: {len(cross_fostered_fathers)} birds have different genetic vs nest fathers")
            logger.info(f"Father mismatch rate: {len(cross_fostered_fathers) / len(xfoster_birds) * 100:.1f}%")
        else:
            logger.info("No birds found with different genetic vs nest fathers")

        # Get file locations for cross-fostered birds
        logger.info("Getting file locations for cross-fostered birds...")
        bird_file_locations = get_file_locations_for_birds(cross_fostered_fathers)

        logger.info("File locations for cross-fostered birds:")
        logger.info("=" * 50)

        # Show first few birds as examples
        for i, (bird_name, file_paths) in enumerate(bird_file_locations.items()):
            if i < 3:  # Show first 3 birds as examples
                logger.info(f"{bird_name} ({len(file_paths)} files):")
                for j, file_path in enumerate(file_paths):
                    if j < 5:  # Show first 5 files per bird
                        logger.info(f"  {file_path}")
                    elif j == 5:
                        logger.info(f"  ... and {len(file_paths) - 5} more files")
                        break
            elif i == 3:
                logger.info("... (showing first 3 birds)")
                break

        logger.info(f"\nSummary:")
        logger.info(f"Cross-fostered birds: {len(cross_fostered_fathers)}")
        logger.info(f"Birds with file locations: {len(bird_file_locations)}")
        logger.info(f"Total file entries found: {sum(len(files) for files in bird_file_locations.values())}")

        # Show distribution of file counts
        file_counts = [len(files) for files in bird_file_locations.values()]
        if file_counts:
            logger.info(
                f"Files per bird - Min: {min(file_counts)}, Max: {max(file_counts)}, Average: {sum(file_counts) / len(file_counts):.1f}"
            )

        # Create the notes CSV for cross-fostered birds
        logger.info("Creating cross-fostered birds notes CSV...")
        create_cross_fostered_notes_csv(cross_fostered_fathers, "cross_fostered_birds_notes.csv")

        # Create the home-reared offspring CSV
        logger.info("Creating home-reared offspring CSV...")
        create_home_reared_offspring_csv(cross_fostered_fathers, "home_reared_offspring.csv")

        # Show some summary statistics
        logger.info("\nSummary Statistics:")
        logger.info("-" * 40)

        # Count birds with notes
        con = sqlite3.connect("2026-01-15-db.sqlite3")
        cur = con.cursor()

        birds_with_notes = 0
        for bird in cross_fostered_fathers:
            try:
                uuid, exists = getUUIDfromBands(cur, bird['child_name'])
                if exists:
                    query = "SELECT notes FROM birds_animal WHERE uuid=?"
                    res = cur.execute(query, (uuid,))
                    notes_result = res.fetchone()
                    if notes_result and notes_result[0] and notes_result[0].strip():
                        birds_with_notes += 1
            except:
                continue

        con.close()

        logger.info(f"Cross-fostered birds: {len(cross_fostered_fathers)}")
        logger.info(f"Cross-fostered birds with notes: {birds_with_notes}")
        logger.info(f"Percentage with notes: {birds_with_notes / len(cross_fostered_fathers) * 100:.1f}%")

        # Get home-reared count for summary
        home_reared_offspring = get_home_reared_offspring_for_parents(cross_fostered_fathers)
        logger.info(f"Home-reared offspring of cross-foster parents: {len(home_reared_offspring)}")

        # ========== AUDIO DATA PROCESSING (TIME CONSUMING) ==========
        logger.info("=" * 70)
        logger.info("STARTING AUDIO DATA PROCESSING")
        logger.info("=" * 70)

        # Get root directory
        root_dir = check_sys_for_macaw_root()
        save_file = "cross_fostered_bird_audio_data.json"

        # Handle existing audio data file based on REUSE_AUDIO_DATA flag
        if not REUSE_AUDIO_DATA and os.path.exists(save_file):
            backup_file_if_exists(save_file, backup_dir)
            logger.info(f"Existing audio data file will be replaced (REUSE_AUDIO_DATA={REUSE_AUDIO_DATA})")
        elif REUSE_AUDIO_DATA and os.path.exists(save_file):
            logger.info(f"Reusing existing audio data file (REUSE_AUDIO_DATA={REUSE_AUDIO_DATA})")
        else:
            logger.info("No existing audio data file found, will create new one")

        # Filter bird_file_locations to only include male birds
        logger.info("Filtering to only male birds for audio search...")
        con = sqlite3.connect("2026-01-15-db.sqlite3")
        cur = con.cursor()

        male_bird_file_locations = {}
        for bird_name, file_paths in bird_file_locations.items():
            try:
                uuid, exists = getUUIDfromBands(cur, bird_name)
                if exists:
                    sex_query = "SELECT sex FROM birds_animal WHERE uuid=?"
                    sex_result = cur.execute(sex_query, (uuid,))
                    sex_row = sex_result.fetchone()
                    sex = sex_row[0] if sex_row and sex_row[0] else 'U'

                    # exclude females
                    if sex == 'M' or sex == 'U':
                        male_bird_file_locations[bird_name] = file_paths
                    else:
                        logger.info(f"Skipping audio search for {bird_name} - sex: {sex}")
                else:
                    logger.info(f"Skipping audio search for {bird_name} - not found in database")
            except Exception as e:
                logger.error(f"Error checking sex for {bird_name}: {e}")

        con.close()

        logger.info(
            f"Audio search will process {len(male_bird_file_locations)} male birds out of {len(bird_file_locations)} total cross-fostered birds"
        )

        # Run audio file search
        bird_audio_data = None
        if REUSE_AUDIO_DATA and os.path.exists(save_file):
            # Just load existing data
            bird_audio_data = load_bird_audio_data(save_file)
        else:
            # Process audio files
            bird_audio_data = get_all_audio_files_for_birds(male_bird_file_locations, root_dir, save_file)

        # Show detailed results for first few birds (if data was loaded successfully)
        if bird_audio_data:
            logger.info(f"\nDETAILED RESULTS (first 3 birds):")
            logger.info("-" * 50)

            for i, (bird_name, data) in enumerate(bird_audio_data.items()):
                if i < 3:
                    logger.info(f"\n{bird_name}:")
                    logger.info(f"  Directories searched: {len(data['directories_searched'])}")
                    for dir_path in data['directories_searched']:
                        logger.info(f"    {dir_path}")
                    logger.info(f"  Audio files found: {data['total_count']}")

                    # Show examples of wav files (which now include date files)
                    if 'wav_files' in data and data['wav_files']:
                        logger.info(f"  Example wav files (including date files):")
                        for j, file_path in enumerate(data['wav_files']):
                            if j < 5:
                                logger.info(f"    {os.path.basename(file_path)}")
                            elif j == 5:
                                logger.info(f"    ... and {len(data['wav_files']) - 5} more")
                                break

        # Create final CSV with audio data
        if bird_audio_data and cross_fostered_fathers:
            logger.info("Creating final cross-fostered birds CSV with audio data...")
            create_cross_fostered_birds_csv(cross_fostered_fathers, bird_audio_data)

        logger.info("=" * 70)
        logger.info("SCRIPT COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        # Final summary
        logger.info("Files created this run:")
        logger.info(f"  - cross_fostered_birds_notes.csv")
        logger.info(f"  - home_reared_offspring.csv")
        logger.info(f"  - cross_fostered_birds_summary.csv")
        if not REUSE_AUDIO_DATA:
            logger.info(f"  - {save_file}")
        logger.info(f"  - Log file: logs/cross_foster_analysis_{RUN_TIMESTAMP}.log")
        if backup_dir and os.path.exists(backup_dir) and os.listdir(backup_dir):
            logger.info(f"  - Backup files in: {backup_dir}/")

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.error("Check the log file for detailed error information")
        raise
    finally:
        logger.info(f"Script execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
