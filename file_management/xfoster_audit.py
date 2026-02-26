#!/usr/bin/env python3
"""
Cross-Foster Data Audit Script

This script performs a comprehensive audit of cross-foster bird data by:
1. Finding cross-foster birds (genetic father ≠ nest father)
2. Finding all home-reared offspring of relevant fathers
3. Scanning CSVs for additional birds not in database
4. Creating summary reports and family analysis

Author: Annie Taylor
Date: 2026-02-25
"""

import os
import sys
import csv
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
import sqlite3

# Import your existing functions
from tools.system_utils import check_sys_for_macaw_root
from tools.dbquery import *


def setup_audit_logging():
    """Set up logging for the audit process"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, f"cross_foster_audit_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"=== Cross-Foster Data Audit Started: {timestamp} ===")
    return logger


def create_output_directory():
    """Create output directory for audit results"""
    output_dir = "cross_foster_data_audit"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    return output_dir


def find_cross_fostered_birds_father_only(bird_list, db_path='../refs/2026-01-15-db.sqlite3'):
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

def get_cross_foster_birds_filtered(db_path='../refs/2026-01-15-db.sqlite3', include_females=False):
    """
    Get cross-foster birds, filtered by sex

    Args:
        db_path: Path to database
        include_females: Whether to include female birds

    Returns:
        list: Cross-foster birds meeting criteria
    """
    logger = logging.getLogger(__name__)

    # Get all birds with genetic parents
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
    all_birds = result.fetchall()
    con.close()

    cross_foster_birds = find_cross_fostered_birds_father_only(all_birds, db_path)

    # Filter by sex
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    filtered_birds = []
    for bird in cross_foster_birds:
        try:
            uuid, exists = getUUIDfromBands(cur, bird['child_name'])
            if exists:
                sex_query = "SELECT sex FROM birds_animal WHERE uuid=?"
                sex_result = cur.execute(sex_query, (uuid,))
                sex_row = sex_result.fetchone()
                sex = sex_row[0] if sex_row and sex_row[0] else 'U'

                # Include based on sex criteria
                if include_females or sex in ['M', 'U']:
                    bird['sex'] = sex
                    bird['uuid'] = uuid
                    filtered_birds.append(bird)
                else:
                    logger.info(f"Excluding {bird['child_name']} - sex: {sex}")
        except Exception as e:
            logger.error(f"Error checking sex for {bird['child_name']}: {e}")

    con.close()
    logger.info(f"Found {len(filtered_birds)} cross-foster birds meeting criteria")
    return filtered_birds


def get_all_relevant_fathers(cross_foster_birds):
    """
    Extract all unique fathers (genetic and nest) from cross-foster birds

    Args:
        cross_foster_birds: List of cross-foster bird records

    Returns:
        set: Unique father names
    """
    fathers = set()

    for bird in cross_foster_birds:
        genetic_father = bird.get('genetic_father')
        nest_father = bird.get('nest_father')

        if genetic_father and genetic_father != 'Unknown':
            fathers.add(genetic_father)
        if nest_father and nest_father != 'Unknown':
            fathers.add(nest_father)

    logging.info(f"Found {len(fathers)} unique fathers involved in cross-fostering")
    return fathers


def get_home_reared_for_fathers(father_names, db_path='../refs/2026-01-15-db.sqlite3', include_females=False):
    """
    Get all home-reared offspring for specific fathers

    Args:
        father_names: Set of father names to check
        db_path: Path to database
        include_females: Whether to include female birds

    Returns:
        list: Home-reared birds, dict: Database inconsistencies found
    """
    logger = logging.getLogger(__name__)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    home_reared_birds = []

    for father_name in father_names:
        try:
            # Get father's UUID
            father_uuid, father_exists = getUUIDfromBands(cur, father_name)
            if not father_exists:
                logger.warning(f"Father {father_name} not found in database")
                continue

            # Get all offspring where this bird is listed as genetic parent
            genetic_offspring_query = """
                                      SELECT DISTINCT bgp.genchild_id,
                                                      bc1.abbrv || ba.band_number ||
                                                      CASE
                                                          WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                                              THEN bc2.abbrv || ba.band_number2
                                                          ELSE ''
                                                          END as child_name,
                                                      ba.sex,
                                                      ba.hatch_date
                                      FROM birds_geneticparent bgp
                                               LEFT JOIN birds_animal ba ON bgp.genchild_id = ba.uuid
                                               LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                                               LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                                      WHERE bgp.genparent_id = ?
                                        AND bgp.genchild_id IS NOT NULL
                                        AND bgp.genchild_id != 'NULL' 
                                      """

            result = cur.execute(genetic_offspring_query, (father_uuid,))
            genetic_offspring = result.fetchall()

            # Get all offspring where this bird is listed as nest parent
            nest_offspring_query = """
                                   SELECT DISTINCT bp.child_id,
                                                   bc1.abbrv || ba.band_number ||
                                                   CASE
                                                       WHEN bc2.abbrv IS NOT NULL AND ba.band_number2 IS NOT NULL
                                                           THEN bc2.abbrv || ba.band_number2
                                                       ELSE ''
                                                       END as child_name,
                                                   ba.sex,
                                                   ba.hatch_date
                                   FROM birds_parent bp
                                            LEFT JOIN birds_animal ba ON bp.child_id = ba.uuid
                                            LEFT JOIN birds_color bc1 ON ba.band_color_id = bc1.id
                                            LEFT JOIN birds_color bc2 ON ba.band_color2_id = bc2.id
                                   WHERE bp.parent_id = ?
                                     AND bp.child_id IS NOT NULL
                                     AND bp.child_id != 'NULL' 
                                   """

            result = cur.execute(nest_offspring_query, (father_uuid,))
            nest_offspring = result.fetchall()

            # Process each genetic offspring to check if home-reared
            for child_uuid, child_name, sex, hatch_date in genetic_offspring:
                if not child_name:
                    continue

                # Filter by sex if needed
                if not include_females and sex not in ['M', 'U']:
                    continue

                # Check if this child is also in nest offspring (home-reared)
                is_home_reared = any(nest_child[0] == child_uuid for nest_child in nest_offspring)

                if is_home_reared:
                    # Get full parent information for this child
                    child_info = get_full_parent_info(child_uuid, cur)
                    child_info.update({
                        'child_uuid': child_uuid,
                        'child_name': child_name,
                        'sex': sex,
                        'hatch_date': hatch_date,
                        'rearing_type': 'HR',
                        'in_database': True
                    })
                    home_reared_birds.append(child_info)

        except Exception as e:
            logger.error(f"Error processing father {father_name}: {e}")
            continue

    con.close()
    logger.info(f"Found {len(home_reared_birds)} home-reared birds")

    return home_reared_birds


def get_full_parent_info(child_uuid, cursor):
    """
    Get complete parent information for a child

    Args:
        child_uuid: UUID of the child bird
        cursor: Database cursor

    Returns:
        dict: Parent information
    """
    parent_info = {
        'genetic_father': 'Unknown',
        'genetic_mother': 'Unknown',
        'nest_father': 'Unknown',
        'nest_mother': 'Unknown'
    }

    # Get genetic parents
    genetic_query = """
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

    result = cursor.execute(genetic_query, (child_uuid,))
    genetic_parents = result.fetchall()

    for parent_uuid, parent_name, sex in genetic_parents:
        if sex == 'M':
            parent_info['genetic_father'] = parent_name
        elif sex == 'F':
            parent_info['genetic_mother'] = parent_name

    # Get nest parents
    nest_query = """
                 SELECT bp.parent_id,
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

    result = cursor.execute(nest_query, (child_uuid,))
    nest_parents = result.fetchall()

    for parent_uuid, parent_name, sex in nest_parents:
        if sex == 'M':
            parent_info['nest_father'] = parent_name
        elif sex == 'F':
            parent_info['nest_mother'] = parent_name

    return parent_info


def enhance_birds_with_database_info(birds, db_path='../refs/2026-01-15-db.sqlite3'):
    """
    Add complete database information to bird records

    Args:
        birds: List of bird records to enhance
        db_path: Path to database

    Returns:
        list: Enhanced bird records
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    enhanced_birds = []

    for bird in birds:
        enhanced_bird = bird.copy()

        # Get additional info if not already present
        if 'uuid' in bird:
            uuid = bird['uuid']
        else:
            uuid, exists = getUUIDfromBands(cur, bird['child_name'])
            if not exists:
                continue

        # Get birth date and sex if not present
        if 'hatch_date' not in enhanced_bird or 'sex' not in enhanced_bird:
            info_query = "SELECT hatch_date, sex FROM birds_animal WHERE uuid=?"
            result = cur.execute(info_query, (uuid,))
            info_row = result.fetchone()

            if info_row:
                if 'hatch_date' not in enhanced_bird:
                    enhanced_bird['hatch_date'] = info_row[0] if info_row[0] else 'Unknown'
                if 'sex' not in enhanced_bird:
                    enhanced_bird['sex'] = info_row[1] if info_row[1] else 'U'

        # Get parent info if not complete
        if not all(key in enhanced_bird for key in ['genetic_father', 'genetic_mother', 'nest_father', 'nest_mother']):
            parent_info = get_full_parent_info(uuid, cur)
            enhanced_bird.update(parent_info)

        enhanced_birds.append(enhanced_bird)

    con.close()
    return enhanced_birds



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

def scan_csvs_for_additional_birds(csv_directory, known_birds, include_females=False):
    """
    Scan CSV files for birds not in the database analysis

    Args:
        csv_directory: Path to directory containing CSV files
        known_birds: Set of bird names already found in database
        include_females: Whether to include female birds

    Returns:
        list: Additional birds found in CSVs
    """
    logger = logging.getLogger(__name__)

    additional_birds = []
    bird_sources = {}  # Track which files each bird appears in

    # Get list of known bird names in various formats
    known_variations = set()
    for bird_name in known_birds:
        variations = generate_bird_name_variations(bird_name)
        known_variations.update(var.lower() for var in variations)

    logger.info(f"Scanning CSV directory: {csv_directory}")

    for filename in os.listdir(csv_directory):
        if not filename.endswith(('.csv', '.txt', '.xlsx', '.xls')):
            continue

        filepath = os.path.join(csv_directory, filename)

        try:
            # Load file
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.txt'):
                df = pd.read_csv(filepath, sep='\t')
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            elif filename.endswith('.xls'):
                df = pd.read_excel(filepath)

            logger.info(f"Scanning {filename} ({df.shape[0]} rows, {df.shape[1]} columns)")

            # Extract birds from this file
            file_birds = extract_birds_from_dataframe(df)

            # Find birds not in known set
            for bird_name in file_birds:
                standardized = standardize_bird_band(bird_name)
                if not standardized:
                    continue

                # Check if this bird is already known
                variations = generate_bird_name_variations(standardized)
                is_known = any(var.lower() in known_variations for var in variations)

                if not is_known:
                    # Track this as a new bird
                    if standardized not in bird_sources:
                        bird_sources[standardized] = {
                            'original_formats': set(),
                            'source_files': [],
                            'first_seen_in': filename
                        }

                    bird_sources[standardized]['original_formats'].add(bird_name)
                    if filename not in bird_sources[standardized]['source_files']:
                        bird_sources[standardized]['source_files'].append(filename)

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    # Convert to list format
    for bird_name, info in bird_sources.items():
        additional_birds.append({
            'child_name': bird_name,
            'genetic_father': 'Unknown',
            'genetic_mother': 'Unknown',
            'nest_father': 'Unknown',
            'nest_mother': 'Unknown',
            'hatch_date': 'Unknown',
            'sex': 'Unknown',
            'rearing_type': 'Unknown',
            'in_database': False,
            'source_files': '; '.join(info['source_files']),
            'original_formats': '; '.join(info['original_formats'])
        })

    logger.info(f"Found {len(additional_birds)} additional birds in CSV files")
    return additional_birds


def check_csv_birds_against_database(csv_birds, db_path='../refs/2026-01-15-db.sqlite3'):
    """
    Check which CSV birds exist in the database and populate their data

    Args:
        csv_birds: List of birds found in CSV files
        db_path: Path to database

    Returns:
        tuple: (birds_found_in_csvs_with_db_data, birds_truly_not_in_database)
    """
    logger = logging.getLogger(__name__)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    birds_with_db_data = []
    birds_not_in_db = []

    for csv_bird in csv_birds:
        bird_name = csv_bird['child_name']

        try:
            # Check if this bird exists in the database AT ALL
            uuid, exists = getUUIDfromBands(cur, bird_name)

            if exists:
                # Bird IS in database - get full data
                logger.info(f"Found {bird_name} in database, populating full data")

                # Get complete bird info from database
                enhanced_bird = csv_bird.copy()
                enhanced_bird['child_uuid'] = uuid
                enhanced_bird['in_database'] = True

                # Get full parent info
                parent_info = get_full_parent_info(uuid, cur)
                enhanced_bird.update(parent_info)

                # Get basic bird info (sex, hatch_date, etc.)
                info_query = "SELECT hatch_date, sex FROM birds_animal WHERE uuid=?"
                result = cur.execute(info_query, (uuid,))
                info_row = result.fetchone()

                if info_row:
                    enhanced_bird['hatch_date'] = info_row[0] if info_row[0] else 'Unknown'
                    enhanced_bird['sex'] = info_row[1] if info_row[1] else 'U'

                # Determine rearing type based on parent relationships
                genetic_father = enhanced_bird.get('genetic_father', 'Unknown')
                nest_father = enhanced_bird.get('nest_father', 'Unknown')

                if genetic_father != 'Unknown' and nest_father != 'Unknown':
                    if genetic_father == nest_father:
                        enhanced_bird['rearing_type'] = 'HR'  # Home-reared
                    else:
                        enhanced_bird['rearing_type'] = 'CF'  # Cross-fostered
                else:
                    enhanced_bird['rearing_type'] = 'Unknown'

                birds_with_db_data.append(enhanced_bird)

            else:
                # Bird is NOT in database
                logger.info(f"{bird_name} not found in database")
                csv_bird['in_database'] = False
                birds_not_in_db.append(csv_bird)

        except Exception as e:
            logger.error(f"Error checking {bird_name} against database: {e}")
            csv_bird['in_database'] = False
            birds_not_in_db.append(csv_bird)

    con.close()

    logger.info(f"CSV birds found in database: {len(birds_with_db_data)}")
    logger.info(f"CSV birds NOT in database: {len(birds_not_in_db)}")

    return birds_with_db_data, birds_not_in_db

def extract_birds_from_dataframe(df):
    """
    Extract potential bird names from a DataFrame

    Args:
        df: pandas DataFrame

    Returns:
        set: Bird names found in the DataFrame
    """
    import re

    birds_found = set()

    # Pattern to identify potential bird band strings
    bird_pattern = re.compile(r'\b[a-zA-Z]+\d+[a-zA-Z]*\d*\b')

    # Search all columns
    for column in df.columns:
        for value in df[column].dropna():
            value_str = str(value)
            potential_birds = bird_pattern.findall(value_str)

            for potential_bird in potential_birds:
                birds_found.add(potential_bird)

    return birds_found


def add_song_counts(birds, audio_data_file="cross_fostered_bird_audio_data.json"):
    """
    Add song count information to bird records

    Args:
        birds: List of bird records
        audio_data_file: Path to audio data JSON file

    Returns:
        list: Birds with song count information added
    """
    logger = logging.getLogger(__name__)

    # Load audio data if available
    audio_data = {}
    if os.path.exists(audio_data_file):
        try:
            with open(audio_data_file, 'r') as f:
                audio_data = json.load(f)
            logger.info(f"Loaded audio data for {len(audio_data)} birds")
        except Exception as e:
            logger.warning(f"Could not load audio data: {e}")

    # Add song counts to birds
    enhanced_birds = []
    for bird in birds:
        enhanced_bird = bird.copy()
        bird_name = bird['child_name']

        if bird_name in audio_data:
            audio_info = audio_data[bird_name]
            enhanced_bird.update({
                'total_songs': audio_info.get('total_count', 0),
                'wav_count': audio_info.get('wav_count', 0),
                'cbin_count': audio_info.get('cbin_count', 0),
                'audio_directories': len(audio_info.get('directories_searched', []))
            })
        else:
            enhanced_bird.update({
                'total_songs': 0,
                'wav_count': 0,
                'cbin_count': 0,
                'audio_directories': 0
            })

        enhanced_birds.append(enhanced_bird)

    return enhanced_birds


def create_family_summary(cross_foster_birds, home_reared_birds, csv_birds_in_db=None):
    """
    Create family-level summary using the same logic as the older analysis

    Args:
        cross_foster_birds: List of cross-foster bird records
        home_reared_birds: List of home-reared bird records
        csv_birds_in_db: List of CSV birds found in database (optional)

    Returns:
        list: Family summary records with all three categories
    """
    logger = logging.getLogger(__name__)

    # Combine all birds into one dataset
    all_birds = cross_foster_birds + home_reared_birds

    # Add CSV birds if provided
    if csv_birds_in_db:
        all_birds += csv_birds_in_db
        logger.info(f"Including {len(csv_birds_in_db)} additional birds from CSV files")

    # Create a DataFrame for easier manipulation (matching older analysis)
    birds_data = []
    for bird in all_birds:
        birds_data.append({
            'bird_name': bird['child_name'],
            'genetic_father': bird.get('genetic_father', 'Unknown'),
            'nest_father': bird.get('nest_father', 'Unknown')
        })

    df = pd.DataFrame(birds_data)

    # Remove rows with missing critical data
    df = df.dropna(subset=['bird_name', 'genetic_father', 'nest_father'])
    df = df[df['genetic_father'] != 'Unknown']
    df = df[df['nest_father'] != 'Unknown']

    # Get all unique fathers (both genetic and nest) - SAME AS OLDER ANALYSIS
    all_fathers = set(df['genetic_father'].unique()) | set(df['nest_father'].unique())
    all_fathers = {f for f in all_fathers if pd.notna(f) and f != ''}

    logger.info(f"Analyzing {len(all_fathers)} unique fathers (genetic + nest)")

    # Analyze each father - SAME LOGIC AS OLDER ANALYSIS
    family_summary = []

    for father in sorted(all_fathers):
        # Find all offspring where this bird is genetic father
        genetic_offspring = df[df['genetic_father'] == father]['bird_name'].tolist()

        # Find all offspring where this bird is nest father
        nest_offspring = df[df['nest_father'] == father]['bird_name'].tolist()

        # Categorize offspring - EXACT SAME LOGIC AS OLDER ANALYSIS
        genetic_offspring_set = set(genetic_offspring)
        nest_offspring_set = set(nest_offspring)

        # Both genetic and nest father (home-reared)
        home_reared = genetic_offspring_set & nest_offspring_set

        # Nest father only (cross-fostered IN / tutored offspring)
        cross_fostered_in = nest_offspring_set - genetic_offspring_set

        # Genetic father only (fostered OUT)
        fostered_out = genetic_offspring_set - nest_offspring_set

        # Add song counts for each category
        total_songs = 0
        home_reared_songs = 0
        cf_in_songs = 0
        fostered_out_songs = 0

        for bird in all_birds:
            bird_name = bird['child_name']
            songs = bird.get('total_songs', 0)
            total_songs += songs

            if bird_name in home_reared:
                home_reared_songs += songs
            elif bird_name in cross_fostered_in:
                cf_in_songs += songs
            elif bird_name in fostered_out:
                fostered_out_songs += songs

        # Create result row - MATCHING OLDER ANALYSIS STRUCTURE
        family_record = {
            'genetic_father': father,
            'home_reared_count': len(home_reared),
            'cross_fostered_in_count': len(cross_fostered_in),  # This is "tutored_count"
            'fostered_out_count': len(fostered_out),  # This is "genetic_only_count"
            'total_offspring': len(home_reared) + len(cross_fostered_in) + len(fostered_out),
            'home_reared_birds': '; '.join(sorted(home_reared)),
            'cross_fostered_in_birds': '; '.join(sorted(cross_fostered_in)),
            'fostered_out_birds': '; '.join(sorted(fostered_out)),
            'total_songs': total_songs,
            'home_reared_songs': home_reared_songs,
            'cf_in_songs': cf_in_songs,
            'fostered_out_songs': fostered_out_songs
        }

        # Add analysis suitability flags
        family_record['suitable_for_within_family_analysis'] = (
                family_record['home_reared_count'] >= 2 and family_record['cross_fostered_in_count'] >= 2
        )
        family_record['has_home_reared'] = family_record['home_reared_count'] > 0
        family_record['has_cross_fostered_in'] = family_record['cross_fostered_in_count'] > 0
        family_record['has_fostered_out'] = family_record['fostered_out_count'] > 0

        # Only include fathers that have at least one offspring
        if family_record['total_offspring'] > 0:
            family_summary.append(family_record)

    # Sort by total offspring (largest families first)
    family_summary.sort(key=lambda x: x['total_offspring'], reverse=True)

    logger.info(f"Created family summary for {len(family_summary)} fathers")
    suitable_count = sum(1 for f in family_summary if f['suitable_for_within_family_analysis'])
    logger.info(f"Families suitable for within-family analysis: {suitable_count}")

    return family_summary


def save_birds_to_csv(birds, output_file, output_dir):
    """
    Save bird records to CSV file

    Args:
        birds: List of bird records
        output_file: Output filename
        output_dir: Output directory
    """
    if not birds:
        logging.info(f"No birds to save for {output_file}")
        return

    filepath = os.path.join(output_dir, output_file)

    # Define column order
    columns = [
        'child_name', 'genetic_father', 'genetic_mother', 'nest_father', 'nest_mother',
        'hatch_date', 'sex', 'rearing_type', 'in_database', 'total_songs', 'wav_count',
        'cbin_count', 'audio_directories'
    ]

    # Add extra columns for CSV-only birds
    if any('source_files' in bird for bird in birds):
        columns.extend(['source_files', 'original_formats'])

    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(birds)

        logging.info(f"Saved {len(birds)} birds to {filepath}")

    except Exception as e:
        logging.error(f"Error saving {output_file}: {e}")

def save_family_summary_to_csv(family_summary, output_dir):
    """Save family summary to CSV file with all three categories"""
    if not family_summary:
        logging.info("No families to save")
        return

    filepath = os.path.join(output_dir, "family_summary_by_father.csv")

    columns = [
        'genetic_father', 'total_offspring',
        'home_reared_count', 'cross_fostered_in_count', 'fostered_out_count',
        'home_reared_birds', 'cross_fostered_in_birds', 'fostered_out_birds',
        'total_songs', 'home_reared_songs', 'cf_in_songs', 'fostered_out_songs',
        'suitable_for_within_family_analysis', 'has_home_reared', 'has_cross_fostered_in', 'has_fostered_out'
    ]

    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(family_summary)

        logging.info(f"Saved family summary for {len(family_summary)} families to {filepath}")

    except Exception as e:
        logging.error(f"Error saving family summary: {e}")


def create_audit_summary_report(cross_foster_birds, home_reared_birds, csv_birds,
                                family_summary, output_dir):
    """
    Create a text summary report of the audit findings

    Args:
        cross_foster_birds: List of cross-foster birds
        home_reared_birds: List of home-reared birds
        csv_birds: List of birds found only in CSVs
        family_summary: List of family records
        database_errors: List of database inconsistencies
        output_dir: Output directory
    """
    filepath = os.path.join(output_dir, "audit_summary_report.txt")

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("CROSS-FOSTER DATA AUDIT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Database birds summary
            f.write("DATABASE BIRDS SUMMARY\n")
            f.write("-" * 25 + "\n")
            f.write(f"Cross-foster birds: {len(cross_foster_birds)}\n")
            f.write(f"Home-reared birds: {len(home_reared_birds)}\n")
            f.write(f"Total database birds: {len(cross_foster_birds) + len(home_reared_birds)}\n\n")

            # Sex distribution
            cf_males = sum(1 for b in cross_foster_birds if b.get('sex') in ['M', 'U'])
            hr_males = sum(1 for b in home_reared_birds if b.get('sex') in ['M', 'U'])
            f.write(f"Male cross-foster birds: {cf_males}\n")
            f.write(f"Male home-reared birds: {hr_males}\n")
            f.write(f"Total male birds: {cf_males + hr_males}\n\n")

            # Song data summary
            cf_with_songs = sum(1 for b in cross_foster_birds if b.get('total_songs', 0) > 0)
            hr_with_songs = sum(1 for b in home_reared_birds if b.get('total_songs', 0) > 0)
            total_songs_cf = sum(b.get('total_songs', 0) for b in cross_foster_birds)
            total_songs_hr = sum(b.get('total_songs', 0) for b in home_reared_birds)

            f.write("SONG DATA SUMMARY\n")
            f.write("-" * 17 + "\n")
            f.write(f"Cross-foster birds with songs: {cf_with_songs}/{len(cross_foster_birds)}\n")
            f.write(f"Home-reared birds with songs: {hr_with_songs}/{len(home_reared_birds)}\n")
            f.write(f"Total songs in cross-foster birds: {total_songs_cf}\n")
            f.write(f"Total songs in home-reared birds: {total_songs_hr}\n")

            if cf_with_songs > 0:
                avg_songs_cf = total_songs_cf / cf_with_songs
                f.write(f"Average songs per cross-foster bird (with songs): {avg_songs_cf:.1f}\n")
            if hr_with_songs > 0:
                avg_songs_hr = total_songs_hr / hr_with_songs
                f.write(f"Average songs per home-reared bird (with songs): {avg_songs_hr:.1f}\n")
            f.write("\n")

            # Family analysis summary
            f.write("FAMILY ANALYSIS SUMMARY\n")
            f.write("-" * 23 + "\n")
            f.write(f"Total families identified: {len(family_summary)}\n")

            suitable_families = [f for f in family_summary if f['suitable_for_within_family_analysis']]
            f.write(f"Families suitable for within-family analysis: {len(suitable_families)}\n")

            mixed_families = [f for f in family_summary if f['has_cross_fostered'] and f['has_home_reared']]
            f.write(f"Families with both CF and HR offspring: {len(mixed_families)}\n")

            cf_only_families = [f for f in family_summary if f['has_cross_fostered'] and not f['has_home_reared']]
            hr_only_families = [f for f in family_summary if f['has_home_reared'] and not f['has_cross_fostered']]
            f.write(f"Families with only CF offspring: {len(cf_only_families)}\n")
            f.write(f"Families with only HR offspring: {len(hr_only_families)}\n\n")

            if suitable_families:
                f.write("FAMILIES SUITABLE FOR WITHIN-FAMILY ANALYSIS:\n")
                for family in suitable_families:
                    f.write(
                        f"  {family['genetic_father']}: {family['cross_fostered_count']} CF, {family['home_reared_count']} HR\n")
                f.write("\n")

            # CSV birds summary
            f.write("CSV-ONLY BIRDS SUMMARY\n")
            f.write("-" * 22 + "\n")
            f.write(f"Birds found only in CSV files: {len(csv_birds)}\n")
            if csv_birds:
                # Count unique source files
                all_sources = set()
                for bird in csv_birds:
                    sources = bird.get('source_files', '').split('; ')
                    all_sources.update(sources)
                f.write(f"CSV files containing additional birds: {len(all_sources)}\n")
                f.write("Source files: " + ", ".join(sorted(all_sources)) + "\n")
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")

            if len(suitable_families) >= 3:
                f.write("✓ Sufficient families for within-family analysis\n")
            else:
                f.write("⚠ Limited families for within-family analysis - consider bulk analysis approach\n")

            total_males_with_songs = cf_with_songs + hr_with_songs
            if total_males_with_songs >= 20:
                f.write("✓ Sufficient birds with songs for bulk analysis\n")
            else:
                f.write("⚠ Limited birds with song data - may need to lower song count thresholds\n")

            if csv_birds:
                f.write("⚠ Additional birds found in CSV files - consider adding to database\n")
            else:
                f.write("✓ All CSV birds already in database\n")

        logging.info(f"Audit summary report saved to {filepath}")

    except Exception as e:
        logging.error(f"Error creating audit summary report: {e}")


def main(include_females=False, csv_directory="/Users/annietaylor/Documents/ucsf/brainard/x-foster"):
    """
    Main function to run the complete cross-foster data audit

    Args:
        include_females: Whether to include female birds in analysis
        csv_directory: Path to directory containing additional CSV files
    """
    logger = setup_audit_logging()
    output_dir = create_output_directory()

    try:
        logger.info("=== PHASE 1: EXTRACT CROSS-FOSTER BIRDS ===")
        cross_foster_birds = get_cross_foster_birds_filtered(include_females=include_females)

        logger.info("=== PHASE 2: FIND ALL RELEVANT FATHERS ===")
        all_fathers = get_all_relevant_fathers(cross_foster_birds)

        logger.info("=== PHASE 3: GET HOME-REARED OFFSPRING ===")
        home_reared_birds = get_home_reared_for_fathers(all_fathers, include_females=include_females)

        logger.info("=== PHASE 4: ENHANCE WITH DATABASE INFO ===")
        cross_foster_birds = enhance_birds_with_database_info(cross_foster_birds)
        home_reared_birds = enhance_birds_with_database_info(home_reared_birds)

        # Mark rearing types
        for bird in cross_foster_birds:
            bird['rearing_type'] = 'CF'
        for bird in home_reared_birds:
            bird['rearing_type'] = 'HR'

        logger.info("=== PHASE 5: ADD SONG COUNT DATA ===")
        cross_foster_birds = add_song_counts(cross_foster_birds)
        home_reared_birds = add_song_counts(home_reared_birds)

        logger.info("=== PHASE 6: SCAN CSV FILES FOR ALL BIRDS ===")
        all_database_birds = set(bird['child_name'] for bird in cross_foster_birds + home_reared_birds)
        csv_birds = scan_csvs_for_additional_birds(csv_directory, all_database_birds, include_females=include_females)

        logger.info("=== PHASE 6b: CHECK CSV BIRDS AGAINST DATABASE ===")
        csv_birds_in_db, csv_birds_not_in_db = check_csv_birds_against_database(csv_birds)

        logger.info("=== PHASE 7: CREATE FAMILY SUMMARY (INCLUDING ALL DATABASE BIRDS) ===")
        # Now include ALL database birds (original + CSV birds found in DB)
        all_database_birds_complete = cross_foster_birds + home_reared_birds + csv_birds_in_db
        family_summary = create_family_summary(cross_foster_birds, home_reared_birds, csv_birds_in_db)

        logger.info("=== PHASE 8: SAVE OUTPUT FILES ===")

        # Save individual bird datasets
        all_birds = cross_foster_birds + home_reared_birds
        save_birds_to_csv(all_birds, "master_birds_inventory.csv", output_dir)
        save_birds_to_csv(cross_foster_birds, "cross_fostered_birds.csv", output_dir)
        save_birds_to_csv(home_reared_birds, "home_reared_birds.csv", output_dir)
        save_birds_to_csv(csv_birds, "birds_not_in_database.csv", output_dir)

        # Save family summary
        save_family_summary_to_csv(family_summary, output_dir)

        # Create summary report
        create_audit_summary_report(cross_foster_birds, home_reared_birds, csv_birds,
                                    family_summary, output_dir)

        logger.info("=== AUDIT COMPLETE ===")
        logger.info(f"Output files saved to: {output_dir}/")
        logger.info(f"Cross-foster birds: {len(cross_foster_birds)}")
        logger.info(f"Home-reared birds: {len(home_reared_birds)}")
        logger.info(f"CSV-only birds: {len(csv_birds)}")
        logger.info(f"Families analyzed: {len(family_summary)}")

        # Quick analysis summary
        suitable_families = [f for f in family_summary if f['suitable_for_within_family_analysis']]
        birds_with_songs = sum(1 for b in all_birds if b.get('total_songs', 0) > 0)

        logger.info(f"\nKEY FINDINGS:")
        logger.info(f"  Families suitable for within-family analysis: {len(suitable_families)}")
        logger.info(f"  Birds with song data: {birds_with_songs}/{len(all_birds)}")

        return {
            'cross_foster_birds': cross_foster_birds,
            'home_reared_birds': home_reared_birds,
            'csv_birds': csv_birds,
            'family_summary': family_summary,
            'output_directory': output_dir
        }

    except Exception as e:
        logger.error(f"Audit failed with error: {e}")
        raise

if __name__ == "__main__":
    # Configuration
    INCLUDE_FEMALES = False  # Set to True to include female birds
    CSV_DIRECTORY = "/Users/annietaylor/Documents/ucsf/brainard/x-foster"

    # Run the audit
    results = main(include_females=INCLUDE_FEMALES, csv_directory=CSV_DIRECTORY)

    print("\n" + "=" * 60)
    print("CROSS-FOSTER DATA AUDIT COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {results['output_directory']}")
    print("\nFiles created:")
    print("  - master_birds_inventory.csv")
    print("  - cross_fostered_birds.csv")
    print("  - home_reared_birds.csv")
    print("  - birds_not_in_database.csv")
    print("  - family_summary_by_father.csv")
    print("  - audit_summary_report.txt")