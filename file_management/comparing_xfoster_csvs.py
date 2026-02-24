import csv
import json
import os
import re
import pandas as pd


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


def find_and_count_birds(data, source_name=""):
    """
    Find and count all bird references in a table/array.

    Args:
        data: List of lists, pandas DataFrame, or similar tabular data
        source_name: Optional name for the data source

    Returns:
        dict: Contains standardized bird IDs, counts, and original formats found
    """
    bird_registry = {}  # standardized_id -> {'count': int, 'original_formats': set, 'source': str}

    # Convert data to a format we can iterate over
    if hasattr(data, 'values'):  # pandas DataFrame
        flat_data = data.values.flatten()
    elif isinstance(data, list):
        flat_data = []
        for row in data:
            if isinstance(row, list):
                flat_data.extend(row)
            else:
                flat_data.append(row)
    else:
        flat_data = [data]  # single value

    # Pattern to identify potential bird band strings
    # Look for combinations of letters and numbers
    bird_pattern = re.compile(r'\b[a-zA-Z]+\d+[a-zA-Z]*\d*\b')

    for item in flat_data:
        if item is None:
            continue

        item_str = str(item)
        potential_birds = bird_pattern.findall(item_str)

        for potential_bird in potential_birds:
            standardized = standardize_bird_band(potential_bird)

            if standardized:
                if standardized not in bird_registry:
                    bird_registry[standardized] = {
                        'count': 0,
                        'original_formats': set(),
                        'source': source_name
                    }

                bird_registry[standardized]['count'] += 1
                bird_registry[standardized]['original_formats'].add(potential_bird)

    return bird_registry


def compare_spreadsheets(spreadsheet_data_dict):
    """
    Compare multiple spreadsheets to find overlapping birds and redundancy.

    Args:
        spreadsheet_data_dict: Dict where keys are spreadsheet names and values are the data

    Returns:
        dict: Analysis results including unique birds per sheet and overlaps
    """
    all_results = {}

    # Process each spreadsheet
    for sheet_name, data in spreadsheet_data_dict.items():
        all_results[sheet_name] = find_and_count_birds(data, sheet_name)

    # Find overlaps
    all_birds = set()
    for sheet_results in all_results.values():
        all_birds.update(sheet_results.keys())

    # Create overlap analysis
    overlap_analysis = {
        'total_unique_birds': len(all_birds),
        'birds_by_sheet': {},
        'overlapping_birds': {},
        'unique_to_sheet': {}
    }

    for sheet_name in all_results.keys():
        sheet_birds = set(all_results[sheet_name].keys())
        overlap_analysis['birds_by_sheet'][sheet_name] = {
            'bird_count': len(sheet_birds),
            'birds': sheet_birds
        }

        # Find birds unique to this sheet
        other_sheets_birds = set()
        for other_sheet, other_results in all_results.items():
            if other_sheet != sheet_name:
                other_sheets_birds.update(other_results.keys())

        unique_birds = sheet_birds - other_sheets_birds
        overlap_analysis['unique_to_sheet'][sheet_name] = unique_birds

    # Find birds that appear in multiple sheets
    for bird in all_birds:
        sheets_with_bird = []
        for sheet_name, sheet_results in all_results.items():
            if bird in sheet_results:
                sheets_with_bird.append(sheet_name)

        if len(sheets_with_bird) > 1:
            overlap_analysis['overlapping_birds'][bird] = sheets_with_bird

    return {
        'individual_results': all_results,
        'overlap_analysis': overlap_analysis
    }


def generate_redundancy_report(comparison_results):
    """
    Generate a human-readable report about redundancy between spreadsheets.
    """
    individual_results = comparison_results['individual_results']
    overlap_analysis = comparison_results['overlap_analysis']

    print("=== BIRD BAND ANALYSIS REPORT ===\n")

    print(f"Total unique birds across all spreadsheets: {overlap_analysis['total_unique_birds']}\n")

    print("Birds per spreadsheet:")
    for sheet_name, sheet_info in overlap_analysis['birds_by_sheet'].items():
        print(f"  {sheet_name}: {sheet_info['bird_count']} birds")
    print()

    print("Overlapping birds (appear in multiple spreadsheets):")
    if overlap_analysis['overlapping_birds']:
        for bird, sheets in overlap_analysis['overlapping_birds'].items():
            print(f"  {bird}: appears in {len(sheets)} sheets - {', '.join(sheets)}")
    else:
        print("  No overlapping birds found")
    print()

    print("Birds unique to each spreadsheet:")
    for sheet_name, unique_birds in overlap_analysis['unique_to_sheet'].items():
        print(f"  {sheet_name}: {len(unique_birds)} unique birds")
        if unique_birds:
            print(f"    {', '.join(sorted(unique_birds))}")
    print()

    print("Redundancy summary:")
    total_bird_entries = sum(len(results) for results in individual_results.values())
    redundant_entries = total_bird_entries - overlap_analysis['total_unique_birds']
    if total_bird_entries > 0:
        redundancy_percentage = (redundant_entries / total_bird_entries) * 100
        print(f"  Total bird entries across all sheets: {total_bird_entries}")
        print(f"  Redundant entries: {redundant_entries}")
        print(f"  Redundancy percentage: {redundancy_percentage:.1f}%")


def export_individual_results_csv(comparison_results, output_filename="bird_analysis.csv"):
    """
    Export results to CSV with the specified format.
    """
    import csv

    individual_results = comparison_results['individual_results']

    # Get all unique birds and all document sources
    all_birds = set()
    all_sources = list(individual_results.keys())

    for results in individual_results.values():
        all_birds.update(results.keys())

    # Create header
    header = ['birdid', 'total_counts'] + all_sources + ['original_formats', 'original_sources']

    # Prepare data
    rows = []
    for bird in sorted(all_birds):
        row = [bird]

        # Calculate total count across all sources
        total_count = sum(individual_results[source].get(bird, {}).get('count', 0)
                         for source in all_sources)
        row.append(total_count)

        # Add count for each source
        for source in all_sources:
            count = individual_results[source].get(bird, {}).get('count', 0)
            row.append(count)

        # Collect all original formats and sources for this bird
        all_formats = set()
        sources_with_bird = []

        for source in all_sources:
            if bird in individual_results[source]:
                all_formats.update(individual_results[source][bird]['original_formats'])
                sources_with_bird.append(source)

        # Add original formats and sources
        row.append('; '.join(sorted(all_formats)))
        row.append('; '.join(sources_with_bird))

        rows.append(row)

    # Write to CSV
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Results exported to {output_filename}")


def convert_tab_file_to_csv(input_file="gmm_simdat_hmm_2mod_all_new_thresh_sorted_msb2.txt",
                            output_file="tutor_data.csv"):
    """
    Convert the tab-delimited text file to CSV format

    Args:
        input_file: Path to the input tab-delimited file
        output_file: Path to output CSV file
    """

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Clean up lines and prepare for CSV
        processed_rows = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Split by tabs
            fields = line.split('\t')

            # Clean up each field (remove extra whitespace)
            cleaned_fields = [field.strip() for field in fields]

            processed_rows.append(cleaned_fields)

        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write all rows
            for row in processed_rows:
                writer.writerow(row)

        print(f"Successfully converted {len(processed_rows)} rows to {output_file}")

        # Display info about the conversion
        if processed_rows:
            print(f"Number of columns: {len(processed_rows[0])}")
            print(f"Headers: {processed_rows[0]}")

            if len(processed_rows) > 1:
                print(f"Sample data row: {processed_rows[1]}")

        return output_file

    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
        print("Make sure the file is in the current directory or provide the full path.")
        return None
    except Exception as e:
        print(f"Error converting file: {e}")
        return None


def check_birds_overlap(csv_file="tutor_data.csv", json_file="cross_fostered_bird_audio_data.json"):
    """
    Check which birds from tutor_data.csv are in cross_fostered_bird_audio_data.json

    Args:
        csv_file: Path to the tutor data CSV file
        json_file: Path to the cross-fostered bird audio data JSON file
    """

    try:
        # Load the cross-fostered bird data
        with open(json_file, 'r') as f:
            cross_fostered_data = json.load(f)

        cross_fostered_birds = set(cross_fostered_data.keys())
        print(f"Cross-fostered birds in JSON: {len(cross_fostered_birds)}")

        # Load the tutor data CSV
        tutor_birds = set()
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get the bird name from the 'Bird_tutee' column
                bird_name = row.get('Bird_tutee', '').strip()
                if bird_name:
                    tutor_birds.add(bird_name)

        print(f"Birds in tutor CSV: {len(tutor_birds)}")

        # Find overlaps
        birds_in_both = cross_fostered_birds.intersection(tutor_birds)
        birds_only_in_cross_fostered = cross_fostered_birds - tutor_birds
        birds_only_in_tutor = tutor_birds - cross_fostered_birds

        print(f"\n" + "=" * 60)
        print("OVERLAP ANALYSIS")
        print("=" * 60)

        print(f"Birds in BOTH datasets: {len(birds_in_both)}")
        if birds_in_both:
            print("  Birds found in both:")
            for bird in sorted(birds_in_both):
                print(f"    {bird}")

        print(f"\nBirds ONLY in cross-fostered data: {len(birds_only_in_cross_fostered)}")
        if birds_only_in_cross_fostered:
            print("  Cross-fostered birds not in tutor data:")
            for bird in sorted(birds_only_in_cross_fostered):
                print(f"    {bird}")

        print(f"\nBirds ONLY in tutor data: {len(birds_only_in_tutor)}")
        if len(birds_only_in_tutor) <= 20:  # Show all if not too many
            print("  Tutor birds not in cross-fostered data:")
            for bird in sorted(birds_only_in_tutor):
                print(f"    {bird}")
        else:
            print(f"  First 20 tutor birds not in cross-fostered data:")
            for bird in sorted(list(birds_only_in_tutor)[:20]):
                print(f"    {bird}")
            print(f"    ... and {len(birds_only_in_tutor) - 20} more")

        # Summary statistics
        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        overlap_percentage = (len(birds_in_both) / len(cross_fostered_birds) * 100) if cross_fostered_birds else 0
        print(f"Overlap rate: {overlap_percentage:.1f}% of cross-fostered birds are in tutor data")

        tutor_coverage = (len(birds_in_both) / len(tutor_birds) * 100) if tutor_birds else 0
        print(f"Coverage rate: {tutor_coverage:.1f}% of tutor birds are cross-fostered")

        return {
            'birds_in_both': birds_in_both,
            'birds_only_cross_fostered': birds_only_in_cross_fostered,
            'birds_only_tutor': birds_only_in_tutor,
            'cross_fostered_birds': cross_fostered_birds,
            'tutor_birds': tutor_birds
        }

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None
    except Exception as e:
        print(f"Error analyzing overlap: {e}")
        return None


def create_overlap_summary_csv(overlap_data, output_file="bird_overlap_summary.csv"):
    """
    Create a CSV summary of the bird overlap analysis
    """
    if not overlap_data:
        print("No overlap data to save")
        return

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write headers
            writer.writerow(['Bird_Name', 'From DB', 'From MSB', 'Status'])

            # Get all unique birds
            all_birds = overlap_data['cross_fostered_birds'].union(overlap_data['tutor_birds'])

            for bird in sorted(all_birds):
                in_cross_fostered = bird in overlap_data['cross_fostered_birds']
                in_tutor = bird in overlap_data['tutor_birds']

                if in_cross_fostered and in_tutor:
                    status = 'Both'
                elif in_cross_fostered:
                    status = 'DB only'
                else:
                    status = 'MSB only'

                writer.writerow([bird, in_cross_fostered, in_tutor, status])

        print(f"Overlap summary saved to: {output_file}")

    except Exception as e:
        print(f"Error saving overlap summary: {e}")


def get_detailed_info_for_overlap_birds(overlap_data, csv_file="tutor_data.csv",
                                        json_file="cross_fostered_bird_audio_data.json"):
    """
    Get detailed information for birds that appear in both datasets
    """
    if not overlap_data or not overlap_data['birds_in_both']:
        print("No overlapping birds to analyze")
        return

    try:
        # Load cross-fostered data
        with open(json_file, 'r') as f:
            cross_fostered_data = json.load(f)

        # Load tutor data
        tutor_data = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bird_name = row.get('Bird_tutee', '').strip()
                if bird_name in overlap_data['birds_in_both']:
                    if bird_name not in tutor_data:
                        tutor_data[bird_name] = []
                    tutor_data[bird_name].append(row)

        print(f"\n" + "=" * 70)
        print("DETAILED INFO FOR OVERLAPPING BIRDS")
        print("=" * 70)

        for bird in sorted(overlap_data['birds_in_both']):
            print(f"\n{bird}:")

            # Cross-fostered info
            if bird in cross_fostered_data:
                cf_data = cross_fostered_data[bird]
                print(f"  Audio files: {cf_data.get('total_count', 0)} total")
                print(
                    f"    WAV: {cf_data.get('wav_count', 0)}, CBIN: {cf_data.get('cbin_count', 0)}, Date-format: {cf_data.get('date_format_count', 0)}")
                print(f"  Directories: {len(cf_data.get('directories_searched', []))}")

            # Tutor info
            if bird in tutor_data:
                tutor_entries = tutor_data[bird]
                print(f"  Tutor data entries: {len(tutor_entries)}")
                for i, entry in enumerate(tutor_entries):
                    tutor_nest = entry.get('tutor_nest', 'Unknown')
                    father = entry.get('father', 'Unknown')
                    specgrm_group = entry.get('specgrm_group', 'Unknown')
                    home_reared = entry.get('home_reared', 'Unknown')
                    print(
                        f"    Entry {i + 1}: Tutor={tutor_nest}, Father={father}, Group={specgrm_group}, Home-reared={home_reared}")

        return tutor_data

    except Exception as e:
        print(f"Error getting detailed info: {e}")
        return None


def create_combined_analysis_csv(overlap_data, csv_file="tutor_data.csv",
                                 json_file="cross_fostered_bird_audio_data.json",
                                 output_file="combined_bird_analysis.csv"):
    """
    Create a comprehensive CSV combining cross-fostered and tutor data for overlapping birds
    """
    if not overlap_data or not overlap_data['birds_in_both']:
        print("No overlapping birds to create combined analysis")
        return

    try:
        # Load cross-fostered data
        with open(json_file, 'r') as f:
            cross_fostered_data = json.load(f)

        # Load tutor data
        tutor_data = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bird_name = row.get('Bird_tutee', '').strip()
                if bird_name in overlap_data['birds_in_both']:
                    if bird_name not in tutor_data:
                        tutor_data[bird_name] = []
                    tutor_data[bird_name].append(row)

        # Create combined CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Headers
            headers = [
                'Bird_Name', 'Tutor_Nest', 'Father', 'Spectrogram_Group', 'Home_Reared',
                'Songs_Non_Fosters', 'In_Striata_Fosters', 'Ped_Dat', 'SD_Values',
                'Total_Audio_Files', 'WAV_Count', 'CBIN_Count', 'Date_Format_Count',
                'Audio_Directories_Count', 'Tutor_Entries_Count'
            ]
            writer.writerow(headers)

            # Data rows
            for bird in sorted(overlap_data['birds_in_both']):
                # Get cross-fostered data
                cf_data = cross_fostered_data.get(bird, {})
                total_audio = cf_data.get('total_count', 0)
                wav_count = cf_data.get('wav_count', 0)
                cbin_count = cf_data.get('cbin_count', 0)
                date_format_count = cf_data.get('date_format_count', 0)
                dir_count = len(cf_data.get('directories_searched', []))

                # Get tutor data
                tutor_entries = tutor_data.get(bird, [])
                tutor_count = len(tutor_entries)

                # If multiple tutor entries, create a row for each
                if tutor_entries:
                    for entry in tutor_entries:
                        # Combine SD values if they exist
                        sd_values = []
                        for col in entry.keys():
                            if 'SD' in col or any(char.isdigit() for char in col):
                                val = entry.get(col, '').strip()
                                if val and val != '':
                                    try:
                                        float(val)  # Check if it's a number
                                        sd_values.append(val)
                                    except:
                                        pass

                        row = [
                            bird,
                            entry.get('tutor_nest', ''),
                            entry.get('father', ''),
                            entry.get('specgrm_group', ''),
                            entry.get('home_reared', ''),
                            entry.get('songs_in_non_fosters_from_foster', ''),
                            entry.get('In_striata_fosters_from_screening_songs', ''),
                            entry.get('ped_dat', ''),
                            '; '.join(sd_values[:4]) if sd_values else '',  # First 4 SD values
                            total_audio,
                            wav_count,
                            cbin_count,
                            date_format_count,
                            dir_count,
                            tutor_count
                        ]
                        writer.writerow(row)
                else:
                    # Bird has audio data but no tutor data
                    row = [
                        bird, '', '', '', '', '', '', '', '',
                        total_audio, wav_count, cbin_count, date_format_count,
                        dir_count, 0
                    ]
                    writer.writerow(row)

        print(f"Combined analysis saved to: {output_file}")

    except Exception as e:
        print(f"Error creating combined analysis: {e}")


def analyze_bird_overlap():
    """
    Run complete bird overlap analysis
    """
    print("ANALYZING BIRD OVERLAP BETWEEN DATASETS")
    print("=" * 60)

    # Check overlap
    overlap_data = check_birds_overlap()

    if overlap_data:
        # Create overlap summary CSV
        create_overlap_summary_csv(overlap_data)

        # Get detailed info for overlapping birds
        detailed_data = get_detailed_info_for_overlap_birds(overlap_data)

        # Create combined analysis CSV
        create_combined_analysis_csv(overlap_data)

        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("Files created:")
        print("  - bird_overlap_summary.csv: Summary of which birds are in which datasets")
        print("  - combined_bird_analysis.csv: Detailed data for birds in both datasets")

        return overlap_data, detailed_data
    else:
        print("Could not complete overlap analysis due to errors")
        return None, None


def extract_birds_from_data(data, bird_column_names=None):
    """
    Extract standardized bird IDs from data, looking in specified columns or all columns.
    """
    birds_found = set()

    if bird_column_names is None:
        columns_to_search = data.columns
    else:
        columns_to_search = [col for col in bird_column_names if col in data.columns]

    # Pattern to identify potential bird band strings
    bird_pattern = re.compile(r'\b[a-zA-Z]+\d+[a-zA-Z]*\d*\b')

    for column in columns_to_search:
        for value in data[column].dropna():
            value_str = str(value)
            potential_birds = bird_pattern.findall(value_str)

            for potential_bird in potential_birds:
                standardized = standardize_bird_band(potential_bird)
                if standardized:
                    birds_found.add(standardized)

    return birds_found


def load_master_birds(cross_fostered_csv, home_reared_csv):
    """
    Load birds from master spreadsheets.
    """
    master_birds = set()

    # Load cross-fostered birds
    try:
        cf_df = pd.read_csv(cross_fostered_csv)
        bird_columns = ['bird_id', 'Bird Name', 'birdid']
        cf_birds = extract_birds_from_data(cf_df, bird_columns)
        master_birds.update(cf_birds)
        print(f"Loaded {len(cf_birds)} birds from cross-fostered summary")
    except Exception as e:
        print(f"Error loading cross-fostered birds: {e}")

    # Load home-reared birds
    try:
        hr_df = pd.read_csv(home_reared_csv)
        hr_birds = extract_birds_from_data(hr_df, bird_columns)
        master_birds.update(hr_birds)
        print(f"Loaded {len(hr_birds)} birds from home-reared summary")
    except Exception as e:
        print(f"Error loading home-reared birds: {e}")

    return master_birds


def find_birds_not_in_database(csv_directory_path, cross_fostered_csv, home_reared_csv):
    """
    Find unique birds in CSV files that aren't in your database spreadsheets.
    """
    # Load master birds
    master_birds = load_master_birds(cross_fostered_csv, home_reared_csv)

    # Dictionary to track each missing bird and which files it appears in
    missing_birds_tracker = {}

    for filename in os.listdir(csv_directory_path):
        if not filename.endswith(('.csv', '.txt', '.xlsx', '.xls')):
            continue

        filepath = os.path.join(csv_directory_path, filename)

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

            # Find birds in this file
            file_birds = extract_birds_from_data(df)
            missing_from_file = file_birds - master_birds

            if missing_from_file:
                print(f"{filename}: {len(missing_from_file)} missing birds - {sorted(missing_from_file)}")

                # Track each missing bird and which files it appears in
                for bird in missing_from_file:
                    if bird not in missing_birds_tracker:
                        missing_birds_tracker[bird] = {
                            'source_files': [],
                            'first_occurrence_df': df,
                            'first_occurrence_filename': filename
                        }
                    missing_birds_tracker[bird]['source_files'].append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return missing_birds_tracker


def create_unique_missing_birds_csv(missing_birds_tracker, output_path):
    """
    Create a CSV with unique missing birds and their source files.
    """
    missing_data = []

    for bird, info in missing_birds_tracker.items():
        df = info['first_occurrence_df']
        source_files = '; '.join(info['source_files'])

        # Find rows containing this bird in the first occurrence dataframe
        bird_rows = []
        for idx, row in df.iterrows():
            row_str = ' '.join(str(val) for val in row.values if pd.notna(val)).lower()
            if bird in row_str:
                bird_rows.append(idx)

        # Add to missing data
        missing_data.append({
            'bird_id': bird,
            'source_files': source_files,
            'file_count': len(info['source_files']),
            'first_occurrence_file': info['first_occurrence_filename'],
            'first_occurrence_row': bird_rows[0] if bird_rows else 'Not found',
            'total_occurrences_in_first_file': len(bird_rows)
        })

    # Save results
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('file_count', ascending=False)
        missing_df.to_csv(output_path, index=False)
        print(f"Unique missing birds saved to: {output_path}")
        print(f"Total unique missing birds: {len(missing_data)}")

        return missing_df
    else:
        print("No missing birds found")
        return None


if __name__ == "__main__":
    csv_directory = os.path.join('/Users', 'annietaylor', 'Documents', 'ucsf', 'brainard', 'x-foster')
    cross_fostered_path = os.path.join(os.getcwd(), 'cross_fostered_birds_summary.csv')
    home_reared_path = os.path.join(os.getcwd(), 'home_reared_offspring.csv')

    # Find missing birds
    missing_birds_tracker = find_birds_not_in_database(
        csv_directory, cross_fostered_path, home_reared_path
    )

    if missing_birds_tracker:
        print(f"\n=== SUMMARY ===")
        print(f"Total unique birds missing from database: {len(missing_birds_tracker)}")
        print(f"Missing birds: {sorted(missing_birds_tracker.keys())}")

        # Create unique missing birds CSV
        output_path = os.path.join(csv_directory, 'unique_birds_not_in_database.csv')
        unique_df = create_unique_missing_birds_csv(missing_birds_tracker, output_path)

        # Show birds that appear in multiple files
        multi_file_birds = {bird: info for bird, info in missing_birds_tracker.items()
                            if len(info['source_files']) > 1}
        if multi_file_birds:
            print(f"\nBirds appearing in multiple files ({len(multi_file_birds)}):")
            for bird, info in sorted(multi_file_birds.items(),
                                     key=lambda x: len(x[1]['source_files']), reverse=True):
                print(f"  {bird}: {len(info['source_files'])} files - {', '.join(info['source_files'])}")

        else:
            print("All birds found in CSV files are already in the database spreadsheets!")

#
# if __name__ == "__main__":
#     # Your directory path
#     path_to_csvs = os.path.join('/Users', 'annietaylor', 'Documents', 'ucsf', 'brainard', 'x-foster')
#     csvs = os.listdir(path_to_csvs)
#     csvs.remove('.DS_Store')
#
#     # Read all the files into a dictionary
#     spreadsheet_data = {}
#
#     for filename in csvs:
#         filepath = os.path.join(path_to_csvs, filename)
#
#         # Skip directories
#         if os.path.isdir(filepath):
#             continue
#
#         try:
#             if filename.endswith('.csv'):
#                 df = pd.read_csv(filepath)
#                 spreadsheet_data[filename] = df
#             elif filename.endswith('.xlsx'):
#                 df = pd.read_excel(filepath)
#                 spreadsheet_data[filename] = df
#             elif filename.endswith('.txt'):
#                 df = pd.read_csv(filepath, sep='\t')  # assuming tab-separated
#                 spreadsheet_data[filename] = df
#             # Skip .numbers files for now
#
#             print(f"Loaded {filename}: {df.shape}")
#
#         except Exception as e:
#             print(f"Could not read {filename}: {e}")
#
#     # Run the analysis using your original functions
#     results = compare_spreadsheets(spreadsheet_data)
#     # generate_redundancy_report(results)
#
#     export_individual_results_csv(results, os.path.join(path_to_csvs, "bird_analysis_results.csv"))
#
#     path_to_csvs = os.path.join('/Users', 'annietaylor', 'Documents', 'ucsf', 'brainard', 'x-foster')
#     filepath = os.path.join(path_to_csvs, 'gmm_simdat_hmm_2mod_all_new_thresh_sorted_msb2.txt')
#     df = pd.read_csv(filepath, sep='\t')  # assuming tab-separated
#
#     # Run the overlap analysis
#     overlap_results, detailed_results = analyze_bird_overlap()
#
# #convert_tab_file_to_csv(input_file="/Users/annietaylor/Documents/ucsf/brainard/x-foster/"
# #                                   "gmm_simdat_hmm_2mod_all_new_thresh_sorted_msb2.txt")
