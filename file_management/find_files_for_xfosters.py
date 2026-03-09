import logging
import sqlite3
from tools.dbquery import *
from x_foster_metadata import generate_bird_name_variations, get_all_audio_files_for_birds, load_bird_audio_data
from tools.system_utils import check_sys_for_macaw_root
from x_foster_audio_files import main_enhanced_deduplication


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

def main():
    root_dir = check_sys_for_macaw_root()
    save_file = "xfoster_audio_paths.json"

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

    bird_file_locations = get_file_locations_for_birds(birds, txt_file_path='../refs/MacawAllDirsByBird.txt',
                                                       db_path='../refs/2026-01-15-db.sqlite3')
    get_all_audio_files_for_birds(bird_file_locations, root_directory=root_dir, save_file=save_file)
    main_enhanced_deduplication(input_file="xfoster_audio_paths(new).json",
                                output_file="xfoster_audio_data_deduplicated(new).json",
                                report_file="xfoster_datetime_analysis_report(new).txt",
                                stats_file="comprehensive_dedup_stats(new).json")


if __name__ == "__main__":
    main()