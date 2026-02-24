import os
import logging
from typing import Dict


def extract_base_name(filepath: str) -> str:
    """
    Extract comparable base name from either saved files or metadata paths.

    Examples:
    - 'syllables_or26or1_280923_073834.h5' -> 'or26or1_280923_073834'
    - '/path/rd12_280923.not.mat' -> 'rd12_280923'
    """
    try:
        filename = os.path.basename(filepath)

        # Remove known prefixes and suffixes
        if filename.startswith('syllables_'):
            filename = filename[10:]  # Remove 'syllables_'

        # TODO truthfully I don't remember why all these cases might be needed? Seems like .h5 should be enough
        if filename.endswith('.h5'):
            filename = filename[:-3]
        elif filename.endswith('.wav.not.mat'):
            filename = filename[:-12]
        elif filename.endswith('.wav'):
            filename = filename[:-4]
        elif filename.endswith('.cbin'):
            filename = filename[:-5]

        return filename

    except Exception as e:
        logging.warning(f"Could not extract base name from {filepath}: {e}")
        return None

def load_audio_paths_mapping(bird_folder: str) -> Dict[str, Dict[str, str]]:
    """
    Load the audio paths mapping from bird's audio_paths.txt file.

    Returns:
        Dict mapping filenames to {'local': path_or_none, 'server': path}
    """
    audio_paths_file = os.path.join(bird_folder, 'audio_paths.txt')
    mapping = {}

    if not os.path.exists(audio_paths_file):
        return mapping

    try:
        with open(audio_paths_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) == 3:
                        bird_id, local_path, server_path = parts
                        filename = os.path.basename(server_path)
                        mapping[filename] = {
                            'local': None if local_path == 'NONE' else local_path,
                            'server': server_path
                        }
    except Exception as e:
        logging.error(f"Error loading audio paths mapping from {audio_paths_file}: {e}")

    return mapping


def update_audio_paths_file(bird_folder: str, filename: str,
                            local_path: str = None, server_path: str = None):
    """
    Add or update entry in bird's audio_paths.txt file.
    """
    audio_paths_file = os.path.join(bird_folder, 'audio_paths.txt')

    # Load existing mapping
    mapping = load_audio_paths_mapping(bird_folder)

    # Update or add entry
    if filename not in mapping:
        mapping[filename] = {'local': None, 'server': None}

    if local_path is not None:
        mapping[filename]['local'] = local_path
    if server_path is not None:
        mapping[filename]['server'] = server_path

    # Write back to file
    try:
        with open(audio_paths_file, 'w') as f:
            f.write("# Format: bird_id|local_path|server_path\n")
            bird_id = os.path.basename(bird_folder)
            for fname, paths in mapping.items():
                local = paths['local'] if paths['local'] else 'NONE'
                server = paths['server'] if paths['server'] else 'NONE'
                f.write(f"{bird_id}|{local}|{server}\n")
    except Exception as e:
        logging.error(f"Error updating audio paths file {audio_paths_file}: {e}")


def get_audio_path(bird_folder: str, filepath_or_filename: str, prefer_local: bool = True) -> str:
    """
    Get audio file path with local/server preference.

    Args:
        bird_folder: Path to bird's folder
        filepath_or_filename: Full path or just filename
        prefer_local: If True, try local first, fallback to server
    """
    # Extract base filename and add .wav extension
    base_name = extract_base_name(filepath_or_filename)
    if base_name is None:
        raise ValueError(f"Could not extract base name from {filepath_or_filename}")

    filename = base_name + '.wav'

    # Load mapping
    mapping = load_audio_paths_mapping(bird_folder)

    # Get paths for this file
    if filename not in mapping:
        raise FileNotFoundError(f"No audio path mapping found for {filename}")

    paths = mapping[filename]

    # Try preferred path first
    if prefer_local and paths['local']:
        if os.path.exists(paths['local']):
            return paths['local']
        else:
            logging.warning(f"Local file not found, falling back to server: {paths['local']}")

    # Try server path
    if paths['server'] and os.path.exists(paths['server']):
        return paths['server']

    raise FileNotFoundError(f"Neither local nor server path exists for {filename}")

#
# def get_audio_path(bird_folder: str, filepath_or_filename: str, prefer_local: bool = True) -> str:
#     """
#     Get audio file path with local/server preference.
#
#     Args:
#         bird_folder: Path to bird's folder
#         filepath_or_filename: Full path or just filename
#         prefer_local: If True, try local first, fallback to server
#     """
#     # Extract base filename using your existing function
#     filename = extract_base_name(filepath_or_filename) + '.wav'  # Ensure .wav extension
#
#     # Load mapping (creates file if doesn't exist)
#     mapping = load_audio_paths_mapping(bird_folder)
#
#     # Get paths for this file
#     if filename not in mapping:
#         raise FileNotFoundError(f"No audio path mapping found for {filename}")
#
#     paths = mapping[filename]
#
#     # Try preferred path first
#     if prefer_local and paths['local'] != 'NONE':
#         if os.path.exists(paths['local']):
#             return paths['local']
#         else:
#             logging.warning(f"Local file not found, falling back to server: {paths['local']}")
#
#     # Try server path
#     if os.path.exists(paths['server']):
#         return paths['server']
#
#     raise FileNotFoundError(f"Neither local nor server path exists for {filename}")
