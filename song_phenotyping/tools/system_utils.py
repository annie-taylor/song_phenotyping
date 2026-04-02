from sys import platform
from pathlib import Path
from typing import Union
import os
import tables


def check_sys_for_macaw_root():
    linux_path_to_macaw = '/run/user/1005/gvfs/smb-share:server=macaw.local,share=users/'
    mac_path_to_macaw = '/Volumes/users/'
    pc_path_to_macaw = 'Z:\\'
    if platform == "linux" or platform == "linux2":
        return linux_path_to_macaw
    elif platform == "darwin":
        return mac_path_to_macaw
    elif platform == "win32":
        return pc_path_to_macaw
    else:
        return False


def fix_mixture_of_separators(fname: str) -> str:
    """
    Normalize file path separators for current operating system.

    Parameters
    ----------
    fname : str
        File path that may contain mixed separators.

    Returns
    -------
    str
        File path with appropriate separators for current platform.

    Notes
    -----
    Platform-specific behavior:
        - Windows: Uses backslashes, maps to Z: drive
        - Linux: Uses forward slashes, maps to SMB share
        - MacOS: Uses forward slashes, maps to /Volumes/users

    See Also
    --------
    os.path.join : Platform-independent path joining
    pathlib.Path : Alternative path handling

    Examples
    --------
    >>> # On Windows
    >>> fix_mixture_of_separators('Z:/data/bird/song.wav')
    'Z:\\data\\bird\\song.wav'

    >>> # On MacOS
    >>> fix_mixture_of_separators('Z:\\data\\bird\\song.wav')
    '/Volumes/users/data/bird/song.wav'
    """
    if platform == 'win32':
        fname = fname.replace('/', '\\')  # use Windows-specific separators
        root = 'Z:\\'
        if root not in fname:
            song_file_path = root + '\\'.join(fname.split('\\')[1:])
        else:
            song_file_path = fname
    elif platform == 'linux':
        fname = fname.replace('\\', '/')
        root = '/run/user/1005/gvfs/smb-share:server=macaw.local,share=users/'
        if root not in fname:
            song_file_path = os.path.join(root,'/'.join(fname.split('/')[1:]))
        else: song_file_path = fname
    else:
        fname = fname.replace('\\', '/')  # use MacOS-specific separators
        root = '/Volumes/users/'
        if root not in fname:
            song_file_path =  root + '/'.join(fname.split('/')[1:])
        else: song_file_path = fname
    return song_file_path


def replace_macaw_root(filename: Union[str, Path]) -> Path:
    linux_path_to_macaw = '/run/user/1005/gvfs/smb-share:server=macaw.local,share=users/'
    mac_path_to_macaw = '/Volumes/users/'
    pc_path_to_macaw = 'Z:\\'
    possible_paths = [linux_path_to_macaw, mac_path_to_macaw, pc_path_to_macaw, 'Y:/', 'Y:\\']
    # TODO this may not be working as expected, fix_mixture_of_separators may be a better solution to this problem
    filename = str(filename)
    # convert path separator to match OS
    if platform in ["linux", "linux2"]:
        for path in possible_paths:
            if path in filename:
                filename = filename.replace(path, linux_path_to_macaw)
    elif platform == "darwin":
        for path in possible_paths:
            if path in filename:
                filename = filename.replace(path, mac_path_to_macaw)
    elif platform == "win32":
        for path in possible_paths:
            if path in filename:
                filename = filename.replace(path, pc_path_to_macaw)
    if '/' in filename and os.sep != '/':
        filename = filename.replace('/', '\\')
    elif '\\' in filename and os.sep != '\\':
        filename = filename.replace('\\', '/')
    return Path(filename)


def optimize_pytables_for_network():
    """Configure PyTables for better network performance"""
    # Increase cache sizes
    tables.parameters.CHUNK_CACHE_SIZE = 50 * 1024 * 1024  # 50MB
    tables.parameters.NODE_CACHE_SLOTS = 1024

    # Use larger I/O buffer
    tables.parameters.IO_BUFFER_SIZE = 1024 * 1024  # 1MB