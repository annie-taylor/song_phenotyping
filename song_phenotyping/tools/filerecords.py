from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

from song_phenotyping.signal import parse_audio_filename


@dataclass
class FileRecord:
    """Resolved paths and parsed metadata for a single audio recording.

    Populated by :func:`audio_paths_txt_to_filerecords` or constructed
    directly.  Holds the canonical metadata path, the resolved audio path
    (local copy or server path), and parsed filename fields (bird, day, time).
    """

    # Canonical absolute path to the metadata file (expected: .wav.not.mat or .not.mat)
    metadata_path: Path

    # Resolved audio path if known (local copy or server absolute path). None if unknown.
    audio_path: Optional[Path] = None

    # True if audio_path is a local file on this machine (exists and is local)
    audio_is_local: bool = False

    # Original server path when audio_path is a local copy (can be None)
    server_audio_path: Optional[Path] = None

    # Parsed/validated metadata fields (fill when available)
    bird: Optional[str] = None
    day: Optional[str] = None
    time: Optional[str] = None

    # Where these parsed fields came from: 'metadata', 'filename', 'directory', 'inferred', etc.
    parse_source: Optional[str] = None

    # Optional freeform place to store extra metadata from the .mat (not persisted by default)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict with string paths (safe for JSON/logs)."""
        d = asdict(self)
        d['metadata_path'] = str(self.metadata_path) if self.metadata_path is not None else None
        d['audio_path'] = str(self.audio_path) if self.audio_path is not None else None
        d['server_audio_path'] = str(self.server_audio_path) if self.server_audio_path is not None else None
        return d

    @classmethod
    def from_paths(cls, metadata_path: str, audio_path: Optional[str] = None, **kwargs):
        """Convenience constructor accepting strings."""
        mp = Path(metadata_path)
        ap = Path(audio_path) if audio_path is not None else None
        audio_is_local = bool(ap and ap.exists())
        return cls(metadata_path=mp, audio_path=ap, audio_is_local=audio_is_local, **kwargs)

    def basename_key(self) -> str:
        """A stable 'bird_day_time' key if available, else fallback to metadata basename."""
        if self.bird and self.day and self.time:
            return f"{self.bird}_{self.day}_{self.time}"
        return self.metadata_path.stem

logger = logging.getLogger(__name__)

def audio_paths_txt_to_filerecords(
    audio_paths_txt: str,
    metadata_ext: str = '.wav.not.mat',
    bird_subset: Optional[List[str]] = None,
) -> Dict[str, List[FileRecord]]:
    """
    Read ``audio_paths.txt`` lines of the form ``bird_id|local_path|server_path``
    and return ``Dict[bird_id, List[FileRecord]]``.

    Parameters
    ----------
    audio_paths_txt : str
        Path to the ``audio_paths.txt`` file.
    metadata_ext : str
        Extension used to derive the metadata filename from an audio path.
    bird_subset : list of str, optional
        If given, only records for these bird IDs are returned.
    """
    records_by_bird: Dict[str, List[FileRecord]] = {}

    p = Path(audio_paths_txt)
    if not p.exists():
        raise FileNotFoundError(f"audio_paths_txt not found: {audio_paths_txt}")

    with p.open('r') as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('|')
            if len(parts) < 3:
                logger.warning("Skipping malformed line %d in %s: %r", line_no, audio_paths_txt, line)
                continue

            bird_id, local_path_str, server_path_str = parts[0].strip(), parts[1].strip(), parts[2].strip()

            if bird_subset is not None and bird_id not in bird_subset:
                continue

            local_path = Path(local_path_str) if local_path_str else None
            server_path = Path(server_path_str) if server_path_str else None

            # Derive metadata path (common evsonganaly convention)
            # If local copy exists, metadata probably lives near it with metadata_ext
            metadata_path = None
            if local_path:
                # audio file like ".../bird_xxx.wav" -> metadata ".../bird_xxx.wav.not.mat" or ".wav.not.mat"
                candidate1 = local_path.with_suffix(local_path.suffix + '.not.mat')  # e.g., .wav + .not.mat
                candidate2 = local_path.with_suffix('.not.mat')  # fallback: replace ext with .not.mat
                candidate3 = local_path.with_name(local_path.name + '.not.mat')  # fallback
                # Choose first existing or construct the default from metadata_ext
                if candidate1.exists():
                    metadata_path = candidate1
                elif candidate2.exists():
                    metadata_path = candidate2
                elif candidate3.exists():
                    metadata_path = candidate3
                else:
                    metadata_path = local_path.parent.joinpath(local_path.name + metadata_ext)

            elif server_path:
                # fallback to server path-derived metadata (no local copy)
                metadata_path = server_path.parent.joinpath(server_path.name + metadata_ext)

            # Build FileRecord
            fr = FileRecord.from_paths(
                metadata_path=str(metadata_path) if metadata_path is not None else '',
                audio_path=str(local_path) if (local_path and local_path.exists()) else (str(server_path) if server_path else None),
                bird=None, day=None, time=None, parse_source='audio_paths_txt'
            )

            # store server path explicitly (even if audio_path points to local)
            fr.server_audio_path = server_path if server_path is not None else None
            fr.audio_is_local = bool(local_path and local_path.exists())

            # Try to parse filename info using your existing parsing function if available
            try:
                # parse_audio_filename is expected to return a dict with 'success' and 'bird','day','time'
                if fr.audio_path:
                    parsed = parse_audio_filename(fr.audio_path)
                else:
                    parsed = parse_audio_filename(str(metadata_path))
                if parsed and parsed.get('success'):
                    fr.bird = parsed.get('bird')
                    fr.day = parsed.get('day')
                    fr.time = parsed.get('time')
                    # Mark that parsing used the filename (or metadata fallback)
                    fr.parse_source = 'filename' if fr.audio_path else 'metadata-fallback'
                else:
                    fr.parse_source = fr.parse_source or 'unparsed'
            except Exception as e:
                logger.debug("Parsing failed for %s: %s", fr.audio_path or metadata_path, e)
                fr.parse_source = fr.parse_source or 'parse-error'

            # Add to dict keyed by explicit bird_id from the text file (preserve original grouping)
            if bird_id not in records_by_bird:
                records_by_bird[bird_id] = []
            records_by_bird[bird_id].append(fr)

    return records_by_bird