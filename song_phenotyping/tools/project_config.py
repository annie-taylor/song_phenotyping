"""Machine-local path configuration for the song phenotyping pipeline.

Reads ``config.yaml`` from the project root (or a path you specify). Each
machine keeps its own ``config.yaml`` (gitignored); ``config.yaml.example``
is committed as a template.

Examples
--------
>>> from song_phenotyping.tools.project_config import ProjectConfig
>>> cfg = ProjectConfig.load()

>>> # Resolve a bird directory on the local cache drive
>>> bird_path = cfg.bird_dir('or18or24', experiment='evsong test')
>>> # → /Volumes/Extreme SSD/evsong test/or18or24

>>> # Get the Macaw server root (auto-detected if not set in config)
>>> macaw = cfg.macaw_root

>>> # Access pipeline run settings
>>> pipe = cfg.pipeline
>>> pipe.save_path        # where outputs are written
>>> pipe.evsong_source    # parent dir of evsonganaly bird folders
>>> pipe.birds            # None = all, or list of bird IDs to process
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Pipeline run settings loaded from the ``pipeline:`` block in config.yaml.

    All fields have sensible defaults so that a minimal config.yaml (one that
    only sets ``paths:``) still works — the pipeline will fall back to writing
    outputs next to the source data and processing all discovered birds.

    Sub-section dicts (``spectrogram_params``, ``embedding_params``,
    ``labelling_params``, ``phenotyping_params``) are kept as plain dicts so
    each stage can merge them into its own typed dataclass.
    """

    save_path: Optional[Path]           # Where to write pipeline outputs
    evsong_source: Optional[Path]       # Parent dir containing evsonganaly bird folders
    wseg_metadata: Optional[Path]       # wseg metadata dir; None = skip wseg
    birds: Optional[List[str]]          # None = auto-discover all; or explicit list
    songs_per_bird: Optional[int]       # None = all songs
    songs_seed: Optional[int]           # None = non-deterministic; int = reproducible subset
    spectrogram_params: Optional[dict]  # Merged into SpectrogramParams at call site
    embedding_params: Optional[dict]    # Merged into UMAPParams / grid at call site
    labelling_params: Optional[dict]    # metrics, weights, HDBSCAN grid, flags
    phenotyping_params: Optional[dict]  # Merged into PhenotypingConfig at call site
    generate_catalog: bool              # Whether to run generate_all_catalogs() after Stage E

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        def _path(v) -> Optional[Path]:
            return Path(os.path.expanduser(str(v))) if v is not None else None

        return cls(
            save_path           = _path(d.get('save_path')),
            evsong_source       = _path(d.get('evsong_source')),
            wseg_metadata       = _path(d.get('wseg_metadata')),
            birds               = d.get('birds'),
            songs_per_bird      = d.get('songs_per_bird'),
            songs_seed          = d.get('songs_seed'),
            spectrogram_params  = d.get('spectrograms') or {},
            embedding_params    = d.get('embedding') or {},
            labelling_params    = d.get('labelling') or {},
            phenotyping_params  = d.get('phenotyping') or {},
            generate_catalog    = bool(d.get('generate_catalog', True)),
        )

    @classmethod
    def empty(cls) -> "PipelineConfig":
        """Return a default PipelineConfig for use when config.yaml lacks a pipeline: block."""
        return cls(
            save_path=None, evsong_source=None, wseg_metadata=None,
            birds=None, songs_per_bird=None, songs_seed=None,
            spectrogram_params={}, embedding_params={},
            labelling_params={}, phenotyping_params={},
            generate_catalog=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root(start: Path) -> Path:
    """Walk up from start until we find config.yaml or config.yaml.example."""
    current = start.resolve()
    for _ in range(10):  # don't walk forever
        if (current / 'config.yaml').exists() or (current / 'config.yaml.example').exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fall back to the directory this file lives in (tools/ → project root)
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# ProjectConfig
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Machine-local path settings loaded from config.yaml."""

    local_cache: Path          # root of local data cache (external drive)
    macaw_root: Optional[Path] # Macaw server root; None = auto-detect
    run_registry: Path         # path to SQLite run registry
    pipeline: PipelineConfig   # pipeline run settings (save_path, sources, birds, …)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ProjectConfig":
        """
        Load config from a YAML file.

        If config_path is None, searches upward from cwd for config.yaml,
        then falls back to config.yaml.example.

        Raises FileNotFoundError if neither file is found.
        Raises ImportError if PyYAML is not installed.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for ProjectConfig.load(). "
                "Install it with: conda install pyyaml"
            )

        if config_path is not None:
            path = Path(config_path)
        else:
            root = _find_project_root(Path.cwd())
            path = root / 'config.yaml'
            if not path.exists():
                path = root / 'config.yaml.example'

        if not path.exists():
            raise FileNotFoundError(
                f"No config.yaml found. Copy config.yaml.example to config.yaml "
                f"and edit it for your machine. (Searched from: {Path.cwd()})"
            )

        with open(path) as f:
            raw = yaml.safe_load(f)

        paths = raw.get('paths', {})

        local_cache_raw = paths.get('local_cache')
        if local_cache_raw is None:
            raise ValueError("config.yaml must specify paths.local_cache")
        local_cache = Path(os.path.expanduser(str(local_cache_raw)))

        macaw_raw = paths.get('macaw_root')
        if macaw_raw is not None:
            macaw_root = Path(os.path.expanduser(str(macaw_raw)))
        else:
            macaw_root = cls._autodetect_macaw()

        registry_raw = paths.get('run_registry', 'db.sqlite3')
        registry_path = Path(os.path.expanduser(str(registry_raw)))
        if not registry_path.is_absolute():
            registry_path = path.parent / registry_path

        pipeline = PipelineConfig.from_dict(raw.get('pipeline', {}))

        return cls(
            local_cache=local_cache,
            macaw_root=macaw_root,
            run_registry=registry_path,
            pipeline=pipeline,
        )

    @staticmethod
    def _autodetect_macaw() -> Optional[Path]:
        """Return the Macaw server root for the current OS, or None if not mounted."""
        try:
            from song_phenotyping.tools.system_utils import check_sys_for_macaw_root
        try:
            root = check_sys_for_macaw_root()
            p = Path(root)
            return p if p.exists() else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def bird_dir(self, bird_id: str, experiment: str | None = None) -> Path:
        """
        Return the local cache directory for a bird.

        Parameters
        ----------
        bird_id : str
            Bird identifier, e.g. 'or18or24'.
        experiment : str, optional
            Experiment subdirectory name, e.g. 'evsong test' or 'wseg test'.
            If None, returns local_cache / bird_id directly.

        Examples
        --------
        cfg.bird_dir('or18or24', 'evsong test')
        # → /Volumes/Extreme SSD/evsong test/or18or24
        """
        if experiment:
            return self.local_cache / experiment / bird_id
        return self.local_cache / bird_id

    def macaw_bird_dir(self, *parts: str) -> Optional[Path]:
        """
        Join parts onto macaw_root, or return None if macaw is not mounted.

        Example: cfg.macaw_bird_dir('annietaylor', 'x-foster')
        """
        if self.macaw_root is None:
            return None
        return self.macaw_root.joinpath(*parts)
