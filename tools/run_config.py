"""
RunConfig and RunRegistry: experiment metadata tracking for the song phenotyping pipeline.

RunConfig captures the full parameter set for one pipeline run (spectrogram generation
through phenotyping) so that any output file can be traced back to the exact inputs and
settings that produced it.

RunRegistry is a thin SQLite wrapper for storing and querying RunConfig records across runs.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tools.spectrogram_configs import SpectrogramParams


def _make_serializable(obj):
    """Recursively convert non-JSON-serializable types to serializable equivalents."""
    if isinstance(obj, range):
        return {'__range__': True, 'start': obj.start, 'stop': obj.stop, 'step': obj.step}
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def _restore_serialized(obj):
    """Reverse _make_serializable: reconstruct range objects from their dict form."""
    if isinstance(obj, dict):
        if obj.get('__range__'):
            return range(obj['start'], obj['stop'], obj['step'])
        return {k: _restore_serialized(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_serialized(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Sub-configs imported from their home modules
# UMAPParams lives in C_embedding; HDBSCANParams lives in D_labelling.
# We re-export them here so callers only need one import.
# ---------------------------------------------------------------------------

def _import_umap_params():
    from C_embedding import UMAPParams
    return UMAPParams

def _import_hdbscan_params():
    from D_labelling import HDBSCANParams
    return HDBSCANParams

def _import_phenotyping_config():
    from E_phenotyping import PhenotypingConfig
    return PhenotypingConfig


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Complete parameter record for one pipeline run.

    One RunConfig is created at the start of a run (or reconstructed from a saved JSON).
    The run_id is embedded in every HDF5/pickle output so results can always be traced
    back to the exact settings that produced them.
    """
    run_id: str
    created_at: str          # ISO 8601 timestamp (UTC)
    spec_mode: str           # "syllable" (Stage A) or "slice" (Stage A1)
    bird_ids: List[str]

    # Sub-configs stored as plain dicts for serialization simplicity.
    # Use RunConfig.spec_params_obj etc. to get typed objects back.
    spec_params: dict
    umap_params: dict
    hdbscan_params: dict     # winning/selected params after grid search
    phenotype_params: dict

    notes: str = ""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        spec_mode: str,
        bird_ids: List[str],
        spec_params=None,
        umap_params=None,
        hdbscan_params=None,
        phenotype_params=None,
        notes: str = "",
    ) -> "RunConfig":
        """Create a new RunConfig with a fresh run_id and current timestamp."""
        UMAPParams = _import_umap_params()
        HDBSCANParams = _import_hdbscan_params()
        PhenotypingConfig = _import_phenotyping_config()

        if spec_params is None:
            spec_params = SpectrogramParams()
        if umap_params is None:
            umap_params = UMAPParams()
        if hdbscan_params is None:
            hdbscan_params = HDBSCANParams()
        if phenotype_params is None:
            phenotype_params = PhenotypingConfig()

        def to_dict(obj):
            if hasattr(obj, 'to_dict'):
                d = obj.to_dict()
            elif hasattr(obj, '__dataclass_fields__'):
                d = asdict(obj)
            else:
                d = dict(obj)
            return _make_serializable(d)

        return cls(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            spec_mode=spec_mode,
            bird_ids=list(bird_ids),
            spec_params=to_dict(spec_params),
            umap_params=to_dict(umap_params),
            hdbscan_params=to_dict(hdbscan_params),
            phenotype_params=to_dict(phenotype_params),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    @property
    def spec_params_obj(self) -> SpectrogramParams:
        return SpectrogramParams(**{
            k: v for k, v in self.spec_params.items()
            if k in SpectrogramParams.__dataclass_fields__
        })

    @property
    def umap_params_obj(self):
        UMAPParams = _import_umap_params()
        return UMAPParams.from_dict(self.umap_params)

    @property
    def hdbscan_params_obj(self):
        HDBSCANParams = _import_hdbscan_params()
        return HDBSCANParams.from_dict(self.hdbscan_params)

    @property
    def phenotype_params_obj(self):
        PhenotypingConfig = _import_phenotyping_config()
        cfg = PhenotypingConfig()
        for k, v in self.phenotype_params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'created_at': self.created_at,
            'spec_mode': self.spec_mode,
            'bird_ids': self.bird_ids,
            'spec_params': self.spec_params,
            'umap_params': self.umap_params,
            'hdbscan_params': self.hdbscan_params,
            'phenotype_params': self.phenotype_params,
            'notes': self.notes,
        }

    def save(self, path: str):
        """Save this RunConfig as a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        """Load a RunConfig from a JSON file."""
        data = json.loads(Path(path).read_text())
        for key in ('spec_params', 'umap_params', 'hdbscan_params', 'phenotype_params'):
            if key in data:
                data[key] = _restore_serialized(data[key])
        return cls(**data)


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    spec_mode       TEXT NOT NULL,
    bird_ids        TEXT NOT NULL,   -- JSON array
    spec_params     TEXT NOT NULL,   -- JSON object
    umap_params     TEXT NOT NULL,   -- JSON object
    hdbscan_params  TEXT NOT NULL,   -- JSON object
    phenotype_params TEXT NOT NULL,  -- JSON object
    notes           TEXT DEFAULT ''
);
"""


class RunRegistry:
    """
    SQLite-backed store for RunConfig records.

    Usage::

        registry = RunRegistry("db.sqlite3")
        registry.register(run_config)
        df = registry.query(spec_mode="syllable")
        df = registry.query(bird_id="or18or24")   # searches within bird_ids list
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def register(self, config: RunConfig):
        """Insert a RunConfig. Silently replaces if run_id already exists."""
        row = config.to_dict()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, created_at, spec_mode, bird_ids,
                    spec_params, umap_params, hdbscan_params, phenotype_params, notes)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    row['run_id'],
                    row['created_at'],
                    row['spec_mode'],
                    json.dumps(row['bird_ids']),
                    json.dumps(row['spec_params']),
                    json.dumps(row['umap_params']),
                    json.dumps(row['hdbscan_params']),
                    json.dumps(row['phenotype_params']),
                    row['notes'],
                ),
            )

    def get(self, run_id: str) -> Optional[RunConfig]:
        """Retrieve a single RunConfig by run_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_config(row)

    def query(self, bird_id: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Return a DataFrame of matching runs.

        Parameters
        ----------
        bird_id : str, optional
            Filter to runs that include this bird (substring match on bird_ids JSON).
        **kwargs
            Exact-match filters on top-level columns: spec_mode, run_id, notes.
            To filter on nested params (e.g. n_neighbors=20), filter the returned
            DataFrame directly: df[df['umap_params'].apply(lambda p: p['n_neighbors']) == 20]
        """
        clauses = []
        values = []

        if bird_id is not None:
            clauses.append("bird_ids LIKE ?")
            values.append(f'%{bird_id}%')

        for col, val in kwargs.items():
            clauses.append(f"{col} = ?")
            values.append(val)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM runs {where} ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()

        records = [self._row_to_config(r).to_dict() for r in rows]
        df = pd.DataFrame(records)

        # Expand JSON columns so callers can filter by nested params
        for col in ['spec_params', 'umap_params', 'hdbscan_params', 'phenotype_params']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: v if isinstance(v, dict) else json.loads(v)
                )
        return df

    def all(self) -> pd.DataFrame:
        """Return all runs as a DataFrame."""
        return self.query()

    @staticmethod
    def _row_to_config(row) -> RunConfig:
        return RunConfig(
            run_id=row['run_id'],
            created_at=row['created_at'],
            spec_mode=row['spec_mode'],
            bird_ids=json.loads(row['bird_ids']),
            spec_params=_restore_serialized(json.loads(row['spec_params'])),
            umap_params=_restore_serialized(json.loads(row['umap_params'])),
            hdbscan_params=_restore_serialized(json.loads(row['hdbscan_params'])),
            phenotype_params=_restore_serialized(json.loads(row['phenotype_params'])),
            notes=row['notes'],
        )
