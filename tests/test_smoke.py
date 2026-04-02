"""
Smoke tests for the A→B→C→D→E pipeline.

Each stage test:
  1. Runs the real pipeline function on a small slice of test data
  2. Asserts the output files exist in the expected location
  3. Asserts the output files have the expected HDF5/pickle structure

Tests are parameterised over both bird types (evsonganaly and wseg) where
applicable, and are automatically skipped if the test data is not yet on the SSD.

Speed: songs_per_bird=3 and minimal UMAP params keep the full suite under ~5 min.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import tables

sys.path.insert(0, str(Path(__file__).parent))

from conftest import (
    EVSONG_BIRD, WSEG_BIRD,
    requires_evsong, requires_wseg,
)
from tools.spectrogram_configs import SpectrogramParams

# Minimal params to keep smoke tests fast
_SPEC_PARAMS  = SpectrogramParams(songs_per_bird=3)
# explore_embedding_parameters_robust takes separate lists, not a UMAPParams object
_MIN_DISTS      = [0.3]
_N_NEIGHBORS    = [5]
_METRICS        = ['silhouette', 'dbi', 'ch']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h5_keys(path: str) -> list[str]:
    with tables.open_file(path, 'r') as f:
        return [node._v_name for node in f.list_nodes(f.root)]


def _first_h5(directory: Path) -> Path:
    files = list(directory.glob("*.h5"))
    assert files, f"No .h5 files found in {directory}"
    return files[0]


# ---------------------------------------------------------------------------
# Stage A — Spectrogram saving
# ---------------------------------------------------------------------------

class TestStageA:

    @requires_evsong
    def test_evsonganaly_filepath_discovery(self, evsong_source_dir):
        """filepaths_from_evsonganaly finds or18or24 and returns non-empty dicts."""
        from A_spec_saving import filepaths_from_evsonganaly
        meta, audio = filepaths_from_evsonganaly(
            wav_directory=evsong_source_dir,
            bird_subset=[EVSONG_BIRD],
        )
        assert EVSONG_BIRD in meta, f"Bird {EVSONG_BIRD} not found in metadata dict"
        assert len(meta[EVSONG_BIRD]) > 0, "No metadata files found"

    @requires_evsong
    def test_evsonganaly_spec_saving(self, evsong_source_dir, evsong_bird_dir):
        """save_specs_for_evsonganaly_birds produces HDF5 files with required keys."""
        from A_spec_saving import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds

        meta, audio = filepaths_from_evsonganaly(
            wav_directory=evsong_source_dir,
            bird_subset=[EVSONG_BIRD],
        )
        save_specs_for_evsonganaly_birds(
            metadata_file_paths=meta,
            audio_file_paths=audio,
            save_path=str(evsong_bird_dir),
            songs_per_bird=_SPEC_PARAMS.songs_per_bird,
            params=_SPEC_PARAMS,
        )

        specs_dir = evsong_bird_dir / EVSONG_BIRD / "syllable_data" / "specs"
        assert specs_dir.exists(), f"specs dir not created: {specs_dir}"
        h5_files = list(specs_dir.glob("syllables_*.h5"))
        assert len(h5_files) > 0, "No syllable HDF5 files produced"

        keys = _h5_keys(str(h5_files[0]))
        for required in ("spectrograms", "manual", "position_idxs", "hashes"):
            assert required in keys, f"Missing key '{required}' in {h5_files[0].name}"

    @requires_wseg
    def test_wseg_filepath_discovery(self, wseg_metadata_dir):
        """filepaths_from_wseg finds bu78bu77 and returns non-empty metadata dict."""
        from A_spec_saving import filepaths_from_wseg
        meta, audio = filepaths_from_wseg(
            seg_directory=wseg_metadata_dir,
            bird_subset=[WSEG_BIRD],
        )
        assert WSEG_BIRD in meta, f"Bird {WSEG_BIRD} not found in metadata dict"
        assert len(meta[WSEG_BIRD]) > 0, "No metadata files found"

    @requires_wseg
    def test_wseg_spec_saving(self, wseg_metadata_dir, wseg_bird_dir):
        """save_specs_for_wseg_birds produces HDF5 files with required keys."""
        from A_spec_saving import filepaths_from_wseg, save_specs_for_wseg_birds

        meta, audio = filepaths_from_wseg(
            seg_directory=wseg_metadata_dir,
            bird_subset=[WSEG_BIRD],
        )
        save_specs_for_wseg_birds(
            metadata_file_paths=meta,
            audio_file_paths=audio,
            save_path=str(wseg_bird_dir),
            songs_per_bird=_SPEC_PARAMS.songs_per_bird,
            params=_SPEC_PARAMS,
        )

        specs_dir = wseg_bird_dir / WSEG_BIRD / "syllable_data" / "specs"
        assert specs_dir.exists(), f"specs dir not created: {specs_dir}"
        h5_files = list(specs_dir.glob("syllables_*.h5"))
        assert len(h5_files) > 0, "No syllable HDF5 files produced"

        keys = _h5_keys(str(h5_files[0]))
        for required in ("spectrograms", "position_idxs", "hashes"):
            assert required in keys, f"Missing key '{required}' in {h5_files[0].name}"


# ---------------------------------------------------------------------------
# Stage B — Flattening
# ---------------------------------------------------------------------------

class TestStageB:

    def _run_a_evsong(self, evsong_source_dir, out_dir):
        from A_spec_saving import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
        meta, audio = filepaths_from_evsonganaly(
            wav_directory=evsong_source_dir, bird_subset=[EVSONG_BIRD]
        )
        save_specs_for_evsonganaly_birds(meta, audio, str(out_dir),
                                         songs_per_bird=3, params=_SPEC_PARAMS)

    @requires_evsong
    def test_flatten_produces_h5(self, evsong_source_dir, evsong_bird_dir):
        """flatten_bird_spectrograms creates flattened HDF5 with correct shape."""
        from B_flattening import flatten_bird_spectrograms
        self._run_a_evsong(evsong_source_dir, evsong_bird_dir)

        result = flatten_bird_spectrograms(
            directory=str(evsong_bird_dir), bird=EVSONG_BIRD
        )
        assert result is True, "flatten_bird_spectrograms returned False"

        flat_dir = evsong_bird_dir / EVSONG_BIRD / "syllable_data" / "flattened"
        assert flat_dir.exists()
        h5_files = list(flat_dir.glob("flattened_*.h5"))
        assert len(h5_files) > 0, "No flattened HDF5 files produced"

        # Verify shape: (n_features, n_syllables) = (513*300, n)
        with tables.open_file(str(h5_files[0]), 'r') as f:
            shape = f.root.flattened_specs.shape
        assert shape[0] == 513 * 300, (
            f"Expected {513*300} features, got {shape[0]}"
        )
        assert shape[1] > 0, "Zero syllables in flattened file"


# ---------------------------------------------------------------------------
# Stage C — UMAP Embedding
# ---------------------------------------------------------------------------

class TestStageC:

    def _run_ab_evsong(self, evsong_source_dir, out_dir):
        from A_spec_saving import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
        from B_flattening import flatten_bird_spectrograms
        meta, audio = filepaths_from_evsonganaly(
            wav_directory=evsong_source_dir, bird_subset=[EVSONG_BIRD]
        )
        save_specs_for_evsonganaly_birds(meta, audio, str(out_dir),
                                         songs_per_bird=3, params=_SPEC_PARAMS)
        flatten_bird_spectrograms(str(out_dir), EVSONG_BIRD)

    @requires_evsong
    def test_embedding_produces_pkl(self, evsong_source_dir, evsong_bird_dir):
        """explore_embedding_parameters_robust creates embedding files with shape (n, 2)."""
        from C_embedding import explore_embedding_parameters_robust
        self._run_ab_evsong(evsong_source_dir, evsong_bird_dir)

        explore_embedding_parameters_robust(
            save_path=str(evsong_bird_dir),
            bird=EVSONG_BIRD,
            min_dists=_MIN_DISTS,
            n_neighbors_list=_N_NEIGHBORS,
            use_parallel=False,
        )

        embed_dir = evsong_bird_dir / EVSONG_BIRD / "syllable_data" / "embeddings"
        assert embed_dir.exists(), f"Embeddings dir not created: {embed_dir}"
        h5_files = list(embed_dir.glob("*.h5"))
        assert len(h5_files) > 0, "No embedding files produced"

        with tables.open_file(str(h5_files[0]), 'r') as f:
            emb = f.root.embeddings.read()
        assert emb.ndim == 2, f"Expected 2D embeddings, got shape {emb.shape}"
        assert emb.shape[1] == 2, f"Expected 2 components, got {emb.shape[1]}"
        assert emb.shape[0] > 0, "Zero syllables in embedding"


# ---------------------------------------------------------------------------
# Stage D — Clustering / Labelling
# ---------------------------------------------------------------------------

class TestStageD:

    def _run_abc_evsong(self, evsong_source_dir, out_dir):
        from A_spec_saving import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
        from B_flattening import flatten_bird_spectrograms
        from C_embedding import explore_embedding_parameters_robust
        meta, audio = filepaths_from_evsonganaly(
            wav_directory=evsong_source_dir, bird_subset=[EVSONG_BIRD]
        )
        save_specs_for_evsonganaly_birds(meta, audio, str(out_dir),
                                         songs_per_bird=3, params=_SPEC_PARAMS)
        flatten_bird_spectrograms(str(out_dir), EVSONG_BIRD)
        explore_embedding_parameters_robust(str(out_dir), EVSONG_BIRD,
                                            min_dists=_MIN_DISTS,
                                            n_neighbors_list=_N_NEIGHBORS,
                                            use_parallel=False)

    @requires_evsong
    def test_labelling_produces_cluster_files(self, evsong_source_dir, evsong_bird_dir):
        """label_bird creates cluster label files in the expected directory."""
        from D_labelling import label_bird, HDBSCANParams
        self._run_abc_evsong(evsong_source_dir, evsong_bird_dir)

        result = label_bird(
            save_path=str(evsong_bird_dir),
            bird=EVSONG_BIRD,
            metrics=_METRICS,
            hdbscan_params=[HDBSCANParams(min_cluster_size=5, min_samples=3).to_dict()],
        )
        assert result is True, "label_bird returned False"

        labelling_dir = evsong_bird_dir / EVSONG_BIRD / "syllable_data" / "labelling"
        assert labelling_dir.exists(), f"Labelling dir not created: {labelling_dir}"
        assert any(labelling_dir.rglob("*.h5")), "No cluster label files produced"


# ---------------------------------------------------------------------------
# Stage E — Phenotyping
# ---------------------------------------------------------------------------

class TestStageE:

    @requires_evsong
    def test_phenotyping_produces_json(self, evsong_source_dir, evsong_bird_dir):
        """phenotype_bird produces a JSON analysis file with expected top-level keys."""
        from A_spec_saving import filepaths_from_evsonganaly, save_specs_for_evsonganaly_birds
        from B_flattening import flatten_bird_spectrograms
        from C_embedding import explore_embedding_parameters_robust
        from D_labelling import label_bird, HDBSCANParams
        from E_phenotyping import phenotype_bird, PhenotypingConfig

        meta, audio = filepaths_from_evsonganaly(
            wav_directory=evsong_source_dir, bird_subset=[EVSONG_BIRD]
        )
        save_specs_for_evsonganaly_birds(meta, audio, str(evsong_bird_dir),
                                         songs_per_bird=3, params=_SPEC_PARAMS)
        flatten_bird_spectrograms(str(evsong_bird_dir), EVSONG_BIRD)
        explore_embedding_parameters_robust(str(evsong_bird_dir), EVSONG_BIRD,
                                            min_dists=_MIN_DISTS,
                                            n_neighbors_list=_N_NEIGHBORS,
                                            use_parallel=False)
        label_bird(
            save_path=str(evsong_bird_dir),
            bird=EVSONG_BIRD,
            metrics=_METRICS,
            hdbscan_params=[HDBSCANParams(min_cluster_size=5, min_samples=3).to_dict()],
        )

        bird_path = str(evsong_bird_dir / EVSONG_BIRD)
        result = phenotype_bird(
            bird_path=bird_path,
            config=PhenotypingConfig(generate_plots=False),
        )
        assert result is True, "phenotype_bird returned False"

        # Check phenotype pickle output
        phenotype_dir = evsong_bird_dir / EVSONG_BIRD / "syllable_data" / "phenotype_detailed"
        assert phenotype_dir.exists(), f"Phenotype dir not created: {phenotype_dir}"
        pkl_files = list(phenotype_dir.glob("automated_phenotype_data_rank*.pkl"))
        assert pkl_files, "No automated phenotype pickle files produced"

        with open(pkl_files[0], 'rb') as f:
            data = pickle.load(f)
        phenotype = data['phenotype_results']
        for key in ("bird_name", "vocabulary", "transition_matrix"):
            assert key in phenotype, f"Missing key '{key}' in phenotype_results"
