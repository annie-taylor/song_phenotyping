"""Unit tests for individual pipeline functions (no I/O, no full pipeline run)."""

import numpy as np
import pytest
import tempfile
import os


# ---------------------------------------------------------------------------
# Stage B — flattening
# ---------------------------------------------------------------------------

class TestFlattenSpectrograms:
    from song_phenotyping.flattening import flatten_spectrograms

    def test_basic_shape(self):
        from song_phenotyping.flattening import flatten_spectrograms
        specs = np.random.rand(5, 10, 20).astype(np.float32)
        out = flatten_spectrograms(specs)
        assert out.shape == (10 * 20, 5)
        assert out.dtype == np.float32

    def test_with_inst_freq(self):
        from song_phenotyping.flattening import flatten_spectrograms
        n, nf, nt = 4, 10, 20
        specs = np.random.rand(n, nf, nt).astype(np.float32)
        inst_freq = np.random.rand(n, nf, nt - 1).astype(np.float32)
        out = flatten_spectrograms(specs, inst_freq=inst_freq)
        expected_features = nf * nt + nf * (nt - 1)
        assert out.shape == (expected_features, n)

    def test_with_group_delay(self):
        from song_phenotyping.flattening import flatten_spectrograms
        n, nf, nt = 4, 10, 20
        specs = np.random.rand(n, nf, nt).astype(np.float32)
        group_delay = np.random.rand(n, nf - 1, nt).astype(np.float32)
        out = flatten_spectrograms(specs, group_delay=group_delay)
        expected_features = nf * nt + (nf - 1) * nt
        assert out.shape == (expected_features, n)

    def test_with_duration_weight(self):
        from song_phenotyping.flattening import flatten_spectrograms
        n, nf, nt = 4, 10, 20
        specs = np.random.rand(n, nf, nt).astype(np.float32)
        durations = np.array([0.5, 0.8, 0.3, 1.0])
        out = flatten_spectrograms(specs, durations=durations, duration_feature_weight=1.0)
        # duration block appends nf features per syllable
        assert out.shape == (nf * nt + nf, n)

    def test_duration_disabled_when_weight_zero(self):
        from song_phenotyping.flattening import flatten_spectrograms
        n, nf, nt = 4, 10, 20
        specs = np.random.rand(n, nf, nt).astype(np.float32)
        durations = np.ones(n)
        out_no_dur = flatten_spectrograms(specs, durations=durations, duration_feature_weight=0.0)
        out_baseline = flatten_spectrograms(specs)
        assert out_no_dur.shape == out_baseline.shape

    def test_all_channels_combined(self):
        from song_phenotyping.flattening import flatten_spectrograms
        n, nf, nt = 3, 8, 16
        specs = np.random.rand(n, nf, nt).astype(np.float32)
        inst_freq = np.random.rand(n, nf, nt - 1).astype(np.float32)
        group_delay = np.random.rand(n, nf - 1, nt).astype(np.float32)
        durations = np.array([0.4, 0.6, 0.9])
        out = flatten_spectrograms(specs, inst_freq=inst_freq, group_delay=group_delay,
                                   durations=durations, duration_feature_weight=2.0)
        expected = nf * nt + nf * (nt - 1) + (nf - 1) * nt + nf
        assert out.shape == (expected, n)

    def test_raises_on_empty(self):
        from song_phenotyping.flattening import flatten_spectrograms
        with pytest.raises(ValueError, match="empty"):
            flatten_spectrograms(np.array([]).reshape(0, 10, 20))

    def test_raises_on_wrong_ndim(self):
        from song_phenotyping.flattening import flatten_spectrograms
        with pytest.raises(ValueError, match="3D"):
            flatten_spectrograms(np.random.rand(10, 20))


# ---------------------------------------------------------------------------
# Stage B — helper functions
# ---------------------------------------------------------------------------

class TestExtractSongId:
    def test_typical_filename(self):
        from song_phenotyping.flattening import extract_song_id
        assert extract_song_id("/some/path/syllables_or18or24_20230101.h5") == "or18or24_20230101"

    def test_raises_on_bad_filename(self):
        from song_phenotyping.flattening import extract_song_id
        with pytest.raises(ValueError):
            extract_song_id("/some/path/spectrogram_or18or24.h5")


class TestCreateFlattenedOutputPath:
    def test_creates_flattened_subdir(self):
        from song_phenotyping.flattening import create_flattened_output_path
        with tempfile.TemporaryDirectory() as tmpdir:
            out = create_flattened_output_path(tmpdir, "or18or24_20230101")
            assert out.endswith("flattened_or18or24_20230101.h5")
            assert os.path.isdir(os.path.join(tmpdir, "flattened"))


# ---------------------------------------------------------------------------
# SpectrogramParams validation
# ---------------------------------------------------------------------------

class TestSpectrogramParamsValidation:
    def test_default_params_valid(self):
        from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
        SpectrogramParams().validate_params()  # should not raise

    def test_invalid_nfft(self):
        from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
        with pytest.raises(ValueError, match="nfft"):
            SpectrogramParams(nfft=0).validate_params()

    def test_invalid_freq_range(self):
        from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
        with pytest.raises(ValueError, match="max_freq"):
            SpectrogramParams(min_freq=5000, max_freq=1000).validate_params()

    def test_zero_duration_weight_is_valid(self):
        from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
        SpectrogramParams(duration_feature_weight=0.0).validate_params()

    def test_phase_flags_default_false(self):
        from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
        p = SpectrogramParams()
        assert p.save_inst_freq is False
        assert p.save_group_delay is False
        assert p.duration_feature_weight == 0.0
