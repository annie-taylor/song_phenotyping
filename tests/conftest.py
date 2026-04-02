"""
Pytest configuration and fixtures for song phenotyping smoke tests.

Test bird data lives on the external SSD under:
  /Volumes/Extreme SSD/smoke_test_birds/
    or18or24/source/18-08-2023/   ← evsonganaly source (audio + .not.mat co-located)
    bu78bu77/source_metadata/song/ ← wseg metadata (.not.mat files)
    bu78bu77/source_audio/         ← wseg audio (.wav files in dated subdirs)

All tests that require this data are skipped automatically if the SSD is not
mounted or the expected directories are not yet populated.
"""

from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# SSD paths
# ---------------------------------------------------------------------------

SSD_ROOT = Path("/Volumes/Extreme SSD/smoke_test_birds")

# evsonganaly bird: or18or24
EVSONG_BIRD      = "or18or24"
EVSONG_SOURCE    = SSD_ROOT / EVSONG_BIRD / "source"   # contains 18-08-2023/ subdir

# wseg bird: bu78bu77
WSEG_BIRD        = "bu78bu77"
WSEG_METADATA    = SSD_ROOT / WSEG_BIRD / "source_metadata"  # contains song/ subdir
WSEG_AUDIO       = SSD_ROOT / WSEG_BIRD / "source_audio"     # contains dated subdirs

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _ssd_available() -> bool:
    return SSD_ROOT.exists()

def _evsong_data_available() -> bool:
    return _ssd_available() and any(EVSONG_SOURCE.rglob("*.wav.not.mat"))

def _wseg_data_available() -> bool:
    return (
        _ssd_available()
        and any(WSEG_METADATA.rglob("*.wav.not.mat"))
        and any(WSEG_AUDIO.rglob("*.wav"))
    )

requires_ssd       = pytest.mark.skipif(not _ssd_available(),
                         reason="SSD not mounted at /Volumes/Extreme SSD/smoke_test_birds")
requires_evsong    = pytest.mark.skipif(not _evsong_data_available(),
                         reason="evsonganaly test bird data not yet on SSD")
requires_wseg      = pytest.mark.skipif(not _wseg_data_available(),
                         reason="wseg test bird data not yet on SSD")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def evsong_source_dir():
    """Path to the evsonganaly source directory (containing dated subdirs)."""
    return str(EVSONG_SOURCE)


@pytest.fixture(scope="session")
def wseg_metadata_dir():
    """Path to the wseg metadata directory (parent of song/)."""
    return str(WSEG_METADATA)


@pytest.fixture(scope="session")
def wseg_audio_dir():
    """Path to the wseg audio directory (containing dated subdirs)."""
    return str(WSEG_AUDIO)


@pytest.fixture
def evsong_bird_dir(tmp_path):
    """Fresh per-test output directory for the evsonganaly bird pipeline run."""
    d = tmp_path / EVSONG_BIRD
    d.mkdir()
    return tmp_path   # pipeline expects project_dir, bird is a subdirectory


@pytest.fixture
def wseg_bird_dir(tmp_path):
    """Fresh per-test output directory for the wseg bird pipeline run."""
    d = tmp_path / WSEG_BIRD
    d.mkdir()
    return tmp_path
