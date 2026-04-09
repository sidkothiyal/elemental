"""Fixtures shared across integration tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_experiment_config(config_name: str) -> object:
    """Load an experiment config by relative name (e.g. 'autoencoder/compression_unet')."""
    GlobalHydra.instance().clear()
    config_dir = str((_ROOT / "configs").absolute())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config_root() -> Path:
    """Return the root configs directory path."""
    return _ROOT / "configs"


@pytest.fixture
def mock_wandb(monkeypatch) -> MagicMock:
    """Patch wandb so callbacks that use it never hit the network."""
    mock = MagicMock()
    mock.init.return_value = MagicMock()
    mock.log = MagicMock()
    mock.finish = MagicMock()
    mock.Table = MagicMock
    mock.Histogram = MagicMock
    monkeypatch.setattr("callback.wandb_callback.wandb", mock)
    monkeypatch.setattr("callback.metrics_callback.wandb", mock)
    return mock
