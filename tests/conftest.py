"""
Root pytest configuration: markers, path setup, and shared fixtures.
"""
# pylint: disable=protected-access,broad-exception-caught

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from model.autoregressive_lm import AutoregressiveLM
from model.transformer import Transformer
from model.unet import UNet

# ---------------------------------------------------------------------------
# Unregister any system-level ROS pytest plugins that are incompatible
# with this Python version / pytest version.  The ROS humble packages are
# installed on the system PATH and get picked up as setuptools entrypoints
# even though they target Python 3.10 and use deprecated pytest hooks.
# ---------------------------------------------------------------------------


def pytest_configure(config):  # pylint: disable=missing-function-docstring
    for name in list(config.pluginmanager._name2plugin.keys()):
        if "launch" in name or "ros" in name:
            try:
                config.pluginmanager.unregister(name=name)
            except Exception:  # nosec
                pass

    config.addinivalue_line("markers", "unit: Fast unit tests with no I/O")
    config.addinivalue_line("markers", "integration: Config loading and component wiring tests")
    config.addinivalue_line("markers", "e2e: End-to-end short training runs")
    config.addinivalue_line("markers", "slow: Tests that take >10 s")
    config.addinivalue_line("markers", "hf: Tests requiring HuggingFace model downloads")


# ---------------------------------------------------------------------------
# Path setup — make src/ importable without installing the package
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src"
_TESTS = _ROOT / "tests"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))


# ---------------------------------------------------------------------------
# Marker declarations
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device():
    """CPU device (override with --device=cuda for GPU CI)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def config_dir():
    """Absolute path to the top-level configs/ directory."""
    return _ROOT / "configs"


# ---------------------------------------------------------------------------
# Data-container fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_container():
    """Populated unified data dict matching the trainer contract."""
    return {
        "input/image": torch.randn(2, 3, 32, 32),
        "target": torch.randn(2, 3, 32, 32),
        "pred": torch.randn(2, 3, 32, 32),
        "model": nn.Linear(4, 4),
        "epoch": 0,
    }


@pytest.fixture
def seq_data_container():
    """Unified data dict for sequence-model tests (B=2, T=16, vocab=64)."""
    return {
        "input/tokens": torch.randint(0, 64, (2, 16)),
        "target": torch.randint(0, 64, (2, 16)),
        "pred": torch.randn(2, 16, 64),
        "model": nn.Linear(4, 4),
        "epoch": 0,
    }


# ---------------------------------------------------------------------------
# Small synthetic tensors
# ---------------------------------------------------------------------------


@pytest.fixture
def image_batch():
    """Random (B=2, C=3, H=32, W=32) image tensor."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def seg_batch():
    """Random (B=2, H=32, W=32) integer label map (3 classes)."""
    return torch.randint(0, 3, (2, 32, 32))


# ---------------------------------------------------------------------------
# Tiny encoder/decoder helpers (reusable across model tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_encoder():
    """Conv encoder: (B, 3, H, W) → (B, 8, H/2, W/2)."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
    )


def make_small_unet() -> UNet:
    """Construct a tiny seeded UNet for use in unit and e2e tests."""
    torch.manual_seed(42)
    return UNet(
        in_channels=3,
        out_channels=3,
        channels=[16, 32],
        use_cross_attn=False,
        num_groups=8,
        dropout=0.0,
    )


def make_small_transformer() -> Transformer:
    """Construct a tiny seeded Transformer for use in unit and e2e tests."""
    torch.manual_seed(42)
    return Transformer(
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_embed_dim=64,
        dropout=0.0,
        attention_dropout=0.0,
    )


def make_small_autoregressive_lm(max_length: int = 16) -> AutoregressiveLM:
    """Construct a tiny seeded AutoregressiveLM for use in unit and e2e tests."""
    encoder = make_small_transformer()
    return AutoregressiveLM(
        encoder=encoder,
        token_embedding=None,
        positional_embedding=None,
        embedding_dim=32,
        max_length=max_length,
        vocab_size=65,
    )


@pytest.fixture
def tiny_decoder():
    """ConvTranspose decoder: (B, 8, H, W) → (B, 3, H*2, W*2)."""
    return nn.Sequential(
        nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
    )
