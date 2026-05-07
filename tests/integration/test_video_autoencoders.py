"""Integration tests: video autoencoder configs instantiate and forward correctly."""
# pylint: disable=missing-class-docstring,missing-function-docstring,broad-exception-caught

from collections.abc import Generator
from pathlib import Path

import hydra.utils
import pytest
import torch
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_ROOT = Path(__file__).parent.parent.parent
_MODEL_CONFIG_DIR = _ROOT / "configs" / "model"


def _load_model_config(name: str, overrides: dict | None = None) -> object:
    """Load a model config and apply optional overrides."""
    path = _MODEL_CONFIG_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())
    cfg = OmegaConf.create(raw)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg.get("model", cfg)


class TestSimple3dAutoencoder:
    """Smoke-test for the 3D-conv video autoencoder config."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self) -> Generator[None, None, None]:
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.integration
    def test_instantiates(self) -> None:
        cfg = _load_model_config("simple_3d_autoencoder")
        model = hydra.utils.instantiate(cfg)
        assert model is not None

    @pytest.mark.integration
    def test_forward_preserves_video_shape(self) -> None:
        """(B, T, C, H, W) -> (B, T, C, H, W) round-trip."""
        model = hydra.utils.instantiate(_load_model_config("simple_3d_autoencoder"))
        model.eval()
        x = torch.randn(2, 8, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestPerceiverVideoAutoencoder:
    """Smoke-test for the Perceiver Resampler video autoencoder config."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self) -> Generator[None, None, None]:
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.integration
    def test_instantiates_and_round_trips(self) -> None:
        # Config references global ${clip_length}; supply it so model-only
        # interpolation resolves. Defaults match the experiment config.
        cfg = _load_model_config("perceiver_video_autoencoder", overrides={"clip_length": 8})
        model = hydra.utils.instantiate(cfg)
        model.eval()
        x = torch.randn(2, 8, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
