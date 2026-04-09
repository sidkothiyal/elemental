"""Integration tests: lightweight model configs instantiate and forward correctly."""
# pylint: disable=broad-exception-caught,too-few-public-methods

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

# Models that can be instantiated without HF downloads or GPU
LIGHTWEIGHT_MODELS = [
    "unet_compression",
    "simple_autoencoder",
    "simple_vq_autoencoder",
    "masked_autoencoder",
    "diffusion_unet",
    "autoregressive_lm_small",
    "classification_cnn",
    "transformer_encoder",
    "frame_predictor",
]

# Models that require HF downloads — skip in CI
HF_MODELS = [
    "dinov2_encoder",
    "dinov3_encoder",
    "sam_encoder",
    "radio_encoder",
    "dinov2_decoder_ae",
    "dinov3_decoder_ae",
]


def _load_model_config(name: str, overrides: dict | None = None) -> object:
    """Load and return a model config from disk, applying optional overrides."""
    path = _MODEL_CONFIG_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())
    cfg = OmegaConf.create(raw)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    # Model configs use @package _global_ with a 'model:' wrapper
    return cfg.get("model", cfg)


class TestLightweightModels:
    """Verify that lightweight model configs instantiate and produce correct output shapes."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self) -> Generator[None, None, None]:
        """Reset Hydra global state before and after each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    # Default overrides for configs that use global interpolations
    _MODEL_OVERRIDES: dict[str, dict] = {
        "classification_cnn": {"num_classes": 10},
    }

    @pytest.mark.integration
    @pytest.mark.parametrize("model_name", LIGHTWEIGHT_MODELS)
    def test_model_instantiates(self, model_name: str) -> None:
        """Model config must instantiate without raising an exception."""
        cfg = _load_model_config(model_name, self._MODEL_OVERRIDES.get(model_name))
        try:
            model = hydra.utils.instantiate(cfg)
        except Exception as e:
            pytest.fail(f"Failed to instantiate model '{model_name}': {e}")
        assert model is not None

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_name",
        [
            "unet_compression",
            "simple_autoencoder",
        ],
    )
    def test_model_forward_shape(self, model_name: str) -> None:
        """Model output spatial dimensions must match input spatial dimensions."""
        model = hydra.utils.instantiate(_load_model_config(model_name))
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        # Output shape should match input spatial dims
        if isinstance(out, dict):
            first = next(iter(out.values()))
            assert first.shape[2:] == x.shape[2:]
        else:
            assert out.shape[2:] == x.shape[2:]


class TestHFModelsSkipped:
    """Smoke-test HuggingFace-backed models; skipped in standard CI (requires downloads)."""

    @pytest.mark.integration
    @pytest.mark.hf
    @pytest.mark.parametrize("model_name", HF_MODELS)
    def test_hf_model_instantiates(self, model_name: str) -> None:
        """HF model config must instantiate when the required weights are available."""
        cfg = _load_model_config(model_name)
        model = hydra.utils.instantiate(cfg)
        assert model is not None
