"""Integration tests: all loss configs instantiate and forward on dummy data."""
# pylint: disable=broad-exception-caught

from pathlib import Path

import hydra.utils
import pytest
import torch
import yaml
from omegaconf import OmegaConf

_ROOT = Path(__file__).parent.parent.parent
_LOSS_CONFIG_DIR = _ROOT / "configs" / "loss"

# Configs that use ${num_epochs} interpolation and can't be instantiated standalone
_INTERPOLATION_CONFIGS = {"phased_mse_ssim", "gan_compression", "phased_l1_ssim"}


def _all_loss_config_names() -> list[str]:
    """Return the stems of every YAML file in the loss config directory."""
    return [p.stem for p in sorted(_LOSS_CONFIG_DIR.glob("*.yaml"))]


def _load_loss_cfg(yaml_path: Path, num_epochs: int = 10) -> object:
    """Load a loss YAML and resolve interpolations with a fake num_epochs."""
    raw = yaml.safe_load(yaml_path.read_text())
    cfg = OmegaConf.create(raw)
    # Provide num_epochs for configs that interpolate it
    resolver_cfg = OmegaConf.create({"num_epochs": num_epochs})
    cfg = OmegaConf.merge(resolver_cfg, cfg)
    return cfg


def _get_loss_cfg(cfg: object) -> object:
    """Extract the loss config from a global-package config."""
    if hasattr(cfg, "loss"):
        return cfg.loss
    return cfg


class TestLossConfigsInstantiate:
    """Verify that every loss config can be instantiated and run a forward pass."""

    # Configs with known YAML bugs that prevent instantiation
    @pytest.mark.integration
    @pytest.mark.parametrize("loss_name", _all_loss_config_names())
    def test_loss_instantiates(self, loss_name: str) -> None:
        """Loss config must instantiate without raising an exception."""
        path = _LOSS_CONFIG_DIR / f"{loss_name}.yaml"
        try:
            cfg = _load_loss_cfg(path)
            loss_cfg = _get_loss_cfg(cfg)
            loss = hydra.utils.instantiate(loss_cfg)
        except Exception as e:
            pytest.fail(f"Failed to instantiate loss '{loss_name}': {e}")
        assert loss is not None

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "loss_name",
        [
            "mse",
            "l1",
            "charbonnier",
            "ssim",
        ],
    )
    def test_simple_loss_forward(self, loss_name: str) -> None:
        """Simple LossContainer losses forward on (pred, target) via data dict."""
        path = _LOSS_CONFIG_DIR / f"{loss_name}.yaml"
        cfg = _load_loss_cfg(path)
        loss_cfg = _get_loss_cfg(cfg)
        loss = hydra.utils.instantiate(loss_cfg)
        data = {"pred": torch.randn(2, 3, 32, 32), "target": torch.randn(2, 3, 32, 32)}
        # LossContainer.forward returns (total, component_dict)
        result, _ = loss(data)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert not torch.isnan(result)
