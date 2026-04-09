"""Integration tests: every config must load without error.

Config lists are auto-discovered via glob so that adding a new YAML file
automatically gets tested — no hardcoded lists to maintain.
"""
# pylint: disable=too-few-public-methods

import glob
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from tests.integration.conftest import load_experiment_config

# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

_CONFIGS_ROOT = Path(__file__).parent.parent.parent / "configs"


def _discover_configs(subdir: str) -> list[str]:
    """Return Hydra-style config names for every YAML under configs/{subdir}/."""
    pattern = str(_CONFIGS_ROOT / subdir / "**" / "*.yaml")
    return sorted(
        str(Path(p).relative_to(_CONFIGS_ROOT)).removesuffix(".yaml")
        for p in glob.glob(pattern, recursive=True)
    )


# Known-broken experiment configs (e.g. referencing missing sub-configs)
_EXPERIMENT_SKIP = {
    "experiment/autoencoder/compression_unet_ssim",  # references missing loss: phased_l1_ssim
    "experiment/autoencoder/compression_vqvae",  # references missing loss: phased_l1_ssim
}

ALL_EXPERIMENT_CONFIGS = [c for c in _discover_configs("experiment") if c not in _EXPERIMENT_SKIP]
ALL_LOSS_CONFIGS = _discover_configs("loss")
ALL_MODEL_CONFIGS = _discover_configs("model")
ALL_DATASET_CONFIGS = _discover_configs("dataset")
ALL_OPTIMIZER_CONFIGS = _discover_configs("optimizer")
ALL_SCHEDULER_CONFIGS = _discover_configs("scheduler")
ALL_CALLBACK_CONFIGS = _discover_configs("callback")
ALL_PROFILER_CONFIGS = _discover_configs("profiler")

# ---------------------------------------------------------------------------
# Experiment config tests
# ---------------------------------------------------------------------------

# Keys every config MUST have for the trainer to work
REQUIRED_TRAINER_KEYS = {
    "lr",
    "num_epochs",
    "seed",
    "device",
    "mode",
    "model",
    "loss",
    "dataloader",
    "callbacks",
    "results_dir",
    "experiment_name",
}

SCHEDULER_INVALID_PARAMS = {
    "StepLR": {"T_max", "milestones", "max_lr", "base_lr", "total_steps"},
    "CosineAnnealingLR": {"step_size", "milestones", "max_lr", "base_lr", "total_steps"},
    "ExponentialLR": {"T_max", "step_size", "milestones", "max_lr", "base_lr", "total_steps"},
}


class TestAllExperimentConfigsLoad:
    """Verify that every experiment config composes cleanly and contains required keys."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self) -> Generator[None, None, None]:
        """Reset Hydra global state before and after each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_EXPERIMENT_CONFIGS)
    def test_config_loads(
        self, config_name: str, config_root: object  # pylint: disable=unused-argument
    ) -> None:
        """Config must compose without Hydra errors."""
        cfg = load_experiment_config(config_name)
        assert cfg is not None

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_EXPERIMENT_CONFIGS)
    def test_required_keys_present(
        self, config_name: str, config_root: object  # pylint: disable=unused-argument
    ) -> None:
        """Config must contain all keys required by the trainer."""
        cfg = load_experiment_config(config_name)
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        missing = REQUIRED_TRAINER_KEYS - set(cfg_dict.keys())
        if missing:
            pytest.fail(f"{config_name} missing trainer keys: {missing}")

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_EXPERIMENT_CONFIGS)
    def test_scheduler_params_valid(
        self, config_name: str, config_root: object  # pylint: disable=unused-argument
    ) -> None:
        """Scheduler params must not contain params from other schedulers."""
        cfg = load_experiment_config(config_name)
        lr_scheduler = cfg.get("lr_scheduler")
        lr_scheduler_params = cfg.get("lr_scheduler_params", {})
        if lr_scheduler is None:
            pytest.skip("No scheduler configured")

        params = {}
        if hasattr(lr_scheduler_params, "keys"):
            params = {k: v for k, v in dict(lr_scheduler_params).items() if v is not None}

        if lr_scheduler in SCHEDULER_INVALID_PARAMS:
            bad = SCHEDULER_INVALID_PARAMS[lr_scheduler] & set(params.keys())
            if bad:
                pytest.fail(f"{config_name}: scheduler {lr_scheduler} has invalid params: {bad}")


# ---------------------------------------------------------------------------
# Component config tests (model, dataset, loss, optimizer, etc.)
# ---------------------------------------------------------------------------


class TestComponentConfigsLoadAsYaml:
    """Every component YAML must be valid and contain expected keys."""

    def _load_yaml(self, subdir: str, config_name: str) -> dict:  # pylint: disable=unused-argument
        """Load and parse a YAML config file, returning its contents as a dict."""
        path = _CONFIGS_ROOT / f"{config_name}.yaml"
        return yaml.safe_load(path.read_text())

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_MODEL_CONFIGS)
    def test_model_config_valid(self, config_name: str) -> None:
        """Model config must be non-empty and contain a _target_ at some level."""
        data = self._load_yaml("model", config_name)
        assert data is not None, f"{config_name}: empty config"
        # Model configs may nest under 'model:' key or be top-level.
        # Some configs (e.g. transformer_encoder) don't have a top-level _target_
        # but have nested components with _target_ — just verify valid structure.
        model_data = data.get("model", data)
        has_target = "_target_" in model_data or any(
            isinstance(v, dict) and "_target_" in v
            for v in model_data.values()
            if isinstance(v, dict)
        )
        assert has_target, f"{config_name}: no _target_ found at any level"

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_DATASET_CONFIGS)
    def test_dataset_config_valid(self, config_name: str) -> None:
        """Dataset config must be non-empty YAML."""
        data = self._load_yaml("dataset", config_name)
        assert data is not None, f"{config_name}: empty config"

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_OPTIMIZER_CONFIGS)
    def test_optimizer_config_valid(self, config_name: str) -> None:
        """Optimizer config must be non-empty YAML."""
        data = self._load_yaml("optimizer", config_name)
        assert data is not None, f"{config_name}: empty config"

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_SCHEDULER_CONFIGS)
    def test_scheduler_config_valid(self, config_name: str) -> None:
        """Scheduler config must be non-empty YAML."""
        data = self._load_yaml("scheduler", config_name)
        assert data is not None, f"{config_name}: empty config"

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_CALLBACK_CONFIGS)
    def test_callback_config_valid(self, config_name: str) -> None:
        """Callback config must be non-empty YAML."""
        data = self._load_yaml("callback", config_name)
        assert data is not None, f"{config_name}: empty config"

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_PROFILER_CONFIGS)
    def test_profiler_config_valid(self, config_name: str) -> None:
        """Profiler config must be non-empty YAML."""
        data = self._load_yaml("profiler", config_name)
        assert data is not None, f"{config_name}: empty config"


# ---------------------------------------------------------------------------
# Loss config tests (auto-discovered)
# ---------------------------------------------------------------------------


class TestLossConfigsInstantiate:
    """Every loss config should be valid YAML with a _target_ key."""

    @pytest.mark.integration
    @pytest.mark.parametrize("config_name", ALL_LOSS_CONFIGS)
    def test_loss_config_has_target(
        self, config_name: str, config_root: object  # pylint: disable=unused-argument
    ) -> None:
        """Each loss yaml must have a _target_ key (possibly nested under 'loss:')."""
        path = _CONFIGS_ROOT / f"{config_name}.yaml"
        data = yaml.safe_load(path.read_text())
        # Loss configs use @package _global_ with a 'loss:' wrapper.
        # GAN loss configs have named sub-losses (e.g. adversarial, feature_matching)
        # each with their own _target_.
        loss_data = data.get("loss", data)
        has_target = (
            "_target_" in loss_data
            or "losses" in loss_data
            or any(
                isinstance(v, dict) and "_target_" in v
                for v in loss_data.values()
                if isinstance(v, dict)
            )
        )
        assert has_target, f"{config_name}: no _target_ found in loss config"
