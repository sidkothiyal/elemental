"""
Test to verify that hierarchical Hydra configs produce equivalent results to flat configs.

This test loads both flat and hierarchical configs and compares them to ensure
the hierarchical refactoring maintains backward compatibility.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

# Mapping of flat config names (in legacy folder) to hierarchical config names
# fmt: off
# pylint: disable=line-too-long
FLAT_TO_HIERARCHICAL = {
    "legacy/base_autoencoder": "experiment/autoencoder/depth_estimation",
    "legacy/base_autoregressive_lm": "experiment/autoregressive/wikitext_lm",
    "legacy/base_autoregressive_tiny_shakespeare": "experiment/autoregressive/tiny_shakespeare",
    "legacy/base_classification": "experiment/classification/imagenet",
    "legacy/base_gan": "experiment/gan/cityscapes_seg",
    "legacy/base_masked_autoencoder": "experiment/autoencoder/masked_autoencoder",
    "legacy/compression_autoencoder_conditional_diffusion": "experiment/autoencoder/compression_conditional_diffusion",  # noqa: E501
    "legacy/compression_autoencoder_diffusion": "experiment/autoencoder/compression_diffusion",
    "legacy/compression_autoencoder_gan_unet": "experiment/gan/compression_unet",
    "legacy/compression_autoencoder_mae": "experiment/autoencoder/compression_mae",
    "legacy/compression_autoencoder_transformer_conditional_diffusion": "experiment/autoencoder/compression_transformer_conditional_diffusion",  # noqa: E501
    "legacy/compression_autoencoder_unet": "experiment/autoencoder/compression_unet",
    "legacy/compression_autoencoder_unet_4_8_16_32_64_skip_first": "experiment/autoencoder/compression_unet_l1",  # noqa: E501
    "legacy/compression_autoencoder_unet_4_8_16_32_64_skip_first_ssim": "experiment/autoencoder/compression_unet_ssim",  # noqa: E501
    "legacy/compression_autoencoder_vq_transformer_conditional_diffusion": "experiment/autoencoder/compression_vq_transformer_conditional_diffusion",  # noqa: E501
    "legacy/compression_autoencoder_vq_vae": "experiment/autoencoder/compression_vqvae",
}
# pylint: enable=line-too-long
# fmt: on

# Keys that are expected to differ between flat and hierarchical configs
EXPECTED_DIFF_KEYS = {
    # # experiment_name contains timestamp and may have different base name
    # "experiment_name",
    # These may be added by hierarchical configs but not present in flat
    "dataset_name",
    "target_shape",
    "prediction_type",
    # Hierarchical adds these defaults
    "resume_training",
    # Some configs have these for loading pretrained weights (optional)
    "load_checkpoint_path",
    "load_keys",
}

# Keys that exist in flat but may be structured differently in hierarchical
STRUCTURAL_DIFF_KEYS = {
    # Callbacks are structured as dict in hierarchical, list in flat
    "callbacks",
}


def get_config_dir() -> str:
    """Get the absolute path to the configs directory."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "configs",
        Path.cwd() / "configs",
        Path(__file__).parent.parent.parent / "configs",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path.absolute())

    raise FileNotFoundError("Could not find configs directory")


def load_config(config_name: str) -> DictConfig:
    """Load a Hydra config by name."""
    GlobalHydra.instance().clear()

    config_dir = get_config_dir()

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

    return cfg


def normalize_config(cfg: DictConfig) -> dict:
    """
    Normalize a config for comparison by:
    - Converting to plain dict
    - Removing keys that are expected to differ
    - Sorting nested structures
    """
    # Convert to container (plain Python objects)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Remove keys that are expected to differ
    for key in EXPECTED_DIFF_KEYS:
        cfg_dict.pop(key, None)

    return cfg_dict


def normalize_callbacks(callbacks):
    """
    Normalize callbacks to a comparable format.
    Hierarchical uses dict, flat uses list - convert both to sorted list of dicts.
    """
    if callbacks is None:
        return []

    if isinstance(callbacks, dict):
        # Hierarchical format: {"wandb": {...}, "checkpoint": {...}}
        return sorted(
            [v for v in callbacks.values() if v is not None], key=lambda x: x.get("_target_", "")
        )
    if isinstance(callbacks, list):
        # Flat format: [{...}, {...}]
        return sorted([c for c in callbacks if c is not None], key=lambda x: x.get("_target_", ""))

    return callbacks


def compare_values(flat_val, hier_val, path: str = "") -> list[str]:
    """
    Recursively compare two values and return list of differences.

    Args:
        flat_val: Value from flat config
        hier_val: Value from hierarchical config
        path: Current path in config for error messages

    Returns:
        List of difference descriptions
    """
    differences = []

    # Handle None comparisons
    if flat_val is None and hier_val is None:
        return []
    if flat_val is None or hier_val is None:
        differences.append(f"{path}: flat={flat_val}, hier={hier_val}")
        return differences

    # Handle type mismatches
    if type(flat_val) is not type(hier_val):
        # Special case: int vs float
        if isinstance(flat_val, (int, float)) and isinstance(hier_val, (int, float)):
            if float(flat_val) != float(hier_val):
                differences.append(f"{path}: flat={flat_val}, hier={hier_val}")
            return differences
        differences.append(
            f"{path}: type mismatch flat={type(flat_val).__name__}, hier={type(hier_val).__name__}"
        )
        return differences

    # Compare dicts
    if isinstance(flat_val, dict):
        all_keys = set(flat_val.keys()) | set(hier_val.keys())
        for key in all_keys:
            if key in EXPECTED_DIFF_KEYS:
                continue
            new_path = f"{path}.{key}" if path else key
            if key not in flat_val:
                # Key only in hierarchical - might be an added default
                if key not in EXPECTED_DIFF_KEYS:
                    differences.append(f"{new_path}: missing in flat config")
            elif key not in hier_val:
                differences.append(f"{new_path}: missing in hierarchical config")
            else:
                differences.extend(compare_values(flat_val[key], hier_val[key], new_path))
        return differences

    # Compare lists
    if isinstance(flat_val, list):
        if len(flat_val) != len(hier_val):
            differences.append(
                f"{path}: list length mismatch flat={len(flat_val)}, hier={len(hier_val)}"
            )
            return differences
        for i, (f, h) in enumerate(zip(flat_val, hier_val)):
            differences.extend(compare_values(f, h, f"{path}[{i}]"))
        return differences

    # Compare primitives
    if flat_val != hier_val:
        differences.append(f"{path}: flat={flat_val}, hier={hier_val}")

    return differences


def compare_configs(
    flat_cfg: DictConfig, hier_cfg: DictConfig, config_name: str  # pylint: disable=unused-argument
) -> list[str]:
    """
    Compare flat and hierarchical configs, returning list of significant differences.

    Args:
        flat_cfg: Config loaded from flat file
        hier_cfg: Config loaded from hierarchical file
        config_name: Name of the config being compared (for future use in error messages)

    Returns:
        List of difference descriptions (empty if configs match)
    """
    # Normalize both configs
    flat_dict = normalize_config(flat_cfg)
    hier_dict = normalize_config(hier_cfg)

    # Special handling for callbacks
    if "callbacks" in flat_dict and "callbacks" in hier_dict:
        flat_dict["callbacks"] = normalize_callbacks(flat_dict["callbacks"])
        hier_dict["callbacks"] = normalize_callbacks(hier_dict["callbacks"])

    # Compare the configs
    differences = compare_values(flat_dict, hier_dict)

    return differences


class TestConfigEquivalence:
    """Test class for config equivalence checks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean up Hydra global state before each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.parametrize("flat_name,hier_name", list(FLAT_TO_HIERARCHICAL.items()))
    def test_config_equivalence(self, flat_name: str, hier_name: str):
        """
        Test that a hierarchical config produces equivalent settings to its flat counterpart.

        This test:
        1. Loads both flat and hierarchical configs
        2. Normalizes them for comparison
        3. Checks that all significant settings match
        """
        # Check if flat config exists
        config_dir = Path(get_config_dir())
        flat_path = config_dir / f"{flat_name}.yaml"

        if not flat_path.exists():
            pytest.skip(f"Flat config {flat_name}.yaml not found")

        # Load configs
        try:
            flat_cfg = load_config(flat_name)
        except Exception as e:
            pytest.fail(f"Failed to load flat config {flat_name}: {e}")

        GlobalHydra.instance().clear()

        try:
            hier_cfg = load_config(hier_name)
        except Exception as e:
            pytest.fail(f"Failed to load hierarchical config {hier_name}: {e}")

        # Compare configs
        differences = compare_configs(flat_cfg, hier_cfg, flat_name)

        # Filter out acceptable differences
        significant_diffs = [
            d
            for d in differences
            if not any(
                skip in d
                for skip in [
                    "missing in flat config",  # Hierarchical may add defaults
                    "experiment_name",
                    "dataset_name",
                    "target_shape",
                    "prediction_type",
                    "resume_training",
                    "load_checkpoint_path",
                    "load_keys",
                    # Callback paths that contain experiment_name
                    "save_dir",
                    "exp_name",
                ]
            )
        ]

        if significant_diffs:
            diff_report = "\n".join(significant_diffs[:20])  # Limit output
            if len(significant_diffs) > 20:
                diff_report += f"\n... and {len(significant_diffs) - 20} more differences"
            pytest.fail(f"Config mismatch between {flat_name} and {hier_name}:\n{diff_report}")

    def test_all_flat_configs_have_hierarchical(self):
        """Verify all flat configs have a corresponding hierarchical config."""
        config_dir = Path(get_config_dir())
        legacy_dir = config_dir / "legacy"

        # Find all flat configs (yaml files in legacy folder, excluding node_train)
        flat_configs = []
        if legacy_dir.exists():
            for f in legacy_dir.glob("*.yaml"):
                name = f.stem
                if not name.startswith("node_train"):
                    flat_configs.append(f"legacy/{name}")

        missing = []
        for flat_name in flat_configs:
            if flat_name not in FLAT_TO_HIERARCHICAL:
                missing.append(flat_name)

        if missing:
            pytest.fail(
                f"Flat configs without hierarchical mapping: {missing}\n"
                "Add these to FLAT_TO_HIERARCHICAL in test_config_equivalence.py"
            )

    def test_hierarchical_configs_exist(self):
        """Verify that all mapped hierarchical configs actually exist."""
        config_dir = Path(get_config_dir())

        missing = []
        for flat_name, hier_name in FLAT_TO_HIERARCHICAL.items():
            hier_path = config_dir / f"{hier_name}.yaml"
            if not hier_path.exists():
                missing.append(f"{flat_name} -> {hier_name}")

        if missing:
            pytest.fail("Hierarchical configs that don't exist:\n" + "\n".join(missing))


class TestConfigLoading:
    """Test that configs can be loaded without errors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clean up Hydra global state before each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.parametrize("config_name", list(FLAT_TO_HIERARCHICAL.values()))
    def test_hierarchical_config_loads(self, config_name: str):
        """Test that each hierarchical config can be loaded without errors."""
        try:
            cfg = load_config(config_name)
            assert cfg is not None
            # Verify essential keys exist
            assert "mode" in cfg, f"Config {config_name} missing 'mode' key"
            assert "model" in cfg, f"Config {config_name} missing 'model' key"
        except Exception as e:
            pytest.fail(f"Failed to load hierarchical config {config_name}: {e}")

    @pytest.mark.parametrize("config_name", list(FLAT_TO_HIERARCHICAL.keys()))
    def test_flat_config_loads(self, config_name: str):
        """Test that each flat config can be loaded without errors."""
        config_dir = Path(get_config_dir())
        if not (config_dir / f"{config_name}.yaml").exists():
            pytest.skip(f"Flat config {config_name}.yaml not found")

        try:
            cfg = load_config(config_name)
            assert cfg is not None
            assert "mode" in cfg, f"Config {config_name} missing 'mode' key"
            assert "model" in cfg, f"Config {config_name} missing 'model' key"
        except Exception as e:
            pytest.fail(f"Failed to load flat config {config_name}: {e}")


# Scheduler parameter validation
SCHEDULER_REQUIRED_PARAMS = {
    "StepLR": {"step_size"},
    "CosineAnnealingLR": {"T_max"},
    "ExponentialLR": {"gamma"},
    "ReduceLROnPlateau": set(),  # No required params
    "MultiStepLR": {"milestones"},
    "CyclicLR": {"base_lr", "max_lr"},
    "OneCycleLR": {"max_lr", "total_steps"},
}

SCHEDULER_INVALID_PARAMS = {
    "StepLR": {"T_max", "milestones", "max_lr", "base_lr", "total_steps"},
    "CosineAnnealingLR": {"step_size", "milestones", "max_lr", "base_lr", "total_steps"},
    "ExponentialLR": {"T_max", "step_size", "milestones", "max_lr", "base_lr", "total_steps"},
    "ReduceLROnPlateau": {"T_max", "step_size", "milestones", "max_lr", "base_lr", "total_steps"},
    "MultiStepLR": {"T_max", "step_size", "max_lr", "base_lr", "total_steps"},
}

# Keys required by trainer.py
TRAINER_REQUIRED_KEYS = {
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

# Keys that should be present for scheduler (trainer uses these)
TRAINER_SCHEDULER_KEYS = {
    "lr_scheduler",
    "lr_scheduler_params",
}


class TestConfigRuntimeValidation:
    """Test that configs can be used at runtime without errors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear GlobalHydra before and after each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.parametrize("config_name", list(FLAT_TO_HIERARCHICAL.values()))
    def test_scheduler_params_valid(self, config_name: str):
        """
        Test that scheduler parameters are compatible with the specified scheduler.

        This catches errors like using T_max (CosineAnnealingLR) with StepLR.
        """
        cfg = load_config(config_name)

        # Get scheduler info
        lr_scheduler = cfg.get("lr_scheduler")
        lr_scheduler_params = cfg.get("lr_scheduler_params", {})

        if lr_scheduler is None:
            pytest.skip(f"Config {config_name} has no lr_scheduler")

        # Convert to dict if needed, filtering out None values
        # (None values are used to override defaults and are filtered by trainer.py)
        if hasattr(lr_scheduler_params, "keys"):
            params = {k: v for k, v in dict(lr_scheduler_params).items() if v is not None}
        else:
            params = lr_scheduler_params or {}

        # Check for invalid params
        if lr_scheduler in SCHEDULER_INVALID_PARAMS:
            invalid = SCHEDULER_INVALID_PARAMS[lr_scheduler]
            found_invalid = set(params.keys()) & invalid
            if found_invalid:
                pytest.fail(
                    f"Config {config_name}: scheduler {lr_scheduler} "
                    f"has invalid params: {found_invalid}"
                )

        # Check for required params
        if lr_scheduler in SCHEDULER_REQUIRED_PARAMS:
            required = SCHEDULER_REQUIRED_PARAMS[lr_scheduler]
            missing = required - set(params.keys())
            if missing:
                pytest.fail(
                    f"Config {config_name}: scheduler {lr_scheduler} "
                    f"missing required params: {missing}"
                )

    @pytest.mark.parametrize("config_name", list(FLAT_TO_HIERARCHICAL.values()))
    def test_trainer_required_keys(self, config_name: str):
        """Test that all keys required by trainer.py are present in the config."""
        cfg = load_config(config_name)
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)

        # Check required keys
        missing = TRAINER_REQUIRED_KEYS - set(cfg_dict.keys())
        if missing:
            pytest.fail(f"Config {config_name} missing required keys for trainer.py: {missing}")

        # Check scheduler keys (should be present unless explicitly set to null)
        for key in TRAINER_SCHEDULER_KEYS:
            if key not in cfg_dict:
                pytest.fail(f"Config {config_name} missing scheduler key: {key}")

    @pytest.mark.parametrize("config_name", list(FLAT_TO_HIERARCHICAL.values()))
    def test_instantiation_signatures(self, config_name: str):
        """
        Test that _target_ classes can accept the provided parameters.

        Note: This test is lenient because many models pass params to subcomponents
        via **kwargs or internal logic that we can't easily validate statically.
        We only fail on obvious signature mismatches for simple cases.
        """
        import importlib
        import inspect

        cfg = load_config(config_name)

        # Check loss instantiation - these are usually simple and should match exactly
        loss_cfg = cfg.get("loss")
        if loss_cfg and hasattr(loss_cfg, "_target_"):
            target = loss_cfg._target_
            # Only check standard torch.nn losses which have strict signatures
            if target.startswith("torch.nn."):
                try:
                    module_path, class_name = target.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)

                    sig = inspect.signature(cls.__init__)
                    valid_params = set(sig.parameters.keys()) - {"self"}

                    loss_params = {k for k in dict(loss_cfg).keys() if not k.startswith("_")}

                    # Check for invalid params (excluding **kwargs cases)
                    has_var_keyword = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                    if not has_var_keyword and loss_params:
                        invalid = loss_params - valid_params
                        if invalid:
                            pytest.fail(
                                f"Config {config_name}: loss {target} has invalid params: {invalid}"
                            )
                except (ImportError, AttributeError):
                    pass

    @pytest.mark.parametrize("config_name", list(FLAT_TO_HIERARCHICAL.values()))
    def test_scheduler_instantiation(self, config_name: str):
        """
        Test that the scheduler can be instantiated with the provided params.

        This directly tests what trainer.py does with scheduler params.
        """
        import torch

        cfg = load_config(config_name)

        lr_scheduler = cfg.get("lr_scheduler")
        lr_scheduler_params = cfg.get("lr_scheduler_params", {})

        if lr_scheduler is None:
            pytest.skip(f"Config {config_name} has no lr_scheduler")

        # Convert params, resolving interpolations
        try:
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
            params = resolved_cfg.get("lr_scheduler_params", {}) or {}
        except Exception:
            # If we can't resolve (e.g., missing vars), use unresolved
            params = dict(lr_scheduler_params) if lr_scheduler_params else {}

        # Get the scheduler class
        scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler, None)
        if scheduler_class is None:
            pytest.fail(f"Config {config_name}: unknown scheduler {lr_scheduler}")

        # Create a dummy optimizer for testing
        dummy_model = torch.nn.Linear(10, 10)
        dummy_optim = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Try to instantiate the scheduler
        try:
            # Handle interpolated values that couldn't be resolved
            # Also filter out None values (allows configs to override defaults by setting to null)
            test_params = {}
            for k, v in params.items():
                if v is None:
                    continue  # Skip None values - trainer.py filters these out
                if isinstance(v, str) and "${" in v:
                    # Replace unresolved interpolations with reasonable defaults
                    test_params[k] = 100  # Default for things like T_max
                else:
                    test_params[k] = v

            _ = scheduler_class(dummy_optim, **test_params)  # noqa: F841
        except TypeError as e:
            pytest.fail(
                f"Config {config_name}: failed to instantiate {lr_scheduler} "
                f"with params {test_params}: {e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
