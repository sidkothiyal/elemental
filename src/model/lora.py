"""
LoRA (Low-Rank Adaptation) implementation for HuggingFace models.

Supports wrapping any HuggingFace PreTrainedModel with LoRA adapters for
efficient fine-tuning. Compatible with both language models and vision-language
models like those used in hf_llm.py.

References:
    - LoRA: Low-Rank Adaptation of Large Language Models (https://arxiv.org/abs/2106.09685)
"""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file as safetensors_load_file, save_file as safetensors_save_file
from torch import nn
from transformers import AutoModel, PreTrainedModel


class LoRAConfig:
    """Configuration for LoRA parameters."""

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRA configuration.

        Args:
            rank: Rank of the low-rank decomposition matrices.
            alpha: Scaling factor for LoRA outputs. The actual scaling is alpha/rank.
            target_modules: List of module name patterns to apply LoRA to.
                Defaults to ["q_proj", "k_proj", "v_proj", "o_proj"].
            dropout: Dropout rate applied to LoRA inputs.
        """
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.dropout = dropout

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LoRAConfig":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary containing config parameters.

        Returns:
            LoRAConfig instance.
        """
        return cls(**config_dict)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file.

        Args:
            path: Directory path to save config to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LoRAConfig":
        """Load config from JSON file.

        Args:
            path: Directory path containing lora_config.json.

        Returns:
            LoRAConfig instance.
        """
        with open(Path(path) / "lora_config.json", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Wraps an existing Linear layer and adds low-rank adaptation matrices A and B.
    The output is: original_output + (x @ A^T @ B^T) * scaling
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRA-wrapped Linear layer.

        Args:
            original_linear: The original nn.Linear layer to wrap.
            rank: Rank of the low-rank decomposition.
            alpha: Scaling factor (actual scaling is alpha/rank).
            dropout: Dropout rate for LoRA path.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Original weights (frozen)
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.weight.requires_grad = False

        self.bias: nn.Parameter | None = None
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
            self.bias.requires_grad = False

        # LoRA parameters (trainable)
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Original linear transformation
        result = nn.functional.linear(x, self.weight, self.bias)
        # LoRA adaptation: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return result + lora_out * self.scaling


class LoRAModel(nn.Module):
    """HuggingFace-compatible LoRA wrapper for efficient fine-tuning.

    Wraps any HuggingFace PreTrainedModel and injects LoRA adapters into
    specified target modules. Supports saving/loading both full checkpoints
    and lightweight adapter-only checkpoints.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-7B")
        >>> model = LoRAModel(base, rank=16, alpha=32)
        >>> model.save_pretrained("./adapter", save_lora_only=True)
    """

    # File names for saved artifacts
    LORA_WEIGHTS_NAME = "lora_weights.safetensors"
    LORA_WEIGHTS_BIN_NAME = "lora_weights.bin"
    LORA_CONFIG_NAME = "lora_config.json"

    def __init__(
        self,
        base_model: PreTrainedModel | nn.Module,
        lora_config: LoRAConfig | None = None,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRA model wrapper.

        Args:
            base_model: HuggingFace PreTrainedModel or any nn.Module to wrap.
            lora_config: LoRA configuration. If provided, overrides other LoRA params.
            rank: Rank of LoRA decomposition (ignored if lora_config provided).
            alpha: LoRA scaling factor (ignored if lora_config provided).
            target_modules: Module name patterns to apply LoRA (ignored if lora_config provided).
            dropout: LoRA dropout rate (ignored if lora_config provided).
        """
        super().__init__()

        # Store base model config for HF compatibility (if available)
        self.config = getattr(base_model, "config", None)

        # Build or use provided LoRA config
        if lora_config is not None:
            self.lora_config = lora_config
        else:
            self.lora_config = LoRAConfig(
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                dropout=dropout,
            )

        self.model = base_model
        self._lora_layers: dict[str, LoRALinear] = {}
        self._inject_lora()

    def _inject_lora(self) -> None:
        """Replace target modules with LoRA-wrapped versions."""
        for name, module in list(self.model.named_modules()):
            # Only wrap Linear layers
            if not isinstance(module, nn.Linear):
                continue
            # Check if module name matches any target pattern
            if not any(target in name for target in self.lora_config.target_modules):
                continue

            # Navigate to parent module
            parent = self.model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]

            # Replace with LoRA version
            lora_layer = LoRALinear(
                module,
                rank=self.lora_config.rank,
                alpha=self.lora_config.alpha,
                dropout=self.lora_config.dropout,
            )
            setattr(parent, attr_name, lora_layer)
            self._lora_layers[name] = lora_layer

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    # ==================== State Dict Methods ====================

    def state_dict(self, *args, **kwargs) -> dict:
        """Get full state dict including base and LoRA weights."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dict into model.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce key matching.

        Returns:
            NamedTuple with missing_keys and unexpected_keys.
        """
        has_lora = any("lora_a" in k or "lora_b" in k for k in state_dict.keys())

        # If loading base weights without LoRA and not strict, filter by shape
        if not has_lora and not strict:
            model_state = self.model.state_dict()
            filtered = {
                k: v
                for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            return self.model.load_state_dict(filtered, strict=False)

        return self.model.load_state_dict(state_dict, strict=strict)

    def lora_state_dict(self) -> dict:
        """Get only LoRA parameters (lora_a and lora_b weights).

        Returns:
            Dictionary containing only LoRA weight tensors.
        """
        return {k: v for k, v in self.model.state_dict().items() if "lora_a" in k or "lora_b" in k}

    def base_state_dict(self) -> dict:
        """Get only base model parameters (excluding LoRA).

        Returns:
            Dictionary containing only base model weight tensors.
        """
        return {
            k: v
            for k, v in self.model.state_dict().items()
            if "lora_a" not in k and "lora_b" not in k
        }

    # ==================== HuggingFace Save/Load ====================

    def save_pretrained(
        self,
        save_directory: str | Path,
        save_lora_only: bool = False,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save model in HuggingFace-compatible format.

        Args:
            save_directory: Directory to save model files.
            save_lora_only: If True, save only LoRA weights (~10MB vs full model).
            safe_serialization: Use safetensors format (recommended for safety).
            **kwargs: Additional arguments passed to base model's save_pretrained.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Always save LoRA config
        with open(save_directory / self.LORA_CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump(self.lora_config.to_dict(), f, indent=2)

        lora_state = self.lora_state_dict()

        if save_lora_only:
            # Save only LoRA weights (lightweight adapter)
            self._save_lora_weights(save_directory, lora_state, safe_serialization)

            # Save adapter metadata
            with open(save_directory / "adapter_model.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lora_only": True,
                        "base_model_name_or_path": getattr(self.config, "_name_or_path", None),
                    },
                    f,
                    indent=2,
                )
        else:
            # Save full model using HF's method (if available)
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(
                    save_directory,
                    safe_serialization=safe_serialization,
                    **kwargs,
                )
            else:
                # Fallback: save full state dict
                self._save_weights(save_directory, self.state_dict(), safe_serialization)

            # Also save LoRA weights separately for easy adapter swapping
            self._save_lora_weights(save_directory, lora_state, safe_serialization)

    def _save_lora_weights(
        self,
        directory: Path,
        state_dict: dict,
        safe_serialization: bool,
    ) -> None:
        """Save LoRA weights to directory.

        Args:
            directory: Directory to save to.
            state_dict: LoRA state dict to save.
            safe_serialization: Whether to use safetensors format.
        """
        if safe_serialization:
            safetensors_save_file(state_dict, directory / self.LORA_WEIGHTS_NAME)
        else:
            torch.save(state_dict, directory / self.LORA_WEIGHTS_BIN_NAME)

    def _save_weights(
        self,
        directory: Path,
        state_dict: dict,
        safe_serialization: bool,
    ) -> None:
        """Save full model weights to directory.

        Args:
            directory: Directory to save to.
            state_dict: Full state dict to save.
            safe_serialization: Whether to use safetensors format.
        """
        if safe_serialization:
            safetensors_save_file(state_dict, directory / "model.safetensors")
        else:
            torch.save(state_dict, directory / "model.bin")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        base_model_name_or_path: str | Path | None = None,
        lora_config: LoRAConfig | None = None,
        model_class: type | None = None,
        device_map: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "LoRAModel":
        """Load a LoRA model from pretrained checkpoint.

        Handles multiple scenarios:
        1. Full LoRA checkpoint (base + LoRA weights saved together)
        2. LoRA adapter only (requires base_model_name_or_path)
        3. Plain HF model (creates fresh LoRA layers for new fine-tuning)

        Args:
            pretrained_model_name_or_path: Path to saved model or HF model ID.
            base_model_name_or_path: Path to base model (required for adapter-only).
            lora_config: LoRA config for fresh LoRA (when loading plain HF model).
            model_class: HuggingFace model class to use for loading base model.
                Defaults to AutoModel. Use AutoModelForCausalLM, AutoModelForVision2Seq, etc.
            device_map: Device placement strategy ("auto", "cuda:0", etc.).
            torch_dtype: Model dtype (torch.float16, torch.bfloat16, etc.).
            **kwargs: Additional arguments for model loading.

        Returns:
            LoRAModel instance.

        Raises:
            ValueError: If adapter-only checkpoint loaded without base_model_name_or_path.
        """
        path = Path(pretrained_model_name_or_path)
        model_class = model_class or AutoModel

        # Check if this is a LoRA checkpoint
        lora_config_path = path / cls.LORA_CONFIG_NAME if path.exists() else None
        has_lora_config: bool = bool(lora_config_path and lora_config_path.exists())

        # Check if this is adapter-only
        adapter_info_path = path / "adapter_model.json" if path.exists() else None
        is_adapter_only = adapter_info_path and adapter_info_path.exists()

        if is_adapter_only:
            return cls._load_adapter_only(
                path,
                base_model_name_or_path,
                lora_config_path,
                has_lora_config,
                lora_config,
                model_class,
                device_map,
                torch_dtype,
                **kwargs,
            )
        if has_lora_config:
            return cls._load_full_checkpoint(
                path,
                lora_config_path,
                model_class,
                device_map,
                torch_dtype,
                **kwargs,
            )
        return cls._load_base_model_with_fresh_lora(
            pretrained_model_name_or_path,
            lora_config,
            model_class,
            device_map,
            torch_dtype,
            **kwargs,
        )

    @classmethod
    def _load_adapter_only(
        cls,
        path: Path,
        base_model_name_or_path: str | Path | None,
        lora_config_path: Path | None,
        has_lora_config: bool,
        lora_config: LoRAConfig | None,
        model_class: type,
        device_map: str | None,
        torch_dtype: torch.dtype | None,
        **kwargs,
    ) -> "LoRAModel":
        """Load adapter-only checkpoint onto base model."""
        with open(path / "adapter_model.json", encoding="utf-8") as f:
            adapter_info = json.load(f)

        # Determine base model path
        if base_model_name_or_path is None:
            base_model_name_or_path = adapter_info.get("base_model_name_or_path")

        if base_model_name_or_path is None:
            raise ValueError(
                "This is a LoRA adapter checkpoint. Please provide base_model_name_or_path."
            )

        # Load base model
        base_model = model_class.from_pretrained(
            base_model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        # Load LoRA config
        if has_lora_config and lora_config_path:
            with open(lora_config_path, encoding="utf-8") as f:
                lora_config = LoRAConfig.from_dict(json.load(f))
        elif lora_config is None:
            raise ValueError("No LoRA config found and none provided.")

        # Create LoRA model and load weights
        model = cls(base_model, lora_config=lora_config)
        lora_weights = cls._load_lora_weights(path)
        model.load_lora_state_dict(lora_weights)

        return model

    @classmethod
    def _load_full_checkpoint(
        cls,
        path: Path,
        lora_config_path: Path | None,
        model_class: type,
        device_map: str | None,
        torch_dtype: torch.dtype | None,
        **kwargs,
    ) -> "LoRAModel":
        """Load full checkpoint with both base and LoRA weights."""
        base_model = model_class.from_pretrained(
            path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        assert lora_config_path is not None
        with open(lora_config_path, encoding="utf-8") as f:
            lora_config = LoRAConfig.from_dict(json.load(f))

        # Create wrapper (will re-inject LoRA layers)
        model = cls(base_model, lora_config=lora_config)

        # Load LoRA weights if saved separately
        try:
            lora_weights = cls._load_lora_weights(path)
            model.load_lora_state_dict(lora_weights)
        except FileNotFoundError:
            # LoRA weights might be in the main checkpoint
            pass

        return model

    @classmethod
    def _load_base_model_with_fresh_lora(
        cls,
        pretrained_model_name_or_path: str | Path,
        lora_config: LoRAConfig | None,
        model_class: type,
        device_map: str | None,
        torch_dtype: torch.dtype | None,
        **kwargs,
    ) -> "LoRAModel":
        """Load plain HF model and wrap with fresh LoRA layers."""
        base_model = model_class.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        if lora_config is None:
            lora_config = LoRAConfig()

        return cls(base_model, lora_config=lora_config)

    @classmethod
    def _load_lora_weights(cls, path: Path) -> dict:
        """Load LoRA weights from directory.

        Args:
            path: Directory containing LoRA weights.

        Returns:
            Dictionary of LoRA weight tensors.

        Raises:
            FileNotFoundError: If no LoRA weights found in directory.
        """
        safetensors_path = path / cls.LORA_WEIGHTS_NAME
        bin_path = path / cls.LORA_WEIGHTS_BIN_NAME

        if safetensors_path.exists():
            return safetensors_load_file(safetensors_path)
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"No LoRA weights found in {path}")

    def load_lora_state_dict(self, lora_state: dict) -> None:
        """Load LoRA weights into model.

        Args:
            lora_state: Dictionary containing LoRA weight tensors.
        """
        model_state = self.model.state_dict()
        model_state.update(lora_state)
        self.model.load_state_dict(model_state)

    # ==================== Utilities ====================

    def merge_and_unload(self) -> PreTrainedModel | nn.Module:
        """Merge LoRA weights into base model and return unwrapped model.

        Useful for inference optimization or exporting to standard HF format.
        After merging, the model no longer has separate LoRA parameters.

        Returns:
            The base model with LoRA weights merged into original weights.
        """
        # Merge LoRA weights into base weights
        for layer in self._lora_layers.values():
            layer.weight.data += (layer.lora_b @ layer.lora_a) * layer.scaling

        # Replace LoRA layers with regular Linear layers
        for name, layer in self._lora_layers.items():
            parent = self.model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]

            new_linear = nn.Linear(
                layer.weight.shape[1],
                layer.weight.shape[0],
                bias=layer.bias is not None,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            new_linear.weight.data = layer.weight.data
            if layer.bias is not None:
                new_linear.bias.data = layer.bias.data

            setattr(parent, attr_name, new_linear)

        return self.model

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over trainable LoRA parameters.

        Yields:
            LoRA A and B parameters from all LoRA layers.
        """
        for layer in self._lora_layers.values():
            yield layer.lora_a
            yield layer.lora_b

    def num_trainable_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Total number of trainable LoRA parameters.
        """
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_parameters(self) -> int:
        """Get total number of parameters in the model.

        Returns:
            Total number of parameters (base + LoRA).
        """
        return sum(p.numel() for p in self.model.parameters())

    def print_trainable_parameters(self) -> None:
        """Print summary of trainable vs total parameters."""
        trainable = self.num_trainable_parameters()
        total = self.num_total_parameters()
        percentage = 100 * trainable / total if total > 0 else 0
        print(f"Trainable: {trainable:,} / {total:,} ({percentage:.2f}%)")

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        # Try to get device from first parameter
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32


if __name__ == "__main__":
    # Example usage demonstrating compatibility with different model types
    print("LoRA Module - Example Usage")
    print("=" * 50)

    # Example 1: Wrap a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            out = self.o_proj(q + k + v)
            return self.fc(out)

    base = SimpleModel()
    lora_model = LoRAModel(base, rank=4, alpha=8, target_modules=["q_proj", "v_proj"])

    print("\nSimple model with LoRA:")
    lora_model.print_trainable_parameters()
    print(f"LoRA layers injected: {list(lora_model._lora_layers.keys())}")

    # Test forward pass
    dummy_x = torch.randn(2, 64)
    dummy_out = lora_model(dummy_x)
    print(f"Output shape: {dummy_out.shape}")

    # Example 2: Save and load adapter
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save adapter only
        lora_model.save_pretrained(tmpdir, save_lora_only=True)
        print(f"\nSaved adapter to: {tmpdir}")
        print("Saved files:", os.listdir(tmpdir))

    print("\n" + "=" * 50)
    print("For HuggingFace models, use:")
    print("  from transformers import AutoModelForCausalLM")
    print("  base = AutoModelForCausalLM.from_pretrained('model-name')")
    print("  lora_model = LoRAModel(base, rank=16)")
