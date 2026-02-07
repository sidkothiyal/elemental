"""
Generic Router module for combining backbones and task heads.

Supports various backbone types (VLM, image encoders like SAM/DINO, etc.)
and task heads (diffusion models, image decoders, action predictors, etc.).
"""

from typing import Any

import torch
from torch import nn


class Router(nn.Module):
    """
    Generic Router that connects backbones to multiple task heads.

    The router takes inputs, optionally renames them for the backbone,
    passes them through the backbone, then routes the backbone output
    (along with other inputs) to one or more task heads with optional
    input/output renaming.

    Example use cases:
        - VLA: VLM backbone -> diffusion head for action prediction
        - Image reconstruction: VLM backbone -> image decoder head
        - Multi-task: Image encoder backbone -> [segmentation head, depth head]
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: dict[str, nn.Module] | None = None,
        rename_backbone_input: dict[str, str] | None = None,
        rename_backbone_output: str | dict[str, str] | None = None,
        rename_head_input: dict[str, dict[str, str]] | None = None,
        rename_head_output: dict[str, dict[str, str] | str] | None = None,
        backbone_output_key: str = "backbone",
    ):
        """Initialize the Router.

        Args:
            backbone: The backbone module (e.g., HFVLM, image encoder).
            heads: Dictionary of task heads {name: module}. Can be None for
                backbone-only usage.
            rename_backbone_input: Mapping to rename inputs before passing to backbone.
                {backbone_param_name: input_kwarg_name}
                Example: {"texts": "prompt", "images": "observation"} means
                backbone receives texts=kwargs["prompt"], images=kwargs["observation"]
            rename_backbone_output: How to store backbone output in the context.
                - str: Store entire output under this key
                - dict: If backbone returns dict, rename specific keys
                  {backbone_output_key: new_key_name}
                - None: Store under backbone_output_key
            rename_head_input: Per-head input renaming.
                {head_name: {head_param_name: context_key}}
                Example: {"diffusion": {"cond": "backbone", "x": "noise"}}
            rename_head_output: Per-head output renaming.
                {head_name: new_name} for single output, or
                {head_name: {output_key: new_key}} for dict outputs
            backbone_output_key: Default key for storing backbone output.
        """
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads) if heads else nn.ModuleDict()
        self.rename_backbone_input = rename_backbone_input or {}
        self.rename_backbone_output = rename_backbone_output
        self.rename_head_input = rename_head_input or {}
        self.rename_head_output = rename_head_output or {}
        self.backbone_output_key = backbone_output_key

    def _rename_inputs(
        self,
        rename_map: dict[str, str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Rename inputs according to a mapping.

        Args:
            rename_map: {new_param_name: context_key}
            context: Available inputs/context.

        Returns:
            Renamed inputs dictionary.
        """
        renamed = {}
        for param_name, context_key in rename_map.items():
            if context_key not in context:
                raise KeyError(
                    f"Input '{context_key}' not found in context. "
                    f"Available keys: {list(context.keys())}"
                )
            renamed[param_name] = context[context_key]
        return renamed

    def _store_output(
        self,
        output: Any,
        rename_spec: str | dict[str, str] | None,
        default_key: str,
        outputs: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Store output in outputs dict and context with optional renaming.

        Args:
            output: The output to store.
            rename_spec: Renaming specification.
            default_key: Default key if no rename specified.
            outputs: Output dictionary to update.
            context: Context dictionary to update.
        """
        if rename_spec is None:
            outputs[default_key] = output
            context[default_key] = output
        elif isinstance(rename_spec, str):
            outputs[rename_spec] = output
            context[rename_spec] = output
        elif isinstance(rename_spec, dict):
            # Output is expected to be a dict, rename specific keys
            if isinstance(output, dict):
                for old_key, new_key in rename_spec.items():
                    if old_key in output:
                        outputs[new_key] = output[old_key]
                        context[new_key] = output[old_key]
            else:
                # Treat as single output, use first mapping
                first_key = next(iter(rename_spec.values()))
                outputs[first_key] = output
                context[first_key] = output

    def forward_backbone(
        self,
        context: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Run the backbone with renamed inputs.

        Args:
            context: Input context dictionary.

        Returns:
            Tuple of (backbone_output, updated_context, outputs).
        """
        # Prepare backbone inputs
        if self.rename_backbone_input:
            backbone_kwargs = self._rename_inputs(self.rename_backbone_input, context)
        else:
            # Pass all context as kwargs (let backbone handle what it needs)
            backbone_kwargs = context.copy()

        # Run backbone
        backbone_output = self.backbone(**backbone_kwargs)

        # Store backbone output
        outputs: dict[str, Any] = {}
        self._store_output(
            backbone_output,
            self.rename_backbone_output,
            self.backbone_output_key,
            outputs,
            context,
        )

        return backbone_output, context, outputs

    def forward_head(
        self,
        head_name: str,
        context: dict[str, Any],
    ) -> Any:
        """Run a specific head with renamed inputs.

        Args:
            head_name: Name of the head to run.
            context: Input context dictionary.

        Returns:
            Head output.
        """
        if head_name not in self.heads:
            raise KeyError(f"Head '{head_name}' not found. Available: {list(self.heads.keys())}")

        head = self.heads[head_name]

        # Prepare head inputs
        if head_name in self.rename_head_input:
            head_kwargs = self._rename_inputs(self.rename_head_input[head_name], context)
        else:
            # Default: pass backbone output as first positional or as 'x'
            head_kwargs = {"x": context.get(self.backbone_output_key)}

        return head(**head_kwargs)

    def forward(
        self,
        heads_to_run: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass through backbone and selected heads.

        Args:
            heads_to_run: List of head names to run. If None, runs all heads.
            **kwargs: Input arguments that form the initial context.

        Returns:
            Dictionary with backbone output and all head outputs.
        """
        outputs: dict[str, Any] = {}
        context = dict(kwargs)  # Mutable copy of inputs

        # Run backbone
        _, context, backbone_outputs = self.forward_backbone(context)
        outputs.update(backbone_outputs)

        # Determine which heads to run
        if heads_to_run is None:
            heads_to_run = list(self.heads.keys())

        # Run selected heads
        for head_name in heads_to_run:
            if head_name not in self.heads:
                raise KeyError(
                    f"Head '{head_name}' not found. Available: {list(self.heads.keys())}"
                )

            head_output = self.forward_head(head_name, context)

            # Store head output with optional renaming
            rename_spec = self.rename_head_output.get(head_name)
            self._store_output(head_output, rename_spec, head_name, outputs, context)

        return outputs


if __name__ == "__main__":
    # Example usage demonstrating the Router

    # Simple backbone that extracts features
    class DummyBackbone(nn.Module):  # pylint: disable=too-few-public-methods
        """Dummy backbone that extracts features."""

        def __init__(self, hidden_dim: int = 256):
            super().__init__()
            self.proj = nn.Linear(512, hidden_dim)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """Forward pass of the dummy backbone."""
            return self.proj(features)

    # Diffusion head for action prediction
    class DummyDiffusionHead(nn.Module):  # pylint: disable=too-few-public-methods
        """Dummy diffusion head for action prediction."""

        def __init__(self, cond_dim: int = 256, action_dim: int = 7):
            super().__init__()
            self.proj = nn.Linear(cond_dim, action_dim)

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            """Forward pass of the dummy diffusion head."""
            # x is noise, cond is conditioning from backbone
            return self.proj(cond) + x.mean(dim=-1, keepdim=True)

    # Image decoder head
    class DummyDecoderHead(nn.Module):  # pylint: disable=too-few-public-methods
        """Dummy image decoder head."""

        def __init__(self, hidden_dim: int = 256, out_channels: int = 3):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, out_channels * 64 * 64)
            self.out_channels = out_channels

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            """Forward pass of the dummy decoder head."""
            # features from backbone
            out = self.proj(features.mean(dim=1))  # [B, C*H*W]
            return out.view(-1, self.out_channels, 64, 64)

    print("=== Router Example ===\n")

    # Create components
    dummy_backbone = DummyBackbone(hidden_dim=256)
    dummy_diffusion_head = DummyDiffusionHead(cond_dim=256, action_dim=7)
    dummy_decoder_head = DummyDecoderHead(hidden_dim=256, out_channels=3)

    # Create router with input/output renaming
    router = Router(
        backbone=dummy_backbone,
        heads={
            "action_diffusion": dummy_diffusion_head,
            "image_decoder": dummy_decoder_head,
        },
        # Rename inputs for backbone
        rename_backbone_input={
            # backbone expects 'features', we provide 'observation_features'
            "features": "observation_features",
        },
        # Store backbone output as 'encoded_features' (backbone output key)
        rename_backbone_output="encoded_features",
        # Rename inputs for each head
        rename_head_input={
            "action_diffusion": {
                # head expects 'x', use 'action_noise' from context
                "x": "action_noise",
                # head expects 'cond', use backbone output
                "cond": "encoded_features",
            },
            "image_decoder": {
                # head expects 'features', use backbone output
                "features": "encoded_features",
            },
        },
        # Rename outputs
        rename_head_output={
            "action_diffusion": "predicted_action",
            "image_decoder": "reconstructed_image",
        },
    )

    # Test inputs
    SAMPLE_BATCH_SIZE = 2
    SAMPLE_SEQ_LEN = 10
    observation_features = torch.randn(SAMPLE_BATCH_SIZE, SAMPLE_SEQ_LEN, 512)
    action_noise = torch.randn(SAMPLE_BATCH_SIZE, 7)

    # Run forward pass
    sample_outputs = router(
        observation_features=observation_features,
        action_noise=action_noise,
    )

    print("Input shapes:")
    print(f"  observation_features: {observation_features.shape}")
    print(f"  action_noise: {action_noise.shape}")

    print("\nOutput keys:", list(sample_outputs.keys()))
    print("\nOutput shapes:")
    for key, value in sample_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # Run only specific heads
    print("\n--- Running only action_diffusion head ---")
    outputs_action_only = router(
        observation_features=observation_features,
        action_noise=action_noise,
        heads_to_run=["action_diffusion"],
    )
    print("Output keys:", list(outputs_action_only.keys()))

    print("\n✓ Router test completed successfully!")
