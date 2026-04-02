"""
HuggingFace Vision Model wrapper.

Wraps HuggingFace Vision models (like DINOv3, SAM) to extract hidden state features
from image inputs.
"""

from typing import Literal

import requests
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class HFVisionEncoder(nn.Module):
    """HuggingFace Vision Model wrapper for feature extraction."""

    def __init__(
        self,
        model_name: str,
        return_layers: list[int] | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa",
        device_map: str | dict = "auto",
        trust_remote_code: bool = True,
        freeze_model: bool = True,
        processor_kwargs: dict | None = None,
        interpolate_pos_encoding: bool = False,
        extract_layer: int | str | None = None,
        model_submodule: str | None = None,
        processor_output_filter: list[str] | None = None,
        processor_to_model_key_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the HuggingFace Vision Encoder wrapper.

        Args:
            model_name: HuggingFace model identifier (e.g., "facebook/dinov2-base").
            return_layers: List of indices for intermediate hidden layers to extract
                in addition to the final layer. If provided, the forward pass returns
                a dictionary. Negative indices (e.g., -2 for second to last) are supported.
            torch_dtype: Data type for model weights.
            attn_implementation: Attention implementation to use.
            device_map: Device placement strategy.
            trust_remote_code: Whether to trust remote code from HuggingFace.
            freeze_model: Whether to freeze the model parameters.
            processor_kwargs: Extra keyword arguments forwarded to the processor at every
                call (e.g. ``{"do_resize": False, "do_center_crop": False}`` to preserve
                the original image resolution).
            interpolate_pos_encoding: When True, passes ``interpolate_pos_encoding=True``
                to the model forward to support variable / non-224 input resolutions.
            extract_layer: Controls which representation is returned from the encoder.
                ``None`` (default) → ``last_hidden_state``. A ``str`` (e.g.
                ``"pooler_output"``, ``"last_hidden_state"``) → the named attribute from
                the model output via ``getattr(outputs, extract_layer)``. An ``int``
                (e.g. ``-1``, ``5``) → the hidden state at that transformer block index
                via ``output_hidden_states=True``.
            model_submodule: Dotted attribute path to extract from the loaded model, e.g.
                ``"model.visual"`` for Qwen3-VL. When set, only that submodule is kept,
                which avoids instantiating or running the rest of the model (e.g. the LLM
                decoder) and saves VRAM.
            processor_output_filter: If set, only these keys from the processor output are
                passed to the model forward call. Useful when a processor returns extra
                fields (e.g. ``input_ids``) that the vision submodule doesn't accept.
                Example: ``["pixel_values", "image_grid_thw"]`` for Qwen3-VL.
            processor_to_model_key_map: Rename processor output keys before passing to the model.
                Example: ``{"pixel_values": "hidden_states", "image_grid_thw": "grid_thw"}``
                for Qwen3-VL's vision submodule whose forward signature differs from the
                full model.
        """
        super().__init__()
        self.model_name = model_name
        self.return_layers = return_layers
        self.freeze_model = freeze_model
        self.processor_kwargs: dict = processor_kwargs or {}
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.extract_layer = extract_layer
        self.processor_output_filter = processor_output_filter
        self.processor_to_model_key_map: dict[str, str] = processor_to_model_key_map or {}

        # Load vision processor and model directly
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        # Optionally extract a specific submodule (e.g. vision tower of a VLM)
        if model_submodule:
            submod = self.model
            for attr in model_submodule.split("."):
                if not hasattr(submod, attr):
                    available = [a for a in dir(submod) if not a.startswith("_")]
                    raise AttributeError(
                        f"model_submodule='{model_submodule}': attribute '{attr}' not found "
                        f"on {type(submod).__name__}. Available: {available[:20]}"
                    )
                submod = getattr(submod, attr)
            self.model = submod

        # Freeze model parameters if it's acting as a backbone encoder
        if self.freeze_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return self.model.dtype

    def _load_image(self, image: str | Image.Image | torch.Tensor) -> Image.Image:
        """Load image into PIL format handled by the processor."""
        if isinstance(image, torch.Tensor):
            # Handle 2-D grayscale tensors
            if image.dim() == 2:
                image = image.unsqueeze(0)  # (H, W) → (1, H, W)
            if image.dim() == 3:
                if image.shape[0] in (1, 3, 4):  # (C, H, W)
                    image = image.permute(1, 2, 0)
                # Now (H, W, C) — strip alpha if 4-channel
                if image.shape[-1] == 4:
                    image = image[..., :3]
                # Scale float tensors to [0, 255]
                if image.is_floating_point():
                    if image.max() <= 1.0:
                        image = (image * 255).clamp(0, 255)
                    image = image.to(torch.uint8)
                image = image.cpu().numpy()
            return Image.fromarray(image)
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return Image.open(requests.get(image, stream=True, timeout=10).raw).convert("RGB")
            return Image.open(image).convert("RGB")
        # Assume PIL Image
        return image.convert("RGB") if hasattr(image, "convert") else image

    def _forward_impl(
        self,
        inputs: dict,
        output_all_hidden_states: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Internal implementation for forward logic."""
        # Apply key filtering and remapping before forwarding to the model
        if self.processor_output_filter:
            inputs = {k: v for k, v in inputs.items() if k in self.processor_output_filter}
        if self.processor_to_model_key_map:
            inputs = {self.processor_to_model_key_map.get(k, k): v for k, v in inputs.items()}

        # Extra kwargs applied to every model call
        extra_kwargs: dict = {}
        if self.interpolate_pos_encoding:
            extra_kwargs["interpolate_pos_encoding"] = True

        # Determine whether we need hidden_states from the model
        needs_hidden_states = (
            output_all_hidden_states
            or self.return_layers is not None
            or isinstance(self.extract_layer, int)
        )

        outputs = self.model(
            **inputs,
            return_dict=True,
            output_hidden_states=needs_hidden_states,
            **extra_kwargs,
        )

        # Route based on extract_layer -------------------------------------------

        # Named output attribute (e.g. "pooler_output", "last_hidden_state")
        if isinstance(self.extract_layer, str):
            if not hasattr(outputs, self.extract_layer):
                available = list(outputs.keys()) if hasattr(outputs, "keys") else dir(outputs)
                raise ValueError(
                    f"extract_layer='{self.extract_layer}' not found in model outputs. "
                    f"Available: {available}"
                )
            return getattr(outputs, self.extract_layer)

        # Specific intermediate transformer block by index
        if isinstance(self.extract_layer, int):
            return outputs.hidden_states[self.extract_layer]

        # All hidden states as tuple
        if output_all_hidden_states:
            return outputs.hidden_states

        # Multi-layer extraction (return_layers)
        if self.return_layers is not None:
            hidden_states = outputs.hidden_states
            out_dict = {"final": hidden_states[-1]}
            for idx in self.return_layers:
                out_dict[f"layer_{idx}"] = hidden_states[idx]
            return out_dict

        # Default: last hidden state
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state

        raise ValueError("Model did not return last_hidden_state")

    def _forward_batch_tensor(
        self,
        x: torch.Tensor,
        output_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Run the encoder on a ``[B, C, H, W]`` tensor batch.

        Converts the batch to PIL images for the processor, then forwards
        through the model.  This avoids the per-image type-checking overhead
        of ``_load_image`` and is the path used when the encoder is embedded
        inside an autoencoder that passes raw tensor batches.
        """
        images_pil = [
            Image.fromarray(img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
            for img in x
        ]
        inputs = self.processor(
            images=images_pil,
            return_tensors="pt",
            **self.processor_kwargs,
        ).to(self.device)

        if self.freeze_model:
            with torch.no_grad():
                return self._forward_impl(inputs, output_all_hidden_states)
        return self._forward_impl(inputs, output_all_hidden_states)

    def forward(
        self,
        images: list[str | Image.Image | torch.Tensor] | str | Image.Image | torch.Tensor,
        output_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Forward pass to extract hidden state features from images.

        Args:
            images: List of images (file paths, PIL images, or tensors), a
                single image, or a 4-D ``[B, C, H, W]`` tensor batch.  When a
                4-D tensor is detected, the efficient batch path is used
                automatically (no per-image type checking).
            output_all_hidden_states: Whether to return all hidden layers as a tuple.

        Returns:
            - If output_all_hidden_states=True: A tuple of all hidden layers.
            - If return_layers is set during init: A dict containing requested intermediate
              layers and the "final" hidden state.
            - Otherwise: A single tensor of the final hidden state.
        """
        # Auto-detect 4D batch tensor [B, C, H, W]
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            return self._forward_batch_tensor(images, output_all_hidden_states)

        if not isinstance(images, list):
            images = [images]

        loaded_images = [self._load_image(img) for img in images]

        # Processor handles normalization, resizing, etc.
        # processor_kwargs can override defaults (e.g. do_resize=False to keep native resolution)
        inputs = self.processor(images=loaded_images, return_tensors="pt", **self.processor_kwargs)
        inputs = inputs.to(self.device)

        if self.freeze_model:
            with torch.no_grad():
                return self._forward_impl(inputs, output_all_hidden_states)
        else:
            return self._forward_impl(inputs, output_all_hidden_states)
