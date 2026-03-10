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
        """
        super().__init__()
        self.model_name = model_name
        self.return_layers = return_layers
        self.freeze_model = freeze_model

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
            # Convert tensor to PIL Image
            if image.dim() == 3:
                if image.shape[0] in (1, 3, 4):  # (C, H, W)
                    image = image.permute(1, 2, 0)
                # Now (H, W, C)
                if image.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    if image.max() <= 1.0:
                        image = (image * 255).clamp(0, 255)
                    image = image.to(torch.uint8)
                image = image.cpu().numpy()
            return Image.fromarray(image)
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return Image.open(requests.get(image, stream=True).raw).convert("RGB")
            return Image.open(image).convert("RGB")
        # Assume PIL Image
        return image.convert("RGB") if hasattr(image, "convert") else image

    def _forward_impl(
        self,
        inputs: dict,
        output_all_hidden_states: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Internal implementation for forward logic."""
        # Always output hidden states if we need specific layers
        requires_hidden_states = output_all_hidden_states or (self.return_layers is not None)

        outputs = self.model(
            **inputs,
            return_dict=True,
            output_hidden_states=requires_hidden_states,
        )

        if output_all_hidden_states:
            return outputs.hidden_states

        # If specific layers were requested, return a dictionary of those layers + final layer
        if self.return_layers is not None:
            hidden_states = outputs.hidden_states
            out_dict = {"final": hidden_states[-1]}
            for idx in self.return_layers:
                # Format layer keys nicely: e.g. "layer_0", "layer_-2", etc.
                out_dict[f"layer_{idx}"] = hidden_states[idx]
            return out_dict

        # Default behavior: return just the final hidden state or pooler output
        if requires_hidden_states:
            return outputs.hidden_states[-1]

        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state

        raise ValueError("Model did not return last_hidden_state")

    def forward(
        self,
        images: list[str | Image.Image | torch.Tensor] | str | Image.Image | torch.Tensor,
        output_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Forward pass to extract hidden state features from images.

        Args:
            images: List of images (file paths, PIL images, or tensors) or a single image.
            output_all_hidden_states: Whether to return all hidden layers as a tuple.

        Returns:
            - If output_all_hidden_states=True: A tuple of all hidden layers.
            - If return_layers is set during init: A dict containing requested intermediate
              layers and the "final" hidden state.
            - Otherwise: A single tensor of the final hidden state.
        """
        if not isinstance(images, list):
            images = [images]

        loaded_images = [self._load_image(img) for img in images]

        # Processor handles normalization, resizing, etc.
        inputs = self.processor(images=loaded_images, return_tensors="pt")
        inputs = inputs.to(self.device)

        if self.freeze_model:
            with torch.no_grad():
                return self._forward_impl(inputs, output_all_hidden_states)
        else:
            return self._forward_impl(inputs, output_all_hidden_states)


if __name__ == "__main__":
    import numpy as np

    # Simple testing
    print("Testing HFVisionEncoder with dinov2-small...")
    encoder = HFVisionEncoder(
        model_name="facebook/dinov2-small",
        return_layers=[-2, 2],  # Return second to last layer, and layer 2
        torch_dtype=torch.float32,
        device_map="cpu",  # For simple testing
        freeze_model=True,
    )

    # Create dummy images
    dummy_img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    sample_outputs = encoder.forward([dummy_img1, dummy_img2])
    print(f"Output type: {type(sample_outputs)}")
    if isinstance(sample_outputs, dict):
        for k, v in sample_outputs.items():
            print(f"  {k} shape: {v.shape}")
