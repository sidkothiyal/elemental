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
        """
        super().__init__()
        self.model_name = model_name
        self.return_layers = return_layers
        self.freeze_model = freeze_model
        self.processor_kwargs: dict = processor_kwargs or {}
        self.interpolate_pos_encoding = interpolate_pos_encoding

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
            interpolate_pos_encoding=self.interpolate_pos_encoding,
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
        # processor_kwargs can override defaults (e.g. do_resize=False to keep native resolution)
        inputs = self.processor(images=loaded_images, return_tensors="pt", **self.processor_kwargs)
        inputs = inputs.to(self.device)

        if self.freeze_model:
            with torch.no_grad():
                return self._forward_impl(inputs, output_all_hidden_states)
        else:
            return self._forward_impl(inputs, output_all_hidden_states)


class HFVisionEncoderDecoder(nn.Module):
    """Frozen HuggingFace ViT encoder paired with a custom convolutional decoder.

    The encoder produces a token sequence [B, 1 + H*W, D] (CLS + patch tokens).
    The CLS token is dropped, the remaining patch tokens are reshaped to a 2-D
    spatial feature map [B, D, H//patch_size, W//patch_size], and the decoder
    maps that feature map back to a full-resolution image.

    The patch grid dimensions are computed dynamically from the input tensor shape
    at every forward pass, so the model works at any resolution as long as H and W
    are divisible by ``patch_size`` and the encoder has ``interpolate_pos_encoding``
    enabled (set via ``processor_kwargs`` / ``interpolate_pos_encoding`` on the encoder).

    Examples:
        DINOv3-ViT-S/16 on 224x224  →  feature map [B, 384, 14, 14]  →  16x upsample
        DINOv3-ViT-S/16 on 640x1024 →  feature map [B, 384, 40, 64]  →  16x upsample
    """

    def __init__(
        self,
        encoder: "HFVisionEncoder",
        decoder: nn.Module,
        patch_size: int,
    ) -> None:
        """Initialise the encoder-decoder.

        Args:
            encoder: Frozen HFVisionEncoder backbone.
            decoder: Convolutional decoder that accepts
                [B, embed_dim, H//patch_size, W//patch_size] and returns a
                reconstructed image tensor at the original HxW resolution.
            patch_size: Patch size used by the ViT backbone (e.g. 16 for DINOv3,
                14 for DINOv2).  Used to compute the patch grid from the input shape.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images and reconstruct them via the convolutional decoder.

        Args:
            x: Input image batch [B, C, H, W] with pixel values in [0, 1].
                H and W must be divisible by ``patch_size``.

        Returns:
            Reconstructed image tensor [B, C, H, W].
        """
        img_h, img_w = x.shape[2], x.shape[3]
        patch_grid_h = img_h // self.patch_size
        patch_grid_w = img_w // self.patch_size

        # list(x) splits [B, C, H, W] → [B × (C, H, W)], which HFVisionEncoder handles
        tokens = self.encoder(list(x))  # [B, 1 + patch_grid_h*patch_grid_w, embed_dim]

        # Drop CLS token (index 0) to obtain only spatial patch tokens
        patch_tokens = tokens[:, 1:, :]  # [B, patch_grid_h*patch_grid_w, embed_dim]

        batch_size, _num_tokens, embed_dim = patch_tokens.shape
        # Rearrange token sequence to a 2-D spatial feature map
        features = patch_tokens.permute(0, 2, 1).reshape(
            batch_size, embed_dim, patch_grid_h, patch_grid_w
        )  # [B, embed_dim, patch_grid_h, patch_grid_w]

        return self.decoder(features)


if __name__ == "__main__":
    import numpy as np

    # Simple testing
    print("Testing HFVisionEncoder with dinov2-small...")
    sample_encoder = HFVisionEncoder(
        model_name="facebook/dinov2-small",
        return_layers=[-2, 2],  # Return second to last layer, and layer 2
        torch_dtype=torch.float32,
        device_map="cpu",  # For simple testing
        freeze_model=True,
    )

    # Create dummy images
    dummy_img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    sample_outputs = sample_encoder.forward([dummy_img1, dummy_img2])
    print(f"Output type: {type(sample_outputs)}")
    if isinstance(sample_outputs, dict):
        for k, v in sample_outputs.items():
            print(f"  {k} shape: {v.shape}")
