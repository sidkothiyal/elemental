"""
HuggingFace Vision-Language Model wrapper.

Wraps HuggingFace VLM models (like Qwen3-VL) to extract hidden state features
from text and image inputs.
"""

import collections
from typing import Literal

import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor


class HFVLM(nn.Module):
    """HuggingFace Vision-Language Model wrapper for feature extraction."""

    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        device_map: str | dict = "auto",
        trust_remote_code: bool = True,
        add_generation_prompt: bool = False,
        freeze_model: bool = True,
    ) -> None:
        """Initialize the HuggingFace VLM wrapper.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-VL-2B-Thinking").
            torch_dtype: Data type for model weights.
            attn_implementation: Attention implementation to use.
            device_map: Device placement strategy.
            trust_remote_code: Whether to trust remote code from HuggingFace.
            add_generation_prompt: Whether to add generation prompt in chat template.
            freeze_model: Whether to freeze the model parameters.
        """
        super().__init__()
        self.model_name = model_name
        self.add_generation_prompt = add_generation_prompt
        self.freeze_model = freeze_model

        # Load model and processor
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Freeze model parameters (feature extraction only)
        if freeze_model:
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

    def _load_image(self, image: str | Image.Image | torch.Tensor) -> str | Image.Image:
        """Load image from path, URL, PIL Image, or tensor.

        Args:
            image: Image as file path, URL, PIL Image, or tensor.

        Returns:
            Image in a format the processor can handle.
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            # Assume tensor is (C, H, W) or (H, W, C) in range [0, 1] or [0, 255]
            if image.dim() == 3:
                if image.shape[0] in (1, 3, 4):  # (C, H, W)
                    image = image.permute(1, 2, 0)
                # Now (H, W, C)
                if image.dtype in (torch.float32, torch.float16):
                    if image.max() <= 1.0:
                        image = (image * 255).clamp(0, 255)
                    image = image.to(torch.uint8)
                image = image.cpu().numpy()
            return Image.fromarray(image)
        # str (path or URL) or PIL Image - processor handles these directly
        return image

    def _build_single_message(
        self,
        text: str,
        images: list[str | Image.Image | torch.Tensor] | None = None,
    ) -> list[dict]:
        """Build chat messages for a single sample.

        Args:
            text: Text prompt.
            images: Optional list of images for this sample.

        Returns:
            Messages in chat format for a single sample.
        """
        content = []

        if images is not None:
            for image in images:
                loaded_image = self._load_image(image)
                content.append({"type": "image", "image": loaded_image})

        content.append({"type": "text", "text": text})

        return [{"role": "user", "content": content}]

    def _build_messages_batch(
        self,
        texts: str | list[str],
        images: list[str | Image.Image | torch.Tensor]
        | list[list[str | Image.Image | torch.Tensor]]
        | None = None,
    ) -> list[list[dict]]:
        """Build chat messages for a batch of samples.

        Args:
            texts: Single text prompt or list of text prompts.
            images: Images for a single sample, or list of images per sample.
                For single sample: list of images (e.g., ["img1.jpg", "img2.jpg"])
                For batch: list of list of images (e.g., [["img1.jpg"], ["img2.jpg", "img3.jpg"]])

        Returns:
            List of messages, one per sample in the batch.
        """
        # Normalize texts to list
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)

        # Normalize images to list of lists
        if images is None:
            images_batch: list[list | None] = [None] * batch_size
        elif len(images) > 0 and not isinstance(images[0], list):
            # Single sample's images provided, assume batch_size=1
            images_batch = [images]  # type: ignore
        else:
            images_batch = images  # type: ignore

        # Validate batch sizes match
        if len(images_batch) != batch_size:
            raise ValueError(
                f"Batch size mismatch: {len(texts)} texts but {len(images_batch)} image lists"
            )

        # Build messages for each sample
        return [self._build_single_message(text, imgs) for text, imgs in zip(texts, images_batch)]

    def _forward_impl(
        self,
        inputs: dict,
        output_all_hidden_states: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Internal forward implementation.

        Args:
            inputs: Processed inputs dictionary.
            output_all_hidden_states: Whether to return all hidden states.

        Returns:
            Hidden states tensor or tuple of tensors.
        """
        outputs = self.model.forward(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        if output_all_hidden_states:
            return outputs.hidden_states
        return outputs.hidden_states[-1]

    def forward(
        self,
        texts: str | list[str],
        images: list[str | Image.Image | torch.Tensor]
        | list[list[str | Image.Image | torch.Tensor]]
        | None = None,
        output_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass to extract hidden state features.

        Args:
            texts: Single text prompt or list of text prompts for batched processing.
            images: Images for the input(s). Can be:
                - None: No images
                - List of images for single sample: ["img1.jpg", "img2.jpg"]
                - List of list of images for batch: [["img1.jpg"], ["img2.jpg", "img3.jpg"]]
                Images can be file paths, URLs, PIL Images, or tensors.
            output_all_hidden_states: If True, return hidden states from all layers.
                If False (default), return only the final hidden state.

        Returns:
            If output_all_hidden_states is False: Final hidden state tensor of shape
                (batch_size, sequence_length, hidden_dim).
            If output_all_hidden_states is True: Tuple of hidden states from all layers,
                where each tensor has shape (batch_size, sequence_length, hidden_dim).
        """
        # Build messages for batch
        messages_batch = self._build_messages_batch(texts, images)

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)

        # Forward pass with or without gradient tracking
        if self.freeze_model:
            with torch.no_grad():
                return self._forward_impl(inputs, output_all_hidden_states)
        else:
            return self._forward_impl(inputs, output_all_hidden_states)

    def get_input_tokens(
        self,
        text: str,
        images: list[str | Image.Image | torch.Tensor] | None = None,
    ) -> dict:
        """Get tokenized inputs and their decoded representations.

        Useful for debugging and understanding token composition.

        Args:
            text: Text prompt (single sample only).
            images: Optional list of images for this sample.

        Returns:
            Dictionary with input_ids, decoded tokens, and token counts.
        """
        messages = self._build_single_message(text, images)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0].tolist()
        tokenizer = self.processor.tokenizer

        # Decode each token
        tokens = [{"id": token_id, "text": tokenizer.decode([token_id])} for token_id in input_ids]

        # Count token frequencies
        token_counts = collections.Counter(input_ids)
        frequency = [
            {"id": token_id, "text": tokenizer.decode([token_id]), "count": count}
            for token_id, count in token_counts.most_common()
        ]

        return {
            "input_ids": input_ids,
            "tokens": tokens,
            "token_frequency": frequency,
            "total_tokens": len(input_ids),
        }


if __name__ == "__main__":
    # Example usage
    vlm = HFVLM(
        model_name="Qwen/Qwen3-VL-2B-Thinking",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        freeze_model=True,  # Use no_grad in forward
    )

    # Test with single text (no images)
    hidden = vlm.forward(texts="Hello, how are you?")
    print(f"Single text-only hidden shape: {hidden.shape}")

    # Test with single sample with image
    hidden = vlm.forward(
        texts="Describe this image.",
        images=["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
    )
    print(f"Single image+text hidden shape: {hidden.shape}")

    # Test with batched inputs
    hidden = vlm.forward(
        texts=["Describe this image.", "What do you see?"],
        images=[
            ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
            ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
        ],
    )
    print(f"Batched hidden shape: {hidden.shape}")

    # Get token info (single sample only)
    token_info = vlm.get_input_tokens(
        text="Describe this image.",
        images=["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
    )
    print(f"Total tokens: {token_info['total_tokens']}")
    print("Top 5 token frequencies:")
    for item in token_info["token_frequency"][:5]:
        print(f"  {item['id']:6d} ({item['text']!r}): {item['count']}")
