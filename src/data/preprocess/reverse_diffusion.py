"""Reverse diffusion preprocessing utilities for adding noise to images."""

from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def add_noise_to_image(
    image_path: str | Path,
    output_dir: str | Path,
    num_timesteps: int = 1000,
    timesteps_to_save: list[int] | None = None,
    num_steps_to_save: int = 20,
    noise_schedule: str = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    image_size: tuple[int, int] | None = None,
    device: str = "cpu",
) -> dict[int, torch.Tensor] | None:
    """
    Add noise to an image progressively using forward diffusion process.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save noisy images.
        num_timesteps: Total number of diffusion timesteps.
        timesteps_to_save: List of specific timesteps to save. If None, saves all.
        num_steps_to_save: Number of steps to save.
        noise_schedule: Type of noise schedule ("linear" or "cosine").
        beta_start: Starting beta value for noise schedule.
        beta_end: Ending beta value for noise schedule.
        image_size: Optional tuple (height, width) to resize image.
        device: Device to run computations on ("cpu" or "cuda").

    Returns:
        Dictionary mapping timestep to noisy image tensor.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    if image_size:
        image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)

    # Convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    x0 = transform(image).unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    # Generate noise schedule
    if noise_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(num_timesteps, device=device)
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")

    # Calculate alpha values
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Determine which timesteps to save
    if timesteps_to_save is None:
        # Save at logarithmic intervals for better visualization
        timesteps_to_save = [0] + [
            int(t) for t in np.logspace(0, np.log10(num_timesteps - 1), num_steps_to_save)
        ]
        timesteps_to_save = sorted(set(timesteps_to_save))

    # Store noisy images
    noisy_images = {}

    # Get original image name
    image_name = Path(image_path).stem

    # Save the base image (x0) first
    base_image_path = output_dir / f"{image_name}.png"
    save_tensor_as_image(x0.squeeze(0), base_image_path)

    # Add noise at each timestep and save
    for t in timesteps_to_save:
        # Generate random noise
        noise = torch.randn_like(x0)

        # Add noise according to diffusion process
        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])

        noisy_image = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

        # Store the noisy image
        noisy_images[t] = noisy_image.squeeze(0).cpu()

        # Save the noisy image
        save_path = output_dir / f"{image_name}_timestep_{t:04d}.png"
        save_tensor_as_image(noisy_image.squeeze(0), save_path)

    return noisy_images


def cosine_beta_schedule(timesteps: int, device: str = "cpu", s: float = 0.008) -> torch.Tensor:
    """
    Create cosine beta schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".

    Args:
        timesteps: Number of diffusion timesteps.
        device: Device to create tensor on.
        s: Small offset to prevent beta from being too small near t=0.

    Returns:
        Beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def save_tensor_as_image(tensor: torch.Tensor, save_path: str | Path) -> None:
    """
    Save a normalized tensor as an image.

    Args:
        tensor: Image tensor with values in [-1, 1], shape [C, H, W].
        save_path: Path to save the image.
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Convert to PIL image
    tensor_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    if tensor_np.shape[0] == 3:  # CHW format
        tensor_np = np.transpose(tensor_np, (1, 2, 0))

    image = Image.fromarray(tensor_np)
    image.save(save_path)


def batch_add_noise_to_images(
    image_dir: str | Path,
    output_base_dir: str | Path,
    num_timesteps: int = 1000,
    timesteps_to_save: list[int] | None = None,
    num_steps_to_save: int = 20,
    noise_schedule: str = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    image_size: tuple[int, int] | None = None,
    device: str = "cpu",
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> None:
    """
    Add noise to all images in a directory.

    Args:
        image_dir: Directory containing input images.
        output_base_dir: Base directory to save noisy images.
        num_timesteps: Total number of diffusion timesteps.
        timesteps_to_save: List of specific timesteps to save. If None, saves all.
        num_steps_to_save: Number of steps to save.
        noise_schedule: Type of noise schedule ("linear" or "cosine").
        beta_start: Starting beta value for noise schedule.
        beta_end: Ending beta value for noise schedule.
        image_size: Optional tuple (height, width) to resize images.
        device: Device to run computations on ("cpu" or "cuda").
        image_extensions: Tuple of valid image file extensions.
    """
    image_dir = Path(image_dir)
    output_base_dir = Path(output_base_dir)

    # Get all image files
    image_files: list[Path] = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_path in tqdm(image_files, total=len(image_files)):
        # Create output directory for this image
        output_dir = output_base_dir / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        add_noise_to_image(
            image_path=image_path,
            output_dir=output_dir,
            num_timesteps=num_timesteps,
            timesteps_to_save=timesteps_to_save,
            num_steps_to_save=num_steps_to_save,
            noise_schedule=noise_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            image_size=image_size,
            device=device,
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "add_noise": add_noise_to_image,
            "batch_add_noise": batch_add_noise_to_images,
        }
    )
