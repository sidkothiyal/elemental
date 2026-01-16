"""Diffusion dataset that generates noisy images on-the-fly."""

import torch
from torch.utils import data


class DiffusionDataset(data.Dataset):
    """Dataset wrapper that adds diffusion noise to images on-the-fly.

    This dataset wraps another dataset and applies forward diffusion process
    to generate noisy images during training. This is more flexible and efficient
    than pre-generating all noisy images.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        num_timesteps: int = 1000,
        noise_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        prediction_type: str = "epsilon",
    ):
        """DiffusionDataset constructor.

        Args:
            dataset: Base dataset that returns images. Expected to return dict with
                    'input' or 'target' key containing image tensors normalized to [-1, 1].
            num_timesteps: Total number of diffusion timesteps.
            noise_schedule: Type of noise schedule ("linear" or "cosine").
            beta_start: Starting beta value for noise schedule.
            beta_end: Ending beta value for noise schedule.
            device: Device to store noise schedule tensors on.
            prediction_type: What the model predicts - "epsilon" (noise) or "x0" (clean image).
        """
        self.dataset = dataset
        self.num_timesteps = num_timesteps
        self.noise_schedule = noise_schedule
        self.device = device
        self.prediction_type = prediction_type

        # Generate noise schedule
        if noise_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif noise_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps, device=device)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        # Calculate alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(
        self, timesteps: int, device: str = "cpu", s: float = 0.008
    ) -> torch.Tensor:
        """
        Cosine beta schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".

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

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Get an item from the dataset with added diffusion noise.

        Args:
            idx: Index of the item to get.

        Returns:
            Dictionary with keys:
                - 'input': Noisy image tensor
                - 'target': Clean image tensor
                - 'timestep': Timestep value
        """
        # Get clean image from base dataset
        item = self.dataset[idx]

        # Extract the clean image - try 'target' first, then 'input'
        if "target" in item:
            x0 = item["target"]
        elif "input" in item:
            x0 = item["input"]
        else:
            raise ValueError("Dataset item must contain either 'input' or 'target' key")

        # Ensure x0 is a tensor
        if not isinstance(x0, torch.Tensor):
            raise ValueError(f"Image must be a tensor, got {type(x0)}")

        x0 = x0.to(self.device)
        # Sample a random timestep
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).squeeze()

        # Generate random noise with same shape as x0
        noise = torch.randn_like(x0)

        # Add noise according to diffusion process
        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        noisy_image = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

        # Return noisy image and appropriate target based on prediction type
        # For epsilon prediction: target should be noise
        # For x0 prediction: target should be clean image
        target = noise if self.prediction_type == "epsilon" else x0

        return {
            "input": noisy_image,
            "target": target,
            "t": t,
            "cond": x0,
        }


if __name__ == "__main__":
    from tqdm import tqdm

    # Import a base dataset to wrap
    from dataset.div2k_dataset import DIV2KDataset

    # Test the dataset
    print("Testing DiffusionDataset...")

    # Create base dataset
    base_dataset = DIV2KDataset(root_dir="data/div2k/train", target_shape=(256, 256))
    print(f"Base dataset length: {len(base_dataset)}")

    # Wrap it with DiffusionDataset
    diffusion_dataset = DiffusionDataset(
        dataset=base_dataset,
        num_timesteps=1000,
        noise_schedule="cosine",
        device="cpu",
    )
    print(f"Diffusion dataset length: {len(diffusion_dataset)}")

    # Test dataloader
    dataloader = data.DataLoader(diffusion_dataset, batch_size=4, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nTesting dataloader...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if batch_idx == 0:
            print("\nFirst batch:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(
                        f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                        f"min={v.min():.3f}, max={v.max():.3f}"
                    )
                else:
                    print(f"  {k}: {v}")

        if batch_idx >= 10:
            break

    print("\nDiffusionDataset test completed successfully!")
    print("\nUsage example:")
    print("  base_dataset = DIV2KDataset(root_dir='data/div2k/train', target_shape=(256, 256))")
    print("  diffusion_dataset = DiffusionDataset(dataset=base_dataset, num_timesteps=1000)")
    print("  dataloader = DataLoader(diffusion_dataset, batch_size=8, shuffle=True)")
