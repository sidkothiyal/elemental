"""Diffusion denoising model supporting both DDPM and DDIM sampling."""

from typing import Literal

import torch
from torch import nn


class DenoisingDiffusionModel(nn.Module):
    """
    Denoising diffusion model that supports both DDPM and DDIM sampling methods.

    This model wraps a base denoising network (and optional conditional model) and
    provides iterative denoising capabilities with configurable stopping criteria.

    The model expects the base network to predict noise (epsilon prediction).
    """

    def __init__(
        self,
        model: nn.Module,
        conditional_model: nn.Module | None = None,
        timestep_model: nn.Module | None = None,
        num_timesteps: int = 1000,
        noise_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # TODO: add v-prediction
        prediction_type: Literal["epsilon", "x0"] = "epsilon",
    ):
        """
        Initialize the denoising diffusion model.

        Args:
            model: Base denoising network that predicts noise/clean image.
                  Should accept (x, t) or (x, t, cond) and output same shape as x.
            conditional_model: Optional model for processing conditional inputs.
                             Output will be passed as additional input to base model.
            timestep_model: Model to project the timestep to a tensor of the same shape as
                the input (e.g. a linear layer, embedding layer, etc.).
            num_timesteps: Total number of diffusion timesteps (default: 1000).
            noise_schedule: Type of noise schedule ("linear" or "cosine").
            beta_start: Starting beta value for linear schedule.
            beta_end: Ending beta value for linear schedule.
            prediction_type: What the model predicts:
                           - "epsilon": predicts noise (standard)
                           - "x0": predicts clean image directly
                        #    - "v": predicts velocity (v-prediction)
        """
        super().__init__()

        self.model = model
        self.conditional_model = conditional_model
        self.timestep_model = timestep_model
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # Generate noise schedule
        if noise_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif noise_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        self._init_noise_coefficients(betas)

    def _init_noise_coefficients(self, betas: torch.Tensor):
        """
        Initialize noise coefficients.

        Args:
            betas: Beta values for the noise schedule.
        """
        # Pre-compute useful values for diffusion
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register as buffers so they're moved with the model
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Pre-compute values for DDPM sampling
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Pre-compute posterior variance for DDPM
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20))
        )

        # Coefficients for posterior mean in DDPM
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Create cosine beta schedule.

        Args:
            timesteps: Number of timesteps.
            s: Small offset to prevent beta from being too small.

        Returns:
            Beta schedule tensor.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def predict_noise_from_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise given x_t and predicted x0.

        Args:
            x_t: Noisy image at timestep t.
            t: Current timestep.
            x0: Predicted clean image.

        Returns:
            Predicted noise.
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        return (x_t - sqrt_alphas_cumprod_t * x0) / sqrt_one_minus_alphas_cumprod_t

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict clean image x0 from noisy image and predicted noise.

        Args:
            x_t: Noisy image at timestep t.
            t: Current timestep.
            noise: Predicted noise.

        Returns:
            Predicted clean image x0.
        """
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extract values from a tensor at specific timesteps and reshape for broadcasting.

        Args:
            a: Tensor to extract from (1D).
            t: Timestep indices.
            x_shape: Shape of the image tensor for broadcasting.

        Returns:
            Extracted and reshaped tensor.
        """
        batch_size = t.shape[0]
        out = a.gather(0, t).to(t.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def predict_from_model(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get model prediction and convert to both noise and x0 predictions.

        Args:
            x_t: Noisy image at timestep t.
            t: Current timestep.
            cond: Optional conditional input.

        Returns:
            Tuple of (predicted_noise, predicted_x0).
        """
        model_args = {}
        # Process conditional input if provided
        if cond is not None and self.conditional_model is not None:
            cond_features = self.conditional_model(cond)
            model_args["aux_features"] = cond_features

        if self.timestep_model:
            model_args["t"] = self.timestep_model(t)
        model_output = self.model(x_t, **model_args)

        # Convert model output to noise and x0 predictions based on prediction type
        if self.prediction_type == "epsilon":
            # Model predicts noise
            pred_noise = model_output
            pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
        elif self.prediction_type == "x0":
            # Model predicts clean image
            pred_x0 = model_output
            pred_noise = self.predict_noise_from_x0(x_t, t, pred_x0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip x0 to valid range [-1, 1] for stability
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        return pred_noise, pred_x0

    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform one DDPM denoising step.

        Args:
            x_t: Noisy image at timestep t.
            t: Current timestep.
            cond: Optional conditional input.

        Returns:
            Denoised image at timestep t-1.
        """
        # Get model predictions
        pred_noise, pred_x0 = self.predict_from_model(x_t, t, cond)

        # Compute posterior mean
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)

        posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x_t

        # Add noise for t > 0
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)

        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        x_t_minus_1 = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

        return x_t_minus_1

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        cond: torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform one DDIM denoising step.

        Args:
            x_t: Noisy image at timestep t.
            t: Current timestep.
            t_prev: Previous timestep (target timestep).
            cond: Optional conditional input.
            eta: DDIM stochasticity parameter (0 = deterministic, 1 = DDPM).

        Returns:
            Denoised image at timestep t_prev.
        """
        # Get model predictions
        pred_noise, pred_x0 = self.predict_from_model(x_t, t, cond)

        # Get alpha values
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        # Compute variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev)
            / (1 - alpha_cumprod_t)
            * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        # Compute predicted x_{t-1}
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        # sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1 - alpha_cumprod_t_prev)

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * pred_noise

        # Deterministic denoising
        x_t_prev = sqrt_alpha_cumprod_t_prev * pred_x0 + pred_dir

        # Add stochastic noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise

        return x_t_prev

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        start_timestep: int | None = None,
        num_denoising_steps: int | None = None,
        denoising_delta: float | None = None,
        method: Literal["ddpm", "ddim"] = "ddim",
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for training: predicts noise or clean image from noisy input.

        During training, this method is used to predict either noise (epsilon) or
        the clean image (x0) from a noisy input at a given timestep.

        Args:
            x: Noisy input image tensor [B, C, H, W], normalized to [-1, 1].
            t: Timestep tensor [B] indicating the noise level.
            cond: Optional conditional/auxiliary input for the conditional model.
            num_denoising_steps: Number of denoising steps to take (for inference).
                               If None, uses all timesteps (self.num_timesteps).
            denoising_delta: Optional threshold for change between steps (for inference).
                           If provided, stops denoising when max change < delta.
            method: Denoising method to use ("ddpm" or "ddim") (for inference).
            eta: DDIM stochasticity parameter (only for DDIM, 0=deterministic) (for inference).
            start_timestep: Starting timestep (for inference). If None, starts from num_timesteps-1.
            return_intermediates: If True, returns list of all intermediate results (for inference).

        Returns:
            Predicted noise (if prediction_type="epsilon") or clean image (if prediction_type="x0").
        """
        # Get model predictions
        pred_noise, pred_x0 = self.predict_from_model(x, t, cond)

        if self.prediction_type == "epsilon":
            return pred_noise
        if self.prediction_type == "x0":
            return pred_x0
        raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # device = x.device
        # batch_size = x.shape[0]

        # # Determine timestep sequence
        # if num_denoising_steps is None:
        #     num_denoising_steps = self.num_timesteps

        # if start_timestep is None:
        #     start_timestep = self.num_timesteps - 1

        # # Create timestep sequence
        # if method == "ddim":
        #     # For DDIM, we can skip timesteps
        #     timesteps = torch.linspace(start_timestep, 0, num_denoising_steps, dtype=torch.long)
        #     timesteps = timesteps.to(device)
        # else:
        #     # For DDPM, we need consecutive timesteps
        #     timesteps = torch.arange(start_timestep, -1, -1, dtype=torch.long)
        #     timesteps = timesteps[:num_denoising_steps].to(device)

        # # Initialize
        # x_t = x
        # intermediates = [x_t.cpu()] if return_intermediates else []

        # # Iterative denoising
        # for i, t_curr in enumerate(timesteps):
        #     # Prepare timestep tensor
        #     t = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)

        #     # Perform denoising step
        #     if method == "ddpm":
        #         x_t_prev = self.ddpm_step(x_t, t, cond)
        #     elif method == "ddim":
        #         # Get next timestep
        #         if i < len(timesteps) - 1:
        #             t_prev = timesteps[i + 1]
        #         else:
        #             t_prev = torch.tensor(0, device=device)

        #         t_prev_tensor = torch.full(
        #             (batch_size,), t_prev, device=device, dtype=torch.long
        #         )
        #         x_t_prev = self.ddim_step(x_t, t, t_prev_tensor, cond, eta)
        #     else:
        #         raise ValueError(f"Unknown method: {method}")

        #     # Check convergence if denoising_delta is specified
        #     if denoising_delta is not None:
        #         max_change = torch.abs(x_t_prev - x_t).max().item()
        #         if max_change < denoising_delta:
        #             x_t = x_t_prev
        #             if return_intermediates:
        #                 intermediates.append(x_t.cpu())
        #             break

        #     x_t = x_t_prev

        #     if return_intermediates:
        #         intermediates.append(x_t.cpu())

        # if return_intermediates:
        #     return x_t, intermediates
        # return x_t

    def sample(
        self,
        shape: tuple[int, int, int, int],
        cond: torch.Tensor | None = None,
        num_denoising_steps: int | None = None,
        method: Literal["ddpm", "ddim"] = "ddim",
        eta: float = 0.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples from pure noise.

        Args:
            shape: Shape of samples to generate [B, C, H, W].
            cond: Optional conditional input.
            num_denoising_steps: Number of denoising steps.
            method: Denoising method ("ddpm" or "ddim").
            eta: DDIM stochasticity parameter.
            device: Device to generate samples on.

        Returns:
            Generated samples tensor.
        """
        # Start from pure noise
        x_T = torch.randn(shape, device=device)

        # Denoise
        x_0 = self.forward(
            x_T,
            cond=cond,
            num_denoising_steps=num_denoising_steps,
            method=method,
            eta=eta,
        )

        return x_0


if __name__ == "__main__":
    # Simple test with a dummy model
    class DummyDenoiser(nn.Module):
        """Dummy denoiser for testing."""

        def __init__(self, channels=3):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, channels, 3, padding=1)
            self.time_embed = nn.Embedding(1000, 64)

        def forward(self, x, t, cond=None):
            # Simple U-Net style processing
            t_emb = self.time_embed(t)
            # Broadcast time embedding
            t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)

            h = self.conv1(x)
            h = h + t_emb  # Add time embedding
            h = torch.relu(h)
            out = self.conv2(h)
            return out

    print("Testing DenoisingDiffusionModel...")

    # Create dummy model
    dummy_model = DummyDenoiser()

    # Create diffusion model
    diffusion = DenoisingDiffusionModel(
        model=dummy_model,
        num_timesteps=1000,
        noise_schedule="linear",
    )

    # Test with random noisy input
    x_noisy = torch.randn(2, 3, 64, 64)

    print(f"\nInput shape: {x_noisy.shape}")
    print(f"Input range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")

    # Test DDIM denoising (fast)
    print("\n--- Testing DDIM (50 steps) ---")
    x_denoised_ddim = diffusion(
        x_noisy,
        num_denoising_steps=50,
        method="ddim",
        eta=0.0,
    )
    print(f"Output shape: {x_denoised_ddim.shape}")
    print(f"Output range: [{x_denoised_ddim.min():.3f}, {x_denoised_ddim.max():.3f}]")

    # Test DDPM denoising
    print("\n--- Testing DDPM (100 steps) ---")
    x_denoised_ddpm = diffusion(
        x_noisy,
        num_denoising_steps=100,
        method="ddpm",
    )
    print(f"Output shape: {x_denoised_ddpm.shape}")
    print(f"Output range: [{x_denoised_ddpm.min():.3f}, {x_denoised_ddpm.max():.3f}]")

    # Test with denoising delta
    print("\n--- Testing with denoising_delta ---")
    x_denoised_delta, intermediates = diffusion(
        x_noisy,
        num_denoising_steps=1000,
        denoising_delta=0.01,
        method="ddim",
        return_intermediates=True,
    )
    print(f"Converged after {len(intermediates)} steps")

    # Test sampling from noise
    print("\n--- Testing sampling from noise ---")
    samples = diffusion.sample(
        shape=(2, 3, 64, 64),
        num_denoising_steps=50,
        method="ddim",
        device="cpu",
    )
    print(f"Generated samples shape: {samples.shape}")
    print(f"Generated samples range: [{samples.min():.3f}, {samples.max():.3f}]")

    print("\n✓ All tests completed successfully!")
