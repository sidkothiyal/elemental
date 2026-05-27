"""Diffusion denoising model supporting both DDPM and DDIM sampling."""

import logging
import os
from typing import Literal

import torch
from torch import nn
from torchvision.utils import save_image

from model.utils import cosine_beta_schedule

logger = logging.getLogger(__name__)


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
        # Inference settings
        start_timestep: int | None = None,
        num_denoising_steps: int | None = None,
        denoising_delta: float | None = None,
        inference_method: Literal["ddpm", "ddim"] = "ddim",
        eta: float = 0.0,
        # Debug settings
        debug_mode: bool = False,
        debug_dir: str | None = None,
        debug_save_interval: int = 50,
        # Data normalization - set to True if data is in [-1, 1], False if in [0, 1]
        data_normalized_minus1_1: bool = True,
        # Whether to clamp predictions during forward pass
        # Set to False if your data is not strictly bounded (e.g. normalized actions)
        clamp_predictions: bool = True,
        # If True, skip iterative denoising and directly predict x0 (useful for debugging)
        direct_prediction: bool = False,
    ):
        """
        Initialize the denoising diffusion model.

        Args:
            model: Base denoising network that predicts noise/clean image.
                  Should accept (x, t) or (x, t, cond) and output same shape as x.
            conditional_model: Optional model for processing conditional inputs.
                             Output will be passed as additional input to base model.
                             If None, and cond is provided in forward step, it will be passed to
                             the model as "aux_features".
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
            start_timestep: Starting timestep for inference.
            num_denoising_steps: Number of denoising steps for inference.
            denoising_delta: Threshold for denoising delta for inference.
            inference_method: Method to use for inference ("ddpm" or "ddim").
            eta: DDIM stochasticity parameter for inference.
            debug_mode: Enable debug logging and image saving.
            debug_dir: Directory to save debug images (required if debug_mode=True).
            debug_save_interval: Interval to save debug images.
            data_normalized_minus1_1: Whether the data is normalized to [-1, 1].
            direct_prediction: Whether to skip iterative denoising and directly predict x0.
        """
        super().__init__()

        self.model = model
        self.conditional_model = conditional_model
        self.timestep_model = timestep_model
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.clamp_predictions = clamp_predictions

        # Inference settings
        self.start_timestep = start_timestep
        self.num_denoising_steps = num_denoising_steps
        self.denoising_delta = denoising_delta
        self.inference_method = inference_method
        self.eta = eta

        # If True, skip iterative denoising and directly predict x0 (useful for debugging)
        self.direct_prediction = direct_prediction

        # Debug settings
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.debug_save_interval = debug_save_interval
        self._debug_batch_counter = 0

        # Data normalization setting
        # True: data is in [-1, 1] range (standard for diffusion)
        # False: data is in [0, 1] range
        self.data_normalized_minus1_1 = data_normalized_minus1_1
        if not self.data_normalized_minus1_1:
            logger.warning(
                "Data is not normalized to [-1, 1], this is not recommended for "
                "denoising diffusion models."
            )

        # Initialize debug directory if needed
        if self.debug_mode and self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

        self._init_inference_settings()

        # Generate noise schedule
        if noise_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif noise_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        self._init_noise_coefficients(betas)

    def _init_inference_settings(self):
        """Initialize inference settings."""
        if self.start_timestep is None:
            self.start_timestep = self.num_timesteps - 1
        # Clamp start_timestep to valid range [0, num_timesteps - 1]
        if self.start_timestep >= self.num_timesteps:
            logger.warning(
                "start_timestep (%s) >= num_timesteps (%s), clamping to %s",
                self.start_timestep,
                self.num_timesteps,
                self.num_timesteps - 1,
            )
            self.start_timestep = self.num_timesteps - 1
        if self.num_denoising_steps is None:
            self.num_denoising_steps = self.num_timesteps
        if self.denoising_delta is None:
            self.denoising_delta = 0.001

        if self.inference_method == "ddim":
            timesteps = torch.linspace(
                self.start_timestep, 0, self.num_denoising_steps, dtype=torch.long
            )
        elif self.inference_method == "ddpm":
            timesteps = torch.arange(self.start_timestep, -1, -1, dtype=torch.long)
            timesteps = timesteps[: self.num_denoising_steps]
        else:
            raise ValueError(f"Unknown inference method: {self.inference_method}")

        self.register_buffer("timesteps", timesteps)

    def _init_noise_coefficients(self, betas: torch.Tensor):
        """
        Initialize noise coefficients based on inference method.

        Args:
            betas: Beta values for the noise schedule.
        """
        # Pre-compute useful values for diffusion (needed by both DDPM and DDIM)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register as buffers so they're moved with the model
        # Note: betas and alphas are only used during initialization, not stored
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Pre-compute values used by both DDPM and DDIM (for predict_noise_from_x0
        # and predict_x0_from_noise)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Pre-compute posterior variance and mean coefficients only for DDPM
        if self.inference_method == "ddpm":
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.register_buffer("posterior_variance", posterior_variance)

            # Coefficients for posterior mean in DDPM
            posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            posterior_mean_coef2 = (
                (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
            )
            self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
            self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

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
        if cond is not None:
            aux_features = cond
            if self.conditional_model is not None:
                if isinstance(aux_features, dict):
                    aux_features = self.conditional_model(**aux_features)
                else:
                    aux_features = self.conditional_model(aux_features)
            model_args["aux_features"] = aux_features

        if self.timestep_model:
            model_args["t"] = self.timestep_model(t)
        model_output = self.model(x_t, **model_args)

        if self.clamp_predictions:
            clamp_range = (-1.0, 1.0) if self.data_normalized_minus1_1 else (0.0, 1.0)

        if self.prediction_type == "epsilon":
            pred_noise = model_output
            pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
            if self.clamp_predictions:
                pred_x0 = torch.clamp(pred_x0, *clamp_range)
        elif self.prediction_type == "x0":
            pred_x0 = model_output
            if self.clamp_predictions:
                # Clamp first, then derive pred_noise from the clamped x0 for DDIM consistency
                pred_x0 = torch.clamp(pred_x0, *clamp_range)
            pred_noise = self.predict_noise_from_x0(x_t, t, pred_x0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

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
        _, pred_x0 = self.predict_from_model(x_t, t, cond)

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

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * pred_noise

        # Deterministic denoising
        x_t_prev = sqrt_alpha_cumprod_t_prev * pred_x0 + pred_dir

        # Add stochastic noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise

        return x_t_prev

    def _normalize_and_clamp_for_vis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize and clamp tensor for visualization.

        Converts data from [-1, 1] to [0, 1] if data_normalized_minus1_1 is True,
        otherwise just clones the tensor. Then clamps to [0.0, 1.0].

        Args:
            x: Input tensor to normalize and clamp.

        Returns:
            Normalized and clamped tensor in [0, 1] range.
        """
        if self.data_normalized_minus1_1:
            # Data is in [-1, 1], convert to [0, 1]
            x_vis = (x + 1.0) / 2.0
        else:
            # Data is already in [0, 1]
            x_vis = x.clone()
        x_vis = torch.clamp(x_vis, 0.0, 1.0)
        return x_vis

    def _save_debug_images(
        self,
        x_t: torch.Tensor,
        batch_idx: int,
        timestep: int,
        cond: torch.Tensor | None = None,
        force_save: bool = False,
    ) -> None:
        """
        Save debug images for each sample in the batch at the current timestep.

        Args:
            x_t: Image tensor at timestep t [B, C, H, W].
            batch_idx: Current batch index.
            timestep: Current timestep value.
            cond: Optional conditional tensor [B, C, H, W] to save.
            force_save: If True, save regardless of interval (for start/end timesteps).
        """
        if not self.debug_mode or self.debug_dir is None:
            return

        # Only save at intervals to avoid filling up disk
        if not force_save and timestep % self.debug_save_interval != 0:
            return

        batch_size = x_t.shape[0]

        # Denormalize to [0, 1] for visualization based on data normalization setting
        x_t_vis = self._normalize_and_clamp_for_vis(x_t)

        # Process cond tensor if provided
        cond_vis = None
        if cond is not None:
            logger.debug("Cond shape: %s", cond.shape)
            # Handle different possible shapes of cond
            if cond.ndim == 4:  # [B, C, H, W] - image-like
                cond_vis = self._normalize_and_clamp_for_vis(cond)
            elif cond.ndim == 2:  # [B, D] - might be embeddings, skip for now
                cond_vis = None
            # Add other shape handling if needed

        for sample_idx in range(batch_size):
            sample_dir = os.path.join(
                self.debug_dir, f"batch_{batch_idx:04d}", f"sample_{sample_idx:04d}"
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Save main image
            image_path = os.path.join(sample_dir, f"timestep_{timestep:04d}.png")
            save_image(x_t_vis[sample_idx : sample_idx + 1], image_path)

            # Save conditional image if available (only once)
            if cond_vis is not None:
                cond_path = os.path.join(sample_dir, "cond.png")
                if not os.path.exists(cond_path):
                    save_image(cond_vis[sample_idx : sample_idx + 1], cond_path)

    def _get_batch_idx(self, batch_idx: int | None) -> int:
        """Return batch_idx, auto-incrementing the internal counter if not provided."""
        if batch_idx is None:
            batch_idx = self._debug_batch_counter
            self._debug_batch_counter += 1
        return batch_idx

    def _compute_timesteps_from_start(self, start_t: int, device: torch.device) -> torch.Tensor:
        """
        Compute the denoising timestep schedule starting from a specific timestep.

        Args:
            start_t: Starting timestep for denoising.
            device: Device to place the tensor on.

        Returns:
            Tensor of timesteps to iterate through during denoising.
        """
        # Clamp start_t to valid range
        start_t = max(0, min(start_t, self.num_timesteps - 1))

        if self.inference_method == "ddim":
            # For DDIM, create evenly spaced steps from start_t to 0
            num_steps = min(self.num_denoising_steps or (start_t + 1), start_t + 1)
            timesteps = torch.linspace(start_t, 0, num_steps, dtype=torch.long, device=device)
        elif self.inference_method == "ddpm":
            # For DDPM, go from start_t down to 0, one step at a time
            timesteps = torch.arange(start_t, -1, -1, dtype=torch.long, device=device)
            timesteps = timesteps[: self.num_denoising_steps]
        else:
            raise ValueError(f"Unknown inference method: {self.inference_method}")

        return timesteps

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        return_intermediates: bool = False,
        batch_idx: int | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass for training and inference.

        During training, returns a dict with both ``pred_noise`` and
        ``pred_x0`` so that downstream consumers (loss, metrics,
        visualisation) can each choose the appropriate prediction.

        During inference (eval mode), performs iterative denoising and
        returns ``{"pred_x0": denoised_result}``.

        Args:
            x: Noisy input image tensor [B, C, H, W], normalized to [-1, 1].
            t: Timestep tensor [B] indicating the noise level of the input.
               During inference: if provided, used as starting timestep for denoising.
               If None, uses self.start_timestep.
            cond: Optional conditional/auxiliary input for the conditional model.
            return_intermediates: If True, returns list of all intermediate results (for inference).
            batch_idx: Optional batch index for debugging. If None and debug_mode is enabled,
                      will use an internal counter.

        Returns:
            A dict ``{"pred_noise": ..., "pred_x0": ...}``.  During
            training both come from a single-step prediction; during
            inference ``pred_x0`` is the iterative-denoising result while
            ``pred_noise`` is a single-step prediction at the initial
            timestep.  When *return_intermediates* is True, returns
            ``(pred_dict, intermediates_list)``.
        """
        # Get model predictions for training or direct prediction mode
        # Direct prediction mode: skip iterative denoising, just predict x0 directly
        # Useful for testing if model can predict without accumulated DDIM errors
        if self.training or self.direct_prediction:
            pred_noise, pred_x0 = self.predict_from_model(x, t, cond)

            # Direct prediction mode: optional debug logging
            if self.debug_mode and self.direct_prediction:
                batch_idx = self._get_batch_idx(batch_idx)
                t_val = t[0].item() if t is not None else 0
                logger.info("Direct prediction mode: single-step x0 prediction at t=%d", t_val)
                self._save_debug_images(x, batch_idx, int(t_val), cond, force_save=True)
                self._save_debug_images(pred_x0, batch_idx, 0, None, force_save=True)

            return {"pred_noise": pred_noise, "pred_x0": pred_x0}

        # Inference mode: iterative denoising
        batch_size = x.shape[0]
        device = x.device

        batch_idx = self._get_batch_idx(batch_idx)

        # Determine starting timestep for inference
        # If t is provided, use it as the starting point (input is noised at that level)
        # Otherwise, use self.start_timestep (assumes input is pure noise)
        if t is not None:
            # Use the provided timestep as starting point
            # For batched inference, use the max timestep in the batch for safety
            actual_start_t = int(t.max().item()) if t.numel() > 1 else int(t.item())
            timesteps = self._compute_timesteps_from_start(actual_start_t, device)
        else:
            # No timestep provided - use pre-computed schedule from start_timestep
            x = torch.randn_like(x)
            timesteps = self.timesteps.to(device)

        # Initialize
        x_t = x
        intermediates = [x_t.cpu()] if return_intermediates else []

        # Single-step prediction at the initial timestep so that pred_noise
        # is available during inference (e.g. for the loss that compares
        # predicted noise against noise targets during validation).
        t_initial = torch.full((batch_size,), timesteps[0], device=device, dtype=torch.long)
        pred_noise, _ = self.predict_from_model(x_t, t_initial, cond)

        # Save initial state if debugging (force save start)
        if self.debug_mode:
            initial_timestep = timesteps[0].item()
            self._save_debug_images(x_t, batch_idx, initial_timestep, cond, force_save=True)
            logger.info(
                (
                    "Debug: Batch %d, Starting denoising with %d steps, method=%s, "
                    "initial_timestep=%d (input_t=%s)"
                ),
                batch_idx,
                len(timesteps),
                self.inference_method,
                initial_timestep,
                t[0].item() if t is not None else "None",
            )

        # Iterative denoising
        for i, t_curr in enumerate(timesteps):
            t_curr = torch.full((batch_size,), t_curr, device=x_t.device, dtype=torch.long)
            t_curr_val = t_curr[0].item()

            if self.inference_method == "ddpm":
                t_prev_val = t_curr_val - 1 if t_curr_val > 0 else 0
                x_t_prev = self.ddpm_step(x_t, t_curr, cond)
            elif self.inference_method == "ddim":
                t_next_raw = (
                    timesteps[i + 1]
                    if i < len(timesteps) - 1
                    else torch.tensor(0, dtype=torch.long, device=device)
                )
                t_prev_val = t_next_raw.item()
                t_prev = torch.full((batch_size,), t_next_raw, device=x_t.device, dtype=torch.long)
                x_t_prev = self.ddim_step(x_t, t_curr, t_prev, cond, self.eta)
            else:
                raise ValueError(f"Unknown inference method: {self.inference_method}")

            # Calculate change metrics
            change = torch.abs(x_t_prev - x_t)
            max_change = change.max().item()
            mean_change = change.mean().item()

            if self.debug_mode:
                logger.info(
                    "Debug: Batch %d, Timestep %d -> %d, Max change: %.6f, Mean change: %.6f",
                    batch_idx,
                    t_curr_val,
                    t_prev_val,
                    max_change,
                    mean_change,
                )
                is_final = t_prev_val == 0
                self._save_debug_images(x_t_prev, batch_idx, t_prev_val, cond, force_save=is_final)

            # Check convergence if denoising_delta is specified
            if self.denoising_delta is not None:
                if max_change < self.denoising_delta:
                    if self.debug_mode:
                        logger.info(
                            "Debug: Batch %d, Converged at timestep %d (change: %.6f < %.6f)",
                            batch_idx,
                            t_curr_val,
                            max_change,
                            self.denoising_delta,
                        )
                    x_t = x_t_prev
                    if return_intermediates:
                        intermediates.append(x_t.cpu())
                    break

            x_t = x_t_prev

            if return_intermediates:
                intermediates.append(x_t.cpu())

        if self.debug_mode:
            logger.info("Debug: Batch %d, Denoising complete", batch_idx)

        result = {"pred_noise": pred_noise, "pred_x0": x_t}
        if return_intermediates:
            return result, intermediates
        return result


if __name__ == "__main__":
    # Simple test with a dummy model
    class DummyDenoiser(nn.Module):
        """Dummy denoiser for testing."""

        def __init__(self, channels=3):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, channels, 3, padding=1)
            self.time_embed = nn.Embedding(1000, 64)

        def forward(self, x, t, cond=None):  # pylint: disable=unused-argument
            """Forward pass for the dummy denoiser."""
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
    diffusion_ddim = DenoisingDiffusionModel(
        model=dummy_model,
        num_timesteps=1000,
        noise_schedule="linear",
        inference_method="ddim",
        num_denoising_steps=50,
        eta=0.0,
    )
    diffusion_ddim.eval()
    x_denoised_ddim = diffusion_ddim(x_noisy)
    print(f"Output shape: {x_denoised_ddim.shape}")
    print(f"Output range: [{x_denoised_ddim.min():.3f}, {x_denoised_ddim.max():.3f}]")

    # Test DDPM denoising
    print("\n--- Testing DDPM (100 steps) ---")
    diffusion_ddpm = DenoisingDiffusionModel(
        model=dummy_model,
        num_timesteps=1000,
        noise_schedule="linear",
        inference_method="ddpm",
        num_denoising_steps=100,
    )
    diffusion_ddpm.eval()
    x_denoised_ddpm = diffusion_ddpm(x_noisy)
    print(f"Output shape: {x_denoised_ddpm.shape}")
    print(f"Output range: [{x_denoised_ddpm.min():.3f}, {x_denoised_ddpm.max():.3f}]")

    # Test with denoising delta
    print("\n--- Testing with denoising_delta ---")
    diffusion_delta = DenoisingDiffusionModel(
        model=dummy_model,
        num_timesteps=1000,
        noise_schedule="linear",
        inference_method="ddim",
        num_denoising_steps=1000,
        denoising_delta=0.01,
    )
    diffusion_delta.eval()
    x_denoised_delta, sample_intermediates = diffusion_delta(x_noisy, return_intermediates=True)
    print(f"Converged after {len(sample_intermediates)} steps")
