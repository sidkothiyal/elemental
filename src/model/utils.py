"""Utility functions for model initialization."""

import inspect
from typing import Type

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.

    Args:
        timesteps: Number of diffusion timesteps.
        s: Small offset to prevent beta from being too small near t=0.

    Returns:
        Beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def create_normalization(
    norm: Type[nn.Module] | None,
    out_channels: int,
    norm_kwargs: dict | None = None,
) -> nn.Module | None:
    """Create a normalization layer.

    Args:
        norm: Normalization class (e.g., nn.BatchNorm2d, nn.GroupNorm, None).
        out_channels: Number of output channels (used for normalization layers
            that require num_channels or num_features).
        norm_kwargs: Keyword arguments for initializing the normalization layer.

    Returns:
        Initialized normalization module or None if norm is None.

    Raises:
        ValueError: If normalization initialization fails or required parameters are missing.
    """
    if norm is None:
        return None

    norm_kwargs = norm_kwargs or {}

    # Get the signature of the norm class's __init__ method
    try:
        sig = inspect.signature(norm.__init__)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to inspect signature of {norm}: {e}") from e

    # Make a copy to avoid modifying the original dict
    norm_kwargs_copy = norm_kwargs.copy()

    # Check if the signature requires num_channels or num_features
    # and add out_channels if needed
    param_names = set(sig.parameters.keys())

    # Skip 'self' parameter
    param_names.discard("self")

    # Check if the signature requires num_channels or num_features
    # and add out_channels if needed
    if "num_channels" in param_names:
        norm_kwargs_copy["num_channels"] = out_channels
    elif "num_features" in param_names:
        norm_kwargs_copy["num_features"] = out_channels

    # Check if all required parameters (non-optional) are provided
    missing_params = []
    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue

        # Skip if parameter has a default value (it's optional)
        if param.default != inspect.Parameter.empty:
            continue

        # Check if the parameter is provided in norm_kwargs
        if param_name not in norm_kwargs_copy:
            missing_params.append(param_name)

    if missing_params:
        raise ValueError(
            f"Missing required parameters for {norm}: {missing_params}. "
            f"Provided parameters: {list(norm_kwargs.keys())}"
        )

    # Try to initialize the normalization layer
    try:
        return norm(**norm_kwargs_copy)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Failed to initialize {norm} with out_channels={out_channels} "
            f"and norm_kwargs={norm_kwargs}. Error: {e}"
        ) from e


def create_activation(
    act: Type[nn.Module] | None,
    act_kwargs: dict | None = None,
) -> nn.Module | None:
    """Create an activation layer.

    Args:
        act: Activation class (e.g., nn.ReLU, nn.SiLU, None).
        act_kwargs: Keyword arguments for initializing the activation layer.

    Returns:
        Initialized activation module or None if act is None.
    """
    if act is None:
        return None

    act_kwargs = act_kwargs or {}
    return act(**act_kwargs)


def assert_list_length(
    variable: list | None,
    expected_length: int,
    variable_name: str,
) -> None:
    """Assert that a variable (if not None) has the expected length.

    Args:
        variable: The variable to check (can be None for optional parameters).
        expected_length: The expected length of the variable.
        variable_name: Name of the variable for error messages.

    Raises:
        ValueError: If variable is not None and its length doesn't match expected_length.
    """
    if variable is not None and len(variable) != expected_length:
        raise ValueError(
            f"{variable_name} length ({len(variable)}) must match "
            f"expected length ({expected_length})."
        )
