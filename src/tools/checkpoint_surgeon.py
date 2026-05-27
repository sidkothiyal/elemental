#!/usr/bin/env python3
"""
Checkpoint Surgeon Tool

A tool to rename layers in PyTorch checkpoints based on configuration files.
This is useful when model structure changes but you want to load old checkpoints.

Usage:
    python checkpoint_surgeon.py \
        --config config.yaml \
        --checkpoint checkpoint.pt \
        --output fixed_checkpoint.pt

The config file should contain a 'checkpoint_surgery' section with rename rules:

    checkpoint_surgery:
      rename_rules:
        - old: "encoder.pos_embeddings"
          new: "pos_embeddings"
        - old: "encoder.to_patch_embedding"
          new: "to_patch_embedding"
        - old: "encoder.transformer.layers"
          new: "encoder.layers"
"""

from pathlib import Path
from typing import Any

import fire
import torch
import yaml


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load a PyTorch checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The loaded checkpoint dictionary.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any], output_path: str) -> None:
    """Save a PyTorch checkpoint.

    Args:
        checkpoint: The checkpoint dictionary to save.
        output_path: Path where to save the checkpoint.
    """
    print(f"Saving checkpoint to: {output_path}")
    # Create parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print("Checkpoint saved successfully!")


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the config file.

    Returns:
        The loaded configuration dictionary.
    """
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_rename_rule(key: str, old_pattern: str, new_pattern: str) -> tuple[str, bool]:
    """Apply a rename rule to a key.

    Args:
        key: The original key name.
        old_pattern: The old pattern to match.
        new_pattern: The new pattern to replace with.

    Returns:
        A tuple of (new_key, was_changed).
    """
    # Check if the key starts with the old pattern
    if key.startswith(old_pattern):
        # Replace the old pattern with the new pattern
        new_key = new_pattern + key[len(old_pattern) :]
        return new_key, True
    return key, False


def rename_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    rename_rules: list[dict[str, str]],
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Rename keys in a state_dict based on rename rules.

    Args:
        state_dict: The state dictionary with old keys.
        rename_rules: List of rename rules, each with 'old' and 'new' keys.
        verbose: Whether to print verbose output.

    Returns:
        A new state dictionary with renamed keys.
    """
    new_state_dict = {}
    rename_count = 0

    for old_key, value in state_dict.items():
        new_key = old_key
        renamed = False

        # Apply each rename rule in order
        for rule in rename_rules:
            old_pattern = rule["old"]
            new_pattern = rule["new"]
            new_key, changed = apply_rename_rule(new_key, old_pattern, new_pattern)
            if changed:
                renamed = True
                rename_count += 1
                if verbose:
                    print(f"  Renamed: {old_key} -> {new_key}")
                break  # Only apply first matching rule

        if not renamed and verbose:
            print(f"  Kept: {old_key}")

        new_state_dict[new_key] = value

    print(f"\nTotal keys renamed: {rename_count}/{len(state_dict)}")
    return new_state_dict


def perform_surgery(
    checkpoint: str,
    output: str,
    config: str,
    checkpoint_key: str = "model_state_dict",
    quiet: bool = False,
) -> None:
    """Perform checkpoint surgery by renaming layers.

    Args:
        checkpoint: Path to the input checkpoint file (required).
        output: Path to save the fixed checkpoint (required).
        config: Path to config YAML file with checkpoint_surgery section (required).
        checkpoint_key: Key in checkpoint dict containing the state_dict
            (default: "model_state_dict").
        quiet: Suppress verbose output (default: False).
    """
    # Load config and get rename rules
    config_data = load_config(config)

    if (
        "checkpoint_surgery" not in config_data
        or "rename_rules" not in config_data["checkpoint_surgery"]
    ):
        print("Error: No 'checkpoint_surgery.rename_rules' found in config")
        print("You can add a section like this to your config:")
        print(
            """
checkpoint_surgery:
  rename_rules:
    - old: "encoder.pos_embeddings"
      new: "pos_embeddings"
    - old: "encoder.to_patch_embedding"
      new: "to_patch_embedding"
    - old: "encoder.transformer.layers"
      new: "encoder.layers"
        """
        )
        raise ValueError("No rename rules found in config")

    rename_rules = config_data["checkpoint_surgery"]["rename_rules"]
    print("Using rename rules from config file")

    # Load checkpoint
    checkpoint_data = load_checkpoint(checkpoint)

    # Get the state_dict
    if checkpoint_key in checkpoint_data:
        state_dict = checkpoint_data[checkpoint_key]
        print(f"\nFound '{checkpoint_key}' in checkpoint with {len(state_dict)} keys")
    else:
        # Assume the checkpoint is just a state_dict
        state_dict = checkpoint_data
        checkpoint_key = None
        print(f"\nCheckpoint is a state_dict with {len(state_dict)} keys")

    # Apply rename rules
    print("\nApplying rename rules:")
    for i, rule in enumerate(rename_rules, 1):
        print(f"  Rule {i}: '{rule['old']}' -> '{rule['new']}'")
    print()

    new_state_dict = rename_state_dict_keys(state_dict, rename_rules, verbose=not quiet)

    # Update checkpoint with new state_dict
    if checkpoint_key is not None:
        checkpoint_data[checkpoint_key] = new_state_dict
    else:
        checkpoint_data = new_state_dict

    # Save the fixed checkpoint
    save_checkpoint(checkpoint_data, output)

    print("\n" + "=" * 80)
    print("Surgery completed successfully!")
    print("=" * 80)


def main():
    """Main entry point for the checkpoint surgeon tool."""
    fire.Fire(perform_surgery)


if __name__ == "__main__":
    main()
