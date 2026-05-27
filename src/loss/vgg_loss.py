"""A VGG-based perceptual loss function for PyTorch."""

import logging

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0-1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at one or more layers of the pretrained
    model specified by :attr:`model` (either ``'vgg16'`` (default) or ``'vgg19'``).

    When multiple layers are specified, features are extracted from all layers,
    and losses are computed at each layer. The losses from higher layers (with
    smaller spatial dimensions) are automatically upscaled to match the largest
    spatial dimensions (from the lowest layer index) before being averaged.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.

    Args:
        model: VGG model to use, either 'vgg16' (default) or 'vgg19'
        layers: Layer index or list of layer indices to extract features from.
                Default is 8 (relu2_2). Common layers for VGG16:
                    - 3: relu1_2
                    - 8: relu2_2
                    - 15: relu3_3
                    - 22: relu4_3
        reduction: Reduction mode: 'mean', 'sum', or 'none' (default 'mean')
    """

    models = {"vgg16": models.vgg16, "vgg19": models.vgg19}
    VALID_LAYERS = [3, 8, 15, 22]

    def __init__(self, model="vgg16", layers=8, reduction="mean"):
        super().__init__()

        # Handle both single layer and list of layers
        if isinstance(layers, int):
            layers = [layers]

        self.layers = sorted(layers)  # Sort to ensure lowest layer is first

        # Validate layers
        for layer in self.layers:
            if layer not in self.VALID_LAYERS:
                raise ValueError(f"Invalid layer: {layer}. Valid layers are: {self.VALID_LAYERS}")

        self.reduction = reduction

        # Build feature extractors for each layer
        vgg = self.models[model](pretrained=True)
        self.feature_extractors = nn.ModuleDict()

        for layer_idx in self.layers:
            self.feature_extractors[str(layer_idx)] = vgg.features[: layer_idx + 1]

        # Set all to eval mode and freeze parameters
        for extractor in self.feature_extractors.values():
            extractor.eval()
            extractor.requires_grad_(False)

        logger.info(
            "VGGLoss initialized with model: %s, layers: %s, reduction: %s",
            model,
            self.layers,
            reduction,
        )

    def get_features(self, input):
        """Extract features from the VGG model at all specified layers.

        Args:
            input: Input tensor of shape (B, 3, H, W) with values in [0, 1]

        Returns:
            If multiple layers: List of feature tensors, one per layer
        """
        features = []
        for layer_idx in self.layers:
            feat = self.feature_extractors[str(layer_idx)](input)
            features.append(feat)
        return features

    def _scale_to_target_shape(self, tensor, target_shape):
        """Scale tensor to target spatial shape using bilinear interpolation.

        Args:
            tensor: Tensor to scale (B, C, H, W)
            target_shape: Target spatial shape (H_target, W_target)

        Returns:
            Scaled tensor (B, C, H_target, W_target)
        """
        if tensor.shape == target_shape:
            return tensor

        return F.interpolate(tensor, size=target_shape[2:], mode="bilinear", align_corners=False)

    def forward(self, input, target):
        """Compute VGG perceptual loss.

        Args:
            input: Input tensor of shape (B, 3, H, W) with values in [0, 1]
            target: Target tensor of shape (B, 3, H, W) with values in [0, 1],
                   or precomputed features if target_is_features=True

        Returns:
            Perceptual loss value (scalar if reduction='mean' or 'sum')
        """
        sep = input.shape[0]
        batch = torch.cat([input, target])
        feats = self.get_features(batch)
        # Split features back into input and target
        input_feats = [f[:sep] for f in feats]
        target_feats = [f[sep:] for f in feats]

        # Compute loss
        # Multiple layers - compute loss at each layer and scale to largest spatial size
        # The lowest layer index has the largest spatial dimensions
        target_shape = input_feats[0].shape
        loss = None

        for inp_feat, tgt_feat in zip(input_feats, target_feats):
            # Compute per-element loss
            layer_loss = F.mse_loss(inp_feat, tgt_feat, reduction=self.reduction)
            if self.reduction == "none":
                layer_loss = layer_loss.mean(dim=1, keepdim=True).repeat(1, input.shape[1], 1, 1)

            if loss is None:
                loss = layer_loss
                continue

            # Scale to target shape if needed (spatial dimensions)
            if layer_loss.shape != target_shape:
                loss += self._scale_to_target_shape(layer_loss, target_shape)

        return loss


if __name__ == "__main__":
    loss = VGGLoss(model="vgg16", layers=[3, 8], reduction="none")
    input_data = torch.randn(2, 3, 640, 1024)
    target_data = torch.randn(2, 3, 640, 1024)
    loss_value = loss(input_data, target_data)
    print(loss_value.shape)
