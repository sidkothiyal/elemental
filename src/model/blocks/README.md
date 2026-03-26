# src/model/blocks — Building Blocks

Reusable convolutional and linear building blocks for constructing model architectures.

## Files

### `conv_block.py` — ConvBlock

Conv2d + optional normalization (BatchNorm, GroupNorm, etc.) + activation (ReLU, GELU, etc.).

```python
ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
          norm=nn.BatchNorm2d, activation=nn.ReLU)
```

### `convtranspose_block.py` — ConvTransposeBlock

ConvTranspose2d + optional normalization + activation. Used in decoder / upsampling paths.

### `resnet_block.py` — ResnetBlock

Residual block: two ConvBlocks with identity shortcut. Optional 1x1 projection when input/output channels differ.

### `linear_block.py` — LinearBlock

Linear + optional normalization + activation. Used in MLP layers.

### `upsample_block.py` — UpsampleBlock

`nn.Upsample(scale_factor=2)` followed by a ConvBlock. Cleaner alternative to ConvTranspose2d for avoiding checkerboard artifacts.

### `conditional_conv_block.py` — ConditionalConvBlock

ConvBlock extended for diffusion models. Adds:
- **Time embedding injection**: MLP processes time embedding `t` → adds to feature map
- **Conditioning injection**: condition embedding `cond` injected via FiLM (feature-wise linear modulation) or cross-attention
- Optional cross-attention module

```python
ConditionalConvBlock(in_channels=128, out_channels=128,
                     time_embed_dim=256, cond_embed_dim=512,
                     use_cross_attn=True, cross_attn_num_heads=8)
```
