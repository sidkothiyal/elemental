"""
ViT -> Vision Transformer
References: https://github.com/lucidrains/vit-pytorch
Paper: https://openreview.net/pdf?id=YicbFdNTTy


Note from paper to clarify:
(Section 3.1) The classification head is implemented by a MLP with one hidden layer at pre-training
time and by a single linear layer at fine-tuning time.
"""

import einops
from torch import nn

class ViT(nn.Module):
    def __init__(self, patch_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
