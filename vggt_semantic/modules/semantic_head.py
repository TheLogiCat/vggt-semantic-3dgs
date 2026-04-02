"""
Module 3 – SemanticHead
========================

A lightweight decoder head that converts deep transformer tokens into a
dense 32-dim semantic feature map.

Architecture overview
---------------------

                        aggregated_tokens_list
                               │
                   ┌───────────┴──────────────┐
                   │ select intermediate layers│  (e.g. layers 17 & 23)
                   └──────────────────────────┘
                               │  List of [B, S, P, 2*embed_dim]
                               ▼
               patch tokens only  [B*S, N_patch, 2*embed_dim]
                               │
                            LayerNorm
                               │
                         [B*S, N_patch, dim_in]
                               │
              reshape to spatial grid  [B*S, dim_in, h_p, w_p]
                               │
                          Conv2d 1×1   →  [B*S, feature_dim, h_p, w_p]
                               │
                   UpSample (ConvTranspose2d 4×)  →  [B*S, feature_dim, h_p*4, w_p*4]
                               │
                        GELU activation
                               │
                   Conv2d 1×1  →  [B*S, sem_dim, ...]
                               │
             Bilinear interpolate to (H, W)  →  [B*S, sem_dim, H, W]
                               │
                        GroupNorm (spatial normalisation)
                               │
                   reshape  →  [B, S, sem_dim, H, W]

Notes
-----
* `dim_in` equals `2 * embed_dim` because VGGT concatenates frame-attention
  and global-attention token outputs before the heads.
* The head uses two intermediate layers for a mild multi-scale fusion.
  `intermediate_layer_idx` can be adjusted to match available depth.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class SemanticHead(nn.Module):
    """
    Lightweight semantic decoder head.

    Args:
        dim_in       (int): Token feature dimension (= 2*embed_dim after
                           concatenation in VGGT aggregator, e.g. 2048).
        sem_dim      (int): Output semantic feature channels (default 32).
        patch_size   (int): Spatial patch size of the token grid (default 14).
        features     (int): Intermediate channel count in the decoder (default 128).
        intermediate_layer_idx (list[int]):
                           Which aggregator output layers to use.
                           Defaults to the last two layers [-2, -1].
                           When a single index is given, multi-scale fusion
                           is skipped.
    """

    def __init__(
        self,
        dim_in: int,
        sem_dim: int = 32,
        patch_size: int = 14,
        features: int = 128,
        intermediate_layer_idx: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.sem_dim = sem_dim
        self.patch_size = patch_size

        # Default: use the last two aggregator outputs for a light multi-scale view
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [-2, -1]
        self.intermediate_layer_idx = intermediate_layer_idx

        num_layers = len(intermediate_layer_idx)

        # ---- per-layer norm + projection ----
        # Each selected layer contributes dim_in features.
        # We first normalise and project each layer independently, then fuse.
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(dim_in) for _ in range(num_layers)]
        )
        # Project each layer to `features` channels
        self.layer_projs = nn.ModuleList(
            [nn.Conv2d(dim_in, features, kernel_size=1) for _ in range(num_layers)]
        )

        # ---- fusion (sum over layers, or single-layer pass-through) ----
        # After element-wise addition of projected features, we apply a 3×3 conv
        fused_channels = features
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, features, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )

        # ---- upsampler ----
        # Two-stage upsampling to reach the original image resolution.
        # Stage 1: 4× ConvTranspose (patch_size/4 → partial res)
        # Stage 2: bilinear to exact (H, W) in forward()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.Conv2d(features, sem_dim, kernel_size=1),
        )

        # ---- output normalisation ----
        # GroupNorm with 8 groups keeps the channel dimension interpretable.
        num_groups = min(8, sem_dim)  # handle small sem_dim gracefully
        self.out_norm = nn.GroupNorm(num_groups, sem_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        aggregated_tokens_list: List[Tensor],
        images: Tensor,
        patch_start_idx: int,
    ) -> Tensor:
        """
        Decode transformer tokens into a dense semantic feature map.

        Args:
            aggregated_tokens_list : list[Tensor]
                Each element has shape [B, S, P_total, dim_in] where
                P_total = N_special + N_patch.
            images : Tensor
                Input images [B, S, 3, H, W] – used only to determine
                target spatial resolution (H, W).
            patch_start_idx : int
                Index from which patch tokens begin in the token sequence
                (equals camera + register count).

        Returns:
            F_sem_hat : Tensor  [B, S, sem_dim, H, W]
                Dense semantic feature map, one per view.
        """
        B, S, _, H, W = images.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # ---- extract & project each selected layer ----
        projected: List[Tensor] = []

        for idx_i, layer_idx in enumerate(self.intermediate_layer_idx):
            # tokens: [B, S, P_total, dim_in]
            tokens = aggregated_tokens_list[layer_idx]

            # Keep only patch tokens: [B, S, N_patch, dim_in]
            patch_tokens = tokens[:, :, patch_start_idx:, :]

            # Flatten batch × views: [B*S, N_patch, dim_in]
            x = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])

            # LayerNorm along feature axis
            x = self.layer_norms[idx_i](x)

            # Rearrange to spatial feature map: [B*S, dim_in, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape(B * S, -1, patch_h, patch_w)

            # 1×1 projection: [B*S, features, patch_h, patch_w]
            x = self.layer_projs[idx_i](x)
            projected.append(x)

        # ---- fuse layers (element-wise sum) ----
        # [B*S, features, patch_h, patch_w]
        fused = projected[0]
        for extra in projected[1:]:
            fused = fused + extra
        fused = self.fuse_conv(fused)

        # ---- upsample ----
        # After 4× ConvTranspose: [B*S, sem_dim, patch_h*4, patch_w*4]
        up = self.upsample(fused)

        # Bilinear to exact target resolution [B*S, sem_dim, H, W]
        if up.shape[-2:] != (H, W):
            up = F.interpolate(up, size=(H, W), mode="bilinear", align_corners=False)

        # ---- output norm ----
        out = self.out_norm(up)  # [B*S, sem_dim, H, W]

        # Reshape to [B, S, sem_dim, H, W]
        F_sem_hat = out.reshape(B, S, self.sem_dim, H, W)
        return F_sem_hat
