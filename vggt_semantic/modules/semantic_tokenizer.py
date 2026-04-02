"""
Module 1 – SemanticTokenizer
============================

Extracts 32-dim semantic tokens from multi-view images and aligns them
spatially with the patch tokens produced by VGGT's RGB patch embedding.

Pipeline
--------
  images [B, S, 3, H, W]
       │
       ▼  (reshape to [B*S, 3, H, W], then backbone)
  raw_feats [B*S, N_patch, backbone_dim]   (e.g. 768 for DINOv2-ViT-L/14)
       │
       ▼  (MLP projection)
  T_sem [B*S, N_patch, sem_dim=32]
       │
       ▼  (reshape)
  T_sem [B, S, N_patch, 32]

Backbone options
----------------
* "dinov2"      – frozen DINOv2-ViT-L/14 loaded via the `transformers`
                  package. Requires an internet connection on the first run.
                  To replace with a different DINOv2 variant, edit
                  _DINOV2_MODEL_ID below.
* "placeholder" – lightweight two-conv stub that produces tokens of the
                  same spatial resolution without pretrained weights.
                  Suitable for unit tests and CI.

To swap in a custom backbone, implement the BackboneProvider protocol:

    class MyBackbone(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            # x: [B, 3, H, W]
            # returns: [B, N_patch, backbone_dim]
            ...

and pass it to SemanticTokenizer as `backbone_provider=MyBackbone(...)`.
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# DINOv2 model identifier (HuggingFace transformers)
_DINOV2_MODEL_ID = "facebook/dinov2-large"   # hidden dim: 1024 per patch token
_DINOV2_PATCH_DIM = 1024                      # ViT-L hidden dim


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

class PlaceholderBackbone(nn.Module):
    """
    Lightweight substitute for DINOv2 used in tests and offline environments.

    Uses a single strided convolution to produce one output vector per
    input patch, matching the spatial resolution of DINOv2.

    Args:
        out_dim  (int): Feature dimension of output tokens.
        patch_size (int): Stride equals patch_size so that spatial resolution
                          matches that of DINOv2 (H/patch_size × W/patch_size).
    """

    def __init__(self, out_dim: int = 1024, patch_size: int = 14) -> None:
        super().__init__()
        self.patch_size = patch_size
        # Lightweight conv backbone (no pretrained weights)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=1, bias=True),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] in range [0, 1]

        Returns:
            tokens: [B, N_patch, out_dim]
                    N_patch = (H // patch_size) * (W // patch_size)
        """
        # [B, out_dim, H/P, W/P]
        feats = self.stem(x)
        B, C, h, w = feats.shape
        # [B, N_patch, out_dim]
        tokens = feats.permute(0, 2, 3, 1).reshape(B, h * w, C)
        tokens = self.norm(tokens)
        return tokens


def _build_dinov2_backbone(patch_size: int) -> nn.Module:
    """
    Try to build a frozen DINOv2 backbone via HuggingFace `transformers`.

    Falls back to PlaceholderBackbone with a warning if the package or
    weights are unavailable.

    Returns
    -------
    backbone : nn.Module
        Module with signature forward(x: [B, 3, H, W]) -> [B, N_patch, dim]
    backbone_dim : int
        Feature dimension of the returned tokens.
    """
    try:
        from transformers import Dinov2Model  # type: ignore

        logger.info("Loading DINOv2 backbone from %s ...", _DINOV2_MODEL_ID)
        model = Dinov2Model.from_pretrained(_DINOV2_MODEL_ID)

        # Freeze all DINOv2 parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # Wrap into a thin adapter that extracts patch tokens only
        class _DINOv2Adapter(nn.Module):
            """Wraps HuggingFace Dinov2Model to return patch tokens."""

            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            @torch.no_grad()
            def forward(self, x: Tensor) -> Tensor:
                """
                Args:
                    x: [B, 3, H, W] float32 in [0, 1]

                Returns:
                    [B, N_patch, hidden_size]
                """
                outputs = self.inner(pixel_values=x, output_hidden_states=False)
                # last_hidden_state: [B, 1+N_reg+N_patch, hidden_size]
                # Skip the CLS token (index 0); keep only patch tokens.
                # DINOv2 with registers: layout = [CLS, reg_0..reg_k, patch_0..patch_n]
                patch_tokens = outputs.last_hidden_state[:, 1:, :]
                # Some DINOv2 variants prepend register tokens.  We keep them
                # out by computing N_patch from H/W at runtime in the tokenizer.
                return patch_tokens

        return _DINOv2Adapter(model), model.config.hidden_size

    except Exception as exc:
        logger.warning(
            "DINOv2 backbone unavailable (%s). Falling back to PlaceholderBackbone. "
            "To use real DINOv2, install transformers>=4.38 and ensure network access.",
            exc,
        )
        stub = PlaceholderBackbone(out_dim=_DINOV2_PATCH_DIM, patch_size=patch_size)
        return stub, _DINOV2_PATCH_DIM


# ---------------------------------------------------------------------------
# SemanticTokenizer (Module 1)
# ---------------------------------------------------------------------------

class SemanticTokenizer(nn.Module):
    """
    Module 1 – extracts 32-dim semantic tokens from multi-view images.

    The backbone (DINOv2 or placeholder) is always frozen; only the
    projection MLP is learnable.

    Args:
        sem_dim      (int): Output semantic feature dimension (default 32).
        patch_size   (int): Patch size, must match VGGT's patch embed (default 14).
        backbone     (str): "dinov2" | "placeholder"
        backbone_provider (nn.Module | None):
                          Supply a custom backbone module directly.  If given,
                          ``backbone`` is ignored.  The module must implement
                          forward(x: [B, 3, H, W]) -> [B, N_patch, backbone_dim].
        backbone_dim (int): Output dimension of `backbone_provider` (required
                          when `backbone_provider` is not None).
    """

    def __init__(
        self,
        sem_dim: int = 32,
        patch_size: int = 14,
        backbone: str = "dinov2",
        backbone_provider: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.sem_dim = sem_dim
        self.patch_size = patch_size

        # ---- backbone ----
        if backbone_provider is not None:
            # User-supplied custom backbone
            if backbone_dim is None:
                raise ValueError(
                    "backbone_dim must be specified when backbone_provider is given."
                )
            self.backbone = backbone_provider
            _backbone_dim = backbone_dim
            # Freeze the custom backbone to match DINOv2 frozen convention
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif backbone == "placeholder":
            self.backbone = PlaceholderBackbone(
                out_dim=_DINOV2_PATCH_DIM, patch_size=patch_size
            )
            _backbone_dim = _DINOV2_PATCH_DIM
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif backbone == "dinov2":
            self.backbone, _backbone_dim = _build_dinov2_backbone(patch_size)
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Choose 'dinov2' or 'placeholder'."
            )

        # ---- projection MLP: backbone_dim → sem_dim ----
        # Two-layer MLP with a hidden layer and LayerNorm for stability.
        hidden = max(sem_dim * 4, 128)
        self.proj = nn.Sequential(
            nn.LayerNorm(_backbone_dim),       # normalise backbone features
            nn.Linear(_backbone_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, sem_dim),
            nn.LayerNorm(sem_dim),             # output norm for stable fusion
        )

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: [B, S, 3, H, W] float32 in range [0, 1]
                    B = batch, S = views/frames, H/W = image resolution.

        Returns:
            T_sem: [B, S, N_patch, sem_dim]
                   N_patch = (H // patch_size) * (W // patch_size)

        Notes
        -----
        DINOv2 may output extra CLS / register tokens.  This method
        truncates the output to exactly N_patch spatial tokens so that
        T_sem aligns pixel-to-pixel with VGGT's RGB patch tokens.
        """
        B, S, C, H, W = images.shape

        N_patch = (H // self.patch_size) * (W // self.patch_size)

        # Reshape to [B*S, 3, H, W] for the backbone
        imgs_flat = images.reshape(B * S, C, H, W)

        # Extract features: [B*S, N_tokens, backbone_dim]
        # N_tokens may be > N_patch (CLS / register tokens prepended)
        with torch.no_grad():
            raw = self.backbone(imgs_flat)

        # Keep only the last N_patch tokens (spatial patch tokens)
        raw = raw[:, -N_patch:, :]          # [B*S, N_patch, backbone_dim]

        # Project to sem_dim: [B*S, N_patch, sem_dim]
        T_sem = self.proj(raw)

        # Reshape back: [B, S, N_patch, sem_dim]
        T_sem = T_sem.reshape(B, S, N_patch, self.sem_dim)
        return T_sem
