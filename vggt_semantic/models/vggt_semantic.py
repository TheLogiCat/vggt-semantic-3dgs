"""
VGGTSemantic – Main model integrating Modules 1, 2, and 3
==========================================================

This model extends VGGT with a semantic branch:

    images [B, S, 3, H, W]
        │
        ├──► SemanticTokenizer (Module 1) ──► T_sem [B, S, N, 32]
        │                                          │
        └──► SemanticGuidedAggregator (Module 2) ◄─┘
                    │  aggregated_tokens_list
                    │
        ┌───────────┼────────────────────────────────────────┐
        ▼           ▼           ▼                            ▼
   CameraHead   DPTHead     DPTHead (pts)          SemanticHead (Module 3)
   pose_enc     depth        world_points           sem_feat [B,S,32,H,W]
                depth_conf   world_points_conf

Forward output dictionary keys
-------------------------------
Geometry keys (preserved from original VGGT):
    pose_enc          [B, S, 9]
    pose_enc_list     list of [B, S, 9]
    depth             [B, S, H, W, 1]
    depth_conf        [B, S, H, W]
    world_points      [B, S, H, W, 3]
    world_points_conf [B, S, H, W]
    track             [B, S, N, 2]   (only if query_points given)
    vis               [B, S, N]
    conf              [B, S, N]
    images            [B, S, 3, H, W]

Semantic keys (new):
    sem_feat          [B, S, 32, H, W]  (only when semantic.enabled=True)

Compatibility note
------------------
When `semantic.enabled=False`, VGGTSemantic uses the standard Aggregator and
produces no semantic output, making it behaviourally equivalent to VGGT.

Module-4 hook points
--------------------
After Module 4 (closed-loop 3DGS training) is implemented, the recommended
integration points are:
  1. `sem_feat`  → used as semantic Gaussian attribute initialiser.
  2. `T_sem`     → available via `self._last_T_sem` after each forward pass
                   for auxiliary supervision losses.
  3. `VGGTSemantic.aggregator` → swap `SemanticGuidedAggregator` for a fully
     differentiable version once DINOv2 is fine-tuned.
"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from vggt_semantic.config import SemanticConfig, VGGTSemanticConfig
from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer
from vggt_semantic.modules.semantic_head import SemanticHead

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry head imports – try real vggt first, fall back to stubs
# ---------------------------------------------------------------------------
try:
    from vggt.heads.camera_head import CameraHead
    from vggt.heads.dpt_head import DPTHead
    from vggt.heads.track_head import TrackHead
    _GEO_HEADS_AVAILABLE = True
except ImportError:
    _GEO_HEADS_AVAILABLE = False
    logger.warning(
        "vggt geometry heads not found – using stub heads for testing. "
        "Install vggt to enable full geometry prediction."
    )

if not _GEO_HEADS_AVAILABLE:
    # ---- Minimal stub geometry heads ----

    class _StubCameraHead(nn.Module):
        """Stub camera head that returns zero pose encodings."""

        def __init__(self, dim_in: int) -> None:
            super().__init__()
            self.proj = nn.Linear(dim_in, 9)

        def forward(self, aggregated_tokens_list, num_iterations: int = 1):
            tokens = aggregated_tokens_list[-1]       # [B, S, P, 2*C]
            pose = self.proj(tokens[:, :, 0])         # camera token → [B, S, 9]
            return [pose]

    class _StubDPTHead(nn.Module):
        """Stub DPT head that returns zero feature maps."""

        def __init__(self, dim_in: int, output_dim: int = 2) -> None:
            super().__init__()
            self.output_dim = output_dim
            self.proj = nn.Linear(dim_in, output_dim + 1)  # +1 conf

        def forward(self, aggregated_tokens_list, images: Tensor, patch_start_idx: int):
            B, S, _, H, W = images.shape
            # Return zeros at original resolution
            pred = torch.zeros(B, S, H, W, self.output_dim, device=images.device, dtype=images.dtype)
            conf = torch.ones(B, S, H, W, device=images.device, dtype=images.dtype)
            if self.output_dim == 1:
                return pred.squeeze(-1), conf
            return pred, conf

    class _StubTrackHead(nn.Module):
        def __init__(self, dim_in: int, patch_size: int = 14) -> None:
            super().__init__()

        def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points):
            B, S, N, _ = query_points.shape if len(query_points.shape) == 4 else (*query_points.shape[:2], query_points.shape[-2], 2)
            zeros = torch.zeros_like(query_points)
            vis  = torch.ones(B, S, N, device=query_points.device)
            conf = torch.ones(B, S, N, device=query_points.device)
            return [zeros], vis, conf

    CameraHead = _StubCameraHead   # type: ignore[assignment]
    DPTHead    = _StubDPTHead      # type: ignore[assignment]
    TrackHead  = _StubTrackHead    # type: ignore[assignment]


# ---------------------------------------------------------------------------
# VGGTSemantic
# ---------------------------------------------------------------------------

class VGGTSemantic(nn.Module):
    """
    VGGT with semantic extension (Modules 1, 2, 3).

    Args:
        img_size      (int):  Input image size (default 518).
        patch_size    (int):  Patch size (default 14).
        embed_dim     (int):  Transformer embedding dimension (default 1024).
        enable_camera (bool): Enable camera-pose head.
        enable_point  (bool): Enable 3D-point head.
        enable_depth  (bool): Enable depth head.
        enable_track  (bool): Enable point-tracking head.
        semantic      (SemanticConfig | None):
                       Semantic extension config.  Pass None to disable
                       the semantic branch entirely (= original VGGT).

    Example::

        from vggt_semantic import VGGTSemantic, SemanticConfig
        from vggt_semantic.config import SemanticGuidanceConfig

        model = VGGTSemantic(
            img_size=518,
            semantic=SemanticConfig(
                enabled=True,
                dim=32,
                backbone="placeholder",        # use "dinov2" in production
                guidance=SemanticGuidanceConfig(enabled=True),
            ),
        )
        images = torch.randn(1, 3, 3, 518, 518)
        out = model(images)
        print(out["sem_feat"].shape)   # [1, 3, 32, 518, 518]
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_depth: bool = True,
        enable_track: bool = False,
        semantic: Optional[SemanticConfig] = None,
    ) -> None:
        super().__init__()

        # ---- resolve semantic config ----
        if semantic is None:
            semantic = SemanticConfig(enabled=False)
        self.semantic_cfg = semantic
        sem_enabled = semantic.enabled
        sem_guidance = sem_enabled and semantic.guidance.enabled

        # ---- aggregator ----
        # Select between the semantically extended and the plain aggregator.
        if sem_guidance:
            from vggt_semantic.modules.aggregator import SemanticGuidedAggregator
            self.aggregator = SemanticGuidedAggregator(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                use_semantic_guidance=True,
                sem_dim=semantic.dim,
            )
        else:
            # Fall back to standard Aggregator (or stub if vggt not installed)
            try:
                from vggt.models.aggregator import Aggregator
                self.aggregator = Aggregator(
                    img_size=img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                )
            except ImportError:
                from vggt_semantic.modules.aggregator import SemanticGuidedAggregator
                self.aggregator = SemanticGuidedAggregator(
                    img_size=img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    use_semantic_guidance=False,
                )

        # head input dim = 2 * embed_dim (frame + global concatenated)
        dim_heads = 2 * embed_dim

        # ---- geometry heads (Module 3 – geometry track) ----
        self.camera_head = CameraHead(dim_in=dim_heads) if enable_camera else None
        self.point_head  = (
            DPTHead(dim_in=dim_heads, output_dim=4,
                    activation="inv_log", conf_activation="expp1")
            if enable_point else None
        )
        self.depth_head  = (
            DPTHead(dim_in=dim_heads, output_dim=2,
                    activation="exp", conf_activation="expp1")
            if enable_depth else None
        )
        self.track_head  = (
            TrackHead(dim_in=dim_heads, patch_size=patch_size)
            if enable_track else None
        )

        # ---- Module 1: semantic tokenizer ----
        self.sem_tokenizer = (
            SemanticTokenizer(
                sem_dim=semantic.dim,
                patch_size=patch_size,
                backbone=semantic.backbone,
            )
            if sem_enabled
            else None
        )

        # ---- Module 3: semantic head ----
        self.sem_head = (
            SemanticHead(dim_in=dim_heads, sem_dim=semantic.dim, patch_size=patch_size)
            if sem_enabled
            else None
        )

        # Internal store: accessible for external hooks (e.g. Module 4)
        self._last_T_sem: Optional[Tensor] = None

    # ------------------------------------------------------------------

    def forward(
        self,
        images: Tensor,
        query_points: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            images        : [S, 3, H, W] or [B, S, 3, H, W]  in range [0, 1].
            query_points  : [N, 2] or [B, N, 2] pixel coords for tracking.

        Returns:
            dict with keys described in the module docstring.
        """
        # ---- normalise batch dimension ----
        if images.ndim == 4:
            images = images.unsqueeze(0)        # [1, S, 3, H, W]
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)

        # ---- Module 1: semantic tokenizer ----
        T_sem: Optional[Tensor] = None
        if self.sem_tokenizer is not None:
            T_sem = self.sem_tokenizer(images)  # [B, S, N_patch, sem_dim]
            self._last_T_sem = T_sem            # store for Module 4 hooks

        # ---- Module 2: semantic-guided aggregation ----
        if isinstance(self.aggregator, SemanticGuidedAggregator):
            aggregated_tokens_list, patch_start_idx = self.aggregator(images, sem_tokens=T_sem)
        else:
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions: Dict[str, Any] = {}

        # ---- geometry heads (autocast disabled for precision) ----
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"]      = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"]      = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"]      = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"]   = vis
            predictions["conf"]  = conf

        # ---- Module 3: semantic head ----
        if self.sem_head is not None:
            sem_feat = self.sem_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["sem_feat"] = sem_feat  # [B, S, 32, H, W]

        if not self.training:
            predictions["images"] = images

        return predictions

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: VGGTSemanticConfig) -> "VGGTSemantic":
        """Construct from a VGGTSemanticConfig dataclass."""
        return cls(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=cfg.enable_camera,
            enable_point=cfg.enable_point,
            enable_depth=cfg.enable_depth,
            enable_track=cfg.enable_track,
            semantic=cfg.semantic,
        )

    def freeze_backbone(self) -> None:
        """Freeze the patch-embed backbone and semantic tokenizer for fine-tuning."""
        if hasattr(self.aggregator, "patch_embed"):
            for p in self.aggregator.patch_embed.parameters():
                p.requires_grad_(False)
        if self.sem_tokenizer is not None:
            for p in self.sem_tokenizer.backbone.parameters():
                p.requires_grad_(False)

    def semantic_parameters(self):
        """Iterator over parameters that belong to the semantic extension."""
        modules = []
        if self.sem_tokenizer is not None:
            modules.append(self.sem_tokenizer.proj)  # only the projection MLP
        if self.sem_head is not None:
            modules.append(self.sem_head)
        # semantic guidance scale/head-weight parameters inside aggregator blocks
        if hasattr(self.aggregator, "frame_blocks"):
            for blk in self.aggregator.frame_blocks:
                if hasattr(blk, "attn") and hasattr(blk.attn, "sem_log_scale"):
                    yield blk.attn.sem_log_scale
                    yield blk.attn.sem_head_weight
        for m in modules:
            yield from m.parameters()


# Convenience import alias
from vggt_semantic.modules.aggregator import SemanticGuidedAggregator  # noqa: E402
