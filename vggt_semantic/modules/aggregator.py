"""
SemanticGuidedAggregator
========================

Extends VGGT's Aggregator to thread semantic tokens (from Module 1)
into each frame-attention block (Module 2).

Key design choices
------------------
1. We subclass `Aggregator` and override only `_process_frame_attention`.
2. Global-attention blocks receive NO semantic guidance – across-frame
   geometric consistency is the goal there and semantic masking could harm
   multi-view reconstruction.  (This is a toggle: set
   `guide_global_attn=True` to enable it for ablations.)
3. Semantic tokens are stored as `_sem_tokens` before the forward pass and
   cleared afterwards; this avoids changing the Aggregator forward signature
   while keeping the code thread-safe for sequential inference.
4. When semantic guidance is disabled or `sem_tokens=None`, the aggregator
   behaves identically to the original.

Frame-attention semantic token shape
--------------------------------------
    self._sem_tokens : [B, S, N_patch, sem_dim]
    After reshape for frame attention (B*S frames):
        sem_tokens_bs : [B*S, N_patch, sem_dim]
"""

import logging
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the real Aggregator; fall back to a stub for testing
# ---------------------------------------------------------------------------
try:
    from vggt.models.aggregator import Aggregator as _VGGTAggregator
    _VGGT_AVAILABLE = True
except ImportError:
    logger.warning(
        "vggt package not found – using built-in Aggregator fallback. "
        "Install vggt (see requirements.txt) for full functionality."
    )
    _VGGT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal Aggregator fallback (when vggt not installed)
# ---------------------------------------------------------------------------
if not _VGGT_AVAILABLE:
    from vggt_semantic.modules.semantic_attention import SemanticGuidedBlock, _VGGTBlock

    class _FallbackPatchEmbed(nn.Module):
        """Minimal strided-conv patch embed (mirrors PatchEmbed from vggt)."""

        def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=64):
            super().__init__()
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: Tensor) -> Tensor:
            x = self.proj(x)                                # [B, C, h, w]
            B, C, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, h * w, C) # [B, N, C]
            x = self.norm(x)
            return x

    _RESNET_MEAN = [0.485, 0.456, 0.406]
    _RESNET_STD  = [0.229, 0.224, 0.225]

    def _slice_expand_flatten(t, B, S):
        q = t[:, 0:1].expand(B, 1, *t.shape[2:])
        o = t[:, 1: ].expand(B, S - 1, *t.shape[2:])
        return torch.cat([q, o], dim=1).reshape(B * S, *t.shape[2:])

    class _VGGTAggregator(nn.Module):  # type: ignore[no-redef]
        """
        Minimal Aggregator replicating vggt.models.aggregator.Aggregator for
        use when the vggt package is not installed.
        """

        def __init__(
            self,
            img_size=224,
            patch_size=14,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            num_register_tokens=2,
            block_fn=None,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            patch_embed="conv",
            aa_order=None,
            aa_block_size=1,
            qk_norm=False,
            rope_freq=-1,
            init_values=0.01,
        ):
            super().__init__()
            if block_fn is None:
                block_fn = _VGGTBlock
            if aa_order is None:
                aa_order = ["frame", "global"]

            self.patch_embed = _FallbackPatchEmbed(img_size, patch_size, 3, embed_dim)
            self.patch_size  = patch_size
            self.depth       = depth
            self.aa_order    = aa_order
            self.aa_block_size = aa_block_size
            self.aa_block_num  = depth // aa_block_size

            make = lambda: block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                init_values=init_values, qk_norm=qk_norm, rope=None,
            )
            self.frame_blocks  = nn.ModuleList([make() for _ in range(depth)])
            self.global_blocks = nn.ModuleList([make() for _ in range(depth)])

            self.camera_token   = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
            self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
            self.patch_start_idx = 1 + num_register_tokens

            nn.init.normal_(self.camera_token,   std=1e-6)
            nn.init.normal_(self.register_token, std=1e-6)

            mean = torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1)
            std  = torch.FloatTensor(_RESNET_STD ).view(1, 1, 3, 1, 1)
            self.register_buffer("_resnet_mean", mean, persistent=False)
            self.register_buffer("_resnet_std",  std,  persistent=False)
            self.use_reentrant = False
            self.rope = None

        def forward(self, images: Tensor):
            B, S, C_in, H, W = images.shape
            images = (images - self._resnet_mean) / self._resnet_std
            imgs_flat = images.view(B * S, C_in, H, W)
            patch_tokens = self.patch_embed(imgs_flat)          # [B*S, N_p, C]
            _, P_patch, C = patch_tokens.shape

            cam  = _slice_expand_flatten(self.camera_token,   B, S)
            reg  = _slice_expand_flatten(self.register_token, B, S)
            tokens = torch.cat([cam, reg, patch_tokens], dim=1) # [B*S, P, C]
            _, P, C = tokens.shape

            fi, gi = 0, 0
            output_list = []
            for _ in range(self.aa_block_num):
                for atype in self.aa_order:
                    if atype == "frame":
                        tokens, fi, f_inter = self._process_frame_attention(tokens, B, S, P, C, fi)
                    else:
                        tokens, gi, g_inter = self._process_global_attention(tokens, B, S, P, C, gi)
                for i in range(len(f_inter)):
                    output_list.append(
                        torch.cat([f_inter[i], g_inter[i]], dim=-1)
                    )
            return output_list, self.patch_start_idx

        def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
            if tokens.shape != (B * S, P, C):
                tokens = tokens.view(B, S, P, C).view(B * S, P, C)
            intermediates = []
            for _ in range(self.aa_block_size):
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
                frame_idx += 1
                intermediates.append(tokens.view(B, S, P, C))
            return tokens, frame_idx, intermediates

        def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
            if tokens.shape != (B, S * P, C):
                tokens = tokens.view(B, S * P, C)
            intermediates = []
            for _ in range(self.aa_block_size):
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
                global_idx += 1
                intermediates.append(tokens.view(B, S, P, C))
            return tokens, global_idx, intermediates


# ---------------------------------------------------------------------------
# SemanticGuidedAggregator
# ---------------------------------------------------------------------------

class SemanticGuidedAggregator(_VGGTAggregator):
    """
    Aggregator extended with Module-2 semantic guidance.

    Replaces the `block_fn` passed to the parent with `SemanticGuidedBlock`
    (via functools.partial) and overrides `_process_frame_attention` to
    forward semantic tokens into each block.

    Constructor args (beyond Aggregator)
    -------------------------------------
    use_semantic_guidance : bool – master switch (default True)
    guide_global_attn     : bool – also guide global-attention blocks
                                   (default False; see module docstring)
    sem_dim               : int  – semantic token dimension (default 32)

    All other args are forwarded to Aggregator.__init__.
    """

    def __init__(
        self,
        *args,
        use_semantic_guidance: bool = True,
        guide_global_attn: bool = False,
        sem_dim: int = 32,
        **kwargs,
    ) -> None:
        self.use_semantic_guidance = use_semantic_guidance
        self.guide_global_attn     = guide_global_attn
        self.sem_dim               = sem_dim

        # Inject SemanticGuidedBlock as the block factory when guidance is on
        if use_semantic_guidance:
            from vggt_semantic.modules.semantic_attention import SemanticGuidedBlock
            sem_block = partial(
                SemanticGuidedBlock,
                sem_dim=sem_dim,
                use_semantic_guidance=True,
            )
            kwargs["block_fn"] = sem_block

        super().__init__(*args, **kwargs)

        # Storage for semantic tokens; set by VGGTSemantic before each forward
        self._sem_tokens: Optional[Tensor] = None

    # ------------------------------------------------------------------

    def forward(self, images: Tensor, sem_tokens: Optional[Tensor] = None) -> Tuple[List[Tensor], int]:
        """
        Args:
            images     : [B, S, 3, H, W]
            sem_tokens : [B, S, N_patch, sem_dim] from SemanticTokenizer
                         (or None to disable semantic guidance for this call)

        Returns:
            Same as Aggregator.forward: (aggregated_tokens_list, patch_start_idx)
        """
        # Stash semantic tokens for use in overridden _process_* methods
        self._sem_tokens = sem_tokens
        try:
            result = super().forward(images)
        finally:
            # Always clear to avoid stale state on exception
            self._sem_tokens = None
        return result

    # ------------------------------------------------------------------

    def _process_frame_attention(
        self,
        tokens: Tensor,
        B: int, S: int, P: int, C: int,
        frame_idx: int,
        pos=None,
    ) -> Tuple[Tensor, int, List[Tensor]]:
        """
        Override: pass per-frame semantic tokens into each SemanticGuidedBlock.

        Shapes
        ------
        tokens       : [B*S, P, C] (frame-level attention layout)
        sem_tokens_bs: [B*S, N_patch, sem_dim]  (if available)
        """
        from vggt_semantic.modules.semantic_attention import SemanticGuidedBlock

        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        # Prepare semantic tokens for this frame batch
        sem_tokens_bs: Optional[Tensor] = None
        if self.use_semantic_guidance and self._sem_tokens is not None:
            # self._sem_tokens: [B, S, N_patch, sem_dim]
            # Reshape to [B*S, N_patch, sem_dim]
            sem_tokens_bs = self._sem_tokens.reshape(
                B * S, -1, self._sem_tokens.shape[-1]
            )

        intermediates: List[Tensor] = []

        for _ in range(self.aa_block_size):
            blk = self.frame_blocks[frame_idx]
            is_sem_block = self.use_semantic_guidance and isinstance(blk, SemanticGuidedBlock)

            if is_sem_block:
                # Semantic-guided path: pass sem_tokens as positional arg
                # (compatible with torch.utils.checkpoint which passes *args positionally)
                if self.training:
                    # checkpoint(fn, *args) → fn(*args): blk(tokens, pos, sem_tokens_bs)
                    tokens = checkpoint(
                        blk,
                        tokens,
                        pos,
                        sem_tokens_bs,
                        use_reentrant=self.use_reentrant,
                    )
                else:
                    tokens = blk(tokens, pos=pos, sem_tokens=sem_tokens_bs)
            else:
                # Standard path (original Block or guidance disabled)
                if self.training:
                    tokens = checkpoint(blk, tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = blk(tokens, pos=pos)

            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    # ------------------------------------------------------------------

    def _process_global_attention(
        self,
        tokens: Tensor,
        B: int, S: int, P: int, C: int,
        global_idx: int,
        pos=None,
    ) -> Tuple[Tensor, int, List[Tensor]]:
        """
        Override: optionally pass global-level semantic tokens.

        Global attention operates on [B, S*P, C] (all frames concatenated).
        Semantic tokens are reshaped accordingly when guide_global_attn=True.
        """
        from vggt_semantic.modules.semantic_attention import SemanticGuidedBlock

        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S * P, 2)

        # Semantic tokens for global attention: [B, S*N_patch, sem_dim]
        sem_tokens_global: Optional[Tensor] = None
        if self.guide_global_attn and self._sem_tokens is not None:
            # [B, S, N_patch, sem_dim] → [B, S*N_patch, sem_dim]
            sem_tokens_global = self._sem_tokens.reshape(
                B, S * self._sem_tokens.shape[2], self._sem_tokens.shape[3]
            )

        intermediates: List[Tensor] = []

        for _ in range(self.aa_block_size):
            blk = self.global_blocks[global_idx]
            is_sem_block = self.guide_global_attn and isinstance(blk, SemanticGuidedBlock)

            if is_sem_block:
                if self.training:
                    tokens = checkpoint(
                        blk,
                        tokens,
                        pos,
                        sem_tokens_global,
                        use_reentrant=self.use_reentrant,
                    )
                else:
                    tokens = blk(tokens, pos=pos, sem_tokens=sem_tokens_global)
            else:
                if self.training:
                    tokens = checkpoint(blk, tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = blk(tokens, pos=pos)

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


