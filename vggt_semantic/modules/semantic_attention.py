"""
Module 2 – SemanticGuidedAttention & SemanticGuidedBlock
=========================================================

Injects per-patch semantic information into the standard scaled-dot-product
attention used by VGGT's transformer blocks.

Mechanism
---------
Standard attention (logits-level view):

    attn_logits = Q K^T / sqrt(d_head)            # [B, H, N, N]
    A           = softmax(attn_logits) @ V

Semantic modulation (additive, logits level):

    sem_norm     = L2-normalise(T_sem)             # [B, N_patch, sem_dim]
    M_sem        = sem_norm @ sem_norm^T           # [B, N_patch, N_patch]  cosine sim ∈ [-1, 1]
    bias         = alpha * M_sem                   # alpha is a learnable scalar (init 0)
    # Pad M_sem to full sequence length (including special tokens)
    attn_logits' = attn_logits + bias              # [B, H, N, N]  (broadcast over heads)
    A'           = softmax(attn_logits') @ V

Why additive at logits level?
    • Numerically stable – the subsequent softmax renormalises.
    • Fully differentiable.
    • Easily disabled: set alpha → 0 (or use_semantic_guidance=False).

Backward-compatibility
----------------------
When `use_semantic_guidance=False` OR `sem_tokens=None`, the module
degrades exactly to the original `Attention` behaviour.

Shape conventions used in comments
-----------------------------------
B   – batch size (may be B*S for frame attention, or B for global attention)
N   – total token count = N_special + N_patch
N_p – N_patch (spatial patch tokens only)
H   – number of attention heads
d   – head dimension
"""

import math
import logging
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the original VGGT Attention / Block classes.
# If vggt is not installed we fall back to minimal reimplementations so that
# this module can be imported and tested in isolation.
# ---------------------------------------------------------------------------
try:
    from vggt.layers.attention import Attention as _VGGTAttention
    from vggt.layers.block import Block as _VGGTBlock
    _VGGT_AVAILABLE = True
except ImportError:
    logger.warning(
        "vggt package not found – using built-in Attention/Block fallbacks. "
        "Install vggt (see requirements.txt) for full functionality."
    )
    _VGGT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback minimal Attention (mirrors vggt.layers.attention.Attention exactly)
# ---------------------------------------------------------------------------
if not _VGGT_AVAILABLE:

    class _VGGTAttention(nn.Module):  # type: ignore[no-redef]
        """Minimal Attention replicating vggt.layers.attention.Attention."""

        def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: type = nn.LayerNorm,
            qk_norm: bool = False,
            fused_attn: bool = True,
            rope=None,
        ) -> None:
            super().__init__()
            assert dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.fused_attn = fused_attn
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim, bias=proj_bias)
            self.proj_drop = nn.Dropout(proj_drop)
            self.rope = rope

        def forward(self, x: Tensor, pos=None) -> Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


# ---------------------------------------------------------------------------
# Fallback minimal Block
# ---------------------------------------------------------------------------
if not _VGGT_AVAILABLE:

    class _VGGTBlock(nn.Module):  # type: ignore[no-redef]
        """Minimal Block replicating vggt.layers.block.Block."""

        def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = None,
            ffn_layer: Callable[..., nn.Module] = None,
            qk_norm: bool = False,
            fused_attn: bool = True,
            rope=None,
        ) -> None:
            super().__init__()
            if attn_class is None:
                attn_class = _VGGTAttention
            if ffn_layer is None:
                ffn_layer = _FallbackMlp
            self.norm1 = norm_layer(dim)
            self.attn = attn_class(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm,
                fused_attn=fused_attn, rope=rope,
            )
            self.norm2 = norm_layer(dim)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = ffn_layer(
                in_features=dim, hidden_features=mlp_hidden,
                act_layer=act_layer, drop=drop, bias=ffn_bias,
            )
            # layer-scale (identity when init_values is None)
            if init_values:
                self.ls1 = _LayerScale(dim, init_values)
                self.ls2 = _LayerScale(dim, init_values)
            else:
                self.ls1 = nn.Identity()
                self.ls2 = nn.Identity()
            self.sample_drop_ratio = drop_path

        def forward(self, x: Tensor, pos=None) -> Tensor:
            x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x


    class _FallbackMlp(nn.Module):
        def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0, bias=True):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            return self.drop(self.fc2(self.act(self.fc1(x))))


    class _LayerScale(nn.Module):
        def __init__(self, dim, init_values=1e-5):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(dim) * init_values)

        def forward(self, x):
            return self.gamma * x


# ---------------------------------------------------------------------------
# SemanticGuidedAttention (Module 2 core)
# ---------------------------------------------------------------------------

class SemanticGuidedAttention(_VGGTAttention):
    """
    Drop-in replacement for vggt.layers.attention.Attention that optionally
    modulates attention logits with a semantic patch-similarity bias.

    Extra constructor args (beyond Attention)
    -----------------------------------------
    sem_dim               : int  – dimension of semantic tokens (default 32)
    use_semantic_guidance : bool – master switch (default True)

    Extra forward args (beyond Attention)
    --------------------------------------
    sem_tokens : Tensor | None
        Shape [B, N_patch, sem_dim].
        When None (or use_semantic_guidance=False), the module behaves
        exactly like the base Attention.
    """

    def __init__(
        self,
        dim: int,
        *,
        sem_dim: int = 32,
        use_semantic_guidance: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(dim, **kwargs)
        self.use_semantic_guidance = use_semantic_guidance
        self.sem_dim = sem_dim

        if use_semantic_guidance:
            # Learnable log-scale for semantic bias; alpha = exp(sem_log_scale).
            # Initialised at -4 so alpha ≈ 0.018 at the start of training,
            # keeping the semantic modulation negligible until it is learned.
            self.sem_log_scale = nn.Parameter(torch.full((1,), -4.0))  # init ≈ 0
            # Per-head weighting: lets different heads focus differently on semantics
            self.sem_head_weight = nn.Parameter(torch.zeros(self.num_heads, 1, 1))

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        pos=None,
        sem_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x          : [B, N, C]  input tokens (N = N_special + N_patch)
            pos        : positional encoding for RoPE (optional)
            sem_tokens : [B, N_patch, sem_dim] semantic tokens (optional)

        Returns:
            out : [B, N, C]
        """
        B, N, C = x.shape

        # ---- standard QKV projection ----
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        # q, k, v: [B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # ---- decide whether to use semantic guidance ----
        use_sem = (
            self.use_semantic_guidance
            and sem_tokens is not None
        )
        if not use_sem:
            # Fast path: fused SDPA (same as original Attention)
            if self.fused_attn:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out = attn @ v
        else:
            # ---- slow path: explicit logits + semantic bias ----
            # attn_logits: [B, num_heads, N, N]
            attn_logits = (q * self.scale) @ k.transpose(-2, -1)

            # ---- semantic patch-similarity matrix ----
            # sem_tokens: [B, N_patch, sem_dim]
            N_patch = sem_tokens.shape[1]
            N_special = N - N_patch   # camera / register tokens

            # L2-normalise → cosine similarity ∈ [-1, 1]
            sem_norm = F.normalize(sem_tokens.float(), dim=-1)  # [B, N_p, sem_dim]

            # M_sem: [B, N_patch, N_patch]
            M_sem = sem_norm @ sem_norm.transpose(-2, -1)

            # Learnable scale (always positive via exp)
            alpha = self.sem_log_scale.exp()  # scalar

            # Semantic bias: [B, N_patch, N_patch]
            sem_bias = alpha * M_sem

            # Expand to full sequence (pad special-token rows/cols with zeros
            # so special tokens are not affected by semantic modulation)
            if N_special > 0:
                sem_bias_full = torch.zeros(
                    B, N, N,
                    device=attn_logits.device,
                    dtype=attn_logits.dtype,
                )
                # Only the patch-to-patch sub-block gets the semantic bias
                sem_bias_full[:, N_special:, N_special:] = sem_bias.to(attn_logits.dtype)
            else:
                sem_bias_full = sem_bias.to(attn_logits.dtype)  # [B, N, N]

            # Broadcast over heads with per-head weighting
            # sem_head_weight: [num_heads, 1, 1] → scale importance per head
            head_w = self.sem_head_weight.sigmoid()          # [H, 1, 1] ∈ (0, 1)
            sem_bias_full = sem_bias_full.unsqueeze(1) * head_w   # [B, H, N, N]

            # Additive modulation at logits level (before softmax)
            attn_logits = attn_logits + sem_bias_full

            attn = attn_logits.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        # ---- output projection ----
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# SemanticGuidedBlock (Module 2 wrapper)
# ---------------------------------------------------------------------------

class SemanticGuidedBlock(_VGGTBlock):
    """
    Transformer block that passes optional semantic tokens down to
    SemanticGuidedAttention.

    Constructed identically to Block; the only behavioural difference is:
    * The internal Attention is replaced by SemanticGuidedAttention.
    * forward() accepts an extra keyword argument `sem_tokens`.

    When `sem_tokens=None` or `use_semantic_guidance=False`, this block is
    numerically identical to the original Block.

    Constructor args
    ----------------
    sem_dim               (int)  – semantic token dimension (default 32)
    use_semantic_guidance (bool) – master switch (default True)
    All other args forwarded to Block.__init__.
    """

    def __init__(
        self,
        dim: int,
        *,
        sem_dim: int = 32,
        use_semantic_guidance: bool = True,
        **kwargs,
    ) -> None:
        # Remove any user-supplied attn_class so we can inject our own
        kwargs.pop("attn_class", None)

        sem_attn_class = partial(
            SemanticGuidedAttention,
            sem_dim=sem_dim,
            use_semantic_guidance=use_semantic_guidance,
        )
        super().__init__(dim, attn_class=sem_attn_class, **kwargs)
        self.use_semantic_guidance = use_semantic_guidance

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, pos=None, sem_tokens: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x          : [B, N, C]
            pos        : RoPE positional encoding (optional)
            sem_tokens : [B, N_patch, sem_dim] (optional)

        Returns:
            x : [B, N, C]
        """
        # Residual with semantic-guided attention
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, sem_tokens=sem_tokens))
        # Standard FFN residual (semantic information already baked into tokens)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
