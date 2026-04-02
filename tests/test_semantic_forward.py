"""
Minimal forward-pass test for vggt_semantic (Modules 1 / 2 / 3).

Run with:
    pytest tests/test_semantic_forward.py -v

All tests use tiny model dimensions (embed_dim=64, depth=2, img_size=56)
and a PlaceholderBackbone so no pretrained weights or internet are required.
"""

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
# Tiny-model hyperparameters for fast CPU tests
_IMG_SIZE = 56       # 56 / 14 = 4 patches per side  → N_patch = 16
_PATCH_SIZE = 14
_EMBED_DIM = 64      # small, matches stub heads
_SEM_DIM = 32        # must stay 32 per spec
_B = 1               # batch
_S = 2               # views


@pytest.fixture(scope="module")
def tiny_images():
    """Random images [B, S, 3, H, W] in [0, 1]."""
    torch.manual_seed(0)
    return torch.rand(_B, _S, 3, _IMG_SIZE, _IMG_SIZE)


# ---------------------------------------------------------------------------
# Module 1 – SemanticTokenizer
# ---------------------------------------------------------------------------

class TestSemanticTokenizer:
    """Tests for SemanticTokenizer (Module 1)."""

    def test_output_shape_placeholder(self, tiny_images):
        from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer

        tok = SemanticTokenizer(
            sem_dim=_SEM_DIM,
            patch_size=_PATCH_SIZE,
            backbone="placeholder",
        ).eval()

        with torch.no_grad():
            T_sem = tok(tiny_images)

        N_patch = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 4*4 = 16
        assert T_sem.shape == (_B, _S, N_patch, _SEM_DIM), (
            f"Expected T_sem shape {(_B, _S, N_patch, _SEM_DIM)}, got {T_sem.shape}"
        )

    def test_sem_dim_is_32(self, tiny_images):
        """Semantic dimension must be exactly 32 per spec."""
        from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer

        tok = SemanticTokenizer(sem_dim=32, patch_size=_PATCH_SIZE, backbone="placeholder").eval()
        with torch.no_grad():
            T_sem = tok(tiny_images)
        assert T_sem.shape[-1] == 32, "Last dimension must be exactly 32"

    def test_no_grad_in_backbone(self, tiny_images):
        """Backbone parameters must be frozen (no grad)."""
        from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer

        tok = SemanticTokenizer(sem_dim=32, patch_size=_PATCH_SIZE, backbone="placeholder")
        for p in tok.backbone.parameters():
            assert not p.requires_grad, "Backbone parameters must be frozen"

    def test_projection_is_trainable(self, tiny_images):
        """Projection MLP must have trainable parameters."""
        from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer

        tok = SemanticTokenizer(sem_dim=32, patch_size=_PATCH_SIZE, backbone="placeholder")
        trainable = [p for p in tok.proj.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Projection MLP must have trainable parameters"

    def test_custom_backbone_provider(self, tiny_images):
        """Custom backbone_provider is correctly injected."""
        from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer

        # Custom backbone that outputs 256-dim tokens
        class _MyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 256, kernel_size=14, stride=14)
            def forward(self, x):
                f = self.conv(x)
                B, C, h, w = f.shape
                return f.permute(0, 2, 3, 1).reshape(B, h * w, C)

        tok = SemanticTokenizer(
            sem_dim=32,
            patch_size=14,
            backbone_provider=_MyBackbone(),
            backbone_dim=256,
        ).eval()
        with torch.no_grad():
            T_sem = tok(tiny_images)
        N_patch = (_IMG_SIZE // _PATCH_SIZE) ** 2
        assert T_sem.shape == (_B, _S, N_patch, 32)


# ---------------------------------------------------------------------------
# Module 2 – SemanticGuidedAttention + SemanticGuidedBlock
# ---------------------------------------------------------------------------

class TestSemanticGuidedAttention:
    """Tests for SemanticGuidedAttention (Module 2)."""

    def _make_attn(self, use_sem=True, dim=64):
        from vggt_semantic.modules.semantic_attention import SemanticGuidedAttention
        return SemanticGuidedAttention(
            dim, num_heads=4, sem_dim=32,
            use_semantic_guidance=use_sem,
            fused_attn=False,  # explicit path for testing
        ).eval()

    def test_output_shape_no_sem(self):
        """Without sem_tokens, output shape matches input."""
        attn = self._make_attn(use_sem=False)
        x = torch.randn(2, 20, 64)  # [B, N, C]
        with torch.no_grad():
            out = attn(x)
        assert out.shape == x.shape

    def test_output_shape_with_sem(self):
        """With sem_tokens, output shape still matches input."""
        attn = self._make_attn(use_sem=True)
        B, N_spec, N_patch = 2, 5, 16
        N = N_spec + N_patch
        x = torch.randn(B, N, 64)
        sem = torch.randn(B, N_patch, 32)
        with torch.no_grad():
            out = attn(x, sem_tokens=sem)
        assert out.shape == x.shape

    def test_guidance_disabled_equals_no_sem(self):
        """Disabling guidance gives same output as passing sem_tokens=None."""
        torch.manual_seed(42)
        attn_sem  = self._make_attn(use_sem=True)
        torch.manual_seed(42)
        attn_base = self._make_attn(use_sem=False)

        # Make both models have the same weights
        attn_base.load_state_dict(
            {k: v for k, v in attn_sem.state_dict().items()
             if k in attn_base.state_dict()},
            strict=False,
        )

        x   = torch.randn(2, 20, 64)
        sem = torch.randn(2, 16, 32)

        with torch.no_grad():
            out_base = attn_base(x)                  # no sem
            out_sem_none = attn_sem(x, sem_tokens=None)  # sem disabled

        assert torch.allclose(out_base, out_sem_none, atol=1e-5), (
            "Output with guidance disabled must equal output without sem_tokens"
        )


class TestSemanticGuidedBlock:
    """Tests for SemanticGuidedBlock (Module 2 wrapper)."""

    def _make_block(self, use_sem=True):
        from vggt_semantic.modules.semantic_attention import SemanticGuidedBlock
        return SemanticGuidedBlock(
            dim=64, num_heads=4, mlp_ratio=2.0,
            sem_dim=32, use_semantic_guidance=use_sem,
            fused_attn=False,
        ).eval()

    def test_forward_no_sem(self):
        blk = self._make_block(use_sem=True)
        x = torch.randn(2, 20, 64)
        with torch.no_grad():
            out = blk(x)
        assert out.shape == x.shape

    def test_forward_with_sem(self):
        blk = self._make_block(use_sem=True)
        x   = torch.randn(2, 20, 64)
        sem = torch.randn(2, 16, 32)
        with torch.no_grad():
            out = blk(x, sem_tokens=sem)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Module 3 – SemanticHead
# ---------------------------------------------------------------------------

class TestSemanticHead:
    """Tests for SemanticHead (Module 3)."""

    def _make_head(self):
        from vggt_semantic.modules.semantic_head import SemanticHead
        # dim_in = 2 * embed_dim (frame + global concatenated)
        return SemanticHead(
            dim_in=2 * _EMBED_DIM,
            sem_dim=_SEM_DIM,
            patch_size=_PATCH_SIZE,
            features=32,
            intermediate_layer_idx=[-1],   # single layer for speed
        ).eval()

    def _fake_tokens_list(self, depth=2):
        """Build a fake aggregated_tokens_list for testing."""
        P_total = 5 + (_IMG_SIZE // _PATCH_SIZE) ** 2  # 5 special + 16 patch = 21
        return [
            torch.randn(_B, _S, P_total, 2 * _EMBED_DIM)
            for _ in range(depth)
        ]

    def test_output_shape(self, tiny_images):
        head = self._make_head()
        tokens_list = self._fake_tokens_list()
        patch_start_idx = 5
        with torch.no_grad():
            out = head(tokens_list, tiny_images, patch_start_idx)
        assert out.shape == (_B, _S, _SEM_DIM, _IMG_SIZE, _IMG_SIZE), (
            f"Expected {(_B, _S, _SEM_DIM, _IMG_SIZE, _IMG_SIZE)}, got {out.shape}"
        )

    def test_sem_dim_is_32(self, tiny_images):
        """Output semantic dimension must be exactly 32."""
        head = self._make_head()
        tokens_list = self._fake_tokens_list()
        with torch.no_grad():
            out = head(tokens_list, tiny_images, patch_start_idx=5)
        assert out.shape[2] == 32, "Semantic output channel dim must be 32"


# ---------------------------------------------------------------------------
# SemanticGuidedAggregator
# ---------------------------------------------------------------------------

class TestSemanticGuidedAggregator:
    """Integration tests for the modified aggregator."""

    def _make_aggregator(self, use_sem=True):
        from vggt_semantic.modules.aggregator import SemanticGuidedAggregator
        return SemanticGuidedAggregator(
            img_size=_IMG_SIZE,
            patch_size=_PATCH_SIZE,
            embed_dim=_EMBED_DIM,
            depth=2,
            num_heads=4,
            num_register_tokens=2,
            patch_embed="conv",
            aa_order=["frame", "global"],
            aa_block_size=1,
            rope_freq=-1,
            use_semantic_guidance=use_sem,
            sem_dim=_SEM_DIM,
        ).eval()

    def test_forward_no_sem(self, tiny_images):
        agg = self._make_aggregator(use_sem=False)
        with torch.no_grad():
            tokens_list, patch_start_idx = agg(tiny_images)
        assert len(tokens_list) > 0
        # Each entry: [B, S, P, 2*embed_dim]
        assert tokens_list[-1].shape[0] == _B
        assert tokens_list[-1].shape[1] == _S

    def test_forward_with_sem(self, tiny_images):
        from vggt_semantic.modules.semantic_tokenizer import SemanticTokenizer
        agg = self._make_aggregator(use_sem=True)
        tok = SemanticTokenizer(sem_dim=32, patch_size=_PATCH_SIZE, backbone="placeholder").eval()

        with torch.no_grad():
            T_sem = tok(tiny_images)
            tokens_list, patch_start_idx = agg(tiny_images, sem_tokens=T_sem)

        assert len(tokens_list) > 0
        last = tokens_list[-1]
        assert last.shape[:2] == (_B, _S)


# ---------------------------------------------------------------------------
# VGGTSemantic – end-to-end forward test
# ---------------------------------------------------------------------------

class TestVGGTSemantic:
    """End-to-end forward tests for VGGTSemantic."""

    def _make_model(self, sem_enabled=True, guidance=True):
        from vggt_semantic import VGGTSemantic
        from vggt_semantic.config import SemanticConfig, SemanticGuidanceConfig

        return VGGTSemantic(
            img_size=_IMG_SIZE,
            patch_size=_PATCH_SIZE,
            embed_dim=_EMBED_DIM,
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_track=False,
            semantic=SemanticConfig(
                enabled=sem_enabled,
                dim=_SEM_DIM,
                backbone="placeholder",
                guidance=SemanticGuidanceConfig(enabled=guidance),
            ),
        ).eval()

    def test_forward_semantic_enabled(self, tiny_images):
        model = self._make_model(sem_enabled=True, guidance=True)
        with torch.no_grad():
            out = model(tiny_images)

        assert "sem_feat" in out, "sem_feat must be present when semantic is enabled"
        assert out["sem_feat"].shape == (_B, _S, _SEM_DIM, _IMG_SIZE, _IMG_SIZE), (
            f"sem_feat shape mismatch: {out['sem_feat'].shape}"
        )

    def test_sem_feat_dim_is_32(self, tiny_images):
        """sem_feat channel dimension must be exactly 32."""
        model = self._make_model(sem_enabled=True)
        with torch.no_grad():
            out = model(tiny_images)
        assert out["sem_feat"].shape[2] == 32

    def test_forward_semantic_disabled(self, tiny_images):
        """When semantic is disabled, sem_feat must NOT be in predictions."""
        model = self._make_model(sem_enabled=False, guidance=False)
        with torch.no_grad():
            out = model(tiny_images)
        assert "sem_feat" not in out, "sem_feat must be absent when semantic is disabled"

    def test_geometry_keys_present(self, tiny_images):
        """Geometry outputs must always be present (regardless of semantic)."""
        model = self._make_model(sem_enabled=True)
        with torch.no_grad():
            out = model(tiny_images)
        # Camera head
        assert "pose_enc" in out, "pose_enc missing"
        assert out["pose_enc"].shape == (_B, _S, 9)
        # Depth head
        assert "depth" in out, "depth missing"

    def test_forward_single_scene_no_batch(self):
        """Model handles [S, 3, H, W] input (no batch dim) gracefully."""
        model = self._make_model(sem_enabled=True)
        imgs = torch.rand(_S, 3, _IMG_SIZE, _IMG_SIZE)
        with torch.no_grad():
            out = model(imgs)
        # After unsqueeze batch dim = 1
        assert out["sem_feat"].shape == (1, _S, _SEM_DIM, _IMG_SIZE, _IMG_SIZE)

    def test_images_stored_during_inference(self, tiny_images):
        """During eval, predictions dict must contain the input images."""
        model = self._make_model(sem_enabled=True)
        model.eval()
        with torch.no_grad():
            out = model(tiny_images)
        assert "images" in out
