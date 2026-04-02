# vggt-semantic-3dgs

**VGGT as semantic-geometric controller for compositional 3DGS**

This repository extends [VGGT (Visual Geometry Grounded Transformer, CVPR 2025)](https://github.com/facebookresearch/vggt) with a semantic branch that produces dense 32-dim semantic feature maps alongside the original geometry outputs.  The semantic features are intended for use as Gaussian attributes in a downstream compositional 3D Gaussian Splatting (3DGS) pipeline.

---

## Architecture overview

```
images [B, S, 3, H, W]
    │
    ├──► Module 1: SemanticTokenizer ──► T_sem [B, S, N_patch, 32]
    │                                          │
    └──► Module 2: SemanticGuidedAggregator ◄──┘
                │  aggregated_tokens_list
                │
    ┌───────────┼──────────────────────────────────────────────┐
    ▼           ▼           ▼                                  ▼
CameraHead   DPTHead   DPTHead(pts)           Module 3: SemanticHead
pose_enc     depth      world_points           sem_feat [B,S,32,H,W]
```

### Module 1 – SemanticTokenizer

*File: `vggt_semantic/modules/semantic_tokenizer.py`*

Extracts 32-dim semantic tokens from multi-view images:

1. Reshape images from `[B, S, 3, H, W]` to `[B*S, 3, H, W]`
2. Run a **frozen** DINOv2-ViT-L/14 backbone (`backbone="dinov2"`) or a lightweight placeholder (`backbone="placeholder"`) to get raw features
3. Project through a 2-layer MLP with LayerNorm to `T_sem [B, S, N_patch, 32]`

The backbone is always frozen; only the projection MLP trains.

**Backbone options:**

| `backbone=` | Description |
|---|---|
| `"dinov2"` | Frozen DINOv2-ViT-L/14 (requires `transformers>=4.38`; auto-downloaded on first run) |
| `"placeholder"` | Lightweight 2-conv stub – no pretrained weights, suitable for tests / offline use |
| `backbone_provider=...` | Pass a custom `nn.Module` with signature `forward([B,3,H,W]) → [B,N,dim]` |

---

### Module 2 – SemanticGuidedAttention / SemanticGuidedBlock

*File: `vggt_semantic/modules/semantic_attention.py`*

Injects semantic patch-similarity into the transformer attention logits. For each frame-level attention block:

```
sem_norm       = L2_normalise(T_sem)                 # [B, N_patch, 32]
M_sem          = sem_norm @ sem_norm^T               # [B, N_patch, N_patch]  (cosine sim)
bias           = exp(sem_log_scale) * M_sem          # learnable scalar scale
attn_logits'   = attn_logits + bias.unsqueeze(H) * sem_head_weight
A'             = softmax(attn_logits') @ V
```

* Additive modulation at logits level → numerically stable, differentiable.
* `sem_log_scale` initialised to `-4` (near-zero effect at start of training).
* `sem_head_weight` per-head sigmoid gating `[H, 1, 1]` for flexibility.
* **Ablation switch:** `use_semantic_guidance=False` degrades to vanilla attention.

The `SemanticGuidedAggregator` subclasses VGGT's `Aggregator` and injects `T_sem` into frame-attention blocks only (global attention preserves geometric consistency). Set `guide_global_attn=True` to also guide global blocks.

---

### Module 3 – SemanticHead

*File: `vggt_semantic/modules/semantic_head.py`*

Lightweight decoder producing a dense semantic feature map:

1. Extract patch tokens from selected aggregator layers (`intermediate_layer_idx`)
2. LayerNorm + 1×1 Conv → `[B*S, features, patch_h, patch_w]`
3. Element-wise fuse across layers
4. 4× ConvTranspose → GELU → 1×1 Conv → `[B*S, 32, ...]`
5. Bilinear upsample to `(H, W)`
6. GroupNorm → reshape to `[B, S, 32, H, W]`

Output key: `sem_feat` in the predictions dictionary.

---

## Configuration

Default config: `configs/semantic_default.yaml`

```yaml
model:
  img_size: 518
  patch_size: 14
  embed_dim: 1024
  enable_camera: true
  enable_depth: true
  enable_point: true
  enable_track: false

  semantic:
    enabled: true          # master switch
    dim: 32                # fixed at 32
    backbone: dinov2       # or "placeholder" for tests
    guidance:
      enabled: true        # Module 2 switch (ablation)
```

Python dataclasses mirror the YAML structure:

```python
from vggt_semantic.config import VGGTSemanticConfig, SemanticConfig, SemanticGuidanceConfig

cfg = VGGTSemanticConfig(
    img_size=518,
    semantic=SemanticConfig(
        enabled=True,
        dim=32,
        backbone="dinov2",
        guidance=SemanticGuidanceConfig(enabled=True),
    ),
)
model = VGGTSemantic.from_config(cfg)
```

---

## Quick start

### Installation

```bash
git clone https://github.com/TheLogiCat/vggt-semantic-3dgs
cd vggt-semantic-3dgs
pip install -e .

# Install base VGGT (required for full geometry heads):
pip install vggt @ git+https://github.com/facebookresearch/vggt.git

# Optional – real DINOv2 backbone (Module 1):
pip install transformers>=4.38
```

### Minimal forward pass

```python
import torch
from vggt_semantic import VGGTSemantic
from vggt_semantic.config import SemanticConfig

# Build model with placeholder backbone (no downloads needed)
model = VGGTSemantic(
    img_size=518,
    semantic=SemanticConfig(enabled=True, dim=32, backbone="placeholder"),
).eval()

# Random 3-view input
images = torch.rand(1, 3, 3, 518, 518)  # [B, S, 3, H, W]

with torch.no_grad():
    out = model(images)

print(out["sem_feat"].shape)      # [1, 3, 32, 518, 518]
print(out["pose_enc"].shape)      # [1, 3, 9]
print(out["depth"].shape)         # [1, 3, 518, 518, ...]
```

### Run tests

```bash
pytest tests/test_semantic_forward.py -v
```

---

## Output dictionary

| Key | Shape | Description |
|-----|-------|-------------|
| `pose_enc` | `[B, S, 9]` | Camera pose encoding (last refinement iter) |
| `pose_enc_list` | `list[[B,S,9]]` | All refinement iterations |
| `depth` | `[B, S, H, W, 1]` | Predicted depth maps |
| `depth_conf` | `[B, S, H, W]` | Depth confidence |
| `world_points` | `[B, S, H, W, 3]` | 3D world coordinates |
| `world_points_conf` | `[B, S, H, W]` | Point confidence |
| `track` | `[B, S, N, 2]` | 2D point tracks (if `query_points` given) |
| `vis` | `[B, S, N]` | Track visibility |
| `conf` | `[B, S, N]` | Track confidence |
| `sem_feat` | `[B, S, 32, H, W]` | **Dense semantic features (new)** |
| `images` | `[B, S, 3, H, W]` | Input images (inference only) |

---

## Training notes

The semantic projection MLP and SemanticHead are the primary trainable components. Recommended training schedule:

1. **Phase 1** – freeze backbone + patch embed, train projection + semantic head only
2. **Phase 2** – unfreeze aggregator blocks and fine-tune end-to-end

Helpers:

```python
model.freeze_backbone()          # freeze DINOv2 + patch embed
optim_sem = torch.optim.Adam(model.semantic_parameters(), lr=1e-3)
optim_geo = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

## Module 4 (future work)

Module 4 – Closed-loop 3DGS training – is **not** implemented in this PR. Recommended interface connection points:

| Hook | Location | Purpose |
|------|----------|---------|
| `sem_feat` output | `VGGTSemantic.forward()` | Initialise per-Gaussian semantic attributes |
| `model._last_T_sem` | `VGGTSemantic` instance | Per-patch semantic tokens for auxiliary supervision |
| `model.aggregator` | `VGGTSemantic.__init__()` | Swap in a fully-differentiable aggregator once DINOv2 is fine-tuned |

---

## Repository structure

```
vggt_semantic/
├── config.py                    # dataclass configs
├── __init__.py
├── models/
│   └── vggt_semantic.py         # VGGTSemantic: main model
└── modules/
    ├── semantic_tokenizer.py    # Module 1
    ├── semantic_attention.py    # Module 2 (SemanticGuidedAttention + Block)
    ├── semantic_head.py         # Module 3
    └── aggregator.py            # SemanticGuidedAggregator
configs/
└── semantic_default.yaml
tests/
└── test_semantic_forward.py
```
