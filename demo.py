#!/usr/bin/env python3
"""
Interactive visualisation demo for VGGTSemantic
================================================

Upload 1–4 images and see side-by-side:
  • Depth map prediction        (geometry DPT head → colourised with 'plasma')
  • Depth confidence map        (geometry DPT head → colourised with 'viridis')
  • Semantic feature map        (semantic head → PCA-RGB or channel-energy map)

Usage
-----
    python demo.py

Then open http://127.0.0.1:7860 in a browser.

Notes
-----
* When the real ``vggt`` package is **not** installed the model uses stub
  geometry heads that return zero depth maps.  The semantic head is fully
  functional regardless.
* When ``backbone="placeholder"`` the semantic tokeniser uses a lightweight
  conv stub instead of DINOv2, so semantic features reflect only low-level
  texture cues rather than high-level semantics.  Install
  ``transformers>=4.38`` and set backbone to ``"dinov2"`` for full semantics.
"""

from __future__ import annotations

import io
import warnings
from typing import List, Optional

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Model singleton (built once, reused across requests)
# ---------------------------------------------------------------------------

_MODEL: Optional[object] = None
_MODEL_IMG_SIZE: int = -1
_MODEL_BACKBONE: str = ""


def _build_model(img_size: int, backbone: str):
    """Construct and cache the VGGTSemantic model."""
    global _MODEL, _MODEL_IMG_SIZE, _MODEL_BACKBONE
    if (
        _MODEL is not None
        and _MODEL_IMG_SIZE == img_size
        and _MODEL_BACKBONE == backbone
    ):
        return _MODEL

    from vggt_semantic import VGGTSemantic
    from vggt_semantic.config import SemanticConfig, SemanticGuidanceConfig

    model = VGGTSemantic(
        img_size=img_size,
        patch_size=14,
        embed_dim=64,          # tiny embed for fast CPU inference in demo
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        semantic=SemanticConfig(
            enabled=True,
            dim=32,
            backbone=backbone,
            guidance=SemanticGuidanceConfig(enabled=True),
        ),
    ).eval()

    _MODEL = model
    _MODEL_IMG_SIZE = img_size
    _MODEL_BACKBONE = backbone
    return model


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _load_and_resize(pil_img: Image.Image, img_size: int) -> torch.Tensor:
    """Load a PIL image, resize to (img_size, img_size), return float [0,1] tensor."""
    img = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1)     # [3, H, W]


def _tensor_to_pil(t: np.ndarray) -> Image.Image:
    """Convert a float [0,1] HxWx3 or HxW numpy array to a PIL Image."""
    t = np.clip(t, 0.0, 1.0)
    return Image.fromarray((t * 255).astype(np.uint8))


def _colorise(arr2d: np.ndarray, cmap: str = "plasma") -> np.ndarray:
    """Apply a matplotlib colourmap to a 2-D float array and return HxWx3 RGB."""
    arr2d = arr2d.astype(np.float32)
    mn, mx = arr2d.min(), arr2d.max()
    if mx > mn:
        arr2d = (arr2d - mn) / (mx - mn)
    cm = plt.get_cmap(cmap)
    rgba = cm(arr2d)                  # HxWx4
    return rgba[..., :3].astype(np.float32)


def _sem_pca_rgb(sem: np.ndarray) -> np.ndarray:
    """
    Project 32-dim semantic feature map to 3-D RGB via PCA.

    sem : [32, H, W]  float32
    Returns [H, W, 3] float32 in [0, 1]
    """
    C, H, W = sem.shape
    feats = sem.reshape(C, -1).T       # [H*W, 32]

    # Zero-mean
    feats = feats - feats.mean(axis=0, keepdims=True)

    # SVD-based PCA (no sklearn required)
    try:
        _, _, Vt = np.linalg.svd(feats, full_matrices=False)  # Vt: [32, 32]
        proj = feats @ Vt[:3].T        # [H*W, 3]
    except np.linalg.LinAlgError:
        proj = feats[:, :3]

    proj = proj.reshape(H, W, 3)

    # Per-channel min-max normalise to [0, 1]
    for c in range(3):
        ch = proj[..., c]
        mn, mx = ch.min(), ch.max()
        proj[..., c] = (ch - mn) / (mx - mn + 1e-8)

    return proj.astype(np.float32)


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(
    images_in: List[Optional[Image.Image]],
    img_size: int,
    backbone: str,
) -> List[Optional[Image.Image]]:
    """
    Run VGGTSemantic on the uploaded images and return a flat list of PIL images:

        [orig_0, depth_0, depth_conf_0, sem_0,
         orig_1, depth_1, depth_conf_1, sem_1, ...]

    for up to MAX_VIEWS views.  Slots not filled are None.
    """
    MAX_VIEWS = 4

    # Filter out None / empty slots
    valid: List[Image.Image] = [im for im in images_in if im is not None]
    if not valid:
        raise gr.Error("Please upload at least one image.")

    # Snap img_size to the nearest multiple of 14 (patch size)
    img_size = max(56, (img_size // 14) * 14)

    model = _build_model(img_size, backbone)

    # Build tensor [1, S, 3, H, W]
    frames = torch.stack([_load_and_resize(im, img_size) for im in valid])  # [S,3,H,W]
    frames = frames.unsqueeze(0)     # [1, S, 3, H, W]

    with torch.no_grad():
        out = model(frames)

    S = frames.shape[1]

    results: List[Optional[Image.Image]] = []
    for s in range(MAX_VIEWS):
        if s >= S:
            # Pad with None for unused slots
            results.extend([None, None, None, None])
            continue

        # 1) Original image
        orig = frames[0, s].permute(1, 2, 0).cpu().numpy()   # [H,W,3]
        results.append(_tensor_to_pil(orig))

        # 2) Depth map
        if "depth" in out:
            depth = out["depth"][0, s, ..., 0].cpu().numpy()  # [H, W]
            dep_rgb = _colorise(depth, cmap="plasma")
            results.append(_tensor_to_pil(dep_rgb))
        else:
            results.append(None)

        # 3) Depth confidence
        if "depth_conf" in out:
            dconf = out["depth_conf"][0, s].cpu().numpy()     # [H, W]
            conf_rgb = _colorise(dconf, cmap="viridis")
            results.append(_tensor_to_pil(conf_rgb))
        else:
            results.append(None)

        # 4) Semantic PCA-RGB
        if "sem_feat" in out:
            sem = out["sem_feat"][0, s].cpu().float().numpy()  # [32, H, W]
            sem_rgb = _sem_pca_rgb(sem)
            results.append(_tensor_to_pil(sem_rgb))
        else:
            results.append(None)

    return results


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def _backend_status() -> str:
    lines = []
    try:
        import vggt  # noqa: F401
        lines.append("✅ vggt geometry heads (depth / world-points): **available**")
    except ImportError:
        lines.append(
            "⚠️ vggt not installed → stub depth head (returns zeros). "
            "Run `pip install vggt @ git+https://github.com/facebookresearch/vggt.git` "
            "for real depth predictions."
        )

    try:
        import transformers  # noqa: F401
        lines.append("✅ transformers (DINOv2 backbone): **available**")
    except ImportError:
        lines.append(
            "⚠️ transformers not installed → placeholder backbone used. "
            "Run `pip install transformers>=4.38` and select `dinov2` backbone "
            "for high-level semantic features."
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

MAX_VIEWS = 4

_LABELS = ["Input image", "Depth map (plasma)", "Depth confidence (viridis)", "Semantic PCA-RGB"]

with gr.Blocks(title="VGGTSemantic Demo") as demo:
    gr.Markdown(
        """
# 🔭 VGGTSemantic – Depth & Semantic Prediction Demo

Upload **1–4 images** (multiple views of the same scene, or a single image) and
click **Run Inference** to see:

| Output | Description |
|--------|-------------|
| **Depth map** | Per-pixel depth predicted by the geometry DPT head (plasma colourmap) |
| **Depth confidence** | Prediction certainty map (viridis colourmap) |
| **Semantic features** | 32-dim semantic feature map compressed to RGB via PCA |
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            backbone_dd = gr.Dropdown(
                choices=["placeholder", "dinov2"],
                value="placeholder",
                label="Semantic backbone",
                info=(
                    "placeholder = fast conv stub (no pretrained weights). "
                    "dinov2 = frozen DINOv2-ViT-L/14 (requires transformers>=4.38)."
                ),
            )
            img_size_sl = gr.Slider(
                minimum=56,
                maximum=336,
                step=14,
                value=112,
                label="Image size (pixels, snapped to multiple of 14)",
                info="Smaller = faster. Full-size VGGT uses 518.",
            )

            gr.Markdown("### 📤 Upload images (1–4)")
            inputs = [gr.Image(type="pil", label=f"View {i+1}") for i in range(MAX_VIEWS)]

            run_btn = gr.Button("▶  Run Inference", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results  (Input · Depth · Confidence · Semantic)")
            outputs: List[gr.Image] = []
            for v in range(MAX_VIEWS):
                with gr.Row():
                    for lbl in _LABELS:
                        outputs.append(
                            gr.Image(
                                label=f"View {v+1} – {lbl}",
                                show_label=True,
                                type="pil",
                                height=200,
                            )
                        )

    gr.Markdown("### ℹ️ Backend status")
    status_md = gr.Markdown(_backend_status())

    run_btn.click(
        fn=run_inference,
        inputs=[*inputs, img_size_sl, backbone_dd],
        outputs=outputs,
    )

    gr.Markdown(
        """
---
**Architecture recap**

```
images [B, S, 3, H, W]
    │
    ├──► Module 1: SemanticTokenizer  ──► T_sem [B, S, N_patch, 32]
    │    └─ frozen DINOv2-ViT-L/14 backbone + 2-layer MLP projection        │                                            │
    └──► Module 2: SemanticGuidedAggregator  ◄────────────────────────────┘
         └─ frame-attention logits += α · (T_sem · T_semᵀ)   (learnable α)
                       │
         ┌─────────────┼─────────────────────────────────────┐
         ▼             ▼                                     ▼
    CameraHead     DPTHead (depth)              Module 3: SemanticHead
    pose_enc       depth [B,S,H,W,1]            sem_feat [B,S,32,H,W]
                   depth_conf [B,S,H,W]         └─ multi-layer fusion
                                                   + ConvTranspose 4× upsample
                                                   + GroupNorm
```
"""
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(share=False)
