#!/usr/bin/env python3
"""
Test script to visualize 32D semantic features from family_semantic.pt using PCA.
Performs PCA reduction to 3 dimensions and maps to RGB heatmap, stacked with original image.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assume vggt_semantic is installable or in pythonpath
# based on train_semantic_controller.py context
from vggt_semantic import VGGTSemantic
from vggt_semantic.config import SemanticConfig, SemanticGuidanceConfig

# -----------------------------------------------------------------------------
# Helpers modified from training script for inference
# -----------------------------------------------------------------------------
def _load_rgb_inference(path: Path, img_size: int) -> Tensor:
    """Load and preprocess a single RGB image for inference."""
    # NEAREST interpolation is fine for inference
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) # [3, H, W]
    return tensor.unsqueeze(0).unsqueeze(0) # [1, 1, 3, H, W] (batch=1, views=1)

def setup_model(checkpoint_path: Path, device: torch.device, img_size: int, backbone: str) -> VGGTSemantic:
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Define architecture (must match training configs)
    model = VGGTSemantic(
        img_size=img_size,
        patch_size=14,
        embed_dim=1024, # ViT-Large dim
        patch_embed="dinov2_vitl14_reg",
        semantic=SemanticConfig(
            enabled=True,
            dim=32,
            backbone=backbone,
            guidance=SemanticGuidanceConfig(enabled=True),
        ),
    ).to(device)

    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        exit(1)
    except KeyError:
        print(f"Error: Could not find 'model_state_dict' in {checkpoint_path}. Is it a valid semantic controller checkpoint?")
        exit(1)
        
    model.eval() # Set to evaluation mode
    return model

def visualize_semantic_features(model: VGGTSemantic, image_batch: Tensor, save_path: Path):
    """
    Core function to extract features, perform PCA, and generate RGB heatmap.
    `image_batch` shape: [1, 1, 3, H, W] (single batch, single view)
    """
    device = image_batch.device
    
    # 1. Forward pass to extract features. 
    # Based on training script context, features are stored in `model._last_T_sem` after call.
    with torch.no_grad():
        _ = model(image_batch) # Call forward

    # 2. Extract features: Expected shape [1, 1, N, 32] after this single view call
    if model._last_T_sem is None:
        raise RuntimeError("Semantic features were not stored after forward pass. Check network implementation.")
    
    features_32d = model._last_T_sem.cpu() # Move to CPU
    b, s, n_patches, d = features_32d.shape
    
    # reshape to [N, 32]
    features_flat = features_32d.reshape(b * s * n_patches, d).numpy()
    
    # 3. Perform PCA to reduce from 32 to 3 dimensions
    print(f"Performing PCA on {features_flat.shape} features to reduce to 3D for visualization...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat) # [N, 3]
    
    # 4. Map PCA components to RGB values [0, 255]
    # Normalize components independently to [0, 255] uint8 for direct image creation
    min_val = pca_features.min(axis=0)
    max_val = pca_features.max(axis=0)
    
    # Handle edge case where max == min (prevents divide by zero)
    range_val = np.where(max_val > min_val, max_val - min_val, 1e-6)
    
    norm_features = (pca_features - min_val) / range_val # Scale to [0, 1]
    rgb_features = (norm_features * 255).astype(np.uint8) # Scale to [0, 255]
    
    # 5. Reconstruct RGB semantic heatmap image
    # Calculate grid size (patches per dimension, e.g., 224/14 = 16)
    grid_size = int(np.sqrt(n_patches)) # e.g., 16 if N=256 (224/14=16)
    semantic_map = rgb_features.reshape(grid_size, grid_size, 3)
    
    # Convert to PIL image and resize to match input resolution using NEAREST interpolation
    # to maintain sharp, distinct borders between semantic regions.
    img_size = image_batch.shape[-1]
    img_semantic = Image.fromarray(semantic_map).resize((img_size, img_size), Image.NEAREST)
    
    # 6. Create composite image (original RGB left, semantic heatmap right)
    # Reconstruct original image from tensor
    orig_np = (image_batch[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_orig = Image.fromarray(orig_np)

    composite = Image.new('RGB', (2 * img_size, img_size))
    composite.paste(img_orig, (0, 0))
    composite.paste(img_semantic, (img_size, 0))
    
    # 7. Save result
    save_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(save_path)
    print(f"Saved semantic visualization composite to: {save_path}")

    # Optional interactive display using matplotlib (won't work in headless environment)
    if os.environ.get('DISPLAY'):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_orig)
        plt.title('Original RGB Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_semantic)
        plt.title('32D Semantic PCA (RGB Heatmap)')
        plt.axis('off')
        
        plt.tight_layout()
        print("Opening interactive display. Close the window to continue...")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test script to visualize 32D semantic features from family_semantic.pt using PCA.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the family_semantic.pt checkpoint")
    parser.add_argument("--image", type=Path, required=True, help="Path to a test RGB image (.jpg, .jpeg, or .png)")
    parser.add_argument("--out-dir", type=Path, default=Path("test_results"), help="Directory to save output visualizations")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--backbone", type=str, default="dinov2", help="Backbone used during training (default: dinov2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running visualization test on device: {device}")

    # 1. Setup model from trained checkpoint
    model = setup_model(args.checkpoint, device, args.img_size, args.backbone)

    # 2. Prepare single test image: Load as [1, 1, 3, H, W]
    image_tensor = _load_rgb_inference(args.image, args.img_size).to(device)

    # 3. Perform forward pass, extract features, apply PCA, map to RGB, stack and save.
    out_name = f"{args.image.stem}_semantic_pca.png"
    save_path = args.out_dir / out_name
    
    visualize_semantic_features(model, image_tensor, save_path)

if __name__ == "__main__":
    main()