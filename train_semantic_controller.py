#!/usr/bin/env python3
"""
Minimal training loop for semantic-controller fine-tuning.

This script intentionally freezes the full base network and only trains:
  1) SemanticTokenizer projection MLP + LayerNorm
  2) sem_log_scale (alpha) in semantic-guided attention
  3) Optional SemanticHead (when --train-sem-head is enabled)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from vggt_semantic import VGGTSemantic
from vggt_semantic.config import SemanticConfig, SemanticGuidanceConfig


def _load_rgb(path: Path, img_size: int) -> Tensor:
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_depth(path: Path, img_size: int, depth_scale: float) -> Tensor:
    dep = Image.open(path).resize((img_size, img_size), Image.NEAREST)
    arr = np.asarray(dep, dtype=np.float32) / depth_scale
    return torch.from_numpy(arr)


class RGBDepthPairs(Dataset):
    """
    Expects:
      root/
        rgb/*.png|jpg
        depth/*.png
    with matching file stems between rgb and depth.
    """

    def __init__(self, root: Path, img_size: int, depth_scale: float = 1000.0) -> None:
        self.root = root
        self.img_size = img_size
        self.depth_scale = depth_scale
        rgb_dir = root / "rgb"
        depth_dir = root / "depth"
        if not rgb_dir.exists() or not depth_dir.exists():
            raise FileNotFoundError(f"Missing {rgb_dir} or {depth_dir}")
        self.pairs: List[Tuple[Path, Path]] = []
        for p in sorted(rgb_dir.glob("*")):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            d = depth_dir / f"{p.stem}.png"
            if d.exists():
                self.pairs.append((p, d))
        if not self.pairs:
            raise RuntimeError("No rgb/depth pairs found.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        rgb_p, dep_p = self.pairs[idx]
        return _load_rgb(rgb_p, self.img_size), _load_depth(dep_p, self.img_size, self.depth_scale)


def _build_batch(item: Tuple[Tensor, Tensor], num_views: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    rgb, depth = item  # rgb [3,H,W], depth [H,W]
    images = rgb.unsqueeze(0).repeat(num_views, 1, 1, 1).unsqueeze(0).to(device)   # [1,S,3,H,W]
    depth_gt = depth.unsqueeze(0).repeat(num_views, 1, 1).unsqueeze(0).to(device)   # [1,S,H,W]
    return images, depth_gt


def _depth_loss(depth_pred: Tensor, depth_gt: Tensor) -> Tensor:
    if depth_pred.ndim == 5:
        depth_pred = depth_pred[..., 0]
    valid = depth_gt > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=depth_pred.device, dtype=depth_pred.dtype)
    return F.l1_loss(depth_pred[valid], depth_gt[valid])


def _relation_distill_loss(model: VGGTSemantic, images: Tensor) -> Tensor:
    if model._last_T_sem is None or model.sem_tokenizer is None:
        return torch.tensor(0.0, device=images.device)
    sem = model._last_T_sem  # [B,S,N,32]
    B, S, N_patch, _ = sem.shape
    imgs_flat = images.reshape(B * S, images.shape[2], images.shape[3], images.shape[4])
    with torch.no_grad():
        teacher = model.sem_tokenizer.backbone(imgs_flat)[:, -N_patch:, :]  # [B*S,N,1024]
    sem_f = sem.reshape(B * S, N_patch, -1)
    sem_sim = F.normalize(sem_f, dim=-1) @ F.normalize(sem_f, dim=-1).transpose(-2, -1)
    tea_sim = F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).transpose(-2, -1)
    return F.mse_loss(sem_sim, tea_sim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root with rgb/ and depth/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="PNG depth divisor")
    parser.add_argument("--backbone", type=str, default="dinov2", choices=["dinov2", "placeholder"])
    parser.add_argument("--train-sem-head", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--save-path", type=Path, default=Path("semantic_controller.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RGBDepthPairs(args.data_root, img_size=args.img_size, depth_scale=args.depth_scale)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = VGGTSemantic(
        img_size=args.img_size,
        patch_size=14,
        embed_dim=64 if args.backbone == "placeholder" else 1024,
        patch_embed="conv" if args.backbone == "placeholder" else "dinov2_vitl14_reg",
        semantic=SemanticConfig(
            enabled=True,
            dim=32,
            backbone=args.backbone,
            guidance=SemanticGuidanceConfig(enabled=True),
        ),
    ).to(device)
    model.freeze_base_and_enable_semantic_controller(train_sem_head=args.train_sem_head)
    params = [p for p in model.semantic_controller_parameters(train_sem_head=args.train_sem_head) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for rgb_batch, depth_batch in dl:
            # Keep loop minimal and robust: operate sample-by-sample across batch items.
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.tensor(0.0, device=device)
            for i in range(rgb_batch.shape[0]):
                images, depth_gt = _build_batch((rgb_batch[i], depth_batch[i]), args.num_views, device)
                out = model(images)
                loss_depth = _depth_loss(out["depth"], depth_gt)
                loss = loss_depth
                if args.distill_weight > 0:
                    loss_distill = _relation_distill_loss(model, images)
                    loss = loss + args.distill_weight * loss_distill
                total_loss = total_loss + loss
            total_loss = total_loss / rgb_batch.shape[0]
            total_loss.backward()
            optimizer.step()
            running += float(total_loss.detach().cpu())
        avg = running / max(1, len(dl))
        print(f"epoch={epoch + 1}/{args.epochs} loss={avg:.6f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_sem_head": args.train_sem_head,
            "epochs": args.epochs,
        },
        args.save_path,
    )
    print(f"saved checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()

