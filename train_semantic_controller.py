#!/usr/bin/env python3
"""
Minimal training loop for semantic-controller fine-tuning on Real Data (COLMAP Poses).

This script intentionally freezes the full base network and only trains:
  1) SemanticTokenizer projection MLP + LayerNorm
  2) sem_log_scale (alpha) in semantic-guided attention
  3) Optional SemanticHead (when --train-sem-head is enabled)
"""

from __future__ import annotations

import argparse
import os
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


def qvec2rotmat(qvec):
    """将四元数转换为 3x3 旋转矩阵"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def parse_colmap_poses(sparse_path: Path):
    """
    解析 COLMAP 的 images.txt 和 cameras.txt
    返回: { image_name: (4x4_pose_matrix, 3x3_intrinsics) }
    """
    poses = {}
    intrinsics = {}
    
    # 1. 解析相机内参 (Intrinsic)
    cam_file = sparse_path / "cameras.txt"
    with open(cam_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            elems = line.split()
            cam_id = int(elems[0])
            params = [float(x) for x in elems[4:]]
            K = np.eye(3)
            if elems[1] == "PINHOLE":
                K[0,0], K[1,1], K[0,2], K[1,2] = params[0], params[1], params[2], params[3]
            elif elems[1] == "SIMPLE_PINHOLE":
                K[0,0], K[1,1], K[0,2], K[1,2] = params[0], params[0], params[1], params[2]
            intrinsics[cam_id] = K

    # 2. 解析外参 (Extrinsic)
    img_file = sparse_path / "images.txt"
    with open(img_file, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            line = lines[i]
            if line.startswith("#"): continue
            elems = line.split()
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            cam_id = int(elems[8])
            img_name = elems[9]
            
            # 构建 [R|t] 矩阵 (World-to-Camera)
            R = qvec2rotmat([qw, qx, qy, qz])
            t = np.array([tx, ty, tz]).reshape(3, 1)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t.flatten()
            
            poses[img_name] = (w2c, intrinsics[cam_id])
            
    return poses


class ColmapDataset(Dataset):
    def __init__(self, root: Path, num_views: int = 2, img_size: int = 224, depth_scale: float = 255.0):
        self.root = Path(root)
        self.num_views = num_views
        self.img_size = img_size
        self.depth_scale = depth_scale
        
        self.img_dir = self.root / "images"
        self.depth_dir = self.root / "pseudo_depths"
        self.sparse_dir = self.root / "sparse"
        
        if not self.sparse_dir.exists():
            raise RuntimeError(f"未找到 COLMAP 稀疏重建目录: {self.sparse_dir}")

        # 解析位姿
        self.poses_dict = parse_colmap_poses(self.sparse_dir)
        
        # 提取有位姿记录的图像，并按名称排序保证相邻视角物理重叠度高
        valid_exts = ('.jpg', '.jpeg', '.png')
        all_imgs = [f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_exts)]
        self.image_names = sorted([f for f in all_imgs if f in self.poses_dict])
        
        if len(self.image_names) < self.num_views:
            raise RuntimeError(f"在 {self.root} 下没有找到足够数量且带有位姿的图片。")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        rgbs, depths, poses, intrinsics = [], [], [], []
        
        for v in range(self.num_views):
            # 强制提取连续的视图组成 Pair，取余防止越界
            curr_idx = (idx + v) % len(self.image_names)
            img_name = self.image_names[curr_idx]
            
            # 加载 RGB
            rgb_p = self.img_dir / img_name
            img = Image.open(rgb_p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            rgbs.append(torch.from_numpy(arr).permute(2, 0, 1))

            # 加载 Depth (伪深度扩展名统一替换为.png)
            base_name = os.path.splitext(img_name)[0]
            dep_p = self.depth_dir / f"{base_name}.png"
            dep = Image.open(dep_p).resize((self.img_size, self.img_size), Image.NEAREST)
            dep_arr = np.asarray(dep, dtype=np.float32) / self.depth_scale
            depths.append(torch.from_numpy(dep_arr))

            # 获取位姿与内参
            w2c, K = self.poses_dict[img_name]
            poses.append(torch.from_numpy(w2c).to(torch.float32))
            intrinsics.append(torch.from_numpy(K).to(torch.float32))

        # 输出维度: [num_views, C, H, W], [num_views, H, W], [num_views, 4, 4], [num_views, 3, 3]
        return torch.stack(rgbs), torch.stack(depths), torch.stack(poses), torch.stack(intrinsics)


def _depth_loss(depth_pred: Tensor, depth_gt: Tensor) -> Tensor:
    if depth_pred.ndim == 5:
        depth_pred = depth_pred[..., 0]
    
    valid = depth_gt > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=depth_pred.device, dtype=depth_pred.dtype)
    
    pred_v = depth_pred[valid]
    gt_v = depth_gt[valid]
    
    # 使用中位数进行动态尺度对齐，消除绝对数值的鸿沟
    pred_med = pred_v.median().detach()
    gt_med = gt_v.median().detach()
    
    # 防止除零异常
    pred_aligned = pred_v / (pred_med + 1e-6)
    gt_aligned = gt_v / (gt_med + 1e-6)
    
    return F.l1_loss(pred_aligned, gt_aligned)


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
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root with images/, pseudo_depths/, sparse/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth-scale", type=float, default=255.0, help="Depth normalization scale")
    parser.add_argument("--backbone", type=str, default="dinov2", choices=["dinov2", "placeholder"])
    parser.add_argument("--train-sem-head", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--save-path", type=Path, default=Path("semantic_controller.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化真实数据集
    ds = ColmapDataset(args.data_root, num_views=args.num_views, img_size=args.img_size, depth_scale=args.depth_scale)
    # 开启多进程预读与锁页内存（加速 CPU 到 GPU 的传输）
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,           # 根据你服务器的 CPU 核心数，可调至 8
        pin_memory=True,         # 开启锁页内存
        prefetch_factor=2        # 提前预取 2 个 Batch
    )

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
    
    # 初始化 AMP 梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        running = 0.0
        for rgb_batch, depth_batch, pose_batch, intrinsic_batch in dl:
            optimizer.zero_grad(set_to_none=True)
            
            # non_blocking=True 配合 pin_memory 实现异步数据传输
            images = rgb_batch.to(device, non_blocking=True)
            depth_gt = depth_batch.to(device, non_blocking=True)
            poses = pose_batch.to(device, non_blocking=True)
            intrinsics = intrinsic_batch.to(device, non_blocking=True)
            
            # 开启混合精度上下文
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(images)
                loss_depth = _depth_loss(out["depth"], depth_gt)
                loss = loss_depth
                
                if args.distill_weight > 0:
                    loss_distill = _relation_distill_loss(model, images)
                    loss = loss + args.distill_weight * loss_distill
            
            # 使用 Scaler 进行反向传播和参数更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running += float(loss.detach().cpu())
            
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