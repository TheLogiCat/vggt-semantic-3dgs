import os
import numpy as np
from PIL import Image, ImageDraw
import torch

base_dir = "mv_toy_dataset"
num_scenes = 5
num_views = 2

for s in range(num_scenes):
    scene_dir = f"{base_dir}/scene_{s:03d}"
    os.makedirs(scene_dir, exist_ok=True)
    
    for v in range(num_views):
        # 核心逻辑：用 v * 30 模拟相机的横向平移，产生真实视差
        offset = v * 30 
        
        # 1. 生成 RGB
        img = Image.new('RGB', (224, 224), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20 + offset, 50, 100 + offset, 180], fill=(200, 50, 50))
        img.save(f"{scene_dir}/rgb_{v}.png")
        
        # 2. 生成 Depth
        depth = np.ones((224, 224), dtype=np.uint8) * 200
        depth[50:180, 20 + offset : 100 + offset] = 50
        Image.fromarray(depth).save(f"{scene_dir}/depth_{v}.png")

print(f"多视角数据生成完毕，请检查 {base_dir} 目录。")