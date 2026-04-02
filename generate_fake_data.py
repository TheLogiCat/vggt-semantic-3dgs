import os
import numpy as np
from PIL import Image, ImageDraw

base_dir = "toy_dataset"
os.makedirs(f"{base_dir}/rgb", exist_ok=True)
os.makedirs(f"{base_dir}/depth", exist_ok=True)

print("正在本地生成合成测试数据...")

for i in range(5):
    # 1. 生成 RGB 图片 (画一些简单的彩色方块和圆，让 DINO 有特征可提取)
    rgb_img = Image.new('RGB', (224, 224), color=(50, 50, 50))
    draw = ImageDraw.Draw(rgb_img)
    # 画个矩形
    draw.rectangle([20+i*10, 20+i*10, 100+i*10, 150], fill=(200, 50, 50))
    # 画个圆
    draw.ellipse([120, 80-i*5, 180, 140-i*5], fill=(50, 200, 50))
    
    # 2. 生成 Depth 图片 (模拟深度：矩形离得近，圆形离得远，背景最远)
    depth_arr = np.ones((224, 224), dtype=np.uint8) * 200 # 背景深度 200
    depth_arr[20+i*10:150, 20+i*10:100+i*10] = 50       # 矩形深度 50 (近)
    
    # 画圆的深度稍微麻烦点，我们直接用粗糙的矩形块代替
    depth_arr[80-i*5:140-i*5, 120:180] = 120            # 圆形区域深度 120 (中)
    depth_img = Image.fromarray(depth_arr)

    # 保存
    rgb_img.save(f"{base_dir}/rgb/{i:04d}.png")
    depth_img.save(f"{base_dir}/depth/{i:04d}.png")

print("✅ 合成玩具数据集准备完毕！")