import os
import torch
from PIL import Image
from transformers import pipeline

# 强制使用 HF 镜像源，防止下载模型时被墙
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 载入 Depth Anything V2 模型 (Small 版本跑得最快，显存占用低，足够做验证)
print("正在加载 Depth Anything V2 模型...")
# 如果你的 transformers 版本较老，可以替换为 "LiheYoung/depth-anything-small-hf"
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)

# 请将这里的路径替换为你 Family 数据集真实的图片目录
input_dir = "data/Family/images/Family" 
output_dir = "data/Family/pseudo_depths"
os.makedirs(output_dir, exist_ok=True)

valid_exts = ('.jpg', '.png', '.jpeg')
images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
print(f"共找到 {len(images)} 张 RGB 图片，开始生成深度图...")

for img_name in images:
    img_path = os.path.join(input_dir, img_name)
    image = Image.open(img_path).convert('RGB')
    
    # 前向推理提取深度
    result = pipe(image)
    depth_image = result["depth"]
    
    # 将结果保存为 png 格式
    base_name = os.path.splitext(img_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}.png")
    depth_image.save(save_path)

print(f"✅ 伪深度图全部生成完毕，已保存在: {output_dir}")