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