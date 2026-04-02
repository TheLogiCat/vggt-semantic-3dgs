"""
vggt_semantic – Semantic-geometric controller for compositional 3DGS.

Extends the VGGT (Visual Geometry Grounded Transformer) architecture with
three additional modules:

  Module 1 – SemanticTokenizer:
      Extracts 32-dim semantic tokens from multi-view images using a frozen
      DINOv2 backbone (or a lightweight placeholder for testing).

  Module 2 – SemanticGuidedAttention / SemanticGuidedBlock:
      Injects semantic patch-patch cosine-similarity into the transformer
      attention logits so that geometrically close but semantically distinct
      regions can be treated differently.

  Module 3 – SemanticHead:
      A lightweight decoder head that turns deep transformer tokens into a
      dense 32-dim semantic feature map [B, S, 32, H, W].

The main entry-point is VGGTSemantic in vggt_semantic.models.vggt_semantic.
"""

from vggt_semantic.models.vggt_semantic import VGGTSemantic
from vggt_semantic.config import SemanticConfig

__all__ = ["VGGTSemantic", "SemanticConfig"]
