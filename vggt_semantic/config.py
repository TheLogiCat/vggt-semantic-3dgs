"""
Configuration dataclasses for vggt_semantic.

All semantic-extension knobs live under SemanticConfig so they can be
toggled cleanly without touching the geometry branch.

Example YAML equivalent (configs/semantic_default.yaml):

    model:
      img_size: 518
      patch_size: 14
      embed_dim: 1024
      enable_camera: true
      enable_depth: true
      enable_point: true
      enable_track: false
      semantic:
        enabled: true
        dim: 32
        backbone: dinov2          # or "placeholder" for lightweight tests
        guidance:
          enabled: true
"""

from dataclasses import dataclass, field


@dataclass
class SemanticGuidanceConfig:
    """Controls Module-2 semantic-guided attention."""

    # Master switch: set to False to bypass semantic modulation entirely.
    enabled: bool = True


@dataclass
class SemanticConfig:
    """
    Top-level configuration for the semantic extension.

    Attributes
    ----------
    enabled : bool
        Master switch.  When False, VGGTSemantic behaves identically to the
        original VGGT (no additional parameters are active).
    dim : int
        Semantic feature dimension.  Hard-coded to 32 as specified in the
        project requirements; change only if you know what you are doing.
    backbone : str
        Which backbone to use inside SemanticTokenizer.
        "dinov2"      – frozen DINOv2-ViT-L/14 (default; requires the
                        transformers package and an internet connection on
                        first run)
        "placeholder" – lightweight conv-based stub for unit tests / CI
    guidance : SemanticGuidanceConfig
        Sub-config for Module-2 guidance attention.
    """

    enabled: bool = True
    dim: int = 32                         # fixed at 32 per spec
    backbone: str = "dinov2"             # "dinov2" | "placeholder"
    guidance: SemanticGuidanceConfig = field(default_factory=SemanticGuidanceConfig)


@dataclass
class VGGTSemanticConfig:
    """
    Full configuration for VGGTSemantic.

    Mirrors the constructor arguments of the base VGGT class with the
    semantic extension added as a nested config.
    """

    # ---- geometry (base VGGT) ----
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    patch_embed: str = "dinov2_vitl14_reg"
    enable_camera: bool = True
    enable_point: bool = True
    enable_depth: bool = True
    enable_track: bool = False

    # ---- semantic extension ----
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
