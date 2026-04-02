"""
Setup for vggt_semantic package.
"""

from setuptools import setup, find_packages

setup(
    name="vggt_semantic",
    version="0.1.0",
    description="VGGT as semantic-geometric controller for compositional 3DGS",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "Pillow",
        "einops",
    ],
    extras_require={
        "dinov2": ["transformers>=4.38.0"],
        "dev": ["pytest"],
    },
)
