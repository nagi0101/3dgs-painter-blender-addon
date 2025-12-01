"""
NPR Core: Framework-independent gaussian painting engine

This module contains all core logic for 3D Gaussian Splatting painting,
independent of Blender. It can be tested and used standalone.

Core modules:
- gaussian: Gaussian2D data structure
- scene_data: High-performance array-based scene representation
- brush: BrushStamp and painting logic
- brush_manager: Brush library management
- spline: StrokeSpline for path interpolation
- deformation: Non-rigid deformation (CPU)
- deformation_gpu: GPU-accelerated deformation (CUDA)
- api: Synchronous API for integration
- gpu_context: GPU context management for Blender integration
"""

__version__ = "0.1.0"

# Core data structures
from .gaussian import Gaussian2D, create_test_gaussian
from .scene_data import SceneData
from .brush import BrushStamp, StrokePainter
from .brush_manager import BrushManager, get_brush_manager

# API
from .api import NPRCoreAPI

__all__ = [
    "Gaussian2D",
    "create_test_gaussian",
    "SceneData",
    "BrushStamp",
    "StrokePainter",
    "BrushManager",
    "get_brush_manager",
    "NPRCoreAPI",
]
