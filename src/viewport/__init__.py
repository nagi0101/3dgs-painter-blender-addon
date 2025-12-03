# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
Viewport rendering module for 3DGS Painter.

This module implements GLSL-based instanced rendering for real-time
Gaussian splatting visualization in Blender's 3D viewport.

Architecture:
- GaussianDataManager: Manages GPU texture data (59-float stride format)
- ViewportRenderer: GLSL shader-based rendering with draw handler
- ShaderLoader: Utility for loading and compiling GLSL shaders

Performance Target: 60 FPS @ 10,000 gaussians
"""

from .gaussian_data import GaussianDataManager
from .viewport_renderer import (
    GaussianViewportRenderer,
    register_viewport_operators,
    unregister_viewport_operators,
)
from .shader_loader import load_shader, get_shader_path
from .panels import register_panels, unregister_panels
from .benchmark import ViewportBenchmark, register_benchmark, unregister_benchmark

__all__ = [
    "GaussianDataManager",
    "GaussianViewportRenderer",
    "load_shader",
    "get_shader_path",
    "register_viewport_operators",
    "unregister_viewport_operators",
    "register_panels",
    "unregister_panels",
    "ViewportBenchmark",
    "register_benchmark",
    "unregister_benchmark",
]
