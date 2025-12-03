# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
Shader loading utilities for GLSL shaders.
"""

import os
from pathlib import Path
from typing import Optional, Tuple


def get_shader_path(shader_name: str) -> Path:
    """
    Get the full path to a shader file.
    
    Args:
        shader_name: Name of the shader file (e.g., 'gaussian_vert.glsl')
        
    Returns:
        Path object pointing to the shader file
    """
    # Get the directory containing this module
    module_dir = Path(__file__).parent.parent
    shader_dir = module_dir / "shaders"
    return shader_dir / shader_name


def load_shader(shader_name: str) -> str:
    """
    Load shader source code from file.
    
    Args:
        shader_name: Name of the shader file
        
    Returns:
        Shader source code as string
        
    Raises:
        FileNotFoundError: If shader file doesn't exist
    """
    shader_path = get_shader_path(shader_name)
    
    if not shader_path.exists():
        raise FileNotFoundError(f"Shader file not found: {shader_path}")
    
    with open(shader_path, "r", encoding="utf-8") as f:
        return f.read()


def load_shader_pair(vert_name: str, frag_name: str) -> Tuple[str, str]:
    """
    Load a vertex and fragment shader pair.
    
    Args:
        vert_name: Vertex shader filename
        frag_name: Fragment shader filename
        
    Returns:
        Tuple of (vertex_source, fragment_source)
    """
    vert_source = load_shader(vert_name)
    frag_source = load_shader(frag_name)
    return vert_source, frag_source


def check_shaders_exist() -> bool:
    """
    Check if required shader files exist.
    
    Returns:
        True if all required shaders exist
    """
    required_shaders = [
        "gaussian_vert.glsl",
        "gaussian_frag.glsl",
    ]
    
    for shader_name in required_shaders:
        if not get_shader_path(shader_name).exists():
            return False
    
    return True
