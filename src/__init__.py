bl_info = {
    "name": "3DGS Painter",
    "author": "Jiin Park",
    "description": "Non-photorealistic 3D Gaussian Splatting painting tools",
    "blender": (4, 2, 0),
    "version": (1, 0, 0),
    "location": "View3D > Sidebar > 3DGS Paint",
    "category": "Paint",
}

import bpy
from typing import List, Type

# Will hold all Blender classes to register
_classes: List[Type] = []


def register():
    """Register all Blender classes"""
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister all Blender classes"""
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
