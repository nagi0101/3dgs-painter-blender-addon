# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
VR Mesh Renderer - Renders Gaussians as actual 3D mesh objects for VR visibility.

Since Blender's draw handler doesn't show in VR headset, we create real mesh
objects (ellipsoids/spheres) that render as part of the scene.
"""

import bpy
import bmesh
from mathutils import Vector, Quaternion, Matrix
import numpy as np
from typing import Optional, List, Dict
import time


class VRMeshGaussianManager:
    """
    Manages 3D mesh objects representing Gaussians for VR rendering.
    
    Creates actual mesh objects instead of using GPU draw handlers,
    ensuring visibility in VR headsets.
    """
    
    _instance: Optional["VRMeshGaussianManager"] = None
    
    # Collection name for VR gaussian meshes
    COLLECTION_NAME = "VR_Gaussians"
    
    # Mesh template name
    TEMPLATE_MESH_NAME = "_gaussian_template"
    
    # Material name
    MATERIAL_NAME = "VR_Gaussian_Mat"
    
    def __init__(self):
        self._gaussian_objects: List[bpy.types.Object] = []
        self._template_mesh: Optional[bpy.types.Mesh] = None
        self._material: Optional[bpy.types.Material] = None
        self._collection: Optional[bpy.types.Collection] = None
        self._object_pool: List[bpy.types.Object] = []  # Reusable objects
        self._next_id: int = 0
        
    @classmethod
    def get_instance(cls) -> "VRMeshGaussianManager":
        if cls._instance is None:
            cls._instance = VRMeshGaussianManager()
        return cls._instance
    
    @classmethod
    def destroy_instance(cls):
        if cls._instance is not None:
            cls._instance.cleanup()
            cls._instance = None
    
    def _ensure_collection(self) -> bpy.types.Collection:
        """Ensure the VR Gaussians collection exists."""
        if self._collection is not None and self._collection.name in bpy.data.collections:
            return self._collection
        
        # Check if collection exists
        if self.COLLECTION_NAME in bpy.data.collections:
            self._collection = bpy.data.collections[self.COLLECTION_NAME]
        else:
            # Create new collection
            self._collection = bpy.data.collections.new(self.COLLECTION_NAME)
            bpy.context.scene.collection.children.link(self._collection)
        
        return self._collection
    
    def _create_template_mesh(self) -> bpy.types.Mesh:
        """Create a low-poly UV sphere template for gaussians."""
        if self._template_mesh is not None and self._template_mesh.name in bpy.data.meshes:
            return self._template_mesh
        
        # Create icosphere (lower poly than UV sphere)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=1,  # Low poly for performance
            radius=1.0
        )
        
        # Create mesh data
        mesh = bpy.data.meshes.new(self.TEMPLATE_MESH_NAME)
        bm.to_mesh(mesh)
        bm.free()
        
        self._template_mesh = mesh
        return mesh
    
    def _create_material(self) -> bpy.types.Material:
        """Create a simple emission material for gaussians."""
        if self._material is not None and self._material.name in bpy.data.materials:
            return self._material
        
        # Check if material exists
        if self.MATERIAL_NAME in bpy.data.materials:
            self._material = bpy.data.materials[self.MATERIAL_NAME]
            return self._material
        
        # Create new material
        mat = bpy.data.materials.new(self.MATERIAL_NAME)
        mat.use_nodes = True
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Use Principled BSDF with emission
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (0, 0.8, 1.0, 1.0)  # Cyan
        principled.inputs['Roughness'].default_value = 0.5
        principled.inputs['Alpha'].default_value = 0.8
        
        # Try to set emission (Blender 4.x)
        if 'Emission Color' in principled.inputs:
            principled.inputs['Emission Color'].default_value = (0, 0.8, 1.0, 1.0)
            principled.inputs['Emission Strength'].default_value = 0.3
        elif 'Emission' in principled.inputs:
            principled.inputs['Emission'].default_value = (0, 0.24, 0.3, 1.0)
        
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Enable transparency
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'NONE'
        
        self._material = mat
        return mat
    
    def _get_or_create_object(self) -> bpy.types.Object:
        """Get an object from pool or create a new one."""
        # Check pool for available objects
        if self._object_pool:
            obj = self._object_pool.pop()
            obj.hide_viewport = False
            obj.hide_render = False
            return obj
        
        # Create new object
        collection = self._ensure_collection()
        template = self._create_template_mesh()
        material = self._create_material()
        
        # Create object with linked mesh (not copy, for memory efficiency)
        obj = bpy.data.objects.new(f"VR_Gauss_{self._next_id}", template)
        self._next_id += 1
        
        # Assign material
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
        
        # Add to collection
        collection.objects.link(obj)
        
        return obj
    
    def add_gaussian(
        self,
        position: Vector,
        rotation: Quaternion,
        scale: Vector,
        color: tuple = (0.0, 0.8, 1.0),
        opacity: float = 0.8
    ) -> bpy.types.Object:
        """
        Add a gaussian as a mesh object.
        
        Args:
            position: World position
            rotation: Orientation quaternion
            scale: Scale (x, y, z)
            color: RGB color (0-1)
            opacity: Alpha (0-1)
        
        Returns:
            Created Blender object
        """
        obj = self._get_or_create_object()
        
        # Set transform
        obj.location = position
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = rotation
        obj.scale = scale
        
        # Per-object color using viewport display color
        obj.color = (color[0], color[1], color[2], opacity)
        
        self._gaussian_objects.append(obj)
        return obj
    
    def add_gaussians_batch(self, gaussians: List[dict]) -> List[bpy.types.Object]:
        """
        Add multiple gaussians efficiently.
        
        Args:
            gaussians: List of dicts with position, rotation, scale, color, opacity
        
        Returns:
            List of created objects
        """
        objects = []
        
        for g in gaussians:
            obj = self.add_gaussian(
                position=g.get('position', Vector((0, 0, 0))),
                rotation=g.get('rotation', Quaternion()),
                scale=g.get('scale', Vector((0.01, 0.01, 0.01))),
                color=g.get('color', (0.0, 0.8, 1.0)),
                opacity=g.get('opacity', 0.8)
            )
            objects.append(obj)
        
        return objects
    
    def clear(self):
        """Remove all gaussian objects (return to pool for reuse)."""
        for obj in self._gaussian_objects:
            obj.hide_viewport = True
            obj.hide_render = True
            self._object_pool.append(obj)
        
        self._gaussian_objects.clear()
    
    def cleanup(self):
        """Full cleanup - remove all objects and data."""
        # Remove all objects
        for obj in self._gaussian_objects + self._object_pool:
            try:
                bpy.data.objects.remove(obj)
            except:
                pass
        
        self._gaussian_objects.clear()
        self._object_pool.clear()
        
        # Remove collection
        if self._collection and self._collection.name in bpy.data.collections:
            try:
                bpy.data.collections.remove(self._collection)
            except:
                pass
        
        # Remove template mesh
        if self._template_mesh and self._template_mesh.name in bpy.data.meshes:
            try:
                bpy.data.meshes.remove(self._template_mesh)
            except:
                pass
        
        self._collection = None
        self._template_mesh = None
    
    @property
    def count(self) -> int:
        return len(self._gaussian_objects)


# Singleton accessor
def get_vr_mesh_manager() -> VRMeshGaussianManager:
    return VRMeshGaussianManager.get_instance()
