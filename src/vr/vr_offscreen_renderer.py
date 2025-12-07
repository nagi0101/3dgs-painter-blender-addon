# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
VR Offscreen Gaussian Renderer - Phase 1

Renders Gaussians to an offscreen buffer and displays on a VR-visible Plane.
This is a fallback for VR since draw_handler doesn't work in VR Session.

Architecture:
    1. Create GPUOffScreen buffer
    2. Render Gaussians to offscreen buffer using existing GLSL shader
    3. Create a Plane in 3D scene
    4. Apply rendered texture to Plane material
    5. Plane is visible in VR (native Blender mesh rendering)
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix, Quaternion
import numpy as np
from typing import Optional, Tuple, List

# Singleton instance
_vr_offscreen_renderer: Optional["VROffscreenGaussianRenderer"] = None


def get_vr_offscreen_renderer() -> "VROffscreenGaussianRenderer":
    """Get or create the VR offscreen renderer singleton."""
    global _vr_offscreen_renderer
    if _vr_offscreen_renderer is None:
        _vr_offscreen_renderer = VROffscreenGaussianRenderer()
    return _vr_offscreen_renderer


def cleanup_vr_offscreen_renderer():
    """Cleanup the VR offscreen renderer singleton."""
    global _vr_offscreen_renderer
    if _vr_offscreen_renderer is not None:
        _vr_offscreen_renderer.cleanup()
        _vr_offscreen_renderer = None


class VROffscreenGaussianRenderer:
    """
    Renders Gaussians to an offscreen texture for VR display.
    
    Since Blender's draw_handler doesn't work in VR, we:
    1. Render to GPUOffScreen
    2. Display on a Plane that VR can see
    """
    
    # Display plane settings
    DISPLAY_PLANE_NAME = "VR_Gaussian_Display"
    DISPLAY_MATERIAL_NAME = "VR_Gaussian_Material"
    COLLECTION_NAME = "VR_Gaussians"
    
    def __init__(self):
        self.offscreen: Optional[gpu.types.GPUOffScreen] = None
        self.width = 1024
        self.height = 1024
        
        # Shader for simple gaussian rendering
        self.shader: Optional[gpu.types.GPUShader] = None
        self.batch: Optional[gpu.types.GPUBatch] = None
        
        # Gaussian data
        self.gaussian_positions: List[Vector] = []
        self.gaussian_colors: List[Tuple[float, float, float, float]] = []
        self.gaussian_sizes: List[float] = []
        
        # Display objects
        self.display_plane: Optional[bpy.types.Object] = None
        self.display_image: Optional[bpy.types.Image] = None
        
        self._initialized = False
    
    def initialize(self):
        """Initialize offscreen buffer and shader."""
        if self._initialized:
            return True
            
        try:
            # Create offscreen buffer
            self.offscreen = gpu.types.GPUOffScreen(self.width, self.height)
            
            # Compile simple point shader
            self._compile_shader()
            
            # Create display image for texture
            self._create_display_image()
            
            self._initialized = True
            print(f"[VR Offscreen] Initialized {self.width}x{self.height} buffer")
            return True
            
        except Exception as e:
            print(f"[VR Offscreen] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _compile_shader(self):
        """Compile simple Gaussian point shader."""
        
        # Simple point-based shader for quick visualization
        vert_src = """
        uniform mat4 viewProjectionMatrix;
        uniform float pointSize;
        
        in vec3 position;
        in vec4 color;
        
        out vec4 vColor;
        
        void main() {
            gl_Position = viewProjectionMatrix * vec4(position, 1.0);
            gl_PointSize = pointSize / gl_Position.w * 100.0;
            vColor = color;
        }
        """
        
        frag_src = """
        in vec4 vColor;
        out vec4 fragColor;
        
        void main() {
            // Simple circular point with soft edges
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = length(coord);
            float alpha = smoothstep(1.0, 0.5, dist) * vColor.a;
            fragColor = vec4(vColor.rgb, alpha);
        }
        """
        
        try:
            self.shader = gpu.types.GPUShader(vert_src, frag_src)
            print("[VR Offscreen] Shader compiled successfully")
        except Exception as e:
            print(f"[VR Offscreen] Shader compilation failed: {e}")
            # Use built-in shader as fallback
            self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    def _create_display_image(self):
        """Create Blender image for texture display."""
        image_name = "VR_Gaussian_Texture"
        
        # Remove existing image if any
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])
        
        # Create new image
        self.display_image = bpy.data.images.new(
            image_name,
            width=self.width,
            height=self.height,
            alpha=True,
            float_buffer=True
        )
        
        # Initialize with transparent black
        pixels = [0.0, 0.0, 0.0, 0.0] * (self.width * self.height)
        self.display_image.pixels = pixels
        
        print(f"[VR Offscreen] Created display image: {image_name}")
    
    def add_gaussian(self, position: Vector, color: Tuple[float, float, float], 
                     opacity: float = 0.8, size: float = 0.05):
        """Add a gaussian point for rendering."""
        self.gaussian_positions.append(position.copy())
        self.gaussian_colors.append((color[0], color[1], color[2], opacity))
        self.gaussian_sizes.append(size)
    
    def add_gaussians_batch(self, gaussians: List[dict]):
        """Add multiple gaussians from dict list."""
        for g in gaussians:
            pos = g.get('position', Vector((0, 0, 0)))
            if isinstance(pos, (list, tuple)):
                pos = Vector(pos)
            color = g.get('color', (0.0, 0.8, 1.0))
            opacity = g.get('opacity', 0.8)
            size = g.get('scale', (0.05, 0.05, 0.05))
            if isinstance(size, (list, tuple, Vector)):
                size = max(size[0], size[1], size[2])
            self.add_gaussian(pos, color, opacity, size)
    
    def clear(self):
        """Clear all gaussian data."""
        self.gaussian_positions.clear()
        self.gaussian_colors.clear()
        self.gaussian_sizes.clear()
    
    def render_to_texture(self, view_matrix: Matrix, proj_matrix: Matrix):
        """
        Render gaussians to offscreen texture.
        
        Args:
            view_matrix: View matrix (from VR head or camera)
            proj_matrix: Projection matrix
        """
        if not self._initialized:
            if not self.initialize():
                return
        
        if not self.gaussian_positions:
            return
        
        # Bind offscreen buffer
        self.offscreen.bind()
        
        try:
            # Clear with transparent black
            gpu.state.blend_set('ALPHA')
            
            # Build vertex data
            positions = [(p.x, p.y, p.z) for p in self.gaussian_positions]
            colors = self.gaussian_colors
            
            # Create batch
            batch = batch_for_shader(
                self.shader,
                'POINTS',
                {"position": positions, "color": colors}
            )
            
            # Calculate view-projection matrix
            view_proj = proj_matrix @ view_matrix
            
            # Draw
            self.shader.bind()
            self.shader.uniform_float("viewProjectionMatrix", view_proj)
            self.shader.uniform_float("pointSize", 10.0)
            
            batch.draw(self.shader)
            
        except Exception as e:
            print(f"[VR Offscreen] Render error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.offscreen.unbind()
    
    def copy_to_image(self):
        """Copy offscreen buffer to Blender image."""
        if self.offscreen is None or self.display_image is None:
            return
        
        try:
            # Read pixels from offscreen buffer
            buffer = self.offscreen.texture_color.read()
            
            # Convert to flat list
            buffer.dimensions = self.width * self.height * 4
            pixels = list(buffer)
            
            # Write to image
            self.display_image.pixels = pixels
            self.display_image.update()
            
        except Exception as e:
            print(f"[VR Offscreen] Image copy error: {e}")
    
    def create_display_plane(self, context, location: Vector = None, size: float = 2.0):
        """
        Create a plane to display the rendered texture in VR.
        
        Args:
            context: Blender context
            location: World location for the plane (default: in front of VR head)
            size: Size of the plane in meters
        """
        # Get or create collection
        collection = self._get_or_create_collection()
        
        # Remove existing plane if any
        if self.DISPLAY_PLANE_NAME in bpy.data.objects:
            old_plane = bpy.data.objects[self.DISPLAY_PLANE_NAME]
            bpy.data.objects.remove(old_plane, do_unlink=True)
        
        # Create plane mesh
        bpy.ops.mesh.primitive_plane_add(size=size)
        plane = context.active_object
        plane.name = self.DISPLAY_PLANE_NAME
        
        # Move to collection
        for col in plane.users_collection:
            col.objects.unlink(plane)
        collection.objects.link(plane)
        
        # Set location
        if location is None:
            # Default: 2m in front of origin, facing -Y
            location = Vector((0, -2, 1.5))
        plane.location = location
        
        # Rotate to face +Y (towards user)
        plane.rotation_euler = (1.5708, 0, 0)  # 90 degrees on X
        
        # Create and apply material
        self._create_display_material(plane)
        
        self.display_plane = plane
        print(f"[VR Offscreen] Created display plane at {location}")
        
        return plane
    
    def _get_or_create_collection(self):
        """Get or create the VR Gaussians collection."""
        if self.COLLECTION_NAME in bpy.data.collections:
            return bpy.data.collections[self.COLLECTION_NAME]
        
        collection = bpy.data.collections.new(self.COLLECTION_NAME)
        bpy.context.scene.collection.children.link(collection)
        return collection
    
    def _create_display_material(self, plane_obj):
        """Create material with offscreen texture."""
        # Remove existing material if any
        if self.DISPLAY_MATERIAL_NAME in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[self.DISPLAY_MATERIAL_NAME])
        
        # Create new material
        material = bpy.data.materials.new(self.DISPLAY_MATERIAL_NAME)
        material.use_nodes = True
        material.blend_method = 'BLEND'  # Enable transparency
        
        # Get node tree
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create texture node
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = self.display_image
        tex_node.location = (-300, 0)
        
        # Create emission shader (for unlit appearance)
        emission = nodes.new('ShaderNodeEmission')
        emission.location = (0, 0)
        
        # Create transparent shader
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        transparent.location = (0, -150)
        
        # Create mix shader
        mix = nodes.new('ShaderNodeMixShader')
        mix.location = (200, 0)
        
        # Create output
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (400, 0)
        
        # Link nodes
        links.new(tex_node.outputs['Color'], emission.inputs['Color'])
        links.new(tex_node.outputs['Alpha'], mix.inputs['Fac'])
        links.new(transparent.outputs['BSDF'], mix.inputs[1])
        links.new(emission.outputs['Emission'], mix.inputs[2])
        links.new(mix.outputs['Shader'], output.inputs['Surface'])
        
        # Assign to plane
        if plane_obj.data.materials:
            plane_obj.data.materials[0] = material
        else:
            plane_obj.data.materials.append(material)
        
        print(f"[VR Offscreen] Created display material")
    
    def update_display(self, context):
        """
        Update the display: render to texture and update image.
        
        Call this each frame to update the VR display.
        """
        if not self.gaussian_positions:
            return
        
        # Get view matrix from VR or camera
        view_matrix, proj_matrix = self._get_view_matrices(context)
        
        # Render to offscreen
        self.render_to_texture(view_matrix, proj_matrix)
        
        # Copy to image
        self.copy_to_image()
    
    def _get_view_matrices(self, context) -> Tuple[Matrix, Matrix]:
        """Get view and projection matrices."""
        # Try to get VR matrices first
        wm = context.window_manager
        if hasattr(wm, 'xr_session_state') and wm.xr_session_state is not None:
            xr = wm.xr_session_state
            try:
                # Get head pose
                head_pos = Vector(xr.viewer_pose_location)
                head_rot = Quaternion(xr.viewer_pose_rotation)
                
                # Build view matrix
                view_matrix = head_rot.to_matrix().to_4x4()
                view_matrix.translation = head_pos
                view_matrix = view_matrix.inverted()
                
                # Simple perspective projection
                proj_matrix = self._make_perspective_matrix(90, 1.0, 0.1, 100.0)
                
                return view_matrix, proj_matrix
                
            except Exception:
                pass
        
        # Fallback: use 3D view camera
        region_3d = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                region_3d = area.spaces[0].region_3d
                break
        
        if region_3d:
            return region_3d.view_matrix.copy(), region_3d.window_matrix.copy()
        
        # Ultimate fallback: identity matrices
        return Matrix.Identity(4), Matrix.Identity(4)
    
    def _make_perspective_matrix(self, fov_deg: float, aspect: float, 
                                  near: float, far: float) -> Matrix:
        """Create simple perspective projection matrix."""
        import math
        fov = math.radians(fov_deg)
        f = 1.0 / math.tan(fov / 2.0)
        
        mat = Matrix.Identity(4)
        mat[0][0] = f / aspect
        mat[1][1] = f
        mat[2][2] = (far + near) / (near - far)
        mat[2][3] = -1.0
        mat[3][2] = (2.0 * far * near) / (near - far)
        mat[3][3] = 0.0
        
        return mat
    
    def cleanup(self):
        """Cleanup all resources."""
        # Remove display plane
        if self.display_plane and self.display_plane.name in bpy.data.objects:
            bpy.data.objects.remove(self.display_plane, do_unlink=True)
        
        # Remove image
        if self.display_image and self.display_image.name in bpy.data.images:
            bpy.data.images.remove(self.display_image)
        
        # Remove material
        if self.DISPLAY_MATERIAL_NAME in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[self.DISPLAY_MATERIAL_NAME])
        
        # Free offscreen buffer
        if self.offscreen:
            self.offscreen.free()
            self.offscreen = None
        
        self.clear()
        self._initialized = False
        
        print("[VR Offscreen] Cleaned up")


# ============================================================
# Operators
# ============================================================

class THREEGDS_OT_VROffscreenTest(bpy.types.Operator):
    """Test VR offscreen rendering with sample data"""
    bl_idname = "threegds.vr_offscreen_test"
    bl_label = "Test VR Offscreen"
    bl_description = "Test offscreen rendering with sample gaussians"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        renderer = get_vr_offscreen_renderer()
        
        if not renderer.initialize():
            self.report({'ERROR'}, "Failed to initialize offscreen renderer")
            return {'CANCELLED'}
        
        # Add sample gaussians
        renderer.clear()
        import random
        for i in range(100):
            pos = Vector((
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(0, 2)
            ))
            color = (random.random(), random.random(), random.random())
            renderer.add_gaussian(pos, color, 0.8, 0.1)
        
        # Create display plane
        renderer.create_display_plane(context)
        
        # Update display
        renderer.update_display(context)
        
        self.report({'INFO'}, f"Created test display with {len(renderer.gaussian_positions)} gaussians")
        return {'FINISHED'}


class THREEGDS_OT_VROffscreenCleanup(bpy.types.Operator):
    """Cleanup VR offscreen resources"""
    bl_idname = "threegds.vr_offscreen_cleanup"
    bl_label = "Cleanup VR Offscreen"
    bl_description = "Remove VR offscreen display and resources"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        cleanup_vr_offscreen_renderer()
        self.report({'INFO'}, "VR offscreen resources cleaned up")
        return {'FINISHED'}


# ============================================================
# Registration
# ============================================================

classes = [
    THREEGDS_OT_VROffscreenTest,
    THREEGDS_OT_VROffscreenCleanup,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    cleanup_vr_offscreen_renderer()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
