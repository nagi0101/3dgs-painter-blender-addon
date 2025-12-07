# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
GLSL-based viewport renderer for Gaussian splatting.

Implements real-time rendering using instanced GLSL shaders with
draw handler integration into Blender's 3D viewport.

Performance Target: 60 FPS @ 10,000 gaussians
"""

import bpy
import gpu
from gpu.types import GPUShader, GPUShaderCreateInfo, GPUStageInterfaceInfo
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from typing import Optional, Tuple, Set
import numpy as np

from .gaussian_data import GaussianDataManager, FLOATS_PER_GAUSSIAN


class GaussianViewportRenderer:
    """
    GLSL-based viewport renderer for Gaussian splatting.
    
    Uses instanced rendering with a custom shader to render gaussians
    as billboard quads with proper depth integration.
    
    Attributes:
        data_manager: GaussianDataManager instance
        shader: Compiled GPU shader
        draw_handle: Blender draw handler reference
        enabled: Whether rendering is active
    """
    
    # Class-level singleton instance
    _instance: Optional["GaussianViewportRenderer"] = None
    
    def __init__(self):
        self.data_manager = GaussianDataManager()
        self.shader: Optional[gpu.types.GPUShader] = None
        self.batch: Optional[gpu.types.GPUBatch] = None
        self.draw_handle = None
        self.enabled: bool = False
        self._depth_texture: Optional[gpu.types.GPUTexture] = None
        self._registered_spaces: Set[int] = set()
        
        # Rendering settings
        self.use_depth_test: bool = True
        self.depth_bias: float = 0.0001
        
    @classmethod
    def get_instance(cls) -> "GaussianViewportRenderer":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = GaussianViewportRenderer()
        return cls._instance
    
    @classmethod
    def destroy_instance(cls):
        """Destroy the singleton instance and cleanup."""
        if cls._instance is not None:
            cls._instance.unregister()
            cls._instance = None
    
    def _compile_shader(self) -> bool:
        """
        Compile GLSL shaders using GPUShaderCreateInfo.
        
        Implements proper 3D Gaussian Splatting with:
        - 3D covariance computation from rotation quaternion and scale
        - 3D to 2D covariance projection using Jacobian
        - Proper elliptical Gaussian evaluation in fragment shader
        
        Note: Push constants have a 128-byte limit, so we minimize uniforms.
        
        Returns:
            True if compilation successful
        """
        try:
            # Define vertex-fragment interface
            interface = GPUStageInterfaceInfo("gaussian_interface")
            interface.smooth('VEC4', "vColor")       # RGB + opacity
            interface.smooth('VEC3', "vConic")       # Inverse 2D covariance (a, b, c)
            interface.smooth('VEC2', "vCoordXY")     # Offset from gaussian center in pixels
            
            # Create shader info
            shader_info = GPUShaderCreateInfo()
            
            # Vertex inputs
            shader_info.vertex_in(0, 'VEC2', "position")
            
            # Push constants (must stay under 128 bytes)
            # MAT4 = 64 bytes, VEC4 = 16 bytes, VEC2 = 8 bytes, INT = 4 bytes
            shader_info.push_constant('MAT4', "viewProjectionMatrix")  # 64 bytes
            # Note: We exceed 128 bytes but Blender/GPU drivers typically allow more
            # If issues arise, we can pack data into textures instead
            shader_info.push_constant('MAT4', "viewMatrix")            # 64 bytes - for covariance transform
            shader_info.push_constant('VEC4', "camPosAndFocalX")       # 16 bytes (xyz=pos, w=focalX)
            shader_info.push_constant('VEC4', "viewportAndFocalY")     # 16 bytes (xy=viewport, z=focalY, w=texWidth)
            shader_info.push_constant('INT', "gaussianCount")          # 4 bytes
            
            # Samplers
            shader_info.sampler(0, 'FLOAT_2D', "gaussianData")
            
            # Interface
            shader_info.vertex_out(interface)
            
            # Fragment output
            shader_info.fragment_out(0, 'VEC4', "fragColor")
            
            # Vertex shader source - Full Gaussian Splatting implementation
            vert_source = '''
// Spherical Harmonics constants
#define SH_C0 0.28209479177387814

// Helper to fetch float from packed texture with row wrapping
float fetchFloat(int baseIdx, int offset, int textureWidth) {
    int idx = baseIdx + offset;
    int y = idx / textureWidth;
    int x = idx - y * textureWidth;
    return texelFetch(gaussianData, ivec2(x, y), 0).r;
}

// Build rotation matrix from quaternion (w, x, y, z)
// IMPORTANT: GLSL mat3 uses COLUMN-MAJOR order!
// mat3(a,b,c, d,e,f, g,h,i) creates matrix where:
//   Column 0 = [a,b,c], Column 1 = [d,e,f], Column 2 = [g,h,i]
// So to get the standard rotation matrix R:
//   | R00  R01  R02 |
//   | R10  R11  R12 |
//   | R20  R21  R22 |
// We need: mat3(R00,R10,R20, R01,R11,R21, R02,R12,R22)
mat3 quatToMat(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    
    return mat3(
        // Column 0
        1.0 - 2.0*(yy + zz),
        2.0*(xy + wz),
        2.0*(xz - wy),
        // Column 1
        2.0*(xy - wz),
        1.0 - 2.0*(xx + zz),
        2.0*(yz + wx),
        // Column 2
        2.0*(xz + wy),
        2.0*(yz - wx),
        1.0 - 2.0*(xx + yy)
    );
}

// Compute 3D covariance from scale and rotation
mat3 computeCov3D(vec3 scale, vec4 rot) {
    mat3 R = quatToMat(rot);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 RS = R * S;
    return RS * transpose(RS);
}

// Project 3D covariance to 2D using Jacobian
vec3 computeCov2D(vec3 mean, mat3 cov3D, float focalX, float focalY, vec2 viewport) {
    float z = mean.z;
    float z2 = z * z;
    
    // Jacobian of perspective projection
    // IMPORTANT: GLSL mat3 uses COLUMN-MAJOR order!
    // The constructor takes columns, not rows:
    // mat3(col0, col1, col2) where each col is vec3
    // We want the Jacobian:
    // | focalX/z    0          -focalX*x/z² |
    // | 0           focalY/z   -focalY*y/z² |
    // | 0           0          0            |
    mat3 J = mat3(
        focalX / z, 0.0, 0.0,                                      // Column 0
        0.0, focalY / z, 0.0,                                      // Column 1
        -focalX * mean.x / z2, -focalY * mean.y / z2, 0.0          // Column 2
    );
    
    mat3 cov2D = J * cov3D * transpose(J);
    
    // Add low-pass filter for anti-aliasing
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;
    
    return vec3(cov2D[0][0], cov2D[0][1], cov2D[1][1]);
}

void main() {
    int gaussianIndex = gl_InstanceID;
    
    if (gaussianIndex >= gaussianCount) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }
    
    // Unpack uniforms
    vec3 camPos = camPosAndFocalX.xyz;
    float focalX = camPosAndFocalX.w;
    vec2 viewport = viewportAndFocalY.xy;
    float focalY = viewportAndFocalY.z;
    int textureWidth = int(viewportAndFocalY.w);
    int floatsPerGaussian = 59;
    
    int baseIdx = gaussianIndex * floatsPerGaussian;
    
    // Fetch gaussian data
    vec3 gPos = vec3(
        fetchFloat(baseIdx, 0, textureWidth),
        fetchFloat(baseIdx, 1, textureWidth),
        fetchFloat(baseIdx, 2, textureWidth)
    );
    
    vec4 gRot = vec4(
        fetchFloat(baseIdx, 3, textureWidth),  // w
        fetchFloat(baseIdx, 4, textureWidth),  // x
        fetchFloat(baseIdx, 5, textureWidth),  // y
        fetchFloat(baseIdx, 6, textureWidth)   // z
    );
    
    vec3 gScale = vec3(
        fetchFloat(baseIdx, 7, textureWidth),
        fetchFloat(baseIdx, 8, textureWidth),
        fetchFloat(baseIdx, 9, textureWidth)
    );
    
    float gOpacity = fetchFloat(baseIdx, 10, textureWidth);
    
    // SH degree 0 color
    vec3 gColor = vec3(
        fetchFloat(baseIdx, 11, textureWidth) * SH_C0 + 0.5,
        fetchFloat(baseIdx, 12, textureWidth) * SH_C0 + 0.5,
        fetchFloat(baseIdx, 13, textureWidth) * SH_C0 + 0.5
    );
    
    // Transform to clip space
    vec4 posClip = viewProjectionMatrix * vec4(gPos, 1.0);
    
    // Early frustum culling
    if (posClip.w <= 0.0 || abs(posClip.x/posClip.w) > 1.3 || abs(posClip.y/posClip.w) > 1.3) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }
    
    // Compute view-space position using the view matrix
    vec4 posViewFull = viewMatrix * vec4(gPos, 1.0);
    vec3 posView = posViewFull.xyz;
    
    // Compute 3D covariance in WORLD space
    mat3 cov3D_world = computeCov3D(gScale, gRot);
    
    // Transform covariance to VIEW space: cov3D_view = V * cov3D_world * V^T
    // where V is the 3x3 rotation part of the view matrix
    mat3 V = mat3(viewMatrix);
    mat3 cov3D_view = V * cov3D_world * transpose(V);
    
    // Project to 2D covariance (now correctly in view space)
    vec3 cov2D = computeCov2D(posView, cov3D_view, focalX, focalY, viewport);
    
    // Compute inverse covariance (conic)
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }
    float detInv = 1.0 / det;
    vec3 conic = vec3(
        cov2D.z * detInv,   // a
        -cov2D.y * detInv,  // b
        cov2D.x * detInv    // c
    );
    
    // Compute quad extent (3-sigma)
    float maxRadius = 3.0 * sqrt(max(cov2D.x, cov2D.z));
    maxRadius = clamp(maxRadius, 1.0, 1000.0);  // Clamp to reasonable range
    
    // Convert to NDC
    vec2 quadExtentNDC = vec2(maxRadius) / viewport * 2.0;
    
    // Billboard quad offset
    vec2 quadOffset = (position - 0.5) * 2.0;
    posClip.xy += quadOffset * quadExtentNDC * posClip.w;
    
    gl_Position = posClip;
    
    // Output to fragment shader
    vColor = vec4(clamp(gColor, 0.0, 1.0), gOpacity);
    vConic = conic;
    vCoordXY = quadOffset * maxRadius;  // Pixel offset from center
}
'''
            
            # Fragment shader source - Proper Gaussian evaluation
            frag_source = '''
void main() {
    // Evaluate 2D Gaussian: exp(-0.5 * x^T * conic * x)
    // conic = inverse covariance = [[a, b], [b, c]]
    float power = -0.5 * (
        vConic.x * vCoordXY.x * vCoordXY.x +
        vConic.z * vCoordXY.y * vCoordXY.y +
        2.0 * vConic.y * vCoordXY.x * vCoordXY.y
    );
    
    // Discard if outside 3-sigma
    if (power > 0.0) {
        discard;
    }
    
    float alpha = vColor.a * exp(power);
    
    // Discard nearly transparent fragments (1/255 threshold)
    if (alpha < 0.004) {
        discard;
    }
    
    alpha = min(alpha, 0.99);
    
    fragColor = vec4(vColor.rgb * alpha, alpha);
}
'''
            
            shader_info.vertex_source(vert_source)
            shader_info.fragment_source(frag_source)
            
            # Create shader from info
            self.shader = gpu.shader.create_from_info(shader_info)
            
            print("[ViewportRenderer] Shader compiled successfully (Full Gaussian Splatting)")
            return True
            
        except Exception as e:
            print(f"[ViewportRenderer] Shader compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_batch(self) -> bool:
        """
        Create GPU batch for instanced quad rendering.
        
        Each gaussian is rendered as a quad (4 vertices, triangle strip).
        
        Returns:
            True if batch creation successful
        """
        if self.shader is None:
            return False
        
        # Quad vertices (we use vertex ID in shader to determine position)
        # 4 vertices per instance: TL, TR, BL, BR
        vertices = [
            (0.0, 0.0),  # Vertex 0
            (1.0, 0.0),  # Vertex 1
            (0.0, 1.0),  # Vertex 2
            (1.0, 1.0),  # Vertex 3
        ]
        
        # Triangle strip indices: 0, 1, 2, 3
        indices = [(0, 1, 2), (2, 1, 3)]
        
        try:
            self.batch = batch_for_shader(
                self.shader,
                'TRIS',
                {"position": vertices},
                indices=indices
            )
            return True
        except Exception as e:
            print(f"[ViewportRenderer] Batch creation failed: {e}")
            return False
    
    def register(self) -> bool:
        """
        Register draw handler for viewport rendering.
        
        Returns:
            True if registration successful
        """
        if self.draw_handle is not None:
            return True  # Already registered
        
        # Compile shader if needed
        if self.shader is None:
            if not self._compile_shader():
                return False
        
        # Create batch if needed
        if self.batch is None:
            if not self._create_batch():
                return False
        
        # Register draw handler
        self.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback,
            (),
            'WINDOW',
            'POST_VIEW'
        )
        
        self.enabled = True
        print("[ViewportRenderer] Draw handler registered")
        return True
    
    def unregister(self):
        """Unregister draw handler and cleanup."""
        if self.draw_handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                self.draw_handle,
                'WINDOW'
            )
            self.draw_handle = None
        
        self.enabled = False
        self.shader = None
        self.batch = None
        self._depth_texture = None
        
        print("[ViewportRenderer] Draw handler unregistered")
    
    def _get_camera_params(self, context) -> Tuple[Matrix, Matrix, Vector, Tuple[float, float], Tuple[int, int]]:
        """
        Extract camera parameters from Blender context.
        
        VR Compatible: Uses gpu.matrix for automatic per-eye matrix support
        in VR stereo rendering mode.
        
        Returns:
            Tuple of (view_matrix, projection_matrix, camera_position, 
                     focal_length, viewport_size)
        """
        region = context.region
        
        # ============================================================
        # VR STEREO RENDERING FIX
        # ============================================================
        # Use gpu.matrix instead of region_3d for VR compatibility.
        # gpu.matrix functions automatically provide correct per-eye
        # matrices when rendering in VR mode (left eye / right eye).
        # In regular viewport mode, these return the same as region_3d.
        # ============================================================
        
        # Get matrices from GPU state (VR-aware)
        view_matrix = gpu.matrix.get_model_view_matrix()
        projection_matrix = gpu.matrix.get_projection_matrix()
        
        # Camera position (inverse of view matrix translation)
        view_inv = view_matrix.inverted()
        camera_position = view_inv.translation.copy()
        
        # Viewport size
        if region:
            viewport_size = (region.width, region.height)
        else:
            # Fallback for VR rendering where region might not be available
            viewport_size = (1920, 1080)
        
        # Estimate focal length from projection matrix
        # projection_matrix[0][0] = 2 * near / (right - left) ≈ 2 * f_x / width
        # projection_matrix[1][1] = 2 * near / (top - bottom) ≈ 2 * f_y / height
        focal_x = abs(projection_matrix[0][0]) * viewport_size[0] / 2.0
        focal_y = abs(projection_matrix[1][1]) * viewport_size[1] / 2.0
        focal_length = (focal_x, focal_y)
        
        return view_matrix, projection_matrix, camera_position, focal_length, viewport_size
    
    def _draw_callback(self):
        """
        Main draw callback (called every frame).
        
        This is registered with Blender's draw handler system.
        """
        # Check if we have data to render
        if not self.data_manager.is_valid:
            return
        
        if self.shader is None or self.batch is None:
            return
        
        # Get context
        context = bpy.context
        
        # ============================================================
        # VR COMPATIBILITY NOTE
        # ============================================================
        # In VR mode, draw callbacks are invoked in an offscreen context
        # where area/region might be None or different than expected.
        # We must NOT early-return in VR mode.
        # ============================================================
        
        # Check if we're in VR mode (skip strict area/region checks)
        is_vr_active = (
            hasattr(context.window_manager, 'xr_session_state') and 
            context.window_manager.xr_session_state is not None
        )
        
        # For non-VR mode, perform safety checks
        if not is_vr_active:
            if context.area is None or context.area.type != 'VIEW_3D':
                return
            
            if context.space_data is None or context.region is None:
                return
        
        # Get camera parameters (uses gpu.matrix for VR compatibility)
        try:
            view_matrix, proj_matrix, cam_pos, focal, viewport = self._get_camera_params(context)
        except Exception as e:
            # In VR, silently continue even if there's an error
            if not is_vr_active:
                print(f"[ViewportRenderer] Camera params error: {e}")
            return
        
        # Get texture info
        tex_info = self.data_manager.get_texture_info()
        
        # Compute combined view-projection matrix
        view_proj_matrix = proj_matrix @ view_matrix
        
        # Set GPU state for alpha blending
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)  # Don't write to depth buffer
        gpu.state.blend_set('ALPHA_PREMULT')
        gpu.state.face_culling_set('NONE')  # Billboards need no culling
        
        try:
            # Bind shader
            self.shader.bind()
            
            # Set uniforms - combined view-projection matrix
            self.shader.uniform_float("viewProjectionMatrix", view_proj_matrix)
            
            # View matrix (for covariance transformation to view space)
            self.shader.uniform_float("viewMatrix", view_matrix)
            
            # Pack camera position (xyz) and focal_x (w) into vec4
            self.shader.uniform_float("camPosAndFocalX", (
                cam_pos.x, cam_pos.y, cam_pos.z, focal[0]
            ))
            
            # Pack viewport (xy), focal_y (z), texture_width (w) into vec4
            self.shader.uniform_float("viewportAndFocalY", (
                float(viewport[0]),  # width
                float(viewport[1]),  # height  
                focal[1],            # focal_y
                float(tex_info["texture_width"])  # texture width
            ))
            
            # Gaussian count
            self.shader.uniform_int("gaussianCount", (tex_info["gaussian_count"],))
            
            # Bind gaussian data texture
            texture = self.data_manager.get_texture()
            if texture is not None:
                self.shader.uniform_sampler("gaussianData", texture)
            
            # Draw instances using draw_instanced
            # Each gaussian = 1 instance = 2 triangles (4 vertices)
            gaussian_count = tex_info["gaussian_count"]
            if gaussian_count > 0:
                self.batch.draw_instanced(self.shader, instance_count=gaussian_count)
            
        except Exception as e:
            print(f"[ViewportRenderer] Draw error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore GPU state
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set('NONE')
    
    def update_gaussians(self, scene_data=None, gaussians=None) -> bool:
        """
        Update gaussian data for rendering.
        
        Args:
            scene_data: SceneData instance, or
            gaussians: List of Gaussian2D objects
            
        Returns:
            True if update successful
        """
        if scene_data is not None:
            return self.data_manager.update_from_scene_data(scene_data)
        elif gaussians is not None:
            return self.data_manager.update_from_gaussians(gaussians)
        else:
            return False
    
    def append_gaussians(self, scene_data=None, gaussians=None) -> bool:
        """
        Append new gaussians without full rebuild.
        
        Args:
            scene_data: SceneData with new gaussians, or
            gaussians: List of new Gaussian2D objects
            
        Returns:
            True if append successful
        """
        return self.data_manager.append_gaussians(
            scene_data=scene_data,
            gaussians=gaussians
        )
    
    def clear(self):
        """Clear all gaussian data."""
        self.data_manager.clear()
    
    def set_depth_testing(self, enabled: bool, bias: float = 0.0001):
        """
        Configure depth testing with Blender scene.
        
        Args:
            enabled: Whether to test against scene depth
            bias: Depth bias to prevent z-fighting
        """
        self.use_depth_test = enabled
        self.depth_bias = bias
    
    def request_redraw(self):
        """Request viewport redraw."""
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    
    @property
    def gaussian_count(self) -> int:
        """Get current gaussian count."""
        return self.data_manager.gaussian_count


# Blender Operators for viewport control

class NPR_OT_EnableViewportRendering(bpy.types.Operator):
    """Enable Gaussian splatting viewport rendering"""
    bl_idname = "npr.enable_viewport_rendering"
    bl_label = "Enable Gaussian Rendering"
    bl_description = "Enable real-time Gaussian splatting in the viewport"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        renderer = GaussianViewportRenderer.get_instance()
        
        if renderer.register():
            self.report({'INFO'}, "Viewport rendering enabled")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to enable viewport rendering")
            return {'CANCELLED'}


class NPR_OT_DisableViewportRendering(bpy.types.Operator):
    """Disable Gaussian splatting viewport rendering"""
    bl_idname = "npr.disable_viewport_rendering"
    bl_label = "Disable Gaussian Rendering"
    bl_description = "Disable real-time Gaussian splatting in the viewport"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        renderer = GaussianViewportRenderer.get_instance()
        renderer.unregister()
        
        self.report({'INFO'}, "Viewport rendering disabled")
        return {'FINISHED'}


class NPR_OT_ToggleViewportRendering(bpy.types.Operator):
    """Toggle Gaussian splatting viewport rendering"""
    bl_idname = "npr.toggle_viewport_rendering"
    bl_label = "Toggle Gaussian Rendering"
    bl_description = "Toggle real-time Gaussian splatting in the viewport"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        renderer = GaussianViewportRenderer.get_instance()
        
        if renderer.enabled:
            renderer.unregister()
            self.report({'INFO'}, "Viewport rendering disabled")
        else:
            if renderer.register():
                self.report({'INFO'}, "Viewport rendering enabled")
            else:
                self.report({'ERROR'}, "Failed to enable viewport rendering")
                return {'CANCELLED'}
        
        return {'FINISHED'}


class NPR_OT_ClearGaussians(bpy.types.Operator):
    """Clear all gaussians from viewport"""
    bl_idname = "npr.clear_gaussians"
    bl_label = "Clear Gaussians"
    bl_description = "Remove all gaussians from the viewport"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        renderer = GaussianViewportRenderer.get_instance()
        renderer.clear()
        renderer.request_redraw()
        
        self.report({'INFO'}, "Gaussians cleared")
        return {'FINISHED'}


class NPR_OT_GenerateTestGaussians(bpy.types.Operator):
    """Generate test gaussians for debugging"""
    bl_idname = "npr.generate_test_gaussians"
    bl_label = "Generate Test Gaussians"
    bl_description = "Generate random test gaussians for viewport testing"
    bl_options = {'REGISTER'}
    
    count: bpy.props.IntProperty(
        name="Count",
        description="Number of gaussians to generate",
        default=1000,
        min=1,
        max=500000
    )
    
    def execute(self, context):
        from .gaussian_data import create_test_data
        
        renderer = GaussianViewportRenderer.get_instance()
        
        # Ensure rendering is enabled
        if not renderer.enabled:
            if not renderer.register():
                self.report({'ERROR'}, "Failed to enable viewport rendering")
                return {'CANCELLED'}
        
        # Generate test data
        test_data = create_test_data(self.count)
        
        # Create a minimal scene data mock
        class MockSceneData:
            def __init__(self, data):
                n = data.shape[0]
                self.count = n
                self.positions = data[:, 0:3]
                # Reorder quaternion from (w,x,y,z) to (x,y,z,w) for SceneData format
                self.rotations = np.zeros((n, 4), dtype=np.float32)
                self.rotations[:, 0:3] = data[:, 4:7]  # x, y, z
                self.rotations[:, 3] = data[:, 3]       # w
                self.scales = data[:, 7:10]
                self.opacities = data[:, 10]
                # Convert SH back to color
                SH_C0 = 0.28209479177387814
                self.colors = data[:, 11:14] * SH_C0 + 0.5
        
        mock_scene = MockSceneData(test_data)
        
        if renderer.update_gaussians(scene_data=mock_scene):
            renderer.request_redraw()
            self.report({'INFO'}, f"Generated {self.count} test gaussians")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to update gaussians")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


# Operator classes to register
viewport_operator_classes = [
    NPR_OT_EnableViewportRendering,
    NPR_OT_DisableViewportRendering,
    NPR_OT_ToggleViewportRendering,
    NPR_OT_ClearGaussians,
    NPR_OT_GenerateTestGaussians,
]


def register_viewport_operators():
    """Register viewport-related operators."""
    for cls in viewport_operator_classes:
        bpy.utils.register_class(cls)


def unregister_viewport_operators():
    """Unregister viewport-related operators."""
    # Cleanup renderer
    GaussianViewportRenderer.destroy_instance()
    
    # Unregister operators
    for cls in reversed(viewport_operator_classes):
        bpy.utils.unregister_class(cls)
