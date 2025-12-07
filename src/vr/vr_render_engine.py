# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
VR Gaussian RenderEngine - Phase 2

Tests if bpy.types.RenderEngine.view_draw() is called during VR session.
If it is, we can render GLSL Gaussians directly to VR!

Usage:
1. Register this engine
2. Set active render engine to "VR_GAUSSIAN"
3. Start VR session
4. Check console for "view_draw called" messages
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
import numpy as np
from typing import Optional, List, Tuple


class VRGaussianRenderEngine(bpy.types.RenderEngine):
    """
    Custom RenderEngine for VR Gaussian rendering.
    
    If view_draw() is called in VR, we can render Gaussians directly!
    """
    
    bl_idname = "VR_GAUSSIAN"
    bl_label = "VR Gaussian"
    bl_use_preview = False
    bl_use_eevee_viewport = True  # Use Eevee as base for scene rendering
    
    # Class-level data storage
    _gaussian_positions: List[Vector] = []
    _gaussian_colors: List[Tuple[float, float, float, float]] = []
    _gaussian_sizes: List[float] = []
    _shader: Optional[gpu.types.GPUShader] = None
    _call_count: int = 0
    _vr_call_count: int = 0
    
    def __init__(self):
        super().__init__()
        print(f"[VR Gaussian Engine] Instance created")
    
    # ========================================
    # Required RenderEngine methods
    # ========================================
    
    def render(self, depsgraph):
        """Final render (F12). Not used for viewport."""
        pass
    
    def view_update(self, context, depsgraph):
        """
        Called when viewport needs update (scene changed).
        """
        pass
    
    def view_draw(self, context, depsgraph):
        """
        Called every frame for viewport drawing.
        
        THIS IS THE KEY METHOD - does it get called in VR?
        """
        VRGaussianRenderEngine._call_count += 1
        
        # Check if this is a VR context
        is_vr = self._is_vr_context(context)
        
        if is_vr:
            VRGaussianRenderEngine._vr_call_count += 1
            print(f"[VR Gaussian Engine] ★★★ view_draw CALLED IN VR! ★★★ (count: {VRGaussianRenderEngine._vr_call_count})")
        else:
            if VRGaussianRenderEngine._call_count % 60 == 0:  # Print every 60 frames
                print(f"[VR Gaussian Engine] view_draw called (PC viewport, count: {VRGaussianRenderEngine._call_count})")
        
        # Draw scene first (use Eevee)
        self._draw_scene(context, depsgraph)
        
        # Draw Gaussians on top
        self._draw_gaussians(context)
    
    def _is_vr_context(self, context) -> bool:
        """Check if we're rendering for VR headset."""
        # Check 1: XR session running
        wm = context.window_manager
        if hasattr(wm, 'xr_session_state') and wm.xr_session_state is not None:
            xr = wm.xr_session_state
            if xr.is_running(context):
                # Check 2: No standard region (VR uses offscreen)
                if context.region is None:
                    return True
                    
                # Check 3: Region type check
                if hasattr(context, 'region') and context.region:
                    if context.region.type == 'XR_SESSION':
                        return True
        
        return False
    
    def _draw_scene(self, context, depsgraph):
        """Draw the base scene (delegate to Eevee)."""
        # For now, just clear - proper scene drawing requires more work
        # In a real implementation, you'd composite with Eevee output
        pass
    
    def _draw_gaussians(self, context):
        """Draw Gaussian splats."""
        if not VRGaussianRenderEngine._gaussian_positions:
            return
        
        # Compile shader if needed
        if VRGaussianRenderEngine._shader is None:
            self._compile_shader()
        
        if VRGaussianRenderEngine._shader is None:
            return
        
        try:
            # Get matrices
            view_matrix = gpu.matrix.get_model_view_matrix()
            proj_matrix = gpu.matrix.get_projection_matrix()
            view_proj = proj_matrix @ view_matrix
            
            # Build vertex data
            positions = [(p.x, p.y, p.z) for p in VRGaussianRenderEngine._gaussian_positions]
            colors = VRGaussianRenderEngine._gaussian_colors
            
            # Create batch
            batch = batch_for_shader(
                VRGaussianRenderEngine._shader,
                'POINTS',
                {"position": positions, "color": colors}
            )
            
            # Set GPU state
            gpu.state.blend_set('ALPHA')
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.point_size_set(10.0)
            
            # Draw
            VRGaussianRenderEngine._shader.bind()
            VRGaussianRenderEngine._shader.uniform_float("viewProjectionMatrix", view_proj)
            VRGaussianRenderEngine._shader.uniform_float("pointSize", 20.0)
            
            batch.draw(VRGaussianRenderEngine._shader)
            
            # Reset state
            gpu.state.blend_set('NONE')
            
        except Exception as e:
            print(f"[VR Gaussian Engine] Draw error: {e}")
    
    def _compile_shader(self):
        """Compile simple point shader."""
        vert_src = """
        uniform mat4 viewProjectionMatrix;
        uniform float pointSize;
        
        in vec3 position;
        in vec4 color;
        
        out vec4 vColor;
        
        void main() {
            gl_Position = viewProjectionMatrix * vec4(position, 1.0);
            gl_PointSize = pointSize / max(gl_Position.w, 0.1) * 50.0;
            vColor = color;
        }
        """
        
        frag_src = """
        in vec4 vColor;
        out vec4 fragColor;
        
        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist = length(coord);
            if (dist > 1.0) discard;
            float alpha = smoothstep(1.0, 0.3, dist) * vColor.a;
            fragColor = vec4(vColor.rgb, alpha);
        }
        """
        
        try:
            VRGaussianRenderEngine._shader = gpu.types.GPUShader(vert_src, frag_src)
            print("[VR Gaussian Engine] Shader compiled")
        except Exception as e:
            print(f"[VR Gaussian Engine] Shader error: {e}")
    
    # ========================================
    # Class methods for data management
    # ========================================
    
    @classmethod
    def add_gaussian(cls, position: Vector, color: Tuple[float, float, float], 
                     opacity: float = 0.8, size: float = 0.05):
        """Add a gaussian point."""
        cls._gaussian_positions.append(position.copy())
        cls._gaussian_colors.append((color[0], color[1], color[2], opacity))
        cls._gaussian_sizes.append(size)
    
    @classmethod
    def add_gaussians_batch(cls, gaussians: List[dict]):
        """Add multiple gaussians."""
        for g in gaussians:
            pos = g.get('position', Vector((0, 0, 0)))
            if isinstance(pos, (list, tuple)):
                pos = Vector(pos)
            color = g.get('color', (0.0, 0.8, 1.0))
            opacity = g.get('opacity', 0.8)
            cls.add_gaussian(pos, color, opacity)
    
    @classmethod
    def clear(cls):
        """Clear all gaussian data."""
        cls._gaussian_positions.clear()
        cls._gaussian_colors.clear()
        cls._gaussian_sizes.clear()
        cls._call_count = 0
        cls._vr_call_count = 0
    
    @classmethod
    def get_stats(cls) -> str:
        """Get rendering statistics."""
        return f"Gaussians: {len(cls._gaussian_positions)}, Calls: {cls._call_count}, VR Calls: {cls._vr_call_count}"


# ============================================================
# Operators
# ============================================================

class THREEGDS_OT_VREngineTest(bpy.types.Operator):
    """Test VR RenderEngine with sample data"""
    bl_idname = "threegds.vr_engine_test"
    bl_label = "Test VR Engine"
    bl_description = "Add sample gaussians and switch to VR Gaussian engine"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        # Clear previous data
        VRGaussianRenderEngine.clear()
        
        # Add sample gaussians
        import random
        for i in range(200):
            pos = Vector((
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(0, 3)
            ))
            color = (random.random(), random.random(), random.random())
            VRGaussianRenderEngine.add_gaussian(pos, color, 0.8, 0.1)
        
        # Switch to VR Gaussian engine
        context.scene.render.engine = 'VR_GAUSSIAN'
        
        self.report({'INFO'}, f"VR Engine test: {VRGaussianRenderEngine.get_stats()}")
        self.report({'WARNING'}, "Now start VR session and check console for 'CALLED IN VR' messages!")
        return {'FINISHED'}


class THREEGDS_OT_VREngineStats(bpy.types.Operator):
    """Show VR Engine statistics"""
    bl_idname = "threegds.vr_engine_stats"
    bl_label = "VR Engine Stats"
    bl_description = "Show VR RenderEngine call statistics"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        stats = VRGaussianRenderEngine.get_stats()
        self.report({'INFO'}, stats)
        
        if VRGaussianRenderEngine._vr_call_count > 0:
            self.report({'WARNING'}, "★ VR calls detected! view_draw() WORKS in VR! ★")
        else:
            self.report({'INFO'}, "No VR calls yet. Start VR session with VR_GAUSSIAN engine active.")
        
        return {'FINISHED'}


class THREEGDS_OT_VREngineClear(bpy.types.Operator):
    """Clear VR Engine data"""
    bl_idname = "threegds.vr_engine_clear"
    bl_label = "Clear VR Engine"
    bl_description = "Clear all gaussian data from VR engine"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        VRGaussianRenderEngine.clear()
        self.report({'INFO'}, "VR Engine cleared")
        return {'FINISHED'}


class THREEGDS_OT_SwitchToEevee(bpy.types.Operator):
    """Switch back to Eevee"""
    bl_idname = "threegds.switch_to_eevee"
    bl_label = "Switch to Eevee"
    bl_description = "Switch back to Eevee render engine"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        self.report({'INFO'}, "Switched to Eevee")
        return {'FINISHED'}


# ============================================================
# Registration
# ============================================================

classes = [
    VRGaussianRenderEngine,
    THREEGDS_OT_VREngineTest,
    THREEGDS_OT_VREngineStats,
    THREEGDS_OT_VREngineClear,
    THREEGDS_OT_SwitchToEevee,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    print("[VR Gaussian Engine] Registered")


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    print("[VR Gaussian Engine] Unregistered")
