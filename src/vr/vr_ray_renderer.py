# vr_ray_renderer.py
# VR Controller Ray Visualization for painting feedback

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
import math


class VRRayRenderer:
    """Renders a laser ray from VR controller to show aiming direction."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._draw_handler = None
        self._enabled = False
        
        # Ray settings
        self.ray_length = 10.0  # meters
        self.ray_color = (0.0, 0.8, 1.0, 0.8)  # cyan
        self.ray_width = 2.0
        
        # Hit point settings
        self.hit_color = (1.0, 0.5, 0.0, 1.0)  # orange
        self.hit_size = 0.02  # meters
        
        # Brush preview settings
        self.brush_preview_color = (1.0, 1.0, 0.0, 0.5)  # yellow
        
        # Current state
        self._controller_pos = Vector((0, 0, 0))
        self._controller_dir = Vector((0, 0, -1))
        self._hit_point = None
        self._is_painting = False
        
        # Shader
        self._shader = None
        self._setup_shader()
    
    def _setup_shader(self):
        """Setup basic line shader."""
        try:
            self._shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        except:
            # Fallback for older Blender versions
            self._shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    @property
    def enabled(self):
        return self._enabled
    
    def register(self):
        """Register the draw handler for VR viewport."""
        if self._draw_handler is not None:
            return
        
        # Register for POST_VIEW drawing
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_callback,
            (),
            'WINDOW',
            'POST_VIEW'
        )
        self._enabled = True
        print("[VR Ray] Ray renderer registered")
    
    def unregister(self):
        """Unregister draw handler."""
        if self._draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        self._enabled = False
        print("[VR Ray] Ray renderer unregistered")
    
    def update(self, controller_pos: Vector, controller_dir: Vector, 
               hit_point: Vector = None, is_painting: bool = False):
        """Update ray position and state."""
        self._controller_pos = controller_pos.copy()
        self._controller_dir = controller_dir.normalized()
        self._hit_point = hit_point.copy() if hit_point else None
        self._is_painting = is_painting
        
        # Force viewport refresh
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    
    def _draw_callback(self):
        """Draw the ray and hit point."""
        if not self._enabled:
            return
        
        # Calculate ray end point
        ray_end = self._controller_pos + (self._controller_dir * self.ray_length)
        
        # Use hit point if available
        if self._hit_point:
            ray_end = self._hit_point
        
        # Draw ray line
        self._draw_ray(self._controller_pos, ray_end)
        
        # Draw hit point
        if self._hit_point:
            self._draw_hit_point(self._hit_point)
        
        # Draw brush preview if painting
        if self._is_painting and self._hit_point:
            self._draw_brush_preview(self._hit_point)
    
    def _draw_ray(self, start: Vector, end: Vector):
        """Draw the laser ray line."""
        try:
            gpu.state.blend_set('ALPHA')
            gpu.state.line_width_set(self.ray_width)
            
            coords = [start, end]
            
            # Try using POLYLINE shader for better quality
            try:
                shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'LINES', {"pos": coords})
                
                region = bpy.context.region
                shader.uniform_float("viewportSize", (region.width, region.height))
                shader.uniform_float("lineWidth", self.ray_width)
                shader.uniform_float("color", self.ray_color)
                batch.draw(shader)
            except:
                # Fallback to simple uniform color
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'LINES', {"pos": coords})
                shader.uniform_float("color", self.ray_color)
                batch.draw(shader)
            
            gpu.state.blend_set('NONE')
            gpu.state.line_width_set(1.0)
        except Exception as e:
            pass  # Silent fail for rendering
    
    def _draw_hit_point(self, point: Vector):
        """Draw a small sphere at hit point."""
        try:
            gpu.state.blend_set('ALPHA')
            
            # Create simple cross at hit point
            size = self.hit_size
            coords = [
                point + Vector((-size, 0, 0)), point + Vector((size, 0, 0)),
                point + Vector((0, -size, 0)), point + Vector((0, size, 0)),
                point + Vector((0, 0, -size)), point + Vector((0, 0, size)),
            ]
            
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'LINES', {"pos": coords})
            shader.uniform_float("color", self.hit_color)
            batch.draw(shader)
            
            gpu.state.blend_set('NONE')
        except:
            pass
    
    def _draw_brush_preview(self, center: Vector):
        """Draw brush size preview circle."""
        try:
            gpu.state.blend_set('ALPHA')
            
            # Get brush size from scene
            try:
                brush_size = bpy.context.scene.npr_brush_size * 0.1
            except:
                brush_size = 0.05
            
            # Create circle
            segments = 16
            coords = []
            for i in range(segments):
                angle1 = (i / segments) * 2 * math.pi
                angle2 = ((i + 1) / segments) * 2 * math.pi
                
                p1 = center + Vector((math.cos(angle1) * brush_size, math.sin(angle1) * brush_size, 0))
                p2 = center + Vector((math.cos(angle2) * brush_size, math.sin(angle2) * brush_size, 0))
                coords.extend([p1, p2])
            
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'LINES', {"pos": coords})
            shader.uniform_float("color", self.brush_preview_color)
            batch.draw(shader)
            
            gpu.state.blend_set('NONE')
        except:
            pass


def get_vr_ray_renderer():
    return VRRayRenderer.get_instance()


def register():
    pass


def unregister():
    try:
        VRRayRenderer.get_instance().unregister()
    except:
        pass
