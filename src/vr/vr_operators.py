# vr_operators.py
# VR Painting - B button triggers paint stroke via XR_ACTION event

import bpy
from bpy.types import Operator
import numpy as np
from mathutils import Vector

from .vr_session import get_vr_session_manager
from .vr_input import get_vr_input_manager, ControllerHand
from .action_maps import try_add_paint_action_now


class THREEGDS_OT_VRPaintStroke(Operator):
    """
    Paint with VR B button.
    This operator is called directly by OpenXR when B button is pressed.
    """
    bl_idname = "threegds.vr_paint_stroke"
    bl_label = "VR Paint Stroke"
    bl_options = {'REGISTER', 'UNDO'}
    
    _painting = False
    _last_pos = None
    _scene_data = None
    _stroke_painter = None
    _brush = None
    _renderer = None
    _input_mgr = None
    
    @classmethod
    def poll(cls, context):
        # Always allow - XR system handles the trigger
        return True
    
    def invoke(self, context, event):
        """Called when B button is pressed."""
        # DEBUG: Always print to verify this operator is being called
        print(f"[VR Paint] ====== vr_paint_stroke INVOKE CALLED! ======")
        print(f"[VR Paint] Event type: {event.type}, value: {event.value}")
        
        from ..operators import get_or_create_paint_session
        from ..viewport.viewport_renderer import GaussianViewportRenderer
        
        # Check if this is an XR event
        if event.type == 'XR_ACTION':
            print(f"[VR Paint] XR_ACTION received: {event.xr}")
        
        # Setup paint session
        try:
            session = get_or_create_paint_session(context)
            self._scene_data = session['scene_data']
            self._stroke_painter = session['stroke_painter']
            self._brush = session['brush']
        except Exception as e:
            self.report({'ERROR'}, f"Paint session error: {e}")
            return {'CANCELLED'}
        
        self._input_mgr = get_vr_input_manager()
        
        # Setup renderer
        self._renderer = GaussianViewportRenderer.get_instance()
        if not self._renderer.enabled:
            self._renderer.register()
        
        # Start painting at current controller position
        self._painting = False
        self._start_paint(context)
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        """Handle continuous painting while B is held."""
        
        # XR_ACTION events from B button
        if event.type == 'XR_ACTION':
            # Button released = finish
            if hasattr(event, 'xr') and event.xr:
                if event.xr.state == 0.0:  # Released
                    self._end_paint(context)
                    return {'FINISHED'}
                else:  # Still pressed
                    self._continue_paint(context)
            return {'RUNNING_MODAL'}
        
        # Timer for continuous painting
        if event.type == 'TIMER':
            if self._painting:
                self._continue_paint(context)
            return {'PASS_THROUGH'}
        
        # ESC to cancel
        if event.type == 'ESC':
            self._end_paint(context)
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def _start_paint(self, context):
        """Start a paint stroke at controller position."""
        paint_input = self._input_mgr.get_painting_input()
        if not paint_input:
            print("[VR Paint] No controller input available")
            return
        
        pos = paint_input['position']
        normal = paint_input['normal']
        
        self._painting = True
        self._last_pos = pos.copy()
        
        scene = context.scene
        self._brush.apply_parameters(
            color=np.array(scene.npr_brush_color, dtype=np.float32),
            size_multiplier=scene.npr_brush_size,
            global_opacity=scene.npr_brush_opacity
        )
        
        self._stroke_painter.start_stroke(
            position=np.array(pos, dtype=np.float32),
            normal=np.array(normal, dtype=np.float32),
            enable_deformation=scene.npr_enable_deformation
        )
        
        self._sync()
        print(f"[VR Paint] Stroke started at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
    
    def _continue_paint(self, context):
        """Continue painting at new position."""
        if not self._painting:
            return
        
        paint_input = self._input_mgr.get_painting_input()
        if not paint_input:
            return
        
        pos = paint_input['position']
        normal = paint_input['normal']
        
        # Minimum distance check
        if self._last_pos:
            dist = (pos - self._last_pos).length
            if dist < 0.01:
                return
        
        self._last_pos = pos.copy()
        
        self._stroke_painter.update_stroke(
            position=np.array(pos, dtype=np.float32),
            normal=np.array(normal, dtype=np.float32)
        )
        self._sync()
    
    def _end_paint(self, context):
        """Finish the paint stroke."""
        if not self._painting:
            return
        
        self._painting = False
        self._stroke_painter.finish_stroke(
            enable_deformation=context.scene.npr_enable_deformation,
            enable_inpainting=False
        )
        self._sync()
        
        count = self._scene_data.count if self._scene_data else 0
        print(f"[VR Paint] Stroke finished. Total: {count} gaussians")
        self.report({'INFO'}, f"Painted {count} gaussians")
    
    def _sync(self):
        """Sync to viewport renderer."""
        if self._renderer and self._scene_data and self._scene_data.count > 0:
            self._renderer.update_gaussians(scene_data=self._scene_data)
    
    def cancel(self, context):
        """Handle cancellation."""
        self._end_paint(context)


class THREEGDS_OT_StartVRSession(Operator):
    """Start VR and register paint action"""
    bl_idname = "threegds.start_vr_session"
    bl_label = "Start VR"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        mgr = get_vr_session_manager()
        return mgr.is_vr_available() and not mgr.is_session_running()
    
    def execute(self, context):
        mgr = get_vr_session_manager()
        mgr.ensure_vr_addon_enabled()
        
        # Start VR session first
        if not mgr.start_vr_session():
            self.report({'ERROR'}, "Failed to start VR")
            return {'CANCELLED'}
        
        # Add paint action to blender_default
        if try_add_paint_action_now():
            self.report({'INFO'}, "VR started - Press B button to paint")
        else:
            self.report({'WARNING'}, "VR started but paint action not registered")
        
        return {'FINISHED'}


class THREEGDS_OT_StopVRSession(Operator):
    """Stop VR"""
    bl_idname = "threegds.stop_vr_session"
    bl_label = "Stop VR"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        return get_vr_session_manager().is_session_running()
    
    def execute(self, context):
        if get_vr_session_manager().stop_vr_session():
            self.report({'INFO'}, "VR stopped")
            return {'FINISHED'}
        return {'CANCELLED'}


class THREEGDS_OT_TestVRInput(Operator):
    """Test VR controller position"""
    bl_idname = "threegds.test_vr_input"
    bl_label = "Test VR Position"
    bl_options = {'REGISTER'}
    
    _timer = None
    
    @classmethod
    def poll(cls, context):
        return get_vr_session_manager().is_session_running()
    
    def invoke(self, context, event):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        self.report({'INFO'}, "Testing controller - check console, ESC to stop")
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            mgr = get_vr_input_manager()
            if not mgr.is_vr_active():
                self._cleanup(context)
                return {'CANCELLED'}
            
            right = mgr.get_controller_state(ControllerHand.RIGHT)
            if right.is_active:
                p = right.aim_position
                print(f"[VR] Right: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
        
        if event.type == 'ESC':
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        self.report({'INFO'}, "Test ended")


class THREEGDS_OT_VRRayTrack(Operator):
    """Show laser ray from controller - ESC to stop"""
    bl_idname = "threegds.vr_ray_track"
    bl_label = "VR Ray Tracking"
    bl_options = {'REGISTER'}
    
    _timer = None
    _ray_renderer = None
    
    @classmethod
    def poll(cls, context):
        return get_vr_session_manager().is_session_running()
    
    def invoke(self, context, event):
        from .vr_ray_renderer import get_vr_ray_renderer
        
        self._ray_renderer = get_vr_ray_renderer()
        self._ray_renderer.register()
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)  # ~60fps
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, "Ray tracking started - ESC to stop")
        print("[VR Ray] Tracking started")
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            mgr = get_vr_input_manager()
            if not mgr.is_vr_active():
                self._cleanup(context)
                return {'CANCELLED'}
            
            # Get controller state
            right = mgr.get_controller_state(ControllerHand.RIGHT)
            if right.is_active:
                # Get aim direction from rotation
                import mathutils
                forward = mathutils.Vector((0, 0, -1))
                direction = right.aim_rotation @ forward
                
                # Update ray renderer
                self._ray_renderer.update(
                    controller_pos=right.aim_position,
                    controller_dir=direction,
                    hit_point=None,  # TODO: raycast for hit point
                    is_painting=False
                )
        
        if event.type == 'ESC':
            self._cleanup(context)
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if self._ray_renderer:
            self._ray_renderer.unregister()
        self.report({'INFO'}, "Ray tracking stopped")
        print("[VR Ray] Tracking stopped")


class THREEGDS_OT_OpenXRLayerTest(Operator):
    """Send test Gaussians to OpenXR Layer via shared memory"""
    bl_idname = "threegds.openxr_layer_test"
    bl_label = "Test OpenXR Layer"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from . import vr_shared_memory
        import numpy as np
        
        # Create test gaussians (grid of colored spheres)
        n_gaussians = 100
        
        # Grid positions
        positions = np.zeros((n_gaussians, 3), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                idx = i * 10 + j
                positions[idx] = [i * 0.2 - 1.0, 0.0, j * 0.2 - 1.0]
        
        # Colors (rainbow gradient)
        colors = np.zeros((n_gaussians, 4), dtype=np.float32)
        for i in range(n_gaussians):
            hue = i / n_gaussians
            # Simple HSV to RGB
            if hue < 0.33:
                colors[i] = [1.0 - hue*3, hue*3, 0.0, 1.0]
            elif hue < 0.67:
                colors[i] = [0.0, 1.0 - (hue-0.33)*3, (hue-0.33)*3, 1.0]
            else:
                colors[i] = [(hue-0.67)*3, 0.0, 1.0 - (hue-0.67)*3, 1.0]
        
        # Scales
        scales = np.full((n_gaussians, 3), 0.05, dtype=np.float32)
        
        # Rotations (identity quaternion)
        rotations = np.zeros((n_gaussians, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w component
        
        # Send to shared memory
        writer = vr_shared_memory.get_shared_memory_writer()
        if not writer.is_open():
            if not writer.create():
                self.report({'ERROR'}, "Failed to create shared memory")
                return {'CANCELLED'}
        
        success = writer.write_gaussians_numpy(
            positions=positions,
            colors=colors,
            scales=scales,
            rotations=rotations
        )
        
        if success:
            self.report({'INFO'}, f"Sent {n_gaussians} test gaussians to OpenXR Layer")
            print(f"[OpenXR Layer] Sent {n_gaussians} gaussians via shared memory")
        else:
            self.report({'ERROR'}, "Failed to write gaussians")
            return {'CANCELLED'}
        
        return {'FINISHED'}


classes = [
    THREEGDS_OT_VRPaintStroke,
    THREEGDS_OT_StartVRSession,
    THREEGDS_OT_StopVRSession,
    THREEGDS_OT_TestVRInput,
    THREEGDS_OT_VRRayTrack,
    THREEGDS_OT_OpenXRLayerTest,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
