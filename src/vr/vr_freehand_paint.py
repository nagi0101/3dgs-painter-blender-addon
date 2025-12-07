# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
VR Freehand Paint - Tilt Brush style 3D space painting

Uses timer polling to track controller position and generate
Gaussians directly at controller tip without raycasting.
"""

import bpy
from bpy.types import Operator
from mathutils import Vector, Quaternion, Matrix
import numpy as np
from typing import Optional, List, Tuple


class VRFreehandPaintState:
    """Singleton state manager for VR painting session."""
    
    _instance: Optional["VRFreehandPaintState"] = None
    
    def __init__(self):
        self.is_painting: bool = False
        self.stroke_points: List[Tuple[Vector, Quaternion, float]] = []  # (pos, rot, pressure)
        self.last_sample_pos: Optional[Vector] = None
        self.brush_size: float = 0.05  # meters
        self.brush_color: Tuple[float, float, float] = (0.0, 0.8, 1.0)  # cyan
        self.spacing: float = 0.02  # meters between samples
        self.min_gaussians_per_stamp: int = 5
        
    @classmethod
    def get_instance(cls) -> "VRFreehandPaintState":
        if cls._instance is None:
            cls._instance = VRFreehandPaintState()
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None


def get_controller_tip(context, hand_index: int = 1) -> Optional[Tuple[Vector, Quaternion]]:
    """
    Get controller tip position and rotation.
    
    Args:
        context: Blender context
        hand_index: 0=left, 1=right (default: right hand)
    
    Returns:
        Tuple of (tip_position, rotation) or None if VR not active
    """
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return None
    
    xr = wm.xr_session_state
    
    try:
        # Get grip location and aim rotation (returns tuples, need to convert)
        grip_pos_tuple = xr.controller_grip_location_get(context, hand_index)
        aim_rot_tuple = xr.controller_aim_rotation_get(context, hand_index)
        
        # Convert tuple to Vector (x, y, z)
        grip_pos = Vector(grip_pos_tuple)
        
        # Convert tuple to Quaternion (w, x, y, z)
        # Note: XR API returns (w, x, y, z) format
        aim_rot = Quaternion(aim_rot_tuple)
        
        # Apply tip offset (controller tip is ~8cm forward from grip)
        TIP_OFFSET = 0.08  # meters
        
        # Get forward direction from quaternion and apply offset
        forward_dir = aim_rot @ Vector((0, 0, -TIP_OFFSET))
        tip_pos = grip_pos + forward_dir
        
        return tip_pos, aim_rot
        
    except Exception as e:
        print(f"[VR Paint] Controller tracking error: {e}")
        import traceback
        traceback.print_exc()
        return None


def is_trigger_pressed(context, hand_index: int = 1, threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Check if trigger is pressed using distance-based trigger simulation.
    
    Since OpenXR action binding is complex, we use a simplified approach:
    - For now, return based on VR session active state
    - TODO: Implement proper action state query
    
    Returns:
        Tuple of (is_pressed, pressure)
    """
    # Simplified: Use B button state if available, otherwise simulate
    # This is a placeholder - real implementation needs OpenXR action binding
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return False, 0.0
    
    # For prototype: Always return False, will be controlled by keyboard in testing
    # Real implementation: query xr.action_state_get()
    return False, 0.0


def generate_gaussians_at_point(
    position: Vector,
    rotation: Quaternion,
    size: float,
    color: Tuple[float, float, float],
    count: int = 5
) -> List[dict]:
    """
    Generate a cluster of Gaussians at the given position.
    
    Creates a small group of gaussians to form a "stamp" at the paint point.
    
    Args:
        position: World space position
        rotation: Orientation quaternion
        size: Base size of the gaussian cluster
        color: RGB color (0-1 range)
        count: Number of gaussians in the stamp
    
    Returns:
        List of gaussian dicts with position, rotation, scale, opacity, color
    """
    gaussians = []
    
    for i in range(count):
        # Random offset within the brush radius
        offset = Vector((
            np.random.uniform(-size * 0.5, size * 0.5),
            np.random.uniform(-size * 0.5, size * 0.5),
            np.random.uniform(-size * 0.3, size * 0.3)
        ))
        
        # Transform offset by brush rotation
        offset = rotation @ offset
        
        # Gaussian properties
        gauss = {
            'position': position + offset,
            'rotation': rotation,  # Use brush rotation
            'scale': Vector((size * 0.3, size * 0.3, size * 0.15)),  # Flat ellipsoid
            'opacity': np.random.uniform(0.6, 0.9),
            'color': (
                color[0] + np.random.uniform(-0.1, 0.1),
                color[1] + np.random.uniform(-0.1, 0.1),
                color[2] + np.random.uniform(-0.1, 0.1)
            )
        }
        gaussians.append(gauss)
    
    return gaussians


def pack_gaussians_to_array(gaussians: List[dict]) -> np.ndarray:
    """
    Pack gaussian dicts to numpy array for viewport renderer.
    
    59 floats per gaussian:
    [0-2]: position, [3-6]: rotation (wxyz), [7-9]: scale,
    [10]: opacity, [11-13]: SH0 color coefficients, [14-58]: higher SH (zeros)
    """
    n = len(gaussians)
    data = np.zeros((n, 59), dtype=np.float32)
    
    SH_C0 = 0.28209479177387814
    
    for i, g in enumerate(gaussians):
        # Position
        data[i, 0:3] = g['position']
        
        # Rotation (w, x, y, z)
        rot = g['rotation']
        data[i, 3] = rot.w
        data[i, 4] = rot.x
        data[i, 5] = rot.y
        data[i, 6] = rot.z
        
        # Scale
        data[i, 7:10] = g['scale']
        
        # Opacity
        data[i, 10] = g['opacity']
        
        # Color as SH coefficients (degree 0)
        # SH inverse: (color - 0.5) / SH_C0
        color = g['color']
        data[i, 11] = (color[0] - 0.5) / SH_C0
        data[i, 12] = (color[1] - 0.5) / SH_C0
        data[i, 13] = (color[2] - 0.5) / SH_C0
        
        # Higher SH coefficients are zero
    
    return data


class THREEGDS_OT_VRFreehandPaint(Operator):
    """
    VR Freehand Painting - Timer polling based
    
    Paint in 3D space using VR controller as a pen.
    Uses timer to continuously sample controller position.
    """
    bl_idname = "threegds.vr_freehand_paint"
    bl_label = "VR Freehand Paint"
    bl_description = "Paint Gaussians in 3D space using VR controller"
    bl_options = {'REGISTER'}
    
    # Timer handle
    _timer = None
    
    # Paint state
    _state: Optional[VRFreehandPaintState] = None
    _accumulated_gaussians: List[dict] = []
    
    # Keyboard trigger simulation (for testing)
    _keyboard_triggered: bool = False
    
    @classmethod
    def poll(cls, context):
        """Check if VR session is active."""
        wm = context.window_manager
        return hasattr(wm, 'xr_session_state') and wm.xr_session_state is not None
    
    def invoke(self, context, event):
        """Start the VR painting modal operator."""
        self._state = VRFreehandPaintState.get_instance()
        self._accumulated_gaussians = []
        self._keyboard_triggered = False
        
        # Add timer for continuous polling (10ms = 100Hz)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, "VR Freehand Paint started - Hold SPACE to paint, ESC to exit")
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        """Handle timer events and keyboard input for testing."""
        
        # Exit on ESC or RIGHTMOUSE
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}
        
        # Keyboard trigger simulation (SPACE key for testing)
        if event.type == 'SPACE':
            if event.value == 'PRESS':
                self._keyboard_triggered = True
                self._start_stroke()
            elif event.value == 'RELEASE':
                self._keyboard_triggered = False
                self._end_stroke(context)
        
        # Timer event - sample controller and paint
        if event.type == 'TIMER':
            self._on_timer_tick(context)
        
        return {'RUNNING_MODAL'}
    
    def _on_timer_tick(self, context):
        """Called every timer tick to sample controller and generate paint."""
        
        # Get controller position
        result = get_controller_tip(context, hand_index=1)  # Right hand
        
        if result is None:
            return  # VR not active or controller not tracked
        
        tip_pos, tip_rot = result
        
        # Check if painting (keyboard simulation or actual VR trigger)
        is_painting = self._keyboard_triggered
        
        # TODO: Add actual VR trigger check here
        # vr_pressed, vr_pressure = is_trigger_pressed(context)
        # is_painting = is_painting or vr_pressed
        
        if is_painting:
            self._continue_stroke(context, tip_pos, tip_rot)
        
        # Always update brush preview position
        self._update_preview(context, tip_pos, tip_rot)
    
    def _start_stroke(self):
        """Initialize a new stroke."""
        self._state.is_painting = True
        self._state.stroke_points = []
        self._state.last_sample_pos = None
        print("[VR Paint] Stroke started")
    
    def _continue_stroke(self, context, position: Vector, rotation: Quaternion):
        """Add a point to the current stroke if spacing threshold met."""
        
        if not self._state.is_painting:
            return
        
        # Check spacing threshold
        if self._state.last_sample_pos is not None:
            dist = (position - self._state.last_sample_pos).length
            if dist < self._state.spacing:
                return  # Too close, skip this sample
        
        # Record point
        self._state.stroke_points.append((position.copy(), rotation.copy(), 1.0))
        self._state.last_sample_pos = position.copy()
        
        # Generate gaussians at this point
        gaussians = generate_gaussians_at_point(
            position=position,
            rotation=rotation,
            size=self._state.brush_size,
            color=self._state.brush_color,
            count=self._state.min_gaussians_per_stamp
        )
        
        self._accumulated_gaussians.extend(gaussians)
        
        # Sync to viewport immediately
        self._sync_to_viewport(context)
        
        print(f"[VR Paint] Point added at {position}, total gaussians: {len(self._accumulated_gaussians)}")
    
    def _end_stroke(self, context):
        """Finish the current stroke."""
        self._state.is_painting = False
        
        point_count = len(self._state.stroke_points)
        gauss_count = len(self._accumulated_gaussians)
        
        print(f"[VR Paint] Stroke ended: {point_count} points, {gauss_count} gaussians")
        
        # Final sync
        self._sync_to_viewport(context)
    
    def _sync_to_viewport(self, context):
        """Sync accumulated gaussians to both PC viewport and VR mesh objects."""
        if not self._accumulated_gaussians:
            return
        
        # ============================================================
        # 1. Sync to PC Viewport (draw handler - only visible on PC)
        # ============================================================
        try:
            from ..viewport.viewport_renderer import GaussianViewportRenderer
            
            renderer = GaussianViewportRenderer.get_instance()
            
            if not renderer.enabled:
                renderer.register()
            
            packed_data = pack_gaussians_to_array(self._accumulated_gaussians)
            
            class MockSceneData:
                def __init__(self, data):
                    n = data.shape[0]
                    self.count = n
                    self.positions = data[:, 0:3]
                    self.rotations = np.zeros((n, 4), dtype=np.float32)
                    self.rotations[:, 0:3] = data[:, 4:7]
                    self.rotations[:, 3] = data[:, 3]
                    self.scales = data[:, 7:10]
                    self.opacities = data[:, 10]
                    SH_C0 = 0.28209479177387814
                    self.colors = data[:, 11:14] * SH_C0 + 0.5
            
            mock_scene = MockSceneData(packed_data)
            renderer.update_gaussians(scene_data=mock_scene)
            renderer.request_redraw()
            
        except Exception as e:
            print(f"[VR Paint] PC viewport sync error: {e}")
        
        # ============================================================
        # 2. VR Offscreen Renderer (Phase 1 - visible in VR headset!)
        # ============================================================
        try:
            from .vr_offscreen_renderer import get_vr_offscreen_renderer
            
            offscreen = get_vr_offscreen_renderer()
            
            if not offscreen._initialized:
                offscreen.initialize()
                offscreen.create_display_plane(context)
            
            # Clear and add all gaussians
            offscreen.clear()
            offscreen.add_gaussians_batch(self._accumulated_gaussians)
            
            # Update display
            offscreen.update_display(context)
            
            print(f"[VR Paint] Offscreen display updated: {len(offscreen.gaussian_positions)} gaussians")
            
        except Exception as e:
            print(f"[VR Paint] VR offscreen sync error: {e}")
            import traceback
            traceback.print_exc()
        
        # ============================================================
        # 3. VR Mesh Objects - DISABLED FOR TESTING
        # ============================================================
        # Mesh rendering disabled to test if GLSL works in VR
        # Uncomment below if GLSL doesn't work and mesh fallback is needed
        # ============================================================
        # try:
        #     from .vr_mesh_renderer import get_vr_mesh_manager
        #     
        #     mesh_mgr = get_vr_mesh_manager()
        #     mesh_mgr.clear()
        #     mesh_mgr.add_gaussians_batch(self._accumulated_gaussians)
        #     
        #     print(f"[VR Paint] VR mesh objects: {mesh_mgr.count}")
        #     
        # except Exception as e:
        #     print(f"[VR Paint] VR mesh sync error: {e}")
    
    def _update_preview(self, context, position: Vector, rotation: Quaternion):
        """Update brush preview at controller position."""
        # TODO: Implement 3D mesh-based preview
        # For now, just ensure viewport updates
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    
    def _finish(self, context):
        """Clean up the operator."""
        # Remove timer
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        
        # Final sync
        self._sync_to_viewport(context)
        
        total_gaussians = len(self._accumulated_gaussians)
        self.report({'INFO'}, f"VR Freehand Paint finished: {total_gaussians} gaussians created")


class THREEGDS_OT_VRFreehandPaintClear(Operator):
    """Clear all VR painted gaussians."""
    bl_idname = "threegds.vr_freehand_clear"
    bl_label = "Clear VR Paint"
    bl_description = "Clear all gaussians created by VR freehand painting"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Clear PC viewport gaussians
        try:
            from ..viewport.viewport_renderer import GaussianViewportRenderer
            
            renderer = GaussianViewportRenderer.get_instance()
            renderer.clear()
            renderer.request_redraw()
        except Exception as e:
            print(f"[VR Paint] PC clear error: {e}")
        
        # Clear VR mesh objects
        try:
            from .vr_mesh_renderer import get_vr_mesh_manager
            
            mesh_mgr = get_vr_mesh_manager()
            mesh_mgr.cleanup()  # Full cleanup including objects
        except Exception as e:
            print(f"[VR Paint] VR mesh clear error: {e}")
        
        # Reset state
        VRFreehandPaintState.reset()
        
        self.report({'INFO'}, "VR paint cleared (PC + VR)")
        return {'FINISHED'}


# Registration
vr_freehand_classes = [
    THREEGDS_OT_VRFreehandPaint,
    THREEGDS_OT_VRFreehandPaintClear,
]


def register():
    for cls in vr_freehand_classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(vr_freehand_classes):
        bpy.utils.unregister_class(cls)
