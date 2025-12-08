# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
VR Freehand Paint - Integrated with existing painting pipeline

Uses the same StrokePainter/BrushStamp infrastructure as desktop painting.
Controller input triggers the same deformation/inpainting features.
"""

import bpy
from bpy.types import Operator
from mathutils import Vector, Quaternion, Matrix
import numpy as np
from typing import Optional, List, Tuple

# Import VR action binding for TRIGGER
from .vr_action_binding import register_paint_action, get_paint_button_state, is_actions_registered

# Import existing painting infrastructure
from ..operators import get_or_create_paint_session
from ..viewport.viewport_renderer import GaussianViewportRenderer


class VRPaintSession:
    """Manages VR painting state with existing infrastructure."""
    
    _instance: Optional["VRPaintSession"] = None
    
    def __init__(self):
        self.is_painting: bool = False
        self.last_sample_pos: Optional[Vector] = None
        self.spacing: float = 0.02  # meters between samples
        
        # These are set from get_or_create_paint_session()
        self.scene_data = None
        self.stroke_painter = None
        self.brush = None
        self.viewport_renderer = None
        
    @classmethod
    def get_instance(cls) -> "VRPaintSession":
        if cls._instance is None:
            cls._instance = VRPaintSession()
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


def is_b_button_pressed(context, hand_index: int = 1) -> Tuple[bool, float]:
    """
    Check if VR B button is pressed using Blender's XR action system.
    
    Args:
        context: Blender context
        hand_index: 0=left, 1=right (default: right hand)
    
    Returns:
        Tuple of (is_pressed, value)
    """
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return False, 0.0
    
    xr = wm.xr_session_state
    user_path = "/user/hand/right" if hand_index == 1 else "/user/hand/left"
    
    # Try different action names for B button
    action_sets = ["blender_default", "default"]
    # Quest controllers: b_button, button_b, b, secondary_button
    action_names = ["b_button", "button_b", "b", "secondary", "secondary_button", "menu"]
    
    for action_set in action_sets:
        for action_name in action_names:
            try:
                # action_state_get returns a boolean or float for button actions
                value = xr.action_state_get(context, action_set, action_name, user_path)
                if value is not None:
                    is_pressed = bool(value) if isinstance(value, bool) else float(value) >= 0.5
                    if is_pressed:
                        print(f"[VR Paint] B button detected: {action_set}/{action_name} = {value}")
                    return is_pressed, float(value) if isinstance(value, (int, float)) else (1.0 if value else 0.0)
            except Exception:
                pass  # Action not found, try next
    
    # Fallback: return False if no B button action found
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
    VR Freehand Painting - Uses existing StrokePainter/BrushStamp
    
    Paint in 3D space using VR controller as a pen.
    Integrates with the desktop painting pipeline for consistent results.
    """
    bl_idname = "threegds.vr_freehand_paint"
    bl_label = "VR Freehand Paint"
    bl_description = "Paint Gaussians in 3D space using VR controller"
    bl_options = {'REGISTER'}
    
    # Timer handle
    _timer = None
    
    # Paint session
    _session: Optional[VRPaintSession] = None
    
    # Keyboard trigger simulation (for testing)
    _keyboard_triggered: bool = False
    
    @classmethod
    def poll(cls, context):
        """Check if VR session is active."""
        wm = context.window_manager
        return hasattr(wm, 'xr_session_state') and wm.xr_session_state is not None
    
    def invoke(self, context, event):
        """Start the VR painting modal operator."""
        # Initialize session with existing infrastructure
        self._session = VRPaintSession.get_instance()
        self._keyboard_triggered = False
        
        # Get paint session from main operators.py (same as desktop)
        try:
            paint_session = get_or_create_paint_session(context)
            self._session.scene_data = paint_session['scene_data']
            self._session.stroke_painter = paint_session['stroke_painter']
            self._session.brush = paint_session['brush']
            print("[VR Paint] Using existing painting infrastructure")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to initialize paint session: {e}")
            return {'CANCELLED'}
        
        # Get viewport renderer
        self._session.viewport_renderer = GaussianViewportRenderer.get_instance()
        if not self._session.viewport_renderer.enabled:
            self._session.viewport_renderer.register()
        
        # Register VR TRIGGER action binding
        if not is_actions_registered():
            if register_paint_action(context):
                print("[VR Paint] TRIGGER binding ready")
            else:
                print("[VR Paint] TRIGGER binding failed, SPACE key fallback available")
        
        # Add timer for continuous polling (10ms = 100Hz)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, "VR Freehand Paint started - Hold TRIGGER to paint, ESC to exit")
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
        
        # NOTE: Matrix updates are now handled by continuous updater in vr_operators.py
        # Removed self._update_vr_matrices(context) to prevent conflicting updates
        
        # Check if painting (keyboard simulation OR actual VR trigger)
        vr_pressed, vr_pressure = get_paint_button_state(context)
        is_painting = self._keyboard_triggered or vr_pressed
        
        # Auto-manage stroke state based on trigger
        if is_painting and not self._session.is_painting:
            self._start_stroke(context)
        elif not is_painting and self._session.is_painting and not self._keyboard_triggered:
            self._end_stroke(context)
        
        if is_painting:
            self._continue_stroke(context, tip_pos, tip_rot)
        
        # Always update brush preview position
        self._update_preview(context, tip_pos, tip_rot)
    
    def _start_stroke(self, context):
        """Initialize a new stroke using StrokePainter."""
        scene = context.scene
        
        # Apply brush settings from UI panels
        self._session.brush.apply_parameters(
            color=np.array(scene.npr_brush_color, dtype=np.float32),
            size_multiplier=scene.npr_brush_size,
            global_opacity=scene.npr_brush_opacity
        )
        
        self._session.is_painting = True
        self._session.last_sample_pos = None
        
        print(f"[VR Paint] Stroke started (size={scene.npr_brush_size:.3f}, color={scene.npr_brush_color[:]})")
    
    def _continue_stroke(self, context, position: Vector, rotation: Quaternion):
        """Add a point to the current stroke using StrokePainter."""
        
        if not self._session.is_painting:
            return
        
        scene = context.scene
        
        # Check spacing threshold
        if self._session.last_sample_pos is not None:
            dist = (position - self._session.last_sample_pos).length
            if dist < self._session.spacing:
                return  # Too close, skip this sample
        
        # Convert to numpy arrays
        pos_np = np.array(position, dtype=np.float32)
        
        # Get normal from controller orientation (up direction)
        normal_vec = rotation @ Vector((0, 1, 0))
        normal_np = np.array(normal_vec, dtype=np.float32)
        
        # First point: start stroke
        if self._session.last_sample_pos is None:
            self._session.stroke_painter.start_stroke(
                position=pos_np,
                normal=normal_np,
                enable_deformation=scene.npr_enable_deformation
            )
        else:
            # Update stroke with new point
            self._session.stroke_painter.update_stroke(
                position=pos_np,
                normal=normal_np
            )
        
        self._session.last_sample_pos = position.copy()
        
        # Sync to viewport
        self._sync_to_viewport(context)
        
        gauss_count = len(self._session.scene_data)
        print(f"[VR Paint] Point added at {position}, total: {gauss_count}")
    
    def _end_stroke(self, context):
        """Finish the current stroke with deformation."""
        scene = context.scene
        
        self._session.is_painting = False
        
        # Finish stroke (applies deformation if enabled)
        self._session.stroke_painter.finish_stroke(
            enable_deformation=scene.npr_enable_deformation,
            enable_inpainting=False  # Can enable later
        )
        
        gauss_count = len(self._session.scene_data)
        print(f"[VR Paint] Stroke ended, total gaussians: {gauss_count}")
        
        # Final sync
        self._sync_to_viewport(context)
    
    def _sync_to_viewport(self, context):
        """Sync SceneData to viewport renderer."""
        if self._session.scene_data is None or len(self._session.scene_data) == 0:
            return
        
        try:
            # Use existing viewport renderer with actual SceneData
            renderer = self._session.viewport_renderer
            
            if renderer and renderer.enabled:
                renderer.update_gaussians(scene_data=self._session.scene_data)
                renderer.request_redraw()
                
        except Exception as e:
            print(f"[VR Paint] Viewport sync error: {e}")
        
        # VR Offscreen renderer disabled - uses old _accumulated_gaussians format
        # Will be re-implemented to use SceneData in future
        
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
        
        # ============================================================
        # 4. OpenXR Layer Shared Memory (Phase 5 - native VR rendering!)
        # ============================================================
        try:
            from .vr_shared_memory import write_gaussians_to_vr
            
            # Use SceneData arrays directly
            scene_data = self._session.scene_data
            n = len(scene_data) if scene_data else 0
            
            if n > 0:
                # Get VR view/projection matrices if available
                view_matrix = None
                proj_matrix = None
                try:
                    wm = context.window_manager
                    if hasattr(wm, 'xr_session_state') and wm.xr_session_state:
                        xr = wm.xr_session_state
                        viewer_pos = xr.viewer_pose_location
                        viewer_rot = xr.viewer_pose_rotation
                        
                        from mathutils import Matrix
                        rot_mat = viewer_rot.to_matrix().to_4x4()
                        trans_mat = Matrix.Translation(viewer_pos)
                        view_mat = (trans_mat @ rot_mat).inverted()
                        view_matrix = np.array(view_mat.transposed(), dtype=np.float32).flatten()
                        
                        import math
                        fov = math.radians(90)
                        near, far = 0.1, 100.0
                        f = 1.0 / math.tan(fov / 2)
                        proj = np.zeros((4, 4), dtype=np.float32)
                        proj[0, 0] = f
                        proj[1, 1] = f
                        proj[2, 2] = (far + near) / (near - far)
                        proj[2, 3] = (2 * far * near) / (near - far)
                        proj[3, 2] = -1
                        proj_matrix = proj.T.flatten()
                        
                except Exception as e:
                    print(f"[VR Paint] Matrix extraction error: {e}")
                
                # Get camera rotation from active camera for OpenXR alignment
                camera_rotation = None
                try:
                    camera = context.scene.camera
                    if camera:
                        # Get camera rotation as quaternion (w, x, y, z)
                        cam_rot = camera.rotation_euler.to_quaternion()
                        camera_rotation = (cam_rot.w, cam_rot.x, cam_rot.y, cam_rot.z)
                except Exception as e:
                    print(f"[VR Paint] Camera rotation error: {e}")
                
                # Build colors with opacity as 4th channel
                colors = np.zeros((n, 4), dtype=np.float32)
                colors[:, :3] = scene_data.colors
                colors[:, 3] = scene_data.opacities
                
                write_gaussians_to_vr({
                    'positions': scene_data.positions,
                    'colors': colors,
                    'scales': scene_data.scales,
                    'rotations': scene_data.rotations,
                    'view_matrix': view_matrix,
                    'proj_matrix': proj_matrix,
                    'camera_rotation': camera_rotation
                })
                
        except Exception as e:
            print(f"[VR Paint] OpenXR Layer sync error: {e}")
    
    def _update_vr_matrices(self, context):
        """
        Update VR view/projection matrices every frame for head tracking.
        
        This ensures 3D projection updates when user moves their head,
        even when not actively painting.
        """
        try:
            from .vr_shared_memory import get_shared_memory_writer
            
            wm = context.window_manager
            if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
                return
            
            xr = wm.xr_session_state
            
            # Get viewer pose (position + rotation)
            viewer_pos = xr.viewer_pose_location
            viewer_rot = xr.viewer_pose_rotation
            
            # Build view matrix from viewer pose
            from mathutils import Matrix
            rot_mat = viewer_rot.to_matrix().to_4x4()
            trans_mat = Matrix.Translation(viewer_pos)
            view_mat = (trans_mat @ rot_mat).inverted()
            
            # Transpose for OpenGL (column-major)
            view_matrix = np.array(view_mat.transposed(), dtype=np.float32).flatten()
            
            # Simple perspective projection (90 degree FOV)
            import math
            fov = math.radians(90)
            aspect = 1.0
            near = 0.1
            far = 100.0
            f = 1.0 / math.tan(fov / 2)
            proj = np.zeros((4, 4), dtype=np.float32)
            proj[0, 0] = f / aspect
            proj[1, 1] = f
            proj[2, 2] = (far + near) / (near - far)
            proj[2, 3] = (2 * far * near) / (near - far)
            proj[3, 2] = -1
            proj_matrix = proj.T.flatten()
            
            # Write matrices to shared memory (without Gaussian data update)
            writer = get_shared_memory_writer()
            if writer and writer.is_open():
                # Update only matrices in shared memory header
                writer.update_matrices(view_matrix, proj_matrix)
                
        except Exception as e:
            # Silently ignore errors to avoid spamming console
            pass
    
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
        
        total_gaussians = len(self._session.scene_data) if self._session and self._session.scene_data else 0
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
