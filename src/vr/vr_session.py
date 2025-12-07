# vr_session.py
# OpenXR Session Management for 3DGS Painter

import bpy
from typing import Optional, Tuple, Callable
from mathutils import Vector, Quaternion


class VRSessionManager:
    """Manages OpenXR VR session state."""
    
    _instance: Optional['VRSessionManager'] = None
    
    @classmethod
    def get_instance(cls) -> 'VRSessionManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._session_active = False
        self._on_session_start_callbacks: list[Callable] = []
        self._on_session_end_callbacks: list[Callable] = []
    
    def is_vr_available(self) -> bool:
        """Check if VR/OpenXR is available."""
        try:
            wm = bpy.context.window_manager
            return hasattr(wm, 'xr_session_state')
        except:
            return False
    
    def is_vr_addon_enabled(self) -> bool:
        """Check if VR Scene Inspection addon is enabled."""
        try:
            return 'viewport_vr_preview' in bpy.context.preferences.addons
        except:
            return False
    
    def ensure_vr_addon_enabled(self) -> bool:
        """Enable VR Scene Inspection addon if not already enabled."""
        if self.is_vr_addon_enabled():
            return True
        try:
            bpy.ops.preferences.addon_enable(module='viewport_vr_preview')
            print("[3DGS Painter VR] Enabled VR Scene Inspection addon")
            return True
        except Exception as e:
            print(f"[3DGS Painter VR] Failed to enable VR addon: {e}")
            return False
    
    def is_session_running(self) -> bool:
        """Check if a VR session is currently active."""
        try:
            wm = bpy.context.window_manager
            if hasattr(wm, 'xr_session_state') and wm.xr_session_state:
                return wm.xr_session_state.is_running(bpy.context)
            return False
        except:
            return False
    
    def get_xr_session_state(self) -> Optional['bpy.types.XrSessionState']:
        """Get the current XR session state."""
        try:
            wm = bpy.context.window_manager
            if hasattr(wm, 'xr_session_state'):
                return wm.xr_session_state
            return None
        except:
            return None
    
    def start_vr_session(self) -> bool:
        """Start a VR session."""
        if self.is_session_running():
            return True
        try:
            bpy.ops.wm.xr_session_toggle()
            self._session_active = True
            self._notify_session_start()
            return True
        except Exception as e:
            print(f"[3DGS Painter VR] Failed to start VR session: {e}")
            return False
    
    def stop_vr_session(self) -> bool:
        """Stop the current VR session."""
        if not self.is_session_running():
            return True
        try:
            bpy.ops.wm.xr_session_toggle()
            self._session_active = False
            self._notify_session_end()
            return True
        except Exception as e:
            print(f"[3DGS Painter VR] Failed to stop VR session: {e}")
            return False
    
    def get_viewer_pose(self) -> Tuple[Optional[Vector], Optional[Quaternion]]:
        """Get the current viewer (HMD) pose in world space."""
        xr_state = self.get_xr_session_state()
        if xr_state is None or not self.is_session_running():
            return None, None
        try:
            location = Vector(xr_state.viewer_pose_location)
            rotation = Quaternion(xr_state.viewer_pose_rotation)
            return location, rotation
        except:
            return None, None
    
    def add_session_start_callback(self, callback: Callable):
        if callback not in self._on_session_start_callbacks:
            self._on_session_start_callbacks.append(callback)
    
    def add_session_end_callback(self, callback: Callable):
        if callback not in self._on_session_end_callbacks:
            self._on_session_end_callbacks.append(callback)
    
    def _notify_session_start(self):
        for callback in self._on_session_start_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[3DGS Painter VR] Session start callback error: {e}")
    
    def _notify_session_end(self):
        for callback in self._on_session_end_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[3DGS Painter VR] Session end callback error: {e}")


def get_vr_session_manager() -> VRSessionManager:
    return VRSessionManager.get_instance()
