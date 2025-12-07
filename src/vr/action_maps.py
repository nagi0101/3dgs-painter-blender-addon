# action_maps.py
# Override xr_session_toggle to inject paint action BEFORE session starts

import bpy
from bpy.types import Operator
from bpy.app.handlers import persistent

_original_toggle = None
_paint_action_added = False


def add_paint_action(session_state):
    """Add paint action to blender_default actionmap."""
    global _paint_action_added
    
    try:
        am = session_state.actionmaps.get("blender_default")
        if am is None:
            print("[3DGS VR] blender_default not found")
            return False
        
        # Check if already exists
        if am.actionmap_items.get("threegds_paint"):
            return True
        
        # Create action item (following Blender VR pattern)
        ami = am.actionmap_items.new("threegds_paint", True)
        if not ami:
            return False
        
        ami.type = 'FLOAT'
        ami.user_paths.new("/user/hand/right")
        ami.op = "threegds.vr_paint_stroke"
        ami.op_mode = 'MODAL'
        ami.bimanual = False
        ami.haptic_name = ""
        ami.haptic_match_user_paths = False
        ami.haptic_duration = 0.0
        ami.haptic_frequency = 0.0
        ami.haptic_amplitude = 0.0
        ami.haptic_mode = 'PRESS'
        
        # Oculus binding (Quest 3)
        amb = ami.bindings.new("oculus", True)
        if amb:
            amb.profile = "/interaction_profiles/oculus/touch_controller"
            amb.component_paths.new("/input/b/click")
            amb.threshold = 0.3
            amb.axis0_region = 'ANY'
            amb.axis1_region = 'ANY'
        
        _paint_action_added = True
        print("[3DGS VR] Paint action added to actionmap (B button)")
        return True
        
    except Exception as e:
        print(f"[3DGS VR] Failed to add paint action: {e}")
        return False


class THREEGDS_OT_XRSessionToggleOverride(Operator):
    """Override xr_session_toggle to inject paint action before session starts"""
    bl_idname = "wm.xr_session_toggle"  # Override the original!
    bl_label = "Toggle VR Session"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        wm = context.window_manager
        
        # Check if session is currently running
        is_running = False
        if hasattr(wm, 'xr_session_state') and wm.xr_session_state:
            try:
                is_running = wm.xr_session_state.is_running(context)
            except:
                pass
        
        if is_running:
            # Stopping session - call original
            return self._call_original(context)
        else:
            # Starting session - inject paint action FIRST
            xr = wm.xr_session_state
            if xr:
                add_paint_action(xr)
            
            # Then start the session
            return self._call_original(context)
    
    def _call_original(self, context):
        """Call the original toggle function."""
        global _original_toggle
        if _original_toggle:
            try:
                # Use the stored original operator
                result = _original_toggle(context)
                return result
            except Exception as e:
                print(f"[3DGS VR] Original toggle error: {e}")
        
        # Fallback - try internal toggle
        try:
            bpy.ops.wm.xr_session_toggle_internal()
            return {'FINISHED'}
        except:
            pass
        
        self.report({'ERROR'}, "Could not toggle VR session")
        return {'CANCELLED'}


def try_add_paint_action_now():
    """Try to add paint action to existing actionmap."""
    try:
        xr = bpy.context.window_manager.xr_session_state
        if xr:
            return add_paint_action(xr)
    except:
        pass
    return False


@persistent
def on_load_post(dummy):
    """Re-add paint action after file load."""
    global _paint_action_added
    _paint_action_added = False


def register():
    """Register override operator."""
    global _original_toggle
    
    # Store reference to original operator before overriding
    try:
        # Check if original exists
        if hasattr(bpy.ops.wm, 'xr_session_toggle'):
            # Store the actual operator function
            _original_toggle = bpy.ops.wm.xr_session_toggle
    except:
        pass
    
    # Register our override
    try:
        bpy.utils.register_class(THREEGDS_OT_XRSessionToggleOverride)
        print("[3DGS VR] Overrode xr_session_toggle operator")
    except Exception as e:
        print(f"[3DGS VR] Override registration failed: {e}")
    
    # Register handler
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    """Unregister override operator."""
    global _original_toggle, _paint_action_added
    
    # Unregister handler
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    
    # Unregister our override
    try:
        bpy.utils.unregister_class(THREEGDS_OT_XRSessionToggleOverride)
    except:
        pass
    
    _original_toggle = None
    _paint_action_added = False


# Backwards compatibility
class VRActionMapManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_action_maps(self):
        return try_add_paint_action_now()
    
    def unregister_action_maps(self):
        pass


def get_action_map_manager():
    return VRActionMapManager.get_instance()
