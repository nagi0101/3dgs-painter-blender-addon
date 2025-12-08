"""
VR Action Binding for 3DGS Painter

Registers custom XR actions for VR controller input, such as the B button
for painting. This module creates the action at VR session start and provides
functions to query the action state.

OpenXR B button path: /user/hand/right/input/b/click
"""

import bpy
from typing import Tuple, Optional

# Action set and action names
ACTION_SET_NAME = "3dgs_paint"
PAINT_ACTION_NAME = "paint_button"
B_BUTTON_PATH = "/user/hand/right/input/b/click"

# Touch controller profile (Quest 2/3, Rift S)
OCULUS_PROFILE = "/interaction_profiles/oculus/touch_controller"

# Track if actions are registered
_actions_registered = False


def register_paint_action(context) -> bool:
    """
    Register the paint action for VR B button.
    
    Call this when VR session starts.
    
    Returns:
        True if successful
    """
    global _actions_registered
    
    if _actions_registered:
        return True
    
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        print("[VR ActionBinding] No XR session state")
        return False
    
    xr = wm.xr_session_state
    
    try:
        # Check if our action map already exists
        for am in xr.actionmaps:
            if am.name == ACTION_SET_NAME:
                print(f"[VR ActionBinding] Action map '{ACTION_SET_NAME}' already exists")
                _actions_registered = True
                return True
        
        # Create new action map
        actionmap = xr.actionmaps.new(xr, ACTION_SET_NAME, True)
        print(f"[VR ActionBinding] Created action map: {ACTION_SET_NAME}")
        
        # Create action for paint button (FLOAT type for button)
        action = actionmap.actionmap_items.new(PAINT_ACTION_NAME, True)
        action.type = 'FLOAT'  # Button press as float 0.0 - 1.0
        
        # Add user paths using the collection API (Blender 5.0+)
        # user_paths is a collection with new(), remove(), find() methods
        action.user_paths.new("/user/hand/right")
        action.user_paths.new("/user/hand/left")  # Optional: also allow left hand
        print(f"[VR ActionBinding] Created action: {PAINT_ACTION_NAME}")
        
        # Create binding for Oculus Touch controller profile
        binding = action.bindings.new(OCULUS_PROFILE, True)
        
        # component_paths is also a collection in Blender 5.0+
        binding.component_paths.new(B_BUTTON_PATH)
        print(f"[VR ActionBinding] Created binding: {OCULUS_PROFILE} -> {B_BUTTON_PATH}")
        
        _actions_registered = True
        print("[VR ActionBinding] Paint action registered successfully!")
        return True
        
    except Exception as e:
        print(f"[VR ActionBinding] Error registering action: {e}")
        import traceback
        traceback.print_exc()
        return False


def unregister_paint_action(context):
    """
    Unregister the paint action.
    
    Call this when VR session ends or addon is disabled.
    """
    global _actions_registered
    
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return
    
    xr = wm.xr_session_state
    
    try:
        # Find and remove our action map
        for i, am in enumerate(xr.actionmaps):
            if am.name == ACTION_SET_NAME:
                xr.actionmaps.remove(xr, am)
                print(f"[VR ActionBinding] Removed action map: {ACTION_SET_NAME}")
                break
    except Exception as e:
        print(f"[VR ActionBinding] Error unregistering: {e}")
    
    _actions_registered = False


def get_paint_button_state(context) -> Tuple[bool, float]:
    """
    Get the current state of the paint button (B button).
    
    Args:
        context: Blender context
    
    Returns:
        Tuple of (is_pressed, pressure_value)
    """
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return False, 0.0
    
    xr = wm.xr_session_state
    
    # Try our custom action first
    try:
        value = xr.action_state_get(
            context,
            ACTION_SET_NAME,
            PAINT_ACTION_NAME,
            "/user/hand/right"
        )
        
        if value is not None:
            # Handle different return types
            if isinstance(value, bool):
                return value, 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                is_pressed = float(value) >= 0.5
                return is_pressed, float(value)
            elif hasattr(value, '__len__') and len(value) > 0:
                # Might be a tuple/array
                val = value[0]
                is_pressed = float(val) >= 0.5
                return is_pressed, float(val)
    except Exception as e:
        # First few frames may fail before action is active
        pass
    
    # Fallback: try standard Blender action names
    fallback_names = ["b_button", "button_b", "secondary"]
    fallback_sets = ["blender_default", "default"]
    
    for action_set in fallback_sets:
        for action_name in fallback_names:
            try:
                value = xr.action_state_get(
                    context,
                    action_set,
                    action_name,
                    "/user/hand/right"
                )
                if value is not None:
                    if isinstance(value, bool):
                        return value, 1.0 if value else 0.0
                    elif isinstance(value, (int, float)):
                        is_pressed = float(value) >= 0.5
                        return is_pressed, float(value)
            except:
                pass
    
    return False, 0.0


def is_actions_registered() -> bool:
    """Check if paint actions have been registered."""
    return _actions_registered
