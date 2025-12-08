"""
VR Action Binding for 3DGS Painter

Registers custom XR actions for VR controller input, such as the B button
for painting. This module creates the action at VR session start and provides
functions to query the action state.

OpenXR B button path: /user/hand/right/input/b/click
"""

import bpy
from typing import Tuple, Optional, List

# Action set and action names
ACTION_SET_NAME = "blender_default"  # Use existing blender_default actionmap
PAINT_ACTION_NAME = "threegds_paint"  # This is already registered with /input/b/click!
B_BUTTON_PATH = "/user/hand/right/input/b/click"

# Touch controller profile (Quest 2/3, Rift S)
OCULUS_PROFILE = "/interaction_profiles/oculus/touch_controller"

# Track state
_actions_registered = False
_debug_counter = 0
_enumerated = False
_found_actions: List[Tuple[str, str]] = []  # (action_set, action_name) pairs


def enumerate_all_actions(context) -> List[Tuple[str, str]]:
    """
    Enumerate all existing VR actions and print them for debugging.
    
    Returns:
        List of (action_set_name, action_name) tuples
    """
    global _enumerated, _found_actions
    
    if _enumerated:
        return _found_actions
    
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return []
    
    xr = wm.xr_session_state
    actions = []
    
    print("\n[VR ActionBinding] ========== ENUMERATING ALL VR ACTIONS ==========")
    
    for am in xr.actionmaps:
        print(f"\n[VR ActionBinding] ActionMap: '{am.name}'")
        for item in am.actionmap_items:
            print(f"  - Action: '{item.name}' (type: {item.type})")
            actions.append((am.name, item.name))
            
            # Print bindings
            for binding in item.bindings:
                print(f"    Binding: {binding.name}")
                for cp in binding.component_paths:
                    print(f"      Component: {cp.path}")
    
    print(f"\n[VR ActionBinding] ========== FOUND {len(actions)} ACTIONS ==========\n")
    
    _enumerated = True
    _found_actions = actions
    return actions


def register_paint_action(context) -> bool:
    """
    Register the paint action for VR B button (or find existing one).
    
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
    
    # First, enumerate all existing actions
    actions = enumerate_all_actions(context)
    
    # Enable VR actions if available
    if hasattr(context.scene, 'vr_actions_enable'):
        context.scene.vr_actions_enable = True
        print("[VR ActionBinding] Enabled scene.vr_actions_enable")
    
    _actions_registered = True
    print("[VR ActionBinding] Ready to poll actions")
    return True


def get_paint_button_state(context) -> Tuple[bool, float]:
    """
    Get the current state of the paint button (B button).
    
    Tries all available actions to find one that responds.
    
    Args:
        context: Blender context
    
    Returns:
        Tuple of (is_pressed, pressure_value)
    """
    global _debug_counter
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return False, 0.0
    
    xr = wm.xr_session_state
    _debug_counter += 1
    
    # Try all found actions
    for action_set, action_name in _found_actions:
        for user_path in ["/user/hand/right", "/user/hand/left"]:
            try:
                value = xr.action_state_get(
                    context,
                    action_set,
                    action_name,
                    user_path
                )
                
                if value is not None:
                    # Handle tuple (value, changed) format
                    if isinstance(value, tuple) and len(value) >= 1:
                        val = value[0]
                        if isinstance(val, (int, float)) and float(val) >= 0.5:
                            print(f"[VR ActionBinding] PRESSED! {action_set}/{action_name} = {val}")
                            return True, float(val)
                    # Handle direct value
                    elif isinstance(value, (int, float)) and float(value) >= 0.5:
                        print(f"[VR ActionBinding] PRESSED! {action_set}/{action_name} = {value}")
                        return True, float(value)
                    elif isinstance(value, bool) and value:
                        print(f"[VR ActionBinding] PRESSED! {action_set}/{action_name} = {value}")
                        return True, 1.0
                        
            except Exception:
                pass
    
    # Debug: show first action's value periodically
    if _debug_counter % 200 == 1 and len(_found_actions) > 0:
        action_set, action_name = _found_actions[0]
        try:
            value = xr.action_state_get(context, action_set, action_name, "/user/hand/right")
            print(f"[VR ActionBinding] Sample action '{action_set}/{action_name}' = {value}")
        except Exception as e:
            print(f"[VR ActionBinding] Sample action error: {e}")
    
    return False, 0.0


def is_actions_registered() -> bool:
    """Check if paint actions have been registered."""
    return _actions_registered

