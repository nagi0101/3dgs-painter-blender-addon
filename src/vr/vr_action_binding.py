"""
VR Action Binding for 3DGS Painter

Uses TRIGGER for painting (via existing 'teleport' action that's already 
registered with OpenXR before session starts).

OpenXR trigger path: /user/hand/right/input/trigger/value
"""

import bpy
from typing import Tuple

# Use existing Blender action that's already bound before session
ACTION_SET_NAME = "blender_default"
PAINT_ACTION_NAME = "teleport"  # Uses /input/trigger/value

# State tracking
_initialized = False
_debug_counter = 0


def get_paint_button_state(context) -> Tuple[bool, float]:
    """
    Get the current state of the paint trigger (TRIGGER button).
    
    Uses the 'teleport' action which is already bound to trigger.
    
    Returns:
        Tuple of (is_pressed, pressure_value)
    """
    global _debug_counter
    wm = context.window_manager
    
    if not hasattr(wm, 'xr_session_state') or wm.xr_session_state is None:
        return False, 0.0
    
    xr = wm.xr_session_state
    _debug_counter += 1
    
    # Query the teleport action (trigger button)
    try:
        value = xr.action_state_get(
            context,
            ACTION_SET_NAME,
            PAINT_ACTION_NAME,
            "/user/hand/right"
        )
        
        if value is not None:
            # Handle tuple (value, changed) format
            if isinstance(value, tuple) and len(value) >= 1:
                pressure = float(value[0])
                is_pressed = pressure >= 0.3  # Threshold to avoid accidental triggers
                
                if is_pressed and _debug_counter % 50 == 1:
                    print(f"[VR Paint] TRIGGER pressed: {pressure:.2f}")
                
                return is_pressed, pressure
            
            # Handle direct float
            elif isinstance(value, (int, float)):
                pressure = float(value)
                is_pressed = pressure >= 0.3
                return is_pressed, pressure
                
    except Exception as e:
        if _debug_counter % 500 == 1:
            print(f"[VR Paint] Trigger query error: {e}")
    
    return False, 0.0


def register_paint_action(context) -> bool:
    """
    Initialize paint action binding.
    
    We use the existing 'teleport' action which is already registered
    with OpenXR before the session starts.
    
    Returns:
        True (always succeeds since we use existing action)
    """
    global _initialized
    
    if _initialized:
        return True
    
    print("[VR Paint] Using TRIGGER (teleport action) for painting")
    print("[VR Paint] Hold trigger to paint, release to finish stroke")
    
    _initialized = True
    return True


def is_actions_registered() -> bool:
    """Check if paint actions have been initialized."""
    return _initialized
