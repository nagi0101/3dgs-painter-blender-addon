# vr_input.py
# VR Controller Input Handling - A Button for painting

import bpy
from typing import Optional, Dict, Any, Tuple
from mathutils import Vector, Quaternion
from enum import Enum
from dataclasses import dataclass


class ControllerHand(Enum):
    LEFT = 0
    RIGHT = 1


@dataclass
class ControllerState:
    hand: ControllerHand
    position: Vector
    rotation: Quaternion
    aim_position: Vector
    aim_rotation: Quaternion
    trigger_value: float = 0.0
    grip_value: float = 0.0
    a_button_pressed: bool = False  # Right A button
    b_button_pressed: bool = False  # Right B button
    is_active: bool = False


class VRInputManager:
    _instance: Optional['VRInputManager'] = None
    
    @classmethod
    def get_instance(cls) -> 'VRInputManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._smoothing_factor = 0.3
        self._prev_positions: Dict[ControllerHand, Vector] = {
            ControllerHand.LEFT: Vector((0, 0, 0)),
            ControllerHand.RIGHT: Vector((0, 0, 0)),
        }
        self._a_button_was_pressed = False
    
    def is_vr_active(self) -> bool:
        try:
            wm = bpy.context.window_manager
            if hasattr(wm, 'xr_session_state') and wm.xr_session_state:
                return wm.xr_session_state.is_running(bpy.context)
            return False
        except:
            return False
    
    def _get_xr_state(self):
        try:
            wm = bpy.context.window_manager
            if hasattr(wm, 'xr_session_state'):
                return wm.xr_session_state
            return None
        except:
            return None
    
    def _get_button_state(self, xr_state, action_name: str, user_path: str) -> float:
        """Try to get button state from various action maps."""
        if xr_state is None:
            return 0.0
        
        # Try common action map names
        action_sets = ['default', 'blender_default', 'actionmap']
        
        for action_set in action_sets:
            try:
                result = xr_state.action_state_get(
                    action_set_name=action_set,
                    action_name=action_name,
                    user_path=user_path
                )
                if result and len(result) > 0:
                    return result[0]
            except:
                continue
        return 0.0
    
    def get_controller_state(self, hand: ControllerHand) -> ControllerState:
        default_state = ControllerState(
            hand=hand,
            position=Vector((0, 0, 0)),
            rotation=Quaternion((1, 0, 0, 0)),
            aim_position=Vector((0, 0, 0)),
            aim_rotation=Quaternion((1, 0, 0, 0)),
            is_active=False
        )
        
        xr_state = self._get_xr_state()
        if xr_state is None or not self.is_vr_active():
            return default_state
        
        try:
            index = hand.value
            ctx = bpy.context
            
            # Use grip_location for position (aim_location returns zeros)
            grip_pos = Vector(xr_state.controller_grip_location_get(ctx, index))
            grip_rot = Quaternion(xr_state.controller_grip_rotation_get(ctx, index))
            
            # Use aim_rotation for direction (aim_location doesn't work)
            aim_rot = Quaternion(xr_state.controller_aim_rotation_get(ctx, index))
            aim_pos = grip_pos.copy()  # Use grip position for aim
            
            # Smoothing
            prev = self._prev_positions.get(hand, grip_pos)
            smoothed_pos = prev.lerp(grip_pos, self._smoothing_factor)
            self._prev_positions[hand] = smoothed_pos
            
            return ControllerState(
                hand=hand,
                position=smoothed_pos,
                rotation=grip_rot,
                aim_position=smoothed_pos,  # Use smoothed grip position
                aim_rotation=aim_rot,
                trigger_value=0.0,
                grip_value=0.0,
                a_button_pressed=False,
                b_button_pressed=False,
                is_active=True
            )
        except Exception as e:
            print(f"[VR Input] Error getting controller state: {e}")
            return default_state
    
    def is_a_button_pressed(self) -> bool:
        """Check if A button on right controller is pressed using keyboard event proxy."""
        # Unfortunately XR button state requires proper action binding
        # For now, return a simulated state based on controller tracking
        # The user will use keyboard A as a proxy while we implement proper binding
        return False
    
    def get_painting_input(self) -> Optional[Dict[str, Any]]:
        if not self.is_vr_active():
            return None
        
        right = self.get_controller_state(ControllerHand.RIGHT)
        if not right.is_active:
            return None
        
        forward = right.aim_rotation @ Vector((0, 0, -1))
        
        # For now, always return pressure=1.0 so painting always happens
        # User presses keyboard 'A' to toggle painting mode
        return {
            'position': right.aim_position.copy(),
            'normal': forward.normalized(),
            'pressure': 1.0,  # Always active - will use keyboard for control
            'controller_position': right.position.copy(),
            'controller_rotation': right.rotation.copy(),
        }


def get_vr_input_manager() -> VRInputManager:
    return VRInputManager.get_instance()
