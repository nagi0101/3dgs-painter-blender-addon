# VR Module for 3DGS Painter
from .vr_session import VRSessionManager, get_vr_session_manager
from .vr_input import VRInputManager, get_vr_input_manager, ControllerHand, ControllerState
from .action_maps import VRActionMapManager, get_action_map_manager
from . import vr_operators
from . import vr_panels
from . import vr_freehand_paint
from . import vr_shared_memory

__version__ = "0.1.0"

def register():
    vr_operators.register()
    vr_panels.register()
    vr_freehand_paint.register()
    # Initialize shared memory for OpenXR Layer communication
    if vr_shared_memory.init_shared_memory():
        print("[3DGS Painter VR] Shared memory initialized for OpenXR Layer")
    print("[3DGS Painter VR] VR module registered")

def unregister():
    # Shutdown shared memory
    try:
        vr_shared_memory.shutdown_shared_memory()
    except:
        pass
    try:
        vr_freehand_paint.unregister()
    except:
        pass
    try:
        vr_panels.unregister()
    except:
        pass
    try:
        vr_operators.unregister()
    except:
        pass
    print("[3DGS Painter VR] VR module unregistered")
