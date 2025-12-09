import bpy
from bpy.types import Panel
from .vr_session import get_vr_session_manager


class NPR_PT_VRPanel(Panel):
    """VR Painting Panel"""
    bl_label = "VR Painting"
    bl_idname = "NPR_PT_vr_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '3DGS Paint'
    bl_order = 50
    
    def draw(self, context):
        layout = self.layout
        mgr = get_vr_session_manager()
        
        if not mgr.is_vr_available():
            layout.label(text="VR not available", icon='ERROR')
            return
        
        is_running = mgr.is_session_running()
        
        # ============================================
        # VR Session Control
        # ============================================
        box = layout.box()
        box.label(text="VR Session", icon='VIEW_CAMERA')
        
        row = box.row(align=True)
        if is_running:
            row.operator("threegds.stop_vr_session", text="Stop VR", icon='CANCEL')
            row.label(text="Active", icon='CHECKMARK')
        else:
            row.operator("threegds.start_vr_session", text="Start VR", icon='PLAY')
        
        # Usage instructions
        if not is_running:
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Start VR to paint")
            col.label(text="TRIGGER = Paint")
        
        # ============================================
        # VR Running sections
        # ============================================
        if is_running:
            layout.separator()
            
            # VR Paint Controls
            box = layout.box()
            box.label(text="VR Paint", icon='BRUSH_DATA')
            
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="TRIGGER: Paint")
            col.label(text="Left Stick: Move")
            
            row = box.row(align=True)
            row.operator("threegds.vr_freehand_clear", text="Clear All", icon='X')
            
            layout.separator()
            
            # Debug
            box = layout.box()
            box.label(text="Debug", icon='EXPERIMENTAL')
            row = box.row(align=True)
            row.operator("threegds.test_vr_input", text="Test Controller", icon='OUTLINER_OB_ARMATURE')
            row.operator("threegds.openxr_layer_test", text="Test Gaussians", icon='EXPORT')


classes = [NPR_PT_VRPanel]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
