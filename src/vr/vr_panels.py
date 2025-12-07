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
        
        box = layout.box()
        box.label(text="VR Session", icon='VIEW_CAMERA')
        
        row = box.row(align=True)
        if is_running:
            row.operator("threegds.stop_vr_session", text="Stop VR", icon='CANCEL')
            row.label(text="Active", icon='CHECKMARK')
        else:
            row.operator("threegds.start_vr_session", text="Start VR", icon='PLAY')
        
        if is_running:
            layout.separator()
            
            # VR Freehand Paint (Tilt Brush style)
            box = layout.box()
            box.label(text="VR Freehand Paint", icon='BRUSH_DATA')
            row = box.row(align=True)
            row.operator("threegds.vr_freehand_paint", text="Start Freehand", icon='GREASEPENCIL')
            row.operator("threegds.vr_freehand_clear", text="Clear", icon='X')
            
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="SPACE: Hold to paint")
            col.label(text="ESC: Exit paint mode")
            col.label(text="Paint at controller tip")
            
            layout.separator()
            
            # Phase 1: VR Offscreen Display (gpu.offscreen + Plane)
            box = layout.box()
            box.label(text="VR Offscreen Display", icon='TEXTURE')
            row = box.row(align=True)
            row.operator("threegds.vr_offscreen_test", text="Test Display", icon='IMAGE_DATA')
            row.operator("threegds.vr_offscreen_cleanup", text="Cleanup", icon='TRASH')
            
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Phase 1: Renders to 2D Plane")
            col.label(text="Visible in VR headset!")
            
            layout.separator()
            
            # Legacy VR Paint (surface-based)
            box = layout.box()
            box.label(text="VR Surface Paint (Legacy)", icon='MOD_SMOOTH')
            box.operator("threegds.vr_paint_stroke", text="Enter Surface Paint", icon='SCULPTMODE_HLT')
            
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text="Right Trigger: Paint")
            col.label(text="Left Grip: Exit")
            
            layout.separator()
            
            box = layout.box()
            box.label(text="Testing", icon='EXPERIMENTAL')
            box.operator("threegds.test_vr_input", text="Test Controller", icon='OUTLINER_OB_ARMATURE')


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
