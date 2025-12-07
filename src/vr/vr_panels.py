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
        # VR Session Control (always visible)
        # ============================================
        box = layout.box()
        box.label(text="VR Session", icon='VIEW_CAMERA')
        
        row = box.row(align=True)
        if is_running:
            row.operator("threegds.stop_vr_session", text="Stop VR", icon='CANCEL')
            row.label(text="Active", icon='CHECKMARK')
        else:
            row.operator("threegds.start_vr_session", text="Start VR", icon='PLAY')
        
        layout.separator()
        
        # ============================================
        # Phase 2: VR RenderEngine Test (ALWAYS VISIBLE!)
        # ============================================
        box = layout.box()
        box.label(text="Phase 2: VR RenderEngine", icon='SCENE')
        
        current_engine = context.scene.render.engine
        box.label(text=f"Engine: {current_engine}")
        
        row = box.row(align=True)
        row.operator("threegds.vr_engine_test", text="Test VR Engine", icon='PLAY')
        row.operator("threegds.vr_engine_stats", text="Stats", icon='INFO')
        
        row = box.row(align=True)
        row.operator("threegds.vr_engine_clear", text="Clear", icon='X')
        row.operator("threegds.switch_to_eevee", text="Eevee", icon='SHADING_RENDERED')
        
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="1. Test VR Engine (PC first)")
        col.label(text="2. Start VR Session")
        col.label(text="3. Check Stats!")
        
        # ============================================
        # VR Running sections
        # ============================================
        if is_running:
            layout.separator()
            
            # VR Freehand Paint
            box = layout.box()
            box.label(text="VR Freehand Paint", icon='BRUSH_DATA')
            row = box.row(align=True)
            row.operator("threegds.vr_freehand_paint", text="Freehand", icon='GREASEPENCIL')
            row.operator("threegds.vr_freehand_clear", text="Clear", icon='X')
            
            layout.separator()
            
            # Testing
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
