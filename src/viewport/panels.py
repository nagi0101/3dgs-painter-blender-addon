# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
UI panels for 3DGS Painter viewport.
"""

import bpy
from bpy.types import Panel


class NPR_PT_ViewportPanel(Panel):
    """Main viewport panel for 3DGS Painter"""
    bl_label = "3DGS Painter"
    bl_idname = "NPR_PT_viewport_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    
    def draw(self, context):
        layout = self.layout
        
        # Get renderer instance
        from .viewport_renderer import GaussianViewportRenderer
        renderer = GaussianViewportRenderer.get_instance()
        
        # Viewport rendering controls
        box = layout.box()
        box.label(text="Viewport Rendering", icon='SHADING_RENDERED')
        
        row = box.row(align=True)
        if renderer.enabled:
            row.operator("npr.disable_viewport_rendering", text="Disable", icon='PAUSE')
        else:
            row.operator("npr.enable_viewport_rendering", text="Enable", icon='PLAY')
        
        # Stats
        if renderer.enabled and renderer.data_manager.is_valid:
            col = box.column(align=True)
            col.label(text=f"Gaussians: {renderer.gaussian_count:,}")
            tex_info = renderer.data_manager.get_texture_info()
            col.label(text=f"Texture: {tex_info['texture_width']}×{tex_info['texture_height']}")
        
        # Rendering settings
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        
        col = box.column(align=True)
        col.prop(context.scene, "npr_use_depth_test", text="Depth Test")
        col.prop(context.scene, "npr_depth_bias", text="Depth Bias")
        
        # Test controls
        box = layout.box()
        box.label(text="Testing", icon='EXPERIMENTAL')
        
        row = box.row(align=True)
        row.operator("npr.generate_test_gaussians", text="Generate Test", icon='MESH_UVSPHERE')
        row.operator("npr.clear_gaussians", text="Clear", icon='TRASH')
        
        col = box.column(align=True)
        col.operator("npr.run_benchmark", text="Run Benchmark", icon='TIME')


class NPR_PT_DependenciesPanel(Panel):
    """Dependencies panel for 3DGS Painter"""
    bl_label = "Dependencies"
    bl_idname = "NPR_PT_dependencies_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Dependency status
        box = layout.box()
        box.label(text="Package Management", icon='PACKAGE')
        
        col = box.column(align=True)
        col.operator("threegds.check_dependencies", text="Check Status", icon='FILE_REFRESH')
        col.operator("threegds.install_dependencies", text="Install Packages", icon='IMPORT')
        
        col.separator()
        col.operator("threegds.test_subprocess", text="Test PyTorch", icon='GHOST_ENABLED')
        col.operator("threegds.test_subprocess_cuda", text="Test CUDA", icon='OUTLINER_DATA_LIGHTPROBE')


# Scene properties for rendering settings
def _register_scene_props():
    """Register scene-level properties for viewport rendering."""
    bpy.types.Scene.npr_use_depth_test = bpy.props.BoolProperty(
        name="Use Depth Test",
        description="Test gaussian depth against Blender scene geometry",
        default=True,
        update=_on_depth_test_changed
    )
    
    bpy.types.Scene.npr_depth_bias = bpy.props.FloatProperty(
        name="Depth Bias",
        description="Small offset to prevent z-fighting with scene geometry",
        default=0.0001,
        min=0.0,
        max=0.01,
        precision=5,
        update=_on_depth_bias_changed
    )


def _unregister_scene_props():
    """Unregister scene-level properties."""
    del bpy.types.Scene.npr_use_depth_test
    del bpy.types.Scene.npr_depth_bias


def _on_depth_test_changed(self, context):
    """Callback when depth test setting changes."""
    from .viewport_renderer import GaussianViewportRenderer
    renderer = GaussianViewportRenderer.get_instance()
    renderer.use_depth_test = context.scene.npr_use_depth_test
    renderer.request_redraw()


def _on_depth_bias_changed(self, context):
    """Callback when depth bias setting changes."""
    from .viewport_renderer import GaussianViewportRenderer
    renderer = GaussianViewportRenderer.get_instance()
    renderer.depth_bias = context.scene.npr_depth_bias
    renderer.request_redraw()


# Panel classes
panel_classes = [
    NPR_PT_ViewportPanel,
    NPR_PT_DependenciesPanel,
]


def register_panels():
    """Register UI panels."""
    _register_scene_props()
    
    for cls in panel_classes:
        bpy.utils.register_class(cls)


def unregister_panels():
    """Unregister UI panels."""
    for cls in reversed(panel_classes):
        bpy.utils.unregister_class(cls)
    
    _unregister_scene_props()
