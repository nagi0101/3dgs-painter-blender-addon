# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
UI panels for 3DGS Painter viewport.
"""

import bpy
from bpy.types import Panel, UIList, PropertyGroup
from bpy.props import StringProperty, IntProperty, CollectionProperty


# =============================================================================
# Property Groups for Brush Library
# =============================================================================

class NPR_BrushItem(PropertyGroup):
    """Property group for brush library items"""
    brush_id: StringProperty(
        name="Brush ID",
        description="Unique identifier for the brush"
    )
    name: StringProperty(
        name="Name",
        description="Display name of the brush"
    )
    brush_type: StringProperty(
        name="Type",
        description="Type of brush (programmatic, converted, imported)"
    )
    gaussian_count: IntProperty(
        name="Gaussians",
        description="Number of gaussians in the brush"
    )
    source: StringProperty(
        name="Source",
        description="Source of the brush (circular, line, grid, image)"
    )


class NPR_UL_BrushList(UIList):
    """UIList for displaying brushes in the library"""
    
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        # Choose icon based on source
        icon_map = {
            'circular': 'MESH_CIRCLE',
            'line': 'IPO_LINEAR',
            'grid': 'MESH_GRID',
            'image': 'IMAGE_DATA',
        }
        brush_icon = icon_map.get(item.source, 'BRUSH_DATA')
        
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=item.name, icon=brush_icon)
            row.label(text=f"{item.gaussian_count}g")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon=brush_icon)


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


class NPR_PT_PaintingPanel(Panel):
    """Painting controls panel for 3DGS Painter"""
    bl_label = "Painting"
    bl_idname = "NPR_PT_painting_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Tool activation info
        box = layout.box()
        box.label(text="Gaussian Paint Tool", icon='BRUSH_DATA')
        
        col = box.column(align=True)
        col.label(text="Select 'Gaussian Paint' tool", icon='INFO')
        col.label(text="from the toolbar (T) to paint", icon='BLANK1')
        
        col.separator()
        col.label(text="LMB drag: Paint stroke", icon='DOT')
        col.label(text="Change tool: Exit paint", icon='LOOP_BACK')
        
        # Brush settings
        box = layout.box()
        box.label(text="Brush Settings", icon='BRUSH_DATA')
        
        col = box.column(align=True)
        col.prop(scene, "npr_brush_size", text="Size")
        col.prop(scene, "npr_brush_opacity", text="Opacity")
        col.prop(scene, "npr_brush_spacing", text="Spacing")
        
        col.separator()
        col.prop(scene, "npr_brush_color", text="Color")
        
        # Brush pattern
        col.separator()
        col.prop(scene, "npr_brush_pattern", text="Pattern")
        col.prop(scene, "npr_brush_num_gaussians", text="Gaussians per Stamp")
        
        # Deformation settings
        box = layout.box()
        box.label(text="Deformation", icon='MOD_SIMPLEDEFORM')
        
        col = box.column(align=True)
        col.prop(scene, "npr_enable_deformation", text="Enable Deformation")
        
        # Actions
        box = layout.box()
        box.label(text="Actions", icon='ACTION')
        
        row = box.row(align=True)
        row.operator("threegds.clear_painted_gaussians", text="Clear All", icon='TRASH')


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
        col.operator("threegds.test_gsplat", text="Test gsplat", icon='SHADING_RENDERED')


class NPR_PT_BrushCreationPanel(Panel):
    """Brush creation panel for 3DGS Painter"""
    bl_label = "Brush Creation"
    bl_idname = "NPR_PT_brush_creation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Programmatic brushes section
        box = layout.box()
        box.label(text="Programmatic Brushes", icon='BRUSH_DATA')
        
        row = box.row(align=True)
        row.operator("threegds.create_brush_circular", text="Circular", icon='MESH_CIRCLE')
        row.operator("threegds.create_brush_line", text="Line", icon='IPO_LINEAR')
        row.operator("threegds.create_brush_grid", text="Grid", icon='MESH_GRID')
        
        # Image to Brush section
        box = layout.box()
        box.label(text="Image to Brush", icon='IMAGE_DATA')
        
        col = box.column(align=True)
        col.prop(scene, "npr_conversion_num_gaussians", text="Gaussians")
        col.prop(scene, "npr_conversion_depth_profile", text="Depth Profile")
        
        col.separator()
        col.prop(scene, "npr_conversion_skeleton_weight", text="Skeleton Weight")
        col.prop(scene, "npr_conversion_thickness_weight", text="Thickness Weight")
        
        col.separator()
        col.prop(scene, "npr_conversion_enable_elongation", text="Enable Elongation")
        
        col.separator()
        col.operator("threegds.convert_image_to_brush", text="Convert Image...", icon='IMPORT')


class NPR_PT_BrushLibraryPanel(Panel):
    """Brush library panel for 3DGS Painter"""
    bl_label = "Brush Library"
    bl_idname = "NPR_PT_brush_library_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3DGS Paint"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Refresh button at top
        row = layout.row(align=True)
        row.operator("threegds.refresh_brush_library", text="Refresh", icon='FILE_REFRESH')
        
        # Brush list
        row = layout.row()
        row.template_list(
            "NPR_UL_BrushList", "",
            scene, "npr_brush_library",
            scene, "npr_brush_library_index",
            rows=5
        )
        
        # Action buttons (vertical sidebar)
        col = row.column(align=True)
        col.operator("threegds.select_library_brush", text="", icon='CHECKMARK')
        col.operator("threegds.rename_library_brush", text="", icon='OUTLINER_DATA_GP_LAYER')
        col.operator("threegds.delete_library_brush", text="", icon='TRASH')
        
        # Selected brush info
        if scene.npr_brush_library and 0 <= scene.npr_brush_library_index < len(scene.npr_brush_library):
            selected = scene.npr_brush_library[scene.npr_brush_library_index]
            
            box = layout.box()
            box.label(text="Selected Brush", icon='INFO')
            
            col = box.column(align=True)
            col.label(text=f"Name: {selected.name}")
            col.label(text=f"Type: {selected.brush_type}")
            col.label(text=f"Gaussians: {selected.gaussian_count}")
            
            # Show if this is the active brush for painting
            if scene.npr_selected_brush_id == selected.brush_id:
                col.label(text="✓ Active for painting", icon='CHECKMARK')
        
        # Currently active brush
        box = layout.box()
        box.label(text="Active Brush", icon='BRUSH_DATA')
        
        if scene.npr_selected_brush_id:
            # Find active brush name
            active_name = "Unknown"
            for item in scene.npr_brush_library:
                if item.brush_id == scene.npr_selected_brush_id:
                    active_name = item.name
                    break
            box.label(text=f"Using: {active_name}")
        else:
            box.label(text="Using: Pattern-based (default)")
            box.label(text=f"Pattern: {scene.npr_brush_pattern}")


# Scene properties for rendering settings
def _register_scene_props():
    """Register scene-level properties for viewport rendering."""
    # Viewport settings
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
    
    # Brush settings (Phase 4)
    bpy.types.Scene.npr_brush_size = bpy.props.FloatProperty(
        name="Brush Size",
        description="Size multiplier for brush strokes",
        default=1.0,
        min=0.01,
        max=10.0
    )
    
    bpy.types.Scene.npr_brush_opacity = bpy.props.FloatProperty(
        name="Brush Opacity",
        description="Opacity of brush strokes",
        default=0.8,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.npr_brush_spacing = bpy.props.FloatProperty(
        name="Brush Spacing",
        description="Spacing between stamps relative to brush size",
        default=0.3,
        min=0.05,
        max=2.0
    )
    
    bpy.types.Scene.npr_brush_color = bpy.props.FloatVectorProperty(
        name="Brush Color",
        description="Color of brush strokes",
        subtype='COLOR',
        default=(0.8, 0.2, 0.2),
        min=0.0,
        max=1.0,
        size=3
    )
    
    bpy.types.Scene.npr_brush_pattern = bpy.props.EnumProperty(
        name="Brush Pattern",
        description="Pattern of gaussians in each brush stamp",
        items=[
            ('CIRCULAR', "Circular", "Gaussians arranged in a circle"),
            ('LINE', "Line", "Gaussians arranged in a line"),
            ('GRID', "Grid", "Gaussians arranged in a grid"),
        ],
        default='CIRCULAR'
    )
    
    bpy.types.Scene.npr_brush_num_gaussians = bpy.props.IntProperty(
        name="Gaussians per Stamp",
        description="Number of gaussians in each brush stamp",
        default=20,
        min=1,
        max=500
    )
    
    # Deformation settings (Phase 4)
    bpy.types.Scene.npr_enable_deformation = bpy.props.BoolProperty(
        name="Enable Deformation",
        description="Apply spline-based deformation to strokes",
        default=False
    )
    
    # Brush conversion settings (Phase 4.5)
    bpy.types.Scene.npr_conversion_num_gaussians = bpy.props.IntProperty(
        name="Target Gaussians",
        description="Number of Gaussians to generate from image",
        default=100,
        min=10,
        max=1000
    )
    
    bpy.types.Scene.npr_conversion_depth_profile = bpy.props.EnumProperty(
        name="Depth Profile",
        description="How depth is estimated from image features",
        items=[
            ('FLAT', "Flat", "Minimal depth variation"),
            ('CONVEX', "Convex", "Center bulges outward (skeleton high)"),
            ('CONCAVE', "Concave", "Center depressed inward"),
            ('RIDGE', "Ridge", "Sharp ridge along skeleton"),
        ],
        default='CONVEX'
    )
    
    bpy.types.Scene.npr_conversion_skeleton_weight = bpy.props.FloatProperty(
        name="Skeleton Weight",
        description="Weight of skeleton proximity in depth estimation",
        default=0.7,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.npr_conversion_thickness_weight = bpy.props.FloatProperty(
        name="Thickness Weight",
        description="Weight of thickness in depth estimation",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    bpy.types.Scene.npr_conversion_enable_elongation = bpy.props.BoolProperty(
        name="Enable Elongation",
        description="Elongate Gaussians along stroke direction",
        default=True
    )
    
    # Brush library properties (Phase 4.5)
    bpy.types.Scene.npr_brush_library = bpy.props.CollectionProperty(
        type=NPR_BrushItem,
        name="Brush Library",
        description="Collection of saved brushes"
    )
    
    bpy.types.Scene.npr_brush_library_index = bpy.props.IntProperty(
        name="Active Brush Index",
        description="Index of selected brush in library list",
        default=0
    )
    
    bpy.types.Scene.npr_selected_brush_id = bpy.props.StringProperty(
        name="Selected Brush ID",
        description="ID of the brush selected for painting (empty = use pattern-based)",
        default=""
    )


def _unregister_scene_props():
    """Unregister scene-level properties."""
    props_to_remove = [
        'npr_use_depth_test',
        'npr_depth_bias',
        'npr_brush_size',
        'npr_brush_opacity',
        'npr_brush_spacing',
        'npr_brush_color',
        'npr_brush_pattern',
        'npr_brush_num_gaussians',
        'npr_enable_deformation',
        # Phase 4.5 brush conversion properties
        'npr_conversion_num_gaussians',
        'npr_conversion_depth_profile',
        'npr_conversion_skeleton_weight',
        'npr_conversion_thickness_weight',
        'npr_conversion_enable_elongation',
        # Phase 4.5 brush library properties
        'npr_brush_library',
        'npr_brush_library_index',
        'npr_selected_brush_id',
    ]
    
    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)


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
    NPR_PT_PaintingPanel,
    NPR_PT_BrushCreationPanel,
    NPR_PT_BrushLibraryPanel,
    NPR_PT_DependenciesPanel,
]

# PropertyGroup classes (must be registered before panels)
property_classes = [
    NPR_BrushItem,
]

# UIList classes
uilist_classes = [
    NPR_UL_BrushList,
]


def register_panels():
    """Register UI panels."""
    # Register property groups first
    for cls in property_classes:
        bpy.utils.register_class(cls)
    
    # Register UIList classes
    for cls in uilist_classes:
        bpy.utils.register_class(cls)
    
    # Register scene properties (after PropertyGroups)
    _register_scene_props()
    
    # Register panels
    for cls in panel_classes:
        bpy.utils.register_class(cls)


def unregister_panels():
    """Unregister UI panels."""
    # Unregister panels first
    for cls in reversed(panel_classes):
        bpy.utils.unregister_class(cls)
    
    # Unregister scene properties
    _unregister_scene_props()
    
    # Unregister UIList classes
    for cls in reversed(uilist_classes):
        bpy.utils.unregister_class(cls)
    
    # Unregister property groups last
    for cls in reversed(property_classes):
        bpy.utils.unregister_class(cls)
