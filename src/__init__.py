bl_info = {
    "name": "3DGS Painter",
    "author": "Jiin Park",
    "description": "Non-photorealistic 3D Gaussian Splatting painting tools",
    "blender": (4, 2, 0),
    "version": (1, 0, 0),
    "location": "View3D > Sidebar > 3DGS Paint",
    "category": "Paint",
}

# =============================================================================
# Subprocess Actor Pattern (Dream Textures style)
# =============================================================================
# 
# IMPORTANT: PyTorch/CUDA is loaded in a SUBPROCESS, not in Blender's main process.
# This is required because Blender's bundled TBB (tbb12.dll) conflicts with PyTorch's
# c10.dll initialization on Windows.
#
# Architecture:
#   Blender Process (main)
#     ├── GLSL Viewport Rendering (60 FPS)
#     ├── UI / Modal Operators
#     ├── NumPy data processing
#     └── IPC Client (Queue + SharedMemory)
#                    │
#   Subprocess ("__actor__")
#     ├── PyTorch + CUDA (no TBB conflict)
#     ├── gsplat computation
#     └── Heavy operations (deformation, inpainting)
#
# =============================================================================

from multiprocessing import current_process

# Check if we are in the subprocess BEFORE any other imports
_is_actor_process = current_process().name == "__actor__"

if not _is_actor_process:
    # Main Blender process - import bpy and register addon
    import sys
    import os
    import site
    from pathlib import Path
    from typing import List, Type
    
    import bpy
    
    # =========================================================================
    # Path Utilities
    # =========================================================================
    
    def _get_addon_path() -> Path:
        """Get the path to this addon's directory."""
        return Path(__file__).parent

    def _absolute_path(path: str) -> str:
        """Get absolute path relative to addon directory."""
        return str(_get_addon_path() / path)
    
    # =========================================================================
    # Dependency Path Setup (for non-PyTorch packages like NumPy)
    # =========================================================================
    
    def _setup_dependency_paths():
        """
        Add .python_dependencies to sys.path for lightweight packages.
        NOTE: PyTorch/CUDA will be loaded in subprocess only!
        This path setup allows numpy and other non-TBB packages to be used
        in the main Blender process.
        """
        deps_path = _get_addon_path() / ".python_dependencies"
        if deps_path.exists():
            deps_str = str(deps_path)
            if deps_str not in sys.path:
                sys.path.insert(0, deps_str)
                # Also add to site-packages for proper package discovery
                site.addsitedir(deps_str)
                print(f"[3DGS Painter] Added dependency path: {deps_str}")
    
    # Setup paths immediately
    _setup_dependency_paths()

    # =========================================================================
    # Module Imports (main process only)
    # =========================================================================
    
    from . import preferences
    from . import operators
    from .viewport import viewport_renderer
    from .viewport import panels as viewport_panels
    from .viewport import benchmark as viewport_benchmark

    # Will hold all Blender classes to register
    _classes: List[Type] = []

    # =========================================================================
    # Startup Dependency Check
    # =========================================================================

    def _check_dependencies_on_startup():
        """
        Check dependencies when addon loads.
        Show warning if packages are missing.
        """
        try:
            from .npr_core.dependencies import get_missing_packages
            missing = get_missing_packages()
            
            if missing:
                def draw_warning(self, context):
                    layout = self.layout
                    layout.label(text="3DGS Painter: Missing dependencies!", icon='ERROR')
                    layout.label(text="Go to Edit > Preferences > Add-ons > 3DGS Painter")
                    layout.label(text="to install required packages.")
                
                # Use timer to delay popup (allows Blender to fully initialize)
                def show_popup():
                    bpy.context.window_manager.popup_menu(draw_warning, title="Warning", icon='ERROR')
                    return None  # Don't repeat
                
                bpy.app.timers.register(show_popup, first_interval=1.0)
        except ImportError as e:
            print(f"[3DGS Painter] Could not check dependencies: {e}")

    # =========================================================================
    # Registration
    # =========================================================================

    def register():
        """Register all Blender classes"""
        # Register submodules
        preferences.register()
        operators.register()
        viewport_renderer.register_viewport_operators()
        viewport_panels.register_panels()
        viewport_benchmark.register_benchmark()
        
        # Register local classes
        for cls in _classes:
            bpy.utils.register_class(cls)
        
        # Check dependencies on startup (optional, based on preference)
        try:
            prefs = bpy.context.preferences.addons[__package__].preferences
            if prefs.auto_check_dependencies:
                _check_dependencies_on_startup()
        except (KeyError, AttributeError):
            # Preferences not yet available, check anyway
            _check_dependencies_on_startup()
        
        print("[3DGS Painter] Addon registered (Subprocess Actor mode)")

    def unregister():
        """Unregister all Blender classes"""
        # Kill generator subprocess if running
        try:
            from .generator_process import kill_generator
            kill_generator()
        except:
            pass
        
        # Unregister viewport module (cleanup renderer)
        viewport_benchmark.unregister_benchmark()
        viewport_panels.unregister_panels()
        viewport_renderer.unregister_viewport_operators()
        
        # Unregister local classes
        for cls in reversed(_classes):
            bpy.utils.unregister_class(cls)
        
        # Unregister submodules
        operators.unregister()
        preferences.unregister()
        
        print("[3DGS Painter] Addon unregistered")
