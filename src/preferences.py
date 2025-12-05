# preferences.py
# Addon preferences for 3DGS Painter

import bpy
from bpy.types import AddonPreferences
from bpy.props import BoolProperty, EnumProperty, StringProperty


def get_cuda_version_items(self, context):
    """Get available CUDA version options."""
    items = [
        ('AUTO', 'Auto-detect', 'Automatically detect CUDA version from system'),
        ('cu124', 'CUDA 12.4', 'Use CUDA 12.4 (requires CUDA 12.4+)'),
        ('cu121', 'CUDA 12.1', 'Use CUDA 12.1 (requires CUDA 12.1+)'),
        ('cu118', 'CUDA 11.8', 'Use CUDA 11.8 (requires CUDA 11.8+)'),
        ('cpu', 'CPU Only', 'Use CPU version (no GPU acceleration)'),
    ]
    return items


class ThreeDGSPainterPreferences(AddonPreferences):
    """Preferences for 3DGS Painter addon."""
    
    bl_idname = __package__
    
    # Dependency settings
    cuda_version: EnumProperty(
        name="CUDA Version",
        description="Select CUDA version for PyTorch installation",
        items=get_cuda_version_items,
        default=0,  # AUTO
    )
    
    auto_check_dependencies: BoolProperty(
        name="Check Dependencies on Startup",
        description="Automatically check for missing packages when Blender starts",
        default=True,
    )
    
    # Installation log (internal)
    install_log: StringProperty(
        name="Installation Log",
        description="Log of the last installation attempt",
        default="",
    )
    
    # Installation status (internal)
    is_installing: BoolProperty(
        name="Is Installing",
        description="Whether installation is in progress",
        default=False,
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Import here to avoid circular imports
        from .npr_core.dependencies import get_missing_packages, get_package_version
        from .npr_core.installer import PackageInstaller, PlatformInfo
        
        # Platform info
        platform_info = PlatformInfo()
        box = layout.box()
        box.label(text="System Information", icon='INFO')
        col = box.column(align=True)
        col.label(text=f"Platform: {platform_info.system} {platform_info.machine}")
        col.label(text=f"Python: {platform_info.python_version}")
        if platform_info.cuda_available:
            col.label(text=f"CUDA: {platform_info.cuda_version}", icon='CHECKMARK')
        else:
            col.label(text="CUDA: Not detected", icon='X')
        
        layout.separator()
        
        # Dependency status
        missing = get_missing_packages()
        installer = PackageInstaller()
        
        # Check if only gsplat is missing (common case after interrupted install)
        missing_names = [dep.name for dep in missing]
        only_gsplat_missing = missing_names == ['gsplat']
        
        box = layout.box()
        if missing:
            if only_gsplat_missing:
                box.label(text="gsplat Installation Required", icon='ERROR')
                box.label(text="  Other packages are already installed.")
                box.label(text="  Only gsplat needs to be installed.")
            else:
                box.label(text="Missing Dependencies", icon='ERROR')
                col = box.column(align=True)
                for dep in missing:
                    row = col.row()
                    row.label(text=f"  • {dep.name} {dep.version}", icon='X')
            
            box.separator()
            
            # CUDA version selector
            if platform_info.cuda_available:
                row = box.row()
                row.prop(self, "cuda_version")
            
            # Install button
            row = box.row()
            row.scale_y = 1.5
            row.enabled = not self.is_installing
            if self.is_installing:
                row.operator("threegds.install_dependencies", text="Installing...", icon='SORTTIME')
            else:
                if only_gsplat_missing:
                    row.operator("threegds.install_dependencies", text="Install gsplat", icon='IMPORT')
                else:
                    row.operator("threegds.install_dependencies", text="Install Dependencies", icon='IMPORT')
        else:
            box.label(text="All Dependencies Installed", icon='CHECKMARK')
            
            # Show installed versions (with error handling for DLL issues)
            col = box.column(align=True)
            try:
                torch_ver = get_package_version("torch")
                if torch_ver:
                    col.label(text=f"  • PyTorch {torch_ver}")
                else:
                    col.label(text="  • PyTorch (installed, version unknown)")
            except Exception:
                col.label(text="  • PyTorch (installed, failed to load)")
            
            try:
                numpy_ver = get_package_version("numpy")
                if numpy_ver:
                    col.label(text=f"  • NumPy {numpy_ver}")
            except Exception:
                pass
            
            # gsplat status and compile button
            box.separator()
            gsplat_box = box.box()
            gsplat_box.label(text="gsplat (CUDA Kernels)", icon='OUTLINER_DATA_LIGHTPROBE')
            
            try:
                gsplat_ver = get_package_version("gsplat")
                if gsplat_ver:
                    gsplat_box.label(text=f"  Package: v{gsplat_ver}")
                else:
                    gsplat_box.label(text="  Package: Installed")
            except Exception:
                gsplat_box.label(text="  Package: Not installed", icon='X')
            
            # Check if CUDA kernels are compiled
            try:
                # Try importing csrc to check compilation status
                import sys
                deps_path = installer.dependencies_path
                if str(deps_path) not in sys.path:
                    sys.path.insert(0, str(deps_path))
                
                # This is a quick check - don't actually import
                gsplat_box.label(text="  Status: Use 'Test gsplat' to verify")
            except Exception:
                pass
            
            # Compile gsplat button
            row = gsplat_box.row()
            row.scale_y = 1.2
            row.enabled = not self.is_installing and platform_info.cuda_available
            if not platform_info.cuda_available:
                row.label(text="CUDA required for gsplat", icon='ERROR')
            elif self.is_installing:
                row.operator("threegds.compile_gsplat", text="Compiling...", icon='SORTTIME')
            else:
                row.operator("threegds.compile_gsplat", text="Compile CUDA Kernels", icon='FILE_REFRESH')
            
            # MSVC status on Windows
            if platform_info.system == "Windows":
                if platform_info.vcvars64_path:
                    gsplat_box.label(text=f"  Compiler: MSVC (vcvars64.bat) ✓", icon='CHECKMARK')
                else:
                    gsplat_box.label(text="  Compiler: Visual Studio Build Tools not found!", icon='ERROR')
                    gsplat_box.label(text="    Install 'Desktop development with C++' workload")
            
            # Uninstall button
            box.separator()
            row = box.row()
            row.operator("threegds.uninstall_dependencies", text="Uninstall Dependencies", icon='TRASH')
            
            # Test buttons
            box.separator()
            box.label(text="Subprocess Tests", icon='PLAY')
            row = box.row(align=True)
            row.operator("threegds.test_subprocess", text="Test PyTorch", icon='SCRIPT')
            row.operator("threegds.test_subprocess_cuda", text="Test CUDA", icon='OUTLINER_DATA_LIGHTPROBE')
            row = box.row(align=True)
            row.operator("threegds.test_gsplat", text="Test gsplat", icon='SHADING_RENDERED')
            row.operator("threegds.kill_subprocess", text="Kill Subprocess", icon='CANCEL')
        
        layout.separator()
        
        # Settings
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        box.prop(self, "auto_check_dependencies")
        
        # Installation log
        if self.install_log:
            layout.separator()
            box = layout.box()
            box.label(text="Installation Log", icon='TEXT')
            
            # Display log lines (limit to last 30 lines for detailed logs)
            log_lines = self.install_log.split('\n')[-30:]
            col = box.column(align=True)
            for line in log_lines:
                if line.strip():
                    # Truncate long lines
                    display_line = line[:100] + "..." if len(line) > 100 else line
                    col.label(text=display_line)


# Registration
classes = [
    ThreeDGSPainterPreferences,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
