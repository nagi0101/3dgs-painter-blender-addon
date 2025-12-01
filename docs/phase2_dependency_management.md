# Phase 2: 의존성 관리 (Dependency Management)

**기간**: 1주  
**목표**: Blender addon 내에서 Python 패키지 자동 설치 (Dream Textures 참고)

---

## 📋 작업 개요

본 Phase는 **사용자 친화적인 패키지 설치 시스템** 구현입니다:
- ✓ Dream Textures 방식 참고 (pip install inside Blender)
- ✓ Platform detection (Windows/macOS/Linux)
- ✓ Progress feedback (UI 진행 상태 표시)
- ✓ Error handling (네트워크 오류, 권한 문제 등)

---

## 🎯 핵심 작업

### 1. 패키지 목록 정의

#### 1.1 requirements.txt

```
# npr_gaussian_painter/requirements.txt

torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
scipy>=1.11.0
PyYAML>=6.0
gsplat>=0.1.0
```

#### 1.2 패키지 정보 클래스

```python
# npr_core/dependencies.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DependencyInfo:
    """Information about a required package."""
    name: str
    version: str
    import_name: Optional[str] = None  # If different from package name
    platform_specific: Optional[str] = None  # e.g., "windows", "linux"
    
    def __post_init__(self):
        if self.import_name is None:
            self.import_name = self.name

REQUIRED_PACKAGES = [
    DependencyInfo("torch", ">=2.0.0", import_name="torch"),
    DependencyInfo("torchvision", ">=0.15.0", import_name="torchvision"),
    DependencyInfo("numpy", ">=1.24.0", import_name="numpy"),
    DependencyInfo("pillow", ">=10.0.0", import_name="PIL"),
    DependencyInfo("scipy", ">=1.11.0", import_name="scipy"),
    DependencyInfo("pyyaml", ">=6.0", import_name="yaml"),
    DependencyInfo("gsplat", ">=0.1.0", import_name="gsplat"),
]

def get_missing_packages():
    """
    Check which packages are missing.
    
    Returns:
        List[DependencyInfo]: List of missing packages
    """
    missing = []
    
    for dep in REQUIRED_PACKAGES:
        try:
            __import__(dep.import_name)
        except ImportError:
            missing.append(dep)
    
    return missing
```

---

### 2. Installer 구현 (Dream Textures 스타일)

#### 2.1 Core Installer

```python
# npr_core/installer.py

import subprocess
import sys
import os
import platform
from pathlib import Path

class PackageInstaller:
    """
    Install Python packages inside Blender.
    Reference: Dream Textures addon implementation.
    """
    
    def __init__(self):
        self.python_exe = self.get_python_executable()
        self.platform = platform.system()
    
    def get_python_executable(self):
        """
        Get path to Blender's Python executable.
        
        Returns:
            Path: Path to python executable
        """
        # Blender's bundled Python
        if self.platform == "Windows":
            # Windows: <blender>/X.X/python/bin/python.exe
            python_exe = Path(sys.executable).parent.parent / "python" / "bin" / "python.exe"
        elif self.platform == "Darwin":  # macOS
            # macOS: <blender>/X.X/python/bin/python3.x
            python_exe = Path(sys.executable).parent.parent / "python" / "bin" / "python3.10"
        else:  # Linux
            # Linux: <blender>/X.X/python/bin/python3.x
            python_exe = Path(sys.executable).parent.parent / "python" / "bin" / "python3.10"
        
        if not python_exe.exists():
            # Fallback: use sys.executable
            python_exe = Path(sys.executable)
        
        return python_exe
    
    def ensure_pip(self):
        """
        Ensure pip is installed in Blender's Python.
        """
        try:
            import pip
        except ImportError:
            print("Installing pip...")
            subprocess.check_call([
                str(self.python_exe),
                "-m", "ensurepip", "--default-pip"
            ])
    
    def install_package(self, package_name, version_spec="", progress_callback=None):
        """
        Install a single package.
        
        Args:
            package_name: str, name of package
            version_spec: str, version specifier (e.g., ">=2.0.0")
            progress_callback: Optional[callable], callback(message: str)
        
        Returns:
            bool: True if successful
        """
        self.ensure_pip()
        
        # Construct package string
        if version_spec:
            package_str = f"{package_name}{version_spec}"
        else:
            package_str = package_name
        
        # Progress
        if progress_callback:
            progress_callback(f"Installing {package_name}...")
        
        try:
            # Run pip install
            result = subprocess.run([
                str(self.python_exe),
                "-m", "pip", "install",
                package_str,
                "--upgrade",
                "--no-cache-dir"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                if progress_callback:
                    progress_callback(f"✓ {package_name} installed successfully")
                return True
            else:
                if progress_callback:
                    progress_callback(f"✗ Failed to install {package_name}: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback(f"✗ Installation of {package_name} timed out")
            return False
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Error installing {package_name}: {str(e)}")
            return False
    
    def install_all(self, packages, progress_callback=None):
        """
        Install all packages from list.
        
        Args:
            packages: List[DependencyInfo]
            progress_callback: Optional[callable]
        
        Returns:
            tuple: (success: bool, failed_packages: List[str])
        """
        failed = []
        
        for dep in packages:
            success = self.install_package(
                dep.name,
                dep.version,
                progress_callback
            )
            
            if not success:
                failed.append(dep.name)
        
        return len(failed) == 0, failed
```

---

### 3. UI 통합

#### 3.1 Preferences Panel

```python
# ui.py (addon preferences)

import bpy
from bpy.types import AddonPreferences
from bpy.props import BoolProperty, StringProperty
from .npr_core.dependencies import get_missing_packages
from .npr_core.installer import PackageInstaller

class NPRGaussianPainterPreferences(AddonPreferences):
    bl_idname = "npr_gaussian_painter"
    
    # Properties
    auto_check_dependencies: BoolProperty(
        name="Auto Check Dependencies",
        description="Automatically check for missing packages on startup",
        default=True
    )
    
    install_log: StringProperty(
        name="Install Log",
        description="Log of installation process",
        default=""
    )
    
    def draw(self, context):
        layout = self.layout
        
        # Check dependencies
        missing = get_missing_packages()
        
        if missing:
            box = layout.box()
            box.label(text="⚠ Missing Dependencies", icon='ERROR')
            
            for dep in missing:
                row = box.row()
                row.label(text=f"  • {dep.name} {dep.version}")
            
            box.separator()
            
            row = box.row()
            row.scale_y = 1.5
            row.operator("npr_gaussian.install_dependencies", text="Install Dependencies", icon='IMPORT')
        else:
            box = layout.box()
            box.label(text="✓ All Dependencies Installed", icon='CHECKMARK')
        
        # Settings
        layout.separator()
        layout.prop(self, "auto_check_dependencies")
        
        # Install log
        if self.install_log:
            layout.separator()
            box = layout.box()
            box.label(text="Installation Log:")
            for line in self.install_log.split('\n'):
                box.label(text=line)
```

#### 3.2 Install Operator

```python
# operators.py

class InstallDependenciesOperator(bpy.types.Operator):
    """Install missing Python packages"""
    bl_idname = "npr_gaussian.install_dependencies"
    bl_label = "Install Dependencies"
    
    _timer = None
    _thread = None
    
    def __init__(self):
        self.installer = None
        self.missing_packages = []
        self.install_log = []
        self.finished = False
        self.success = False
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            # Check if installation finished
            if self.finished:
                # Update preferences with log
                prefs = context.preferences.addons["npr_gaussian_painter"].preferences
                prefs.install_log = '\n'.join(self.install_log)
                
                # Cleanup
                wm = context.window_manager
                wm.event_timer_remove(self._timer)
                
                if self.success:
                    self.report({'INFO'}, "Dependencies installed successfully")
                    return {'FINISHED'}
                else:
                    self.report({'ERROR'}, "Failed to install some dependencies")
                    return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        from .npr_core.dependencies import get_missing_packages
        from .npr_core.installer import PackageInstaller
        import threading
        
        # Get missing packages
        self.missing_packages = get_missing_packages()
        
        if not self.missing_packages:
            self.report({'INFO'}, "All dependencies already installed")
            return {'FINISHED'}
        
        # Initialize installer
        self.installer = PackageInstaller()
        
        # Progress callback
        def progress_callback(message):
            self.install_log.append(message)
            print(message)
        
        # Install in background thread
        def install_thread():
            self.success, failed = self.installer.install_all(
                self.missing_packages,
                progress_callback
            )
            self.finished = True
        
        self._thread = threading.Thread(target=install_thread)
        self._thread.start()
        
        # Setup timer for modal
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, "Installing dependencies...")
        return {'RUNNING_MODAL'}
```

---

### 4. 시작 시 자동 검사

#### 4.1 Startup Check

```python
# __init__.py

import bpy
from .npr_core.dependencies import get_missing_packages

def check_dependencies_on_startup():
    """
    Check dependencies when addon loads.
    Show warning if packages are missing.
    """
    missing = get_missing_packages()
    
    if missing:
        def draw_warning(self, context):
            layout = self.layout
            layout.label(text="NPR Gaussian Painter: Missing dependencies!", icon='ERROR')
            layout.label(text="Open Preferences > Add-ons to install.")
        
        bpy.context.window_manager.popup_menu(draw_warning, title="Warning", icon='ERROR')

def register():
    # ... register classes ...
    
    # Check dependencies
    prefs = bpy.context.preferences.addons[__name__].preferences
    if prefs.auto_check_dependencies:
        check_dependencies_on_startup()
```

---

### 5. Platform-Specific 처리

#### 5.1 PyTorch Platform Detection

```python
# npr_core/installer.py (additions)

class PackageInstaller:
    # ... existing code ...
    
    def get_torch_install_command(self):
        """
        Get platform-specific PyTorch install command.
        
        Returns:
            list: pip install arguments
        """
        import torch
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            # CUDA version
            cuda_version = torch.version.cuda
            if cuda_version.startswith("11"):
                index_url = "https://download.pytorch.org/whl/cu118"
            elif cuda_version.startswith("12"):
                index_url = "https://download.pytorch.org/whl/cu121"
            else:
                index_url = None
        else:
            # CPU only
            index_url = "https://download.pytorch.org/whl/cpu"
        
        args = ["torch>=2.0.0", "torchvision>=0.15.0"]
        
        if index_url:
            args.extend(["--index-url", index_url])
        
        return args
    
    def install_pytorch(self, progress_callback=None):
        """
        Install PyTorch with platform-specific settings.
        """
        self.ensure_pip()
        
        args = self.get_torch_install_command()
        
        if progress_callback:
            progress_callback(f"Installing PyTorch (this may take a while)...")
        
        try:
            result = subprocess.run([
                str(self.python_exe),
                "-m", "pip", "install"
            ] + args, capture_output=True, text=True, timeout=600)
            
            return result.returncode == 0
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Error installing PyTorch: {str(e)}")
            return False
```

---

## 🧪 테스트 및 검증

### 설치 테스트

```python
# Test script (run inside Blender console)

from npr_core.dependencies import get_missing_packages
from npr_core.installer import PackageInstaller

# Check missing
missing = get_missing_packages()
print(f"Missing packages: {[d.name for d in missing]}")

# Install
installer = PackageInstaller()

def progress_callback(msg):
    print(msg)

success, failed = installer.install_all(missing, progress_callback)

if success:
    print("✓ All packages installed")
else:
    print(f"✗ Failed packages: {failed}")
```

### Platform 테스트

- [ ] Windows 10/11 (CUDA 11.8/12.1)
- [ ] macOS 13+ (Metal)
- [ ] Linux (Ubuntu 22.04, CUDA)

---

## 📚 참고 자료

- Dream Textures addon: https://github.com/carson-katri/dream-textures
- Blender Python API: `sys.executable`, `ensurepip`
- PyTorch installation guide: https://pytorch.org/get-started/locally/

---

## 🎯 완료 기준

- ✓ Preferences panel에서 원클릭 설치 가능
- ✓ 모든 플랫폼에서 PyTorch 정상 설치
- ✓ Progress feedback 및 error handling 구현
- ✓ 시작 시 자동 검사 (선택 가능)
