# installer.py
# Package installer for 3DGS Painter addon
# Reference: Dream Textures addon implementation

import subprocess
import sys
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from enum import Enum


class CUDAVersion(Enum):
    """Supported CUDA versions for PyTorch."""
    CUDA_118 = "cu118"  # CUDA 11.8
    CUDA_121 = "cu121"  # CUDA 12.1
    CUDA_124 = "cu124"  # CUDA 12.4
    CPU = "cpu"         # CPU only
    MPS = "mps"         # Apple Metal (macOS)


class PlatformInfo:
    """Detected platform information."""
    
    def __init__(self):
        self.system = platform.system()  # Windows, Darwin, Linux
        self.machine = platform.machine()  # x86_64, arm64, etc.
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.cuda_version: Optional[str] = None
        self.cuda_available = False
        self.vcvars64_path: Optional[str] = None  # Path to vcvars64.bat on Windows
        self.vs_install_path: Optional[str] = None  # Visual Studio installation path
        
        self._detect_cuda()
        self._detect_visual_studio()
    
    def _detect_visual_studio(self):
        """Detect Visual Studio installation on Windows.
        
        Uses Microsoft's official vswhere.exe tool to locate Visual Studio installations,
        then finds vcvars64.bat for MSVC environment setup.
        """
        if self.system != "Windows":
            return
        
        vcvars64, vs_path = self._find_vcvars64()
        if vcvars64:
            self.vcvars64_path = vcvars64
            self.vs_install_path = vs_path
    
    def _find_vcvars64(self) -> Tuple[Optional[str], Optional[str]]:
        """Find vcvars64.bat using Microsoft's vswhere.exe tool.
        
        vcvars64.bat is required to set up the MSVC compiler environment
        for building CUDA extensions on Windows.
        
        Returns:
            Tuple of (vcvars64.bat path, VS installation path) or (None, None)
        """
        # vswhere.exe is typically located here
        program_files_x86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
        vswhere_path = os.path.join(
            program_files_x86,
            'Microsoft Visual Studio',
            'Installer',
            'vswhere.exe'
        )
        
        if not os.path.exists(vswhere_path):
            return None, None
        
        try:
            # Use vswhere to find Visual Studio installation path
            result = subprocess.run(
                [
                    vswhere_path,
                    '-latest',
                    '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                    '-property', 'installationPath'
                ],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result.returncode == 0 and result.stdout.strip():
                vs_path = result.stdout.strip().split('\n')[0]
                vcvars64_path = os.path.join(
                    vs_path,
                    'VC', 'Auxiliary', 'Build', 'vcvars64.bat'
                )
                if os.path.exists(vcvars64_path):
                    return vcvars64_path, vs_path
            
            # Fallback: Check common paths for Build Tools
            common_paths = [
                r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
                r'C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
                r'C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat',
                r'C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat',
                r'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
                r'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    vs_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
                    return path, vs_path
                    
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None, None
    
    def _detect_cuda(self):
        """Detect CUDA version using nvidia-smi or nvcc."""
        if self.system == "Darwin":
            # macOS doesn't have CUDA, uses Metal/MPS
            return
        
        # Method 1: Try nvidia-smi (most reliable for driver version)
        cuda_version = self._get_cuda_from_nvidia_smi()
        if cuda_version:
            self.cuda_version = cuda_version
            self.cuda_available = True
            return
        
        # Method 2: Try nvcc --version (CUDA toolkit version)
        cuda_version = self._get_cuda_from_nvcc()
        if cuda_version:
            self.cuda_version = cuda_version
            self.cuda_available = True
            return
        
        # Method 3: Check environment variable
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            # Try to extract version from path (e.g., "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1")
            match = re.search(r'v?(\d+\.\d+)', cuda_path)
            if match:
                self.cuda_version = match.group(1)
                self.cuda_available = True
                return
    
    def _get_cuda_from_nvidia_smi(self) -> Optional[str]:
        """Get CUDA version from nvidia-smi output."""
        try:
            # nvidia-smi shows the driver's CUDA version (maximum supported)
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if self.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                # Parse output: "CUDA Version: 12.1"
                match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None
    
    def _get_cuda_from_nvcc(self) -> Optional[str]:
        """Get CUDA version from nvcc compiler."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if self.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                # Parse output: "release 12.1, V12.1.105"
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None
    
    def get_recommended_cuda_version(self) -> CUDAVersion:
        """
        Get recommended PyTorch CUDA version based on detected CUDA.
        
        Returns:
            CUDAVersion enum for PyTorch installation
        """
        if self.system == "Darwin":
            return CUDAVersion.MPS
        
        if not self.cuda_available or not self.cuda_version:
            return CUDAVersion.CPU
        
        try:
            major, minor = map(int, self.cuda_version.split('.')[:2])
            cuda_numeric = major + minor / 10.0
            
            # PyTorch CUDA version mapping:
            # - CUDA 12.4+ -> cu124
            # - CUDA 12.1-12.3 -> cu121
            # - CUDA 11.8-12.0 -> cu118
            # - CUDA < 11.8 -> cu118 (may have compatibility issues)
            
            if cuda_numeric >= 12.4:
                return CUDAVersion.CUDA_124
            elif cuda_numeric >= 12.1:
                return CUDAVersion.CUDA_121
            else:
                return CUDAVersion.CUDA_118
                
        except (ValueError, AttributeError):
            return CUDAVersion.CPU
    
    def __str__(self) -> str:
        cuda_info = f"CUDA {self.cuda_version}" if self.cuda_available else "No CUDA"
        vs_info = ", MSVC" if self.vcvars64_path else ""
        return f"{self.system} {self.machine} Python {self.python_version} ({cuda_info}{vs_info})"


class PackageInstaller:
    """
    Install Python packages inside Blender.
    Reference: Dream Textures addon implementation.
    """
    
    def __init__(self, addon_path: Optional[Path] = None):
        """
        Initialize the package installer.
        
        Args:
            addon_path: Path to the addon directory. If None, uses the directory containing this file.
        """
        if addon_path is None:
            addon_path = Path(__file__).parent.parent
        
        self.addon_path = Path(addon_path)
        self.dependencies_path = self.addon_path / ".python_dependencies"
        self.requirements_path = self.addon_path / "requirements"
        self.platform_info = PlatformInfo()
        self.python_exe = self._get_python_executable()
    
    def _get_python_executable(self) -> Path:
        """
        Get path to Blender's Python executable.
        
        Returns:
            Path to python executable
        """
        # sys.executable in Blender points to the Python interpreter
        return Path(sys.executable)

    def _find_system_python(self, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[Path]:
        """
        Find a suitable system-wide Python 3.11 installation for compiling modules.
        """
        def log(msg):
            if progress_callback:
                progress_callback(f"  [PyFinder] {msg}")

        log("Searching for system-wide Python 3.11 for compilation...")

        blender_python_dir = Path(sys.executable).parent
        log(f"Blender's Python directory: {blender_python_dir.resolve()}")

        # 1. Check common names for Python 3.11
        for py_exe_name in ("python3.11", "python3", "python"):
            log(f"Checking for '{py_exe_name}' in PATH...")
            system_python = shutil.which(py_exe_name)
            
            if not system_python:
                log(f"'{py_exe_name}' not found.")
                continue

            py_path = Path(system_python)
            log(f"Found '{py_exe_name}' at: {py_path}")

            # 2. Ensure it's not Blender's own Python
            if py_path.parent.resolve() == blender_python_dir.resolve():
                log("This is Blender's own Python. Skipping.")
                continue

            # 3. Verify version is 3.11
            try:
                version_result = subprocess.run(
                    [str(py_path), "--version"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
                )
                version_str = version_result.stdout.strip()
                log(f"Version check: {version_str}")
                if " 3.11" not in version_str:
                    log("Version is not 3.11. Skipping.")
                    continue
            except (subprocess.TimeoutExpired, OSError) as e:
                log(f"Could not get version. Error: {e}. Skipping.")
                continue
            
            # 4. Check for 'include/Python.h'
            py_install_dir = py_path.parent
            header_path = py_install_dir / "include" / "Python.h"
            log(f"Checking for development headers at: {header_path}")
            
            if header_path.exists():
                log("Found compatible Python for compilation!")
                return py_path
            else:
                log("Development headers (Python.h) not found at this location. Skipping.")

        log("Could not find any suitable system-wide Python installation.")
        error_msg = (
            "[ERROR] Compatible Python 3.11 not found in PATH.\n"
            "  gsplat requires a full Python 3.11 installation to compile.\n"
            "  Please install Python 3.11 for Windows from python.org and\n"
            "  ensure 'Add python.exe to PATH' and 'Install development headers' are checked."
        )
        if progress_callback:
            for line in error_msg.split('\n'):
                progress_callback(line)
        return None

    def ensure_pip(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Ensure pip is installed in Blender's Python.
        
        Returns:
            True if pip is available
        """
        try:
            import pip
            return True
        except ImportError:
            if progress_callback:
                progress_callback("Installing pip...")
            
            try:
                subprocess.check_call(
                    [str(self.python_exe), "-m", "ensurepip", "--default-pip"],
                    creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
                )
                return True
            except subprocess.CalledProcessError as e:
                if progress_callback:
                    progress_callback(f"Failed to install pip: {e}")
                return False
    
    def get_pytorch_install_args(self, cuda_version: Optional[CUDAVersion] = None) -> List[str]:
        """
        Get pip install arguments for PyTorch based on platform and CUDA version.
        
        IMPORTANT: Use exact versions with +cuXXX suffix to prevent pip from
        installing a newer CPU-only version from PyPI.
        
        Args:
            cuda_version: Override CUDA version selection. If None, auto-detect.
        
        Returns:
            List of pip install arguments
        """
        if cuda_version is None:
            cuda_version = self.platform_info.get_recommended_cuda_version()
        
        # Use exact versions with CUDA suffix to prevent CPU version overwrite
        # PyTorch 2.4.0 is used for gsplat precompiled wheel compatibility
        if cuda_version == CUDAVersion.CPU:
            return [
                "torch==2.4.0+cpu",
                "torchvision==0.19.0+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        elif cuda_version == CUDAVersion.CUDA_118:
            return [
                "torch==2.4.0+cu118",
                "torchvision==0.19.0+cu118",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        elif cuda_version == CUDAVersion.CUDA_121:
            return [
                "torch==2.4.0+cu121",
                "torchvision==0.19.0+cu121",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
        elif cuda_version == CUDAVersion.CUDA_124:
            return [
                "torch==2.4.0+cu124",
                "torchvision==0.19.0+cu124",
                "--index-url", "https://download.pytorch.org/whl/cu124"
            ]
        elif cuda_version == CUDAVersion.MPS:
            # macOS - use default PyPI (has MPS support built-in)
            return ["torch==2.4.0", "torchvision==0.19.0"]
        else:
            # Fallback to CPU
            return [
                "torch==2.4.0+cpu",
                "torchvision==0.19.0+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
    
    def get_requirements_file(self) -> Path:
        """
        Get the appropriate requirements file for the current platform.
        
        Returns:
            Path to requirements file
        """
        system = self.platform_info.system
        
        if system == "Windows":
            if self.platform_info.cuda_available:
                return self.requirements_path / "win_cuda.txt"
            else:
                return self.requirements_path / "win_cpu.txt"
        elif system == "Darwin":
            return self.requirements_path / "mac_mps.txt"
        else:  # Linux
            if self.platform_info.cuda_available:
                return self.requirements_path / "linux_cuda.txt"
            else:
                return self.requirements_path / "win_cpu.txt"  # Use CPU version for Linux without CUDA
    
    def install_requirements(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Install packages from requirements file to target directory.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            Tuple of (success, failed_packages)
        """
        from .dependencies import get_missing_packages
        
        if not self.ensure_pip(progress_callback):
            return False, ["pip"]
        
        # Create dependencies directory
        self.dependencies_path.mkdir(parents=True, exist_ok=True)
        
        requirements_file = self.get_requirements_file()
        if not requirements_file.exists():
            if progress_callback:
                progress_callback(f"Requirements file not found: {requirements_file}")
            return False, [str(requirements_file)]
        
        if progress_callback:
            progress_callback(f"Installing from {requirements_file.name}...")
        
        try:
            result = subprocess.run(
                [
                    str(self.python_exe),
                    "-m", "pip", "install",
                    "-r", str(requirements_file),
                    "--target", str(self.dependencies_path),
                    "--upgrade",
                    "--upgrade-strategy", "only-if-needed",  # Don't upgrade torch if already installed
                    "--no-cache-dir"
                ],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            # pip may return non-zero due to dependency conflicts with other system packages
            # but still successfully install our required packages
            # Check if our target packages are actually installed
            still_missing = get_missing_packages(include_optional=False)
            still_missing_names = [dep.name for dep in still_missing]
            
            if not still_missing:
                if progress_callback:
                    progress_callback("[OK] Base packages installed successfully")
                return True, []
            elif result.returncode == 0:
                if progress_callback:
                    progress_callback("[OK] Base packages installed successfully")
                return True, []
            else:
                # Only report as failure if packages are actually still missing
                if progress_callback:
                    if "dependency conflicts" in result.stderr.lower() or "does not currently take into account" in result.stderr:
                        # This is just a warning about unrelated packages
                        progress_callback("[WARN] pip reported dependency conflicts (unrelated packages)")
                        progress_callback("  Verifying our packages were installed...")
                        if still_missing:
                            progress_callback(f"[FAILED] Still missing: {', '.join(still_missing_names)}")
                            return False, still_missing_names
                        else:
                            progress_callback("[OK] All required packages installed despite warnings")
                            return True, []
                    else:
                        progress_callback(f"[FAILED] Failed to install packages: {result.stderr[:500]}")
                return False, ["requirements"]
                
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback("[TIMEOUT] Installation timed out (10 minutes)")
            return False, ["timeout"]
        except Exception as e:
            if progress_callback:
                progress_callback(f"[ERROR] Error: {str(e)}")
            return False, [str(e)]
    
    def install_pytorch(
        self,
        cuda_version: Optional[CUDAVersion] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Install PyTorch with appropriate CUDA support.
        
        Args:
            cuda_version: Override CUDA version. If None, auto-detect.
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if successful
        """
        if not self.ensure_pip(progress_callback):
            return False
        
        # Create dependencies directory
        self.dependencies_path.mkdir(parents=True, exist_ok=True)
        
        if cuda_version is None:
            cuda_version = self.platform_info.get_recommended_cuda_version()
        
        if progress_callback:
            progress_callback(f"Installing PyTorch ({cuda_version.value})... This may take several minutes.")
        
        args = self.get_pytorch_install_args(cuda_version)
        
        try:
            result = subprocess.run(
                [
                    str(self.python_exe),
                    "-m", "pip", "install",
                    *args,
                    "--target", str(self.dependencies_path),
                    "--upgrade",
                    "--force-reinstall",  # Force reinstall to ensure correct CUDA version
                    "--no-deps",  # Don't install dependencies (handled separately)
                    "--no-cache-dir"
                ],
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minutes for PyTorch
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                if progress_callback:
                    progress_callback(f"[OK] PyTorch ({cuda_version.value}) installed successfully")
                return True
            else:
                if progress_callback:
                    progress_callback(f"[FAILED] Failed to install PyTorch: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback("[TIMEOUT] PyTorch installation timed out (20 minutes)")
            return False
        except Exception as e:
            if progress_callback:
                progress_callback(f"[ERROR] Error installing PyTorch: {str(e)}")
            return False
    
    def _setup_msvc_env(self) -> Optional[dict]:
        """
        Setup MSVC compiler environment by running vcvars64.bat.
        
        On Windows, gsplat's setup.py requires MSVC (cl.exe) to be properly configured.
        vcvars64.bat sets up the necessary environment variables (PATH, INCLUDE, LIB, etc.).
        
        This follows the official gsplat Windows installation guide:
        https://martinresearch.github.io/gsplat/docs/INSTALL_WIN.html
        
        Returns:
            Environment dict with MSVC configured, or None if vcvars64.bat not found
        """
        if self.platform_info.system != "Windows":
            return os.environ.copy()
        
        if not self.platform_info.vcvars64_path:
            return None
        
        # Run vcvars64.bat and capture the resulting environment variables
        # We use a cmd script that runs vcvars64.bat and then prints all env vars
        try:
            # Create a temporary batch script to capture environment after vcvars64.bat
            capture_script = f'''
@echo off
call "{self.platform_info.vcvars64_path}" >nul 2>&1
set
'''
            result = subprocess.run(
                ["cmd", "/c", capture_script],
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if result.returncode != 0:
                return None
            
            # Start with current environment (CRITICAL: pip needs Python paths!)
            env = os.environ.copy()
            
            # Parse the MSVC environment variables from vcvars64.bat output
            vcvars_env = {}
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, _, value = line.partition('=')
                    vcvars_env[key] = value
            
            # Only update MSVC-specific variables, preserve Python environment
            msvc_vars = [
                'PATH', 'INCLUDE', 'LIB', 'LIBPATH',
                'VCToolsInstallDir', 'VCToolsVersion', 'VCToolsRedistDir',
                'VSINSTALLDIR', 'VCINSTALLDIR',
                'WindowsLibPath', 'WindowsSdkDir', 'WindowsSdkVersion',
                'WindowsSDKLibVersion', 'WindowsSDKVersion',
                'UCRTVersion', 'UniversalCRTSdkDir',
                'DevEnvDir', 'ExtensionSdkDir', 'Framework40Version',
                'FrameworkDir', 'FrameworkDir32', 'FrameworkDir64',
                'FrameworkVersion', 'FrameworkVersion32', 'FrameworkVersion64',
                'VSCMD_ARG_app_plat', 'VSCMD_ARG_HOST_ARCH', 'VSCMD_ARG_TGT_ARCH',
                'VSCMD_VER', '__DOTNET_ADD_32BIT', '__DOTNET_ADD_64BIT',
                '__DOTNET_PREFERRED_BITNESS', '__VSCMD_PREINIT_PATH',
            ]
            for key in msvc_vars:
                if key in vcvars_env:
                    env[key] = vcvars_env[key]
            
            # Set TORCH_CUDA_ARCH_LIST to compile only for detected GPU
            # This dramatically speeds up compilation (from 20+ min to ~2-5 min)
            if "TORCH_CUDA_ARCH_LIST" not in env:
                gpu_arch = self._detect_gpu_arch()
                if gpu_arch:
                    env["TORCH_CUDA_ARCH_LIST"] = gpu_arch
                else:
                    env["TORCH_CUDA_ARCH_LIST"] = "7.5;8.6;8.9+PTX"
            
            return env
            
        except (subprocess.TimeoutExpired, OSError) as e:
            return None
    
    def _detect_gpu_arch(self) -> Optional[str]:
        """
        Detect GPU compute capability using nvidia-smi.
        
        Returns:
            GPU architecture string like "7.5" or None if detection fails
        """
        if self.platform_info.system != "Windows" and not self.platform_info.cuda_available:
            return None
        
        try:
            # Use nvidia-smi to get GPU compute capability
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Get first GPU's compute capability (e.g., "7.5")
                arch = result.stdout.strip().split('\n')[0].strip()
                if arch and '.' in arch:
                    return arch
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None
    
    def _clear_torch_extensions_cache(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Clear torch_extensions cache to force recompilation.
        
        This is useful when previous compilation failed and left corrupted cache.
        """
        cache_paths = []
        
        # Windows: %USERPROFILE%\.cache\torch_extensions
        # Linux/Mac: ~/.cache/torch_extensions
        if self.platform_info.system == "Windows":
            cache_paths.append(Path(os.environ.get("USERPROFILE", "")) / ".cache" / "torch_extensions")
            cache_paths.append(Path(os.environ.get("LOCALAPPDATA", "")) / "torch_extensions")
        else:
            cache_paths.append(Path.home() / ".cache" / "torch_extensions")
        
        for cache_path in cache_paths:
            gsplat_cache = cache_path / "py311_cu124"  # Adjust based on Python/CUDA version
            if gsplat_cache.exists():
                try:
                    # Only remove gsplat-related folders
                    for item in gsplat_cache.iterdir():
                        if "gsplat" in item.name.lower():
                            if progress_callback:
                                progress_callback(f"  Clearing cache: {item.name}")
                            shutil.rmtree(item, ignore_errors=True)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"  Warning: Could not clear cache: {e}")
    
    def _verify_gsplat_installation(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Verify gsplat installation by attempting to import and run a simple test.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if gsplat is working correctly
        """
        verify_script = f'''
import sys
sys.path.insert(0, r"{self.dependencies_path}")

try:
    import gsplat
    print(f"gsplat version: {{gsplat.__version__}}")
    
    # Try to import CUDA functions
    from gsplat import spherical_harmonics
    print("[OK] gsplat CUDA functions imported successfully")
    
    # Quick CUDA test
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        N = 5
        degree = 1
        K = (degree + 1) ** 2
        dirs = torch.randn(N, 3, device=device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        coeffs = torch.randn(N, K, 3, device=device)
        result = spherical_harmonics(degree, dirs, coeffs)
        print(f"[OK] CUDA test passed, output shape: {{result.shape}}")
    
    sys.exit(0)
except Exception as e:
    print(f"[FAILED] {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        try:
            result = subprocess.run(
                [str(self.python_exe), "-c", verify_script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.stdout and progress_callback:
                for line in result.stdout.strip().split('\n'):
                    progress_callback(f"  {line}")
            
            if result.returncode == 0:
                return True
            else:
                if result.stderr and progress_callback:
                    progress_callback(f"  [ERROR] {result.stderr[:200]}")
                return False
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"  [ERROR] Verification failed: {e}")
            return False
    
    def _get_gsplat_wheel_index_url(self) -> Optional[str]:
        """
        Get the gsplat precompiled wheel index URL based on PyTorch/CUDA version.
        
        gsplat provides precompiled wheels at https://docs.gsplat.studio/whl/
        Format: pt{pytorch_version}cu{cuda_version}
        Example: pt24cu124 for PyTorch 2.4 + CUDA 12.4
        
        Returns:
            Index URL string or None if no precompiled wheel available
        """
        cuda_version = self.platform_info.get_recommended_cuda_version()
        
        # Map CUDA versions to gsplat wheel index paths
        # gsplat wheels are available for specific PyTorch + CUDA combinations
        # We're using PyTorch 2.4.0, so pt24 prefix
        if cuda_version == CUDAVersion.CUDA_124:
            return "https://docs.gsplat.studio/whl/pt24cu124"
        elif cuda_version == CUDAVersion.CUDA_121:
            return "https://docs.gsplat.studio/whl/pt24cu121"
        elif cuda_version == CUDAVersion.CUDA_118:
            return "https://docs.gsplat.studio/whl/pt24cu118"
        else:
            # No precompiled wheel available for CPU/MPS
            return None
    
    def install_gsplat(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Install gsplat package (REQUIRED for CUDA systems).
        
        This method uses a system-wide Python interpreter for compilation,
        as Blender's Python lacks necessary development headers.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if successful
        
        Raises:
            RuntimeError: If gsplat installation fails on CUDA system
        """
        if not self.platform_info.cuda_available:
            if progress_callback:
                progress_callback("[SKIP] gsplat requires CUDA - skipping on non-CUDA system")
            return True  # Not a failure, just skipped
        
        if progress_callback:
            progress_callback("Installing gsplat (required for 3DGS optimization)...")
            
        # Find a suitable system python for compilation
        system_python_exe = self._find_system_python(progress_callback)
        if not system_python_exe:
            raise RuntimeError("A full Python 3.11 installation with development headers is required to compile gsplat.")

        # Windows-specific: Need to setup MSVC environment first
        env = None
        if self.platform_info.system == "Windows":
            if not self.platform_info.vcvars64_path:
                error_msg = (
                    "[ERROR] Visual Studio Build Tools not found!\n"
                    "  gsplat requires MSVC compiler to build CUDA extensions on Windows.\n"
                    "  Please install Visual Studio Build Tools:\n"
                    "  1. Download from: https://visualstudio.microsoft.com/downloads/\n"
                    "  2. Select 'Desktop development with C++' workload\n"
                    "  3. Ensure 'MSVC v143 - VS 2022 C++ x64/x86 build tools' is selected\n"
                    "  4. Restart Blender after installation"
                )
                if progress_callback:
                    for line in error_msg.split('\n'):
                        progress_callback(line)
                raise RuntimeError("Visual Studio Build Tools required for gsplat installation")
            
            if progress_callback:
                progress_callback(f"  Setting up MSVC environment via vcvars64.bat...")
            
            env = self._setup_msvc_env()
            if env is None:
                error_msg = "[ERROR] Failed to setup MSVC environment from vcvars64.bat"
                if progress_callback:
                    progress_callback(error_msg)
                raise RuntimeError(error_msg)
            
            # Add our dependencies to PYTHONPATH so gsplat's setup.py can find torch
            python_path = env.get("PYTHONPATH", "")
            if python_path:
                env["PYTHONPATH"] = f"{self.dependencies_path};{python_path}"
            else:
                env["PYTHONPATH"] = str(self.dependencies_path)

            if progress_callback:
                progress_callback("  [OK] MSVC environment configured")
        else: # Linux/macOS
            env = os.environ.copy()
            # Add our dependencies to PYTHONPATH for non-Windows too
            python_path = env.get("PYTHONPATH", "")
            if python_path:
                env["PYTHONPATH"] = f"{self.dependencies_path}:{python_path}"
            else:
                env["PYTHONPATH"] = str(self.dependencies_path)
            
            # Set TORCH_CUDA_ARCH_LIST for faster compilation
            if "TORCH_CUDA_ARCH_LIST" not in env:
                gpu_arch = self._detect_gpu_arch()
                if gpu_arch:
                    env["TORCH_CUDA_ARCH_LIST"] = gpu_arch
        
        # Clear any previous failed build cache
        self._clear_torch_extensions_cache(progress_callback)
        
        try:
            # Install build dependencies first (required for --no-build-isolation)
            if progress_callback:
                progress_callback("  Installing build dependencies (wheel, setuptools, ninja)...")
            
            build_deps_result = subprocess.run(
                [
                    str(system_python_exe), # USE SYSTEM PYTHON
                    "-m", "pip", "install",
                    "wheel", "setuptools", "ninja",
                    "--target", str(self.dependencies_path),
                    "--upgrade",
                    "--no-cache-dir"
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if build_deps_result.returncode != 0 and progress_callback:
                progress_callback(f"  [WARN] Build deps install warning: {build_deps_result.stderr[:200]}")
            elif progress_callback:
                progress_callback("  [OK] Build dependencies installed")
            
            if self.platform_info.system == "Windows":
                # Windows: Force source build to use setup.py
                if progress_callback:
                    progress_callback("  Building gsplat from source (this may take 5-15 minutes)...")
                    progress_callback("  Using external Python for compilation.")
                
                pip_args = [
                    str(system_python_exe), # USE SYSTEM PYTHON
                    "-m", "pip", "install",
                    "gsplat>=1.0.0",
                    "--target", str(self.dependencies_path),
                    "--no-binary", ":all:",  # Force source build for gsplat
                    "--no-build-isolation",   # Use system packages (our installed torch)
                    "--no-deps",
                    "--no-cache-dir",
                    "-v"  # Verbose output for debugging
                ]
            else:
                # Linux/macOS: Try precompiled wheels first, fallback to source
                wheel_index_url = self._get_gsplat_wheel_index_url()
                
                pip_args = [
                    str(system_python_exe), # USE SYSTEM PYTHON
                    "-m", "pip", "install",
                    "gsplat>=1.0.0",
                    "--target", str(self.dependencies_path),
                    "--no-deps",
                    "--no-cache-dir"
                ]
                
                if wheel_index_url:
                    if progress_callback:
                        progress_callback(f"  Using precompiled wheel from: {wheel_index_url}")
                    pip_args.extend(["--index-url", wheel_index_url, "--find-links", wheel_index_url])
                else:
                    if progress_callback:
                        progress_callback("  Building from source (no precompiled wheel available)...")
            
            result = subprocess.run(
                pip_args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=1800,  # 30 minutes for source build
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if self.platform_info.system == "Windows" else 0
            )
            
            if result.returncode == 0:
                if progress_callback:
                    progress_callback("[OK] gsplat installed successfully")
                
                # Verify the installation works
                if progress_callback:
                    progress_callback("  Verifying gsplat installation...")
                
                if self._verify_gsplat_installation(progress_callback):
                    if progress_callback:
                        progress_callback("[OK] gsplat verified - CUDA kernels working")
                    return True
                else:
                    error_msg = "[ERROR] gsplat installed but verification failed"
                    if progress_callback:
                        progress_callback(error_msg)
                    raise RuntimeError(error_msg)
            else:
                # Installation failed - show full output for debugging
                full_output = f"pip command failed with exit code {result.returncode}\n"
                if result.stdout:
                    full_output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    full_output += f"STDERR:\n{result.stderr}\n"
                
                error_msg = f"[ERROR] gsplat installation failed:\n{full_output}"
                if progress_callback:
                    progress_callback(error_msg)
                raise RuntimeError(f"gsplat installation failed, see log for details.")
                
        except subprocess.TimeoutExpired:
            error_msg = "[ERROR] gsplat installation timed out (30 minutes)"
            if progress_callback:
                progress_callback(error_msg)
            raise RuntimeError(error_msg)
        except RuntimeError:
            raise  # Re-raise our own errors
        except Exception as e:
            error_msg = f"[ERROR] gsplat installation error: {str(e)}"
            if progress_callback:
                progress_callback(error_msg)
            raise RuntimeError(error_msg)
    
    def compile_gsplat(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Verify gsplat CUDA kernels are working (public method).
        
        NOTE: With the new installation method using --no-binary=gsplat,
        CUDA kernels are compiled during pip install via setup.py.
        This method now just verifies the installation works.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            Tuple of (success: bool, detailed_log: str)
        """
        detailed_log_lines = []
        
        def log(msg):
            detailed_log_lines.append(msg)
            if progress_callback:
                progress_callback(msg)
        
        if not self.platform_info.cuda_available:
            msg = "[SKIP] gsplat requires CUDA - skipped on non-CUDA system"
            log(msg)
            return True, '\n'.join(detailed_log_lines)
        
        log("Verifying gsplat CUDA installation...")
        
        # Check for compiled kernels
        gsplat_dir = self.dependencies_path / "gsplat"
        if gsplat_dir.exists():
            # Look for .pyd (Windows) or .so (Linux/macOS) files
            compiled_files = list(gsplat_dir.rglob("*.pyd")) + list(gsplat_dir.rglob("*.so"))
            if compiled_files:
                log(f"  Found {len(compiled_files)} compiled kernel files")
                for f in compiled_files[:5]:  # Show first 5
                    log(f"    - {f.name}")
        
        # Verify by running a test
        if self._verify_gsplat_installation(progress_callback):
            log("[OK] gsplat CUDA kernels verified and working")
            return True, '\n'.join(detailed_log_lines)
        else:
            log("[ERROR] gsplat verification failed")
            log("  Try reinstalling gsplat with: Preferences -> NPR Core -> Reinstall Dependencies")
            return False, '\n'.join(detailed_log_lines)
    
    def install_all(
        self,
        cuda_version: Optional[CUDAVersion] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Install all required packages.
        
        Only installs missing packages - skips already installed ones.
        
        Installation order:
        1. PyTorch + torchvision (with exact CUDA version) - if missing
        2. Base requirements (with --upgrade-strategy only-if-needed)
        3. gsplat (REQUIRED for CUDA systems) - if missing
        
        Args:
            cuda_version: Override CUDA version for PyTorch
            progress_callback: Optional callback for progress messages
        
        Returns:
            Tuple of (success, failed_packages)
        """
        from .dependencies import is_package_installed, get_missing_packages
        
        failed = []
        
        if progress_callback:
            progress_callback(f"Platform: {self.platform_info}")
            progress_callback(f"Target directory: {self.dependencies_path}")
        
        # Check what's missing
        missing = get_missing_packages(include_optional=False)
        missing_names = [dep.import_name for dep in missing]
        
        if progress_callback:
            if missing:
                progress_callback(f"Missing packages: {', '.join([dep.name for dep in missing])}")
            else:
                progress_callback("All required packages are already installed")
        
        # Step 1: Install PyTorch FIRST with exact CUDA version - only if missing
        if 'torch' in missing_names or 'torchvision' in missing_names:
            if not self.install_pytorch(cuda_version, progress_callback):
                failed.append("torch")
        else:
            if progress_callback:
                progress_callback("[OK] PyTorch already installed - skipping")
        
        # Step 2: Install base requirements (only missing ones, excluding gsplat)
        # Uses --upgrade-strategy only-if-needed to not touch already-installed packages
        # gsplat is handled separately in Step 3
        base_missing = [dep for dep in missing if dep.import_name != 'gsplat']
        if base_missing:
            success, failed_reqs = self.install_requirements(progress_callback)
            if not success:
                failed.extend(failed_reqs)
        else:
            if progress_callback:
                progress_callback("[OK] Base packages already installed - skipping")
        
        # Step 3: Install gsplat (REQUIRED for CUDA systems)
        if not is_package_installed('gsplat'):
            try:
                self.install_gsplat(progress_callback)
            except RuntimeError as e:
                failed.append("gsplat")
                if progress_callback:
                    progress_callback(f"[ERROR] gsplat installation failed: {e}")
        else:
            if progress_callback:
                progress_callback("[OK] gsplat already installed - skipping")
        
        all_success = len(failed) == 0
        
        if progress_callback:
            if all_success:
                progress_callback("=" * 40)
                progress_callback("[OK] All dependencies installed successfully!")
                progress_callback("Please restart Blender to load the packages.")
            else:
                progress_callback("=" * 40)
                progress_callback(f"[FAILED] Some packages failed: {', '.join(failed)}")
        
        return all_success, failed
    
    def uninstall_all(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Remove all installed dependencies.
        
        Args:
            progress_callback: Optional callback for progress messages
        
        Returns:
            True if successful
        """
        if self.dependencies_path.exists():
            if progress_callback:
                progress_callback(f"Removing {self.dependencies_path}...")
            
            try:
                shutil.rmtree(self.dependencies_path)
                if progress_callback:
                    progress_callback("[OK] Dependencies removed successfully")
                return True
            except Exception as e:
                if progress_callback:
                    progress_callback(f"[FAILED] Failed to remove dependencies: {e}")
                return False
        else:
            if progress_callback:
                progress_callback("No dependencies directory found")
            return True
    
    def is_installed(self) -> bool:
        """
        Check if dependencies are installed.
        
        Returns:
            True if dependencies directory exists and is not empty
        """
        if not self.dependencies_path.exists():
            return False
        
        # Check if directory has content (more than just metadata)
        contents = list(self.dependencies_path.iterdir())
        return len(contents) > 2  # Allow for .dist-info directories
