# dependencies.py
# Dependency management for 3DGS Painter addon

from dataclasses import dataclass
from typing import List, Optional
import importlib.util


@dataclass
class DependencyInfo:
    """Information about a required package."""
    name: str
    version: str
    import_name: str = ""  # If different from package name (empty = use name)
    optional: bool = False  # If True, addon can work without this package
    
    def __post_init__(self):
        if not self.import_name:
            self.import_name = self.name


# Required packages for 3DGS Painter
# Note: torch/torchvision are installed separately with platform-specific handling
# PyTorch version: 2.4.0 with CUDA 12.4 (cu124) on Windows/Linux with NVIDIA GPU
# Using 2.4.0 for gsplat precompiled wheel compatibility
REQUIRED_PACKAGES: List[DependencyInfo] = [
    DependencyInfo("torch", "==2.4.0", import_name="torch"),  # +cu124 suffix added by installer
    DependencyInfo("torchvision", "==0.19.0", import_name="torchvision"),  # +cu124 suffix added by installer
    DependencyInfo("numpy", ">=1.24.0", import_name="numpy"),
    DependencyInfo("pillow", ">=10.0.0", import_name="PIL"),
    DependencyInfo("scipy", ">=1.11.0", import_name="scipy"),
    DependencyInfo("pyyaml", ">=6.0", import_name="yaml"),
    DependencyInfo("opencv-python", ">=4.8.0", import_name="cv2"),
    DependencyInfo("scikit-image", ">=0.21.0", import_name="skimage"),
    DependencyInfo("gsplat", ">=0.1.0", import_name="gsplat", optional=False),  # REQUIRED for CUDA systems
]


def is_package_installed(import_name: str) -> bool:
    """
    Check if a package is installed and importable.
    
    Args:
        import_name: The module name to import (may differ from package name)
    
    Returns:
        True if the package is installed and can be imported
    """
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def get_missing_packages(include_optional: bool = False) -> List[DependencyInfo]:
    """
    Check which required packages are missing.
    
    Args:
        include_optional: If True, also check optional packages
    
    Returns:
        List of missing DependencyInfo objects
    """
    missing = []
    
    for dep in REQUIRED_PACKAGES:
        if dep.optional and not include_optional:
            continue
            
        if not is_package_installed(dep.import_name):
            missing.append(dep)
    
    return missing


def get_installed_packages() -> List[DependencyInfo]:
    """
    Get list of required packages that are installed.
    
    Returns:
        List of installed DependencyInfo objects
    """
    installed = []
    
    for dep in REQUIRED_PACKAGES:
        if is_package_installed(dep.import_name):
            installed.append(dep)
    
    return installed


def check_all_dependencies() -> tuple:
    """
    Check all dependencies and return status.
    
    Returns:
        Tuple of (all_installed: bool, missing: List[DependencyInfo], installed: List[DependencyInfo])
    """
    missing = get_missing_packages(include_optional=False)
    installed = get_installed_packages()
    all_installed = len(missing) == 0
    
    return all_installed, missing, installed


def get_package_version(import_name: str) -> Optional[str]:
    """
    Get the installed version of a package.
    
    Args:
        import_name: The module name to check
    
    Returns:
        Version string or None if not installed or failed to load
    """
    try:
        module = __import__(import_name)
        return getattr(module, '__version__', 'unknown')
    except (ImportError, OSError, RuntimeError):
        # OSError can occur when DLLs fail to load (Windows PyTorch issue)
        # RuntimeError can occur with CUDA initialization failures
        return None
