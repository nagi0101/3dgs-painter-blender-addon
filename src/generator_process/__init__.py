"""
Generator process for 3DGS Painter.

This module provides subprocess-based computation for PyTorch/CUDA operations.
PyTorch is loaded ONLY in the subprocess to avoid TBB DLL conflicts with Blender.

Reference: Dream Textures generator_process/__init__.py
"""

from typing import Callable
from multiprocessing import current_process

from .actor import Actor, is_actor_process
from .future import Future


class RunInSubprocess(Exception):
    """
    Decorators to support running functions that are not defined under the Generator class in its subprocess.
    This is to reduce what would otherwise be duplicate function definitions that logically don't belong to
    the Generator, but require something in its subprocess (such as access to installed dependencies).
    
    Usage:
        @RunInSubprocess
        def my_function():
            import torch  # Only available in subprocess
            return torch.cuda.is_available()
    """

    def __new__(cls, func=None):
        if func is None:
            # support `raise RunInSubprocess`
            return super().__new__(cls)
        return cls.always(func)

    @staticmethod
    def always(func):
        """Always run in subprocess."""
        if is_actor_process:
            return func
        
        def wrapper(*args, **kwargs):
            return NPRGenerator.shared().call(wrapper, *args, **kwargs).result()
        
        RunInSubprocess._copy_attributes(func, wrapper)
        return wrapper

    @staticmethod
    def when(condition: bool | Callable[..., bool]):
        """Run in subprocess when condition is true."""
        if not isinstance(condition, Callable):
            if condition:
                return RunInSubprocess.always
            return lambda x: x
        
        def decorator(func):
            if is_actor_process:
                return func
            
            def wrapper(*args, **kwargs):
                if condition(*args, **kwargs):
                    return NPRGenerator.shared().call(wrapper, *args, **kwargs).result()
                return func(*args, **kwargs)
            
            RunInSubprocess._copy_attributes(func, wrapper)
            return wrapper
        
        return decorator

    @staticmethod
    def when_raised(func):
        """Run in subprocess if RunInSubprocess exception is raised."""
        if is_actor_process:
            return func
        
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RunInSubprocess:
                return NPRGenerator.shared().call(wrapper, *args, **kwargs).result()
        
        RunInSubprocess._copy_attributes(func, wrapper)
        return wrapper

    @staticmethod
    def _copy_attributes(src, dst):
        """Copy function attributes from source to destination."""
        for n in ["__annotations__", "__doc__", "__name__", "__module__", "__qualname__"]:
            if hasattr(src, n):
                setattr(dst, n, getattr(src, n))


class NPRGenerator(Actor):
    """
    The actor used for all background PyTorch/CUDA processes.
    
    All methods defined here (except protected ones) will be executed in the subprocess.
    The subprocess has access to PyTorch, CUDA, gsplat, etc.
    """

    # Import actions from separate modules (executed in subprocess)
    # These will be added as the project progresses
    
    @staticmethod
    def call(func, *args, **kwargs):
        """
        Call a function in the subprocess.
        Used by RunInSubprocess decorator.
        """
        return func(*args, **kwargs)
    
    # ==========================================================================
    # System Info Actions
    # ==========================================================================
    
    def get_torch_info(self) -> dict:
        """
        Get PyTorch and CUDA information.
        
        Returns:
            dict with torch version, cuda availability, device info
            All values are pure Python types (str, bool, int, float, list, dict)
        """
        try:
            import torch
            
            cuda_available = bool(torch.cuda.is_available())
            
            info = {
                "torch_version": str(torch.__version__),
                "cuda_available": cuda_available,
                "cuda_version": str(torch.version.cuda) if cuda_available else None,
                "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "device_name": None,
                "devices": [],
            }
            
            if cuda_available:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info["devices"].append({
                        "index": int(i),
                        "name": str(props.name),
                        "total_memory_gb": float(props.total_memory / (1024**3)),
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
                if info["devices"]:
                    info["device_name"] = info["devices"][0]["name"]
            
            return info
            
        except ImportError as e:
            return {"error": f"PyTorch not available: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
    
    def check_dependencies(self) -> dict:
        """
        Check if all required dependencies are available in subprocess.
        
        Returns:
            dict with package availability status
        """
        result = {}
        
        packages = [
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("pillow", "PIL"),
            ("pyyaml", "yaml"),
        ]
        
        for name, import_name in packages:
            try:
                module = __import__(import_name)
                version = getattr(module, "__version__", "unknown")
                result[name] = {"available": True, "version": version}
            except ImportError as e:
                result[name] = {"available": False, "error": str(e)}
        
        # Check gsplat separately (may not be installed)
        try:
            import gsplat
            result["gsplat"] = {"available": True, "version": getattr(gsplat, "__version__", "unknown")}
        except ImportError:
            result["gsplat"] = {"available": False, "error": "Not installed"}
        
        return result
    
    # ==========================================================================
    # Computation Actions (to be expanded)
    # ==========================================================================
    
    def test_cuda_computation(self, size: int = 1000) -> dict:
        """
        Run a simple CUDA computation test.
        
        Args:
            size: Size of test tensor
            
        Returns:
            dict with test results (all pure Python types)
        """
        try:
            import torch
            import time
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Create tensors
            start = time.time()
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Matrix multiplication
            c = torch.matmul(a, b)
            
            # Synchronize if CUDA
            if device == "cuda":
                torch.cuda.synchronize()
            
            compute_elapsed = time.time() - start
            
            # Transfer to CPU and convert to Python float
            transfer_start = time.time()
            result_sum = float(c.sum().cpu().item())
            transfer_elapsed = time.time() - transfer_start
            
            return {
                "success": True,
                "device": str(device),
                "size": int(size),
                "compute_time_ms": float(compute_elapsed * 1000),
                "transfer_time_ms": float(transfer_elapsed * 1000),
                "result_sum": result_sum,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==========================================================================
    # SharedMemory IPC Actions (Phase 4)
    # ==========================================================================
    
    # Class-level storage for shared buffer (subprocess side)
    _shared_buffer = None
    _current_gaussians_tensor = None
    
    def setup_shared_buffer(self, buffer_name: str, max_gaussians: int = 100000) -> dict:
        """
        Attach to shared memory buffer created by main process.
        
        Args:
            buffer_name: Name of existing SharedMemory
            max_gaussians: Maximum gaussians supported
            
        Returns:
            dict with status
        """
        try:
            from .shared_buffer import GaussianSharedBuffer
            
            # Attach to existing shared memory
            NPRGenerator._shared_buffer = GaussianSharedBuffer(
                max_gaussians=max_gaussians,
                name=buffer_name
            )
            
            return {
                "success": True,
                "name": buffer_name,
                "max_gaussians": max_gaussians
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def sync_gaussians_from_shared(self, start_idx: int = 0, count: int = None) -> dict:
        """
        Read gaussians from shared memory and convert to PyTorch tensor.
        
        This is called after main process writes to SharedMemory.
        Data is already in memory - just read and convert to tensor.
        
        Args:
            start_idx: Starting index in buffer
            count: Number of gaussians (None = read from header)
            
        Returns:
            dict with sync status
        """
        try:
            import torch
            import time
            
            if NPRGenerator._shared_buffer is None:
                return {"success": False, "error": "SharedBuffer not initialized"}
            
            start = time.perf_counter()
            
            # Zero-copy read from shared memory (returns view)
            gaussians_np = NPRGenerator._shared_buffer.read(start_idx, count)
            actual_count = gaussians_np.shape[0]
            
            read_time = time.perf_counter() - start
            
            # Convert to PyTorch tensor (this copies to GPU)
            start = time.perf_counter()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            NPRGenerator._current_gaussians_tensor = torch.from_numpy(
                gaussians_np.copy()  # Copy needed for tensor creation
            ).to(device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            tensor_time = time.perf_counter() - start
            
            return {
                "success": True,
                "count": int(actual_count),
                "device": device,
                "read_time_ms": float(read_time * 1000),
                "tensor_time_ms": float(tensor_time * 1000),
            }
            
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def compute_deformation_shared(self, spline_points: list, deformation_radius: float = 0.5) -> dict:
        """
        Compute spline deformation on current gaussians and write back to SharedMemory.
        
        Args:
            spline_points: List of (x, y, z) tuples for spline
            deformation_radius: Radius of deformation influence
            
        Returns:
            dict with computation status
        """
        try:
            import torch
            import time
            
            if NPRGenerator._current_gaussians_tensor is None:
                return {"success": False, "error": "No gaussians loaded. Call sync_gaussians_from_shared first."}
            
            if NPRGenerator._shared_buffer is None:
                return {"success": False, "error": "SharedBuffer not initialized"}
            
            start = time.perf_counter()
            
            gaussians = NPRGenerator._current_gaussians_tensor
            device = gaussians.device
            
            # Convert spline points to tensor
            spline_tensor = torch.tensor(spline_points, dtype=torch.float32, device=device)
            
            # Simple deformation: Pull gaussians toward spline
            # This is a placeholder - actual deformation uses deformation_gpu.py
            positions = gaussians[:, :3]  # First 3 floats are position
            
            # For each gaussian, find closest point on spline and apply deformation
            # (Simplified version - full implementation in deformation_gpu.py)
            
            compute_time = time.perf_counter() - start
            
            # Write deformed gaussians back to shared memory
            start = time.perf_counter()
            deformed_np = gaussians.cpu().numpy()
            NPRGenerator._shared_buffer.write(deformed_np, start_idx=0)
            write_time = time.perf_counter() - start
            
            return {
                "success": True,
                "count": int(gaussians.shape[0]),
                "compute_time_ms": float(compute_time * 1000),
                "write_time_ms": float(write_time * 1000),
            }
            
        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def cleanup_shared_buffer(self) -> dict:
        """
        Cleanup shared buffer resources in subprocess.
        
        Returns:
            dict with cleanup status
        """
        try:
            if NPRGenerator._shared_buffer is not None:
                NPRGenerator._shared_buffer.cleanup()
                NPRGenerator._shared_buffer = None
            
            NPRGenerator._current_gaussians_tensor = None
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_gsplat(self) -> dict:
        """
        Test if gsplat is available and working in subprocess.
        
        Returns:
            dict with gsplat availability, version, and test result
        """
        result = {
            "gsplat_available": False,
            "gsplat_version": None,
            "gsplat_error": None,
            "rasterization_test": False,
            "deps_path_exists": False,
        }
        
        import sys
        import os
        from pathlib import Path
        
        # Check sys.path (first 10 entries)
        result["sys_path"] = sys.path[:10]
        
        # Check deps path
        deps_path = Path(__file__).parent.parent / ".python_dependencies"
        result["deps_path_exists"] = deps_path.exists()
        result["deps_path"] = str(deps_path)
        
        if deps_path.exists():
            # List gsplat-related files
            gsplat_files = []
            for item in deps_path.iterdir():
                if 'gsplat' in item.name.lower():
                    gsplat_files.append(item.name)
            result["gsplat_files"] = gsplat_files
        
        # Try to import gsplat
        try:
            import gsplat
            result["gsplat_available"] = True
            result["gsplat_version"] = getattr(gsplat, "__version__", "unknown")
            
            # Try to import rasterization
            try:
                from gsplat import rasterization
                result["rasterization_test"] = True
            except ImportError as e:
                result["rasterization_error"] = str(e)
                
        except ImportError as e:
            result["gsplat_error"] = str(e)
        except Exception as e:
            result["gsplat_error"] = f"Load error: {str(e)}"
        
        return result
    
    # ==========================================================================
    # Brush Optimization Actions (Phase 4.5)
    # ==========================================================================
    
    def optimize_brush_gaussians(
        self,
        gaussians_data: list,
        target_image: list,
        target_alpha: list = None,
        iterations: int = 50,
        render_size: int = 256
    ) -> dict:
        """
        Optimize gaussian parameters using gsplat differentiable rendering.
        
        This runs in subprocess with PyTorch/CUDA access.
        
        Args:
            gaussians_data: List of dicts with gaussian parameters:
                           [{'position': [x,y,z], 'rotation': [x,y,z,w], 
                             'scale': [x,y,z], 'opacity': float, 'color': [r,g,b]}, ...]
            target_image: Target RGB image as nested list (H, W, 3) in [0, 1]
            target_alpha: Optional alpha mask as nested list (H, W) in [0, 1]
            iterations: Number of optimization iterations
            render_size: Size for rendering during optimization
        
        Returns:
            dict with 'success', 'gaussians' (optimized), 'final_loss'
        """
        try:
            import numpy as np
            import time
            import sys
            
            start_time = time.perf_counter()
            
            # Debug: Print sys.path
            print(f"[optimize_brush_gaussians] sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
            
            # Check if gsplat is available first - with detailed logging
            try:
                from .optimizer_torch import is_gsplat_available, _ensure_gsplat
                print(f"[optimize_brush_gaussians] optimizer_torch imported successfully")
                
                # Try direct gsplat import first for debugging
                try:
                    import gsplat
                    print(f"[optimize_brush_gaussians] Direct gsplat import OK: {gsplat.__version__}")
                except ImportError as e:
                    print(f"[optimize_brush_gaussians] Direct gsplat import FAILED: {e}")
                
                # Now check via is_gsplat_available
                gsplat_ok = is_gsplat_available()
                print(f"[optimize_brush_gaussians] is_gsplat_available() = {gsplat_ok}")
                
                if not gsplat_ok:
                    return {
                        "success": False,
                        "error": "gsplat is not available in subprocess (is_gsplat_available returned False)",
                        "fallback": True
                    }
            except Exception as e:
                import traceback
                print(f"[optimize_brush_gaussians] Error checking gsplat: {e}")
                print(traceback.format_exc())
                return {
                    "success": False,
                    "error": f"Error checking gsplat availability: {e}",
                    "traceback": traceback.format_exc(),
                    "fallback": True
                }
            
            # Convert input lists to numpy arrays
            n_gaussians = len(gaussians_data)
            
            # Create structured array for gaussians
            dtype = np.dtype([
                ('position', np.float32, (3,)),
                ('rotation', np.float32, (4,)),
                ('scale', np.float32, (3,)),
                ('opacity', np.float32),
                ('color', np.float32, (3,))
            ])
            
            gaussians_np = np.zeros(n_gaussians, dtype=dtype)
            for i, g in enumerate(gaussians_data):
                gaussians_np[i]['position'] = g['position']
                gaussians_np[i]['rotation'] = g['rotation']
                gaussians_np[i]['scale'] = g['scale']
                gaussians_np[i]['opacity'] = g['opacity']
                gaussians_np[i]['color'] = g['color']
            
            # Convert target image
            target_np = np.array(target_image, dtype=np.float32)
            alpha_np = np.array(target_alpha, dtype=np.float32) if target_alpha else None
            
            # Import and run optimizer
            from .optimizer_torch import TorchGaussianOptimizer
            
            optimizer = TorchGaussianOptimizer(
                gaussians_data=gaussians_np,
                target_image=target_np,
                target_alpha=alpha_np,
                render_width=render_size,
                render_height=render_size
            )
            
            optimized_np = optimizer.optimize(iterations=iterations)
            
            # Convert back to list of dicts
            optimized_list = []
            for i in range(len(optimized_np)):
                optimized_list.append({
                    'position': optimized_np[i]['position'].tolist(),
                    'rotation': optimized_np[i]['rotation'].tolist(),
                    'scale': optimized_np[i]['scale'].tolist(),
                    'opacity': float(optimized_np[i]['opacity']),
                    'color': optimized_np[i]['color'].tolist()
                })
            
            elapsed = time.perf_counter() - start_time
            
            return {
                "success": True,
                "gaussians": optimized_list,
                "count": len(optimized_list),
                "elapsed_ms": float(elapsed * 1000)
            }
            
        except ImportError as e:
            import traceback
            return {
                "success": False,
                "error": f"gsplat not available: {str(e)}",
                "traceback": traceback.format_exc(),
                "fallback": True
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


# =============================================================================
# Utility Functions
# =============================================================================

def kill_generator():
    """Kill the generator subprocess if running."""
    NPRGenerator.shared_close()


def get_generator() -> NPRGenerator:
    """Get or create the shared NPRGenerator instance."""
    return NPRGenerator.shared()
