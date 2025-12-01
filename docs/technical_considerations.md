# 기술적 고려사항 (Technical Considerations)

**범위**: 전체 프로젝트  
**목적**: 횡단적(cross-cutting) 기술 이슈 및 최적화 전략

---

## 📋 개요

본 문서는 특정 Phase에 국한되지 않는 **공통 기술적 고려사항**을 다룹니다:
- GPU 컨텍스트 관리
- 메모리 최적화 (VRAM/RAM)
- 성능 프로파일링
- 에러 처리
- 플랫폼 호환성

---

## 🎮 1. GPU 컨텍스트 관리

### 1.1 문제점

**Blender + PyTorch + GLSL 공존**:
- Blender: OpenGL 컨텍스트 소유 (3D Viewport)
- PyTorch: CUDA 컨텍스트 소유 (gsplat computation)
- GLSL Shaders: OpenGL 텍스처 공유

**잠재적 충돌**:
- CUDA와 OpenGL 동시 사용 시 컨텍스트 스위칭 오버헤드
- 메모리 중복 할당
- Thread safety 이슈

### 1.2 해결 전략

#### Strategy A: Sequential Execution (현재 권장)

```python
# operators.py (painting operator)

def modal(self, context, event):
    if event.type == 'MOUSEMOVE' and self.painting:
        # 1. Update GLSL viewport FIRST (low latency)
        self.update_viewport_immediate(stamp)
        
        # 2. Queue computation for later (after stroke finishes)
        self.pending_deformations.append(stamp)
        
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}
    
    if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
        # 3. Flush GLSL pipeline
        bgl.glFlush()
        
        # 4. Switch to CUDA context (gsplat)
        torch.cuda.synchronize()
        
        # 5. Process deformations
        self.apply_deformations_batch(self.pending_deformations)
        
        # 6. Sync back to GLSL
        self.sync_to_viewport()
        
        return {'FINISHED'}
```

#### Strategy B: CUDA-OpenGL Interop (advanced, future optimization)

```python
# Advanced: Share memory between CUDA and OpenGL
# Requires: cudaGraphicsRegisterResource

import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

class CUDAGLInterop:
    """
    CUDA-OpenGL interoperability for zero-copy data sharing.
    WARNING: Experimental, platform-dependent.
    """
    
    def __init__(self, gl_texture_id):
        self.gl_texture_id = gl_texture_id
        self.cuda_resource = None
    
    def register_texture(self):
        """Register OpenGL texture with CUDA."""
        import pycuda.gl as cuda_gl
        
        # Register texture
        self.cuda_resource = cuda_gl.RegisteredImage(
            self.gl_texture_id,
            gl.GL_TEXTURE_3D,
            cuda_gl.graphics_map_flags.WRITE_DISCARD
        )
    
    def map_to_cuda(self):
        """Map texture to CUDA tensor (zero-copy)."""
        mapping = self.cuda_resource.map()
        array = mapping.array(0, 0)
        
        # Convert to PyTorch tensor
        # ... requires custom CUDA kernel
        
        return tensor
```

**Recommendation**: Use Strategy A (sequential) initially. Strategy B only if profiling shows sync overhead > 10ms.

---

## 💾 2. 메모리 관리

### 2.1 VRAM 예산

**Target**: 8GB GPU 지원 (RTX 3060 Ti/3070 기준)

#### 메모리 프로필

```
[Viewport Only - 98% 시간]
┌──────────────────────────────────┐
│ GLSL Textures                    │
│  - Gaussian data (59-float): 1MB │  (10k gaussians)
│  - Depth buffer: 8MB             │  (1080p)
│  - Color buffer: 8MB             │
├──────────────────────────────────┤
│ Blender Scene                    │
│  - Mesh geometry: ~500MB         │
│  - Textures: ~1GB                │
├──────────────────────────────────┤
│ Subtotal: ~2.5GB - 4GB           │
└──────────────────────────────────┘

[Computation Active - 2% 시간]
┌──────────────────────────────────┐
│ GLSL (same as above): ~1-2GB     │
├──────────────────────────────────┤
│ PyTorch Tensors                  │
│  - Gaussians: 50MB (10k)         │
│  - Gradients: 50MB               │
│  - Intermediate: ~100MB          │
│  - gsplat render buffer: ~30MB   │
├──────────────────────────────────┤
│ Subtotal: ~3.5GB - 6.5GB         │
└──────────────────────────────────┘

Peak Usage: ~6.5GB (safe for 8GB)
```

### 2.2 메모리 최적화 전략

#### Chunked Processing

```python
# npr_core/deformation_gpu.py

class DeformationGPU:
    def apply_large_batch(self, gaussians, chunk_size=10000):
        """
        Process large batches in chunks to avoid OOM.
        
        Args:
            gaussians: List of Gaussian objects
            chunk_size: Max gaussians per chunk
        
        Returns:
            Deformed gaussians
        """
        results = []
        
        for i in range(0, len(gaussians), chunk_size):
            chunk = gaussians[i:i+chunk_size]
            
            # Process chunk
            deformed_chunk = self.apply(chunk)
            results.extend(deformed_chunk)
            
            # Clear cache
            torch.cuda.empty_cache()
        
        return results
```

#### Gradient Checkpointing

```python
# For inpainting optimization (Phase 5)

from torch.utils.checkpoint import checkpoint

class InpaintingOptimizer:
    def render_with_checkpointing(self, params):
        """
        Use gradient checkpointing to reduce memory.
        Trades compute for memory (2x slower, 50% less memory).
        """
        return checkpoint(self.render_gsplat, params)
```

#### VRAM Monitor

```python
# npr_core/memory_monitor.py

import torch

class VRAMMonitor:
    """Monitor VRAM usage during operations."""
    
    @staticmethod
    def get_usage():
        """
        Get current VRAM usage.
        
        Returns:
            dict: {'allocated': float (GB), 'cached': float (GB), 'free': float (GB)}
        """
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        
        return {
            'allocated': allocated,
            'cached': cached,
            'free': free,
            'total': total
        }
    
    @staticmethod
    def print_summary():
        """Print VRAM usage summary."""
        usage = VRAMMonitor.get_usage()
        if usage:
            print(f"VRAM: {usage['allocated']:.2f}GB / {usage['total']:.2f}GB")
            print(f"  Allocated: {usage['allocated']:.2f}GB")
            print(f"  Cached: {usage['cached']:.2f}GB")
            print(f"  Free: {usage['free']:.2f}GB")
```

---

## ⚡ 3. 성능 최적화

### 3.1 프로파일링 도구

#### GPU Timer

```python
# npr_core/profiling.py

import torch
import time

class GPUTimer:
    """Measure GPU execution time."""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        self.start_event.record()
        return self
    
    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed = self.start_event.elapsed_time(self.end_event)  # ms
    
    def get_elapsed(self):
        """Get elapsed time in milliseconds."""
        return self.elapsed

# Usage
with GPUTimer() as timer:
    deformed = deformation_engine.apply(gaussians)

print(f"Deformation took {timer.get_elapsed():.2f}ms")
```

#### Comprehensive Profiler

```python
# npr_core/profiling.py

class PerformanceProfiler:
    """Profile full painting pipeline."""
    
    def __init__(self):
        self.timings = {}
    
    def measure(self, name):
        """Context manager for timing."""
        import time
        
        class TimingContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                elapsed = (time.perf_counter() - self.start) * 1000
                if self.name not in self.profiler.timings:
                    self.profiler.timings[self.name] = []
                self.profiler.timings[self.name].append(elapsed)
        
        return TimingContext(self, name)
    
    def print_summary(self):
        """Print timing summary."""
        import numpy as np
        
        print("\n=== Performance Profile ===")
        for name, times in self.timings.items():
            avg = np.mean(times)
            std = np.std(times)
            min_t = np.min(times)
            max_t = np.max(times)
            
            print(f"{name}:")
            print(f"  Avg: {avg:.2f}ms (±{std:.2f}ms)")
            print(f"  Range: {min_t:.2f}ms - {max_t:.2f}ms")
            print(f"  Calls: {len(times)}")

# Usage
profiler = PerformanceProfiler()

with profiler.measure("stamp_generation"):
    stamp = brush.place_at(...)

with profiler.measure("viewport_update"):
    viewport_renderer.update_partial(...)

with profiler.measure("deformation"):
    deformed = deformation_engine.apply(...)

profiler.print_summary()
```

### 3.2 Performance Targets

| Operation | Target | Acceptable | Critical |
|-----------|--------|------------|----------|
| Stamp generation | <5ms | <10ms | >20ms |
| Viewport update (incremental) | <2ms | <5ms | >10ms |
| Deformation (100 stamps) | <500ms | <1000ms | >2000ms |
| Inpainting (100 iter) | <5s | <10s | >20s |
| Final render (1080p) | <10s | <30s | >60s |

### 3.3 Bottleneck 분석

**Common Bottlenecks**:

1. **CPU-GPU Transfer** (most common)
   - Symptom: Low GPU utilization, high CPU usage
   - Solution: Batch transfers, use pinned memory

2. **Synchronization Overhead**
   - Symptom: `torch.cuda.synchronize()` taking >5ms
   - Solution: Minimize sync points, use async operations

3. **Texture Upload** (GLSL)
   - Symptom: `glTexSubImage3D` >10ms
   - Solution: Use PBO (Pixel Buffer Objects), smaller updates

---

## 🛡️ 4. 에러 처리

### 4.1 GPU 에러

#### CUDA Out of Memory

```python
# npr_core/error_handling.py

import torch

def safe_gpu_operation(func, *args, fallback_chunk_size=None, **kwargs):
    """
    Safely execute GPU operation with OOM handling.
    
    Args:
        func: Function to execute
        fallback_chunk_size: If OOM, retry with chunking
    
    Returns:
        Result of func or None if failed
    """
    try:
        return func(*args, **kwargs)
    
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        
        if fallback_chunk_size:
            print(f"OOM detected, retrying with chunk size {fallback_chunk_size}")
            
            # Retry with chunking
            # ... implement chunked version ...
            
            return chunked_result
        else:
            raise RuntimeError(
                "GPU out of memory. Try:\n"
                "1. Reduce gaussian count\n"
                "2. Close other GPU applications\n"
                "3. Reduce viewport resolution"
            )
```

#### Device Not Available

```python
# npr_core/gpu_context.py

class BlenderGPUContext:
    def initialize(self):
        """Initialize with fallback to CPU."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.backend = 'cuda'
        else:
            print("⚠ CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.backend = 'cpu'
            
            # Warn user
            import bpy
            def draw_warning(self, context):
                layout = self.layout
                layout.label(text="GPU not available!", icon='ERROR')
                layout.label(text="Performance will be degraded.")
            
            bpy.context.window_manager.popup_menu(draw_warning, title="Warning", icon='ERROR')
```

### 4.2 File I/O 에러

```python
# npr_core/brush_manager.py

import json
from pathlib import Path

class BrushManager:
    def load(self, filepath, retry=True):
        """
        Load brush with error handling.
        
        Args:
            filepath: Path to brush file
            retry: Whether to retry on failure
        
        Returns:
            Brush object or None
        """
        filepath = Path(filepath)
        
        # Check existence
        if not filepath.exists():
            raise FileNotFoundError(f"Brush file not found: {filepath}")
        
        # Check permission
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"Cannot read brush file: {filepath}")
        
        try:
            # Load JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate schema
            self.validate_brush_data(data)
            
            # Create brush
            brush = Brush.from_dict(data)
            return brush
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in brush file: {e}")
        
        except Exception as e:
            if retry:
                print(f"Failed to load brush, retrying: {e}")
                time.sleep(0.5)
                return self.load(filepath, retry=False)
            else:
                raise RuntimeError(f"Failed to load brush: {e}")
```

---

## 🌐 5. 플랫폼 호환성

### 5.1 OS-Specific 이슈

#### Windows

```python
# Windows: Path separator, Python executable location

import platform

if platform.system() == "Windows":
    # Use forward slashes for paths (Blender compatibility)
    addon_path = str(Path(__file__).parent).replace('\\', '/')
    
    # Python executable
    python_exe = Path(sys.executable).parent.parent / "python" / "bin" / "python.exe"
```

#### macOS

```python
# macOS: Metal backend, Python location

if platform.system() == "Darwin":
    # Metal backend (no CUDA)
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
        else:
            device = torch.device('cpu')
    
    # Python executable
    python_exe = Path(sys.executable).parent.parent / "python" / "bin" / "python3.10"
```

#### Linux

```python
# Linux: Distribution differences, CUDA paths

if platform.system() == "Linux":
    # Check CUDA library path
    import os
    
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            os.environ['LD_LIBRARY_PATH'] = path + ":" + os.environ.get('LD_LIBRARY_PATH', '')
            break
```

### 5.2 Blender 버전 호환성

```python
# __init__.py

bl_info = {
    "name": "NPR Gaussian Painter",
    "blender": (3, 6, 0),  # Minimum version
    "category": "Paint",
}

def check_blender_version():
    """Check if Blender version is compatible."""
    import bpy
    
    min_version = (3, 6, 0)
    current = bpy.app.version
    
    if current < min_version:
        raise RuntimeError(
            f"Blender {min_version[0]}.{min_version[1]} or higher required. "
            f"Current version: {current[0]}.{current[1]}"
        )
```

---

## 📊 6. 디버깅 도구

### 6.1 Visualization Helpers

```python
# npr_core/debug_visualizer.py

import bpy
import numpy as np

class DebugVisualizer:
    """Visualize gaussians and debug info in Blender."""
    
    @staticmethod
    def draw_gaussian_centers(scene_data, name="GaussianCenters"):
        """
        Draw gaussian centers as empties in Blender.
        
        Args:
            scene_data: SceneData object
            name: Collection name
        """
        # Create collection
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        
        # Draw centers
        for i, g in enumerate(scene_data.gaussians):
            empty = bpy.data.objects.new(f"G_{i}", None)
            empty.empty_display_size = 0.1
            empty.empty_display_type = 'SPHERE'
            empty.location = g.position
            
            col.objects.link(empty)
    
    @staticmethod
    def draw_spline(spline_points, name="SplineCurve"):
        """
        Draw spline curve in Blender.
        
        Args:
            spline_points: np.ndarray [N, 3]
            name: Curve name
        """
        curve_data = bpy.data.curves.new(name, type='CURVE')
        curve_data.dimensions = '3D'
        
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(len(spline_points) - 1)
        
        for i, point in enumerate(spline_points):
            polyline.points[i].co = (*point, 1.0)
        
        curve_obj = bpy.data.objects.new(name, curve_data)
        bpy.context.scene.collection.objects.link(curve_obj)
```

### 6.2 Log System

```python
# npr_core/logging.py

import logging
from pathlib import Path

def setup_logger(name="npr_gaussian", level=logging.INFO):
    """
    Setup logger for addon.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (Blender temp directory)
    import tempfile
    log_file = Path(tempfile.gettempdir()) / "npr_gaussian.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger

# Usage
logger = setup_logger()
logger.info("Addon loaded")
logger.debug("Debug info")
logger.error("Error occurred")
```

---

## 🔍 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_npr_core.py

import pytest
import numpy as np
from npr_core.brush import Brush
from npr_core.scene_data import SceneData

def test_brush_placement():
    """Test brush stamp placement."""
    brush = Brush.from_file("data/brushes/test.png")
    
    stamp = brush.place_at(
        position=np.array([0, 0, 0]),
        normal=np.array([0, 0, 1]),
        size_multiplier=1.0
    )
    
    assert len(stamp.gaussians) > 0
    assert stamp.center is not None

def test_scene_data_add_remove():
    """Test scene data manipulation."""
    scene = SceneData()
    
    # Add gaussians
    gaussian = Gaussian(
        position=np.array([0, 0, 0]),
        scale=np.array([1, 1, 1]),
        opacity=0.5
    )
    scene.add_gaussian(gaussian)
    
    assert len(scene.gaussians) == 1
    
    # Remove
    scene.remove_gaussian(0)
    assert len(scene.gaussians) == 0
```

### 7.2 Integration Tests (Blender)

```python
# tests/test_blender_integration.py

import bpy
import sys
sys.path.append("path/to/addon")

def test_operator_registration():
    """Test that operators are registered."""
    assert hasattr(bpy.ops, 'gaussian')
    assert hasattr(bpy.ops.gaussian, 'paint')
    assert hasattr(bpy.ops.gaussian, 'inpaint')

def test_painting_workflow():
    """Test full painting workflow."""
    # Load brush
    bpy.ops.gaussian.load_brush(filepath="data/brushes/test.json")
    
    # Start painting
    bpy.ops.gaussian.paint('INVOKE_DEFAULT')
    
    # Simulate stroke
    # ... (requires event simulation)
    
    # Check scene data
    scene_data = bpy.context.scene.gaussian_scene_data
    assert len(scene_data.gaussians) > 0
```

---

## 📚 참고 자료

- PyTorch Performance Tuning: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Blender GPU Module: https://docs.blender.org/api/current/gpu.html
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

## 🎯 체크리스트

### 성능
- [ ] Profiling 도구 구현
- [ ] 모든 operation target 달성
- [ ] VRAM 사용량 < 8GB 유지

### 안정성
- [ ] GPU OOM 처리
- [ ] File I/O 에러 처리
- [ ] Platform compatibility 검증

### 디버깅
- [ ] Logger 시스템 구현
- [ ] Debug visualization 도구
- [ ] Unit tests 작성
