# Phase 4: 페인팅 인터랙션 구현 (Painting Interaction + gsplat Integration)

**기간**: 3주  
**목표**: Real-time painting + Hybrid 데이터 동기화 + gsplat Deformation 통합

---

## 📋 작업 개요

본 Phase는 Hybrid 아키텍처의 **양방향 통합**을 구현합니다:

-   ✓ GLSL Viewport (실시간 페인팅 표시)
-   ✓ gsplat Computation (Deformation 계산)
-   ✓ 데이터 동기화 (NumPy ↔ PyTorch ↔ GLSL)

---

## 🎯 핵심 작업

### 1. Raycasting 및 Surface Interaction

#### 1.1 마우스 좌표 → 3D 위치 변환

```python
# operators.py

from bpy_extras import view3d_utils
from mathutils import Vector

def raycast_mouse_to_surface(context, event):
    """
    Convert mouse coordinates to 3D surface position.

    Args:
        context: bpy.context
        event: Modal operator event

    Returns:
        tuple: (location: Vector, normal: Vector, hit: bool)
    """
    region = context.region
    rv3d = context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)

    # Get ray direction
    view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    # Raycast against scene objects
    result, location, normal, index, obj, matrix = context.scene.ray_cast(
        context.view_layer.depsgraph,
        ray_origin,
        view_vector
    )

    if result:
        return location, normal, True
    else:
        # Fallback: project to XY plane at z=0
        distance = -ray_origin.z / view_vector.z if view_vector.z != 0 else 100
        location = ray_origin + view_vector * distance
        normal = Vector((0, 0, 1))
        return location, normal, False
```

#### 1.2 Tablet Pressure 지원

```python
def get_tablet_pressure(event):
    """
    Get tablet pressure (0-1 range).

    Returns:
        float: pressure value, 1.0 if not using tablet
    """
    if hasattr(event, 'pressure'):
        return event.pressure
    return 1.0
```

---

### 2. Modal Operator (Painting Mode)

#### 2.1 기본 구조

```python
# operators.py

import bpy
from bpy.props import FloatProperty, StringProperty
import numpy as np

class GaussianPaintOperator(bpy.types.Operator):
    """Paint with Gaussian Splat Brushes"""
    bl_idname = "gaussian.paint"
    bl_label = "Paint with Gaussian Brush"
    bl_options = {'REGISTER', 'UNDO'}

    # Properties
    brush_size: FloatProperty(name="Brush Size", default=0.5, min=0.01, max=5.0)
    brush_opacity: FloatProperty(name="Opacity", default=0.5, min=0.0, max=1.0)

    def __init__(self):
        self.stroke_points = []
        self.stroke_normals = []
        self.stroke_pressures = []
        self.painting = False

        # Hybrid architecture components
        self.viewport_renderer = None  # GLSL renderer
        self.npr_core_session = None   # npr_core painting session

    def invoke(self, context, event):
        # Initialize viewport renderer
        from .viewport_renderer import GaussianViewportRenderer
        self.viewport_renderer = context.scene.gaussian_viewport_renderer

        # Initialize npr_core session
        from npr_core.painting_session import PaintingSession
        self.npr_core_session = PaintingSession()

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Start stroke
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.painting = True
            self.stroke_points = []
            self.stroke_normals = []
            self.stroke_pressures = []
            return {'RUNNING_MODAL'}

        # Continue stroke
        if event.type == 'MOUSEMOVE' and self.painting:
            location, normal, hit = raycast_mouse_to_surface(context, event)
            pressure = get_tablet_pressure(event)

            # Add to stroke
            self.stroke_points.append(location)
            self.stroke_normals.append(normal)
            self.stroke_pressures.append(pressure)

            # Generate stamp (npr_core)
            stamp = self.generate_stamp(location, normal, pressure)

            # Update viewport (GLSL)
            self.update_viewport_immediate(stamp)

            # Trigger redraw
            context.area.tag_redraw()

            return {'RUNNING_MODAL'}

        # Finish stroke
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.painting:
            self.painting = False
            self.finish_stroke(context)
            return {'FINISHED'}

        # Cancel
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.painting = False
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def generate_stamp(self, location, normal, pressure):
        """
        Generate brush stamp at location.

        Returns:
            BrushStamp: npr_core stamp object
        """
        from npr_core.brush import BrushStamp

        # Get current brush
        brush = self.npr_core_session.current_brush

        # Place stamp with scaling based on pressure
        size_multiplier = 0.5 + 0.5 * pressure
        stamp = brush.place_at(
            position=np.array(location),
            normal=np.array(normal),
            size_multiplier=size_multiplier,
            opacity_multiplier=self.brush_opacity
        )

        return stamp

    def update_viewport_immediate(self, stamp):
        """
        Update GLSL viewport with new stamp (immediate feedback).

        Args:
            stamp: BrushStamp from npr_core
        """
        # Add stamp to scene data
        self.npr_core_session.scene_data.add_stamp(stamp)

        # Sync to GLSL viewport (incremental update)
        new_gaussians = stamp.gaussians
        start_idx = len(self.npr_core_session.scene_data.gaussians) - len(new_gaussians)

        # Pack new gaussians to 59-float format
        packed_data = self.viewport_renderer.data_manager.pack_gaussians_subset(
            new_gaussians, start_idx
        )

        # Update GPU texture (partial update)
        self.viewport_renderer.data_manager.update_partial(
            start_idx,
            start_idx + len(new_gaussians),
            packed_data
        )

    def finish_stroke(self, context):
        """
        Finish stroke and apply deformation (gsplat computation).
        """
        if len(self.stroke_points) < 2:
            return

        # Start incremental deformation processing
        bpy.ops.gaussian.apply_deformation('INVOKE_DEFAULT',
            stroke_id=id(self.stroke_points)
        )
```

---

### 3. Incremental Deformation (gsplat Computation)

#### 3.1 Deformation Operator

```python
# operators.py

class ApplyDeformationOperator(bpy.types.Operator):
    """Apply Deformation to Stroke (Hybrid: gsplat computation)"""
    bl_idname = "gaussian.apply_deformation"
    bl_label = "Apply Deformation"

    stroke_id: bpy.props.IntProperty()

    def __init__(self):
        self.stamps_to_process = []
        self.current_index = 0
        self.timer = None

        # Hybrid components
        self.gsplat_deformer = None
        self.viewport_renderer = None

    def invoke(self, context, event):
        # Get stroke data from paint operator
        paint_op = context.scene.gaussian_paint_session
        self.stamps_to_process = paint_op.get_stroke_stamps(self.stroke_id)

        # Initialize gsplat deformer
        from npr_core.deformation_gpu import DeformationGPU
        self.gsplat_deformer = DeformationGPU()

        # Initialize viewport renderer
        self.viewport_renderer = context.scene.gaussian_viewport_renderer

        # Setup timer for incremental processing
        self.timer = context.window_manager.event_timer_add(0.01, window=context.window)
        context.window_manager.modal_handler_add(self)

        # Progress bar
        context.window_manager.progress_begin(0, 100)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            # Cancel
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # Process batch (10 stamps at a time)
            batch_size = 10
            batch = self.stamps_to_process[self.current_index:self.current_index + batch_size]

            if batch:
                # Deform batch using gsplat (GPU computation)
                self.deform_batch_gsplat(batch)

                self.current_index += batch_size

                # Update progress
                progress = min(100, int(100 * self.current_index / len(self.stamps_to_process)))
                context.window_manager.progress_update(progress)

                # Redraw viewport
                context.area.tag_redraw()

            if self.current_index >= len(self.stamps_to_process):
                # Finished
                context.window_manager.progress_end()
                self.cleanup(context)
                return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def deform_batch_gsplat(self, batch):
        """
        Apply deformation to batch using gsplat (Hybrid computation).

        Args:
            batch: List of BrushStamp objects
        """
        import torch
        from npr_core.deformation_gpu import apply_spline_deformation

        # 1. Get gaussians as PyTorch tensors
        gaussians_numpy = np.array([
            stamp.gaussians_as_array() for stamp in batch
        ])
        gaussians_tensor = torch.from_numpy(gaussians_numpy).cuda()

        # 2. Compute spline parameters
        spline_points = torch.tensor([
            stamp.center for stamp in batch
        ], device='cuda', dtype=torch.float32)

        # 3. Apply deformation (gsplat GPU computation)
        deformed_tensor = apply_spline_deformation(
            gaussians_tensor,
            spline_points,
            radius=self.deformation_radius
        )

        # 4. Convert back to NumPy
        deformed_numpy = deformed_tensor.cpu().numpy()

        # 5. Update npr_core scene data
        for i, stamp in enumerate(batch):
            stamp.update_gaussians(deformed_numpy[i])

        # 6. Sync to GLSL viewport
        self.sync_to_viewport(batch)

    def sync_to_viewport(self, batch):
        """
        Sync deformed gaussians to GLSL viewport.

        Args:
            batch: List of BrushStamp objects (already deformed)
        """
        # Get indices of affected gaussians
        start_idx = batch[0].gaussian_start_idx
        end_idx = batch[-1].gaussian_end_idx

        # Pack to 59-float format
        packed_data = self.viewport_renderer.data_manager.pack_gaussians_range(
            start_idx, end_idx
        )

        # Update GPU texture
        self.viewport_renderer.data_manager.update_partial(
            start_idx, end_idx, packed_data
        )

    def cancel(self, context):
        """Cancel deformation."""
        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
        context.window_manager.progress_end()

    def cleanup(self, context):
        """Cleanup after completion."""
        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
```

---

### 4. Hybrid 데이터 동기화

#### 4.1 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────┐
│   User Input (Mouse/Tablet)                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│   npr_core: Generate Stamp                 │  ← Python/NumPy
│   - brush.place_at()                       │
│   - Returns BrushStamp (NumPy arrays)      │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌──────────────────┐  ┌────────────────────┐
│  GLSL Viewport   │  │  PyTorch Tensor    │
│  (Immediate)     │  │  (For computation) │
│                  │  │                    │
│  NumPy → Texture │  │  NumPy → Tensor    │
│  Partial update  │  │  Keep in VRAM      │
└────────┬─────────┘  └─────────┬──────────┘
         │                       │
         │                       ▼
         │            ┌──────────────────────┐
         │            │  gsplat Deformation  │
         │            │  (GPU computation)   │
         │            └──────────┬───────────┘
         │                       │
         │                       ▼
         │            ┌──────────────────────┐
         │            │  Tensor → NumPy      │
         │            └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │  Sync to Viewport    │
         │  (Update texture)    │
         └──────────────────────┘
```

#### 4.2 동기화 헬퍼 함수

```python
# hybrid_sync.py

import numpy as np
import torch

class HybridDataSync:
    """
    Manages data synchronization between:
    - NumPy (npr_core)
    - PyTorch (gsplat computation)
    - GLSL Texture (viewport)
    """

    def __init__(self):
        self.numpy_buffer = None
        self.torch_tensor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def numpy_to_torch(self, numpy_array):
        """
        Convert NumPy array to PyTorch tensor (GPU).

        Args:
            numpy_array: np.ndarray

        Returns:
            torch.Tensor on GPU
        """
        tensor = torch.from_numpy(numpy_array).to(self.device)
        return tensor

    def torch_to_numpy(self, tensor):
        """
        Convert PyTorch tensor to NumPy array.

        Args:
            tensor: torch.Tensor

        Returns:
            np.ndarray
        """
        return tensor.detach().cpu().numpy()

    def numpy_to_glsl_texture(self, numpy_array, texture_manager):
        """
        Upload NumPy array to GLSL texture.

        Args:
            numpy_array: np.ndarray, shape (N, 59)
            texture_manager: GaussianDataManager instance
        """
        texture_manager.upload_to_texture(numpy_array)

    def incremental_sync(self, start_idx, end_idx, numpy_array, texture_manager):
        """
        Incremental update for real-time painting.

        Args:
            start_idx: int
            end_idx: int
            numpy_array: np.ndarray, shape (end_idx - start_idx, 59)
            texture_manager: GaussianDataManager instance
        """
        texture_manager.update_partial(start_idx, end_idx, numpy_array)

    def benchmark_sync(self, size=10000):
        """
        Benchmark synchronization overhead.

        Args:
            size: Number of gaussians

        Returns:
            dict: Timing results
        """
        import time

        results = {}

        # Generate test data
        numpy_data = np.random.randn(size, 59).astype(np.float32)

        # NumPy → PyTorch
        start = time.time()
        tensor = self.numpy_to_torch(numpy_data)
        torch.cuda.synchronize()
        results['numpy_to_torch'] = (time.time() - start) * 1000

        # PyTorch computation (dummy)
        start = time.time()
        result_tensor = tensor * 2.0 + 1.0
        torch.cuda.synchronize()
        results['torch_computation'] = (time.time() - start) * 1000

        # PyTorch → NumPy
        start = time.time()
        result_numpy = self.torch_to_numpy(result_tensor)
        results['torch_to_numpy'] = (time.time() - start) * 1000

        return results
```

---

### 5. SharedMemory IPC 구현 (고성능 프로세스 간 통신)

Phase 2에서 구현한 Subprocess Actor 패턴은 Queue 기반 IPC를 사용합니다. 실시간 페인팅에서는 대용량 Gaussian 데이터 전송 시 Queue의 pickle 직렬화 오버헤드(~80ms for 10k gaussians)가 병목이 됩니다. SharedMemory를 통해 **zero-copy** 전송으로 <1ms 지연을 달성합니다.

#### 5.1 성능 요구사항

| IPC 방식         | 10k Gaussians (2.3MB) | 100k Gaussians (23MB) | 용도              |
| ---------------- | --------------------- | --------------------- | ----------------- |
| Queue (pickle)   | ~80ms                 | ~800ms                | 명령, 작은 데이터 |
| **SharedMemory** | **<1ms**              | **<5ms**              | 대용량 NumPy 배열 |

**목표**: 페인팅 중 Gaussian 데이터 동기화 지연 < 1ms

#### 5.2 GaussianSharedBuffer 클래스

```python
# src/generator_process/shared_buffer.py

from multiprocessing.shared_memory import SharedMemory
import numpy as np
from typing import Optional, Tuple
import atexit

class GaussianSharedBuffer:
    """
    SharedMemory wrapper for zero-copy Gaussian data transfer.

    Gaussian 데이터 포맷: (N, 59) float32 배열
    - positions: 3 floats
    - rotations: 4 floats (quaternion)
    - scales: 3 floats
    - colors: 3 floats (RGB)
    - opacity: 1 float
    - SH coefficients: 45 floats (15 * 3)

    Reference: Dream Textures realtime_viewport.py
    """

    # 59 floats per gaussian * 4 bytes = 236 bytes
    FLOATS_PER_GAUSSIAN = 59
    BYTES_PER_GAUSSIAN = FLOATS_PER_GAUSSIAN * 4

    def __init__(self, max_gaussians: int = 100000, name: Optional[str] = None):
        """
        Initialize shared buffer.

        Args:
            max_gaussians: Maximum number of gaussians to support
            name: Optional name for existing shared memory (for receiver side)
        """
        self.max_gaussians = max_gaussians
        self.current_count = 0
        self._shm: Optional[SharedMemory] = None
        self._array: Optional[np.ndarray] = None
        self._is_owner = False

        if name:
            # Attach to existing shared memory (receiver/subprocess side)
            self._attach(name)
        else:
            # Create new shared memory (sender/main process side)
            self._create()

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def _create(self):
        """Create new shared memory buffer."""
        size = self.max_gaussians * self.BYTES_PER_GAUSSIAN
        self._shm = SharedMemory(create=True, size=size)
        self._array = np.ndarray(
            (self.max_gaussians, self.FLOATS_PER_GAUSSIAN),
            dtype=np.float32,
            buffer=self._shm.buf
        )
        self._is_owner = True
        print(f"[SharedBuffer] Created: {self._shm.name}, size={size} bytes")

    def _attach(self, name: str):
        """Attach to existing shared memory buffer."""
        self._shm = SharedMemory(name=name)
        self._array = np.ndarray(
            (self.max_gaussians, self.FLOATS_PER_GAUSSIAN),
            dtype=np.float32,
            buffer=self._shm.buf
        )
        self._is_owner = False
        print(f"[SharedBuffer] Attached: {name}")

    @property
    def name(self) -> str:
        """Get shared memory name for IPC."""
        return self._shm.name if self._shm else ""

    @property
    def array(self) -> np.ndarray:
        """Get numpy array view of shared memory."""
        return self._array

    def write(self, data: np.ndarray, start_idx: int = 0) -> int:
        """
        Write gaussian data to shared buffer (zero-copy).

        Args:
            data: NumPy array of shape (N, 59), dtype=float32
            start_idx: Starting index in buffer

        Returns:
            Number of gaussians written
        """
        if data.shape[1] != self.FLOATS_PER_GAUSSIAN:
            raise ValueError(f"Expected {self.FLOATS_PER_GAUSSIAN} floats per gaussian, got {data.shape[1]}")

        n_gaussians = data.shape[0]
        end_idx = start_idx + n_gaussians

        if end_idx > self.max_gaussians:
            raise ValueError(f"Buffer overflow: {end_idx} > {self.max_gaussians}")

        # Zero-copy write (numpy view of shared memory)
        self._array[start_idx:end_idx] = data
        self.current_count = max(self.current_count, end_idx)

        return n_gaussians

    def read(self, start_idx: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Read gaussian data from shared buffer (zero-copy view).

        Args:
            start_idx: Starting index
            count: Number of gaussians to read (None = all from start_idx)

        Returns:
            NumPy array view (NOT a copy - modifications affect shared memory)
        """
        if count is None:
            count = self.current_count - start_idx

        end_idx = start_idx + count
        return self._array[start_idx:end_idx]

    def read_copy(self, start_idx: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Read gaussian data as a copy (safe for modification).

        Args:
            start_idx: Starting index
            count: Number of gaussians to read

        Returns:
            NumPy array copy
        """
        return self.read(start_idx, count).copy()

    def clear(self):
        """Clear buffer (reset count, optionally zero memory)."""
        self.current_count = 0
        # Optionally: self._array[:] = 0

    def cleanup(self):
        """Release shared memory resources."""
        if self._shm:
            try:
                self._shm.close()
                if self._is_owner:
                    self._shm.unlink()
                    print(f"[SharedBuffer] Unlinked: {self._shm.name}")
            except Exception as e:
                print(f"[SharedBuffer] Cleanup error: {e}")
            finally:
                self._shm = None
                self._array = None

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
```

#### 5.3 Actor 통합 (Hybrid Queue + SharedMemory)

```python
# src/generator_process/actor.py 수정사항

class Actor:
    """
    Hybrid IPC Pattern:
    - Queue: 명령 전송 (작은 데이터, <1KB)
    - SharedMemory: Gaussian 데이터 전송 (대용량, >1KB)
    """

    def __init__(self):
        # ... 기존 코드 ...
        self._shared_buffer: Optional[GaussianSharedBuffer] = None
        self._subprocess_buffer_name: Optional[str] = None

    def _setup_shared_buffer(self, max_gaussians: int = 100000):
        """Initialize shared buffer for gaussian data."""
        if self._shared_buffer is None:
            self._shared_buffer = GaussianSharedBuffer(max_gaussians)
            # Queue를 통해 subprocess에 buffer name 전달
            self._send_buffer_name(self._shared_buffer.name)

    def send_gaussians_shared(self, data: np.ndarray, start_idx: int = 0) -> Future:
        """
        Send gaussian data via SharedMemory (zero-copy).

        Args:
            data: Gaussian array (N, 59) float32
            start_idx: Starting index in shared buffer

        Returns:
            Future that resolves when subprocess acknowledges
        """
        self._setup_shared_buffer()

        # 1. Write to shared memory (zero-copy)
        count = self._shared_buffer.write(data, start_idx)

        # 2. Send notification via Queue (just metadata, not data)
        return self._send(
            action="sync_shared_gaussians",
            start_idx=start_idx,
            count=count
        )

    def receive_gaussians_shared(self, start_idx: int = 0, count: int = None) -> np.ndarray:
        """
        Receive gaussian data from subprocess via SharedMemory.

        Args:
            start_idx: Starting index
            count: Number of gaussians

        Returns:
            NumPy array (copy for safety)
        """
        if self._shared_buffer is None:
            raise RuntimeError("SharedBuffer not initialized")

        return self._shared_buffer.read_copy(start_idx, count)
```

#### 5.4 Subprocess 측 구현

```python
# src/generator_process/__init__.py (NPRGenerator 수정)

class NPRGenerator(Actor):
    """GPU computation actor with SharedMemory support."""

    @classmethod
    def _setup_shared_buffer_subprocess(cls, buffer_name: str, max_gaussians: int):
        """Attach to shared buffer in subprocess."""
        cls._shared_buffer = GaussianSharedBuffer(
            max_gaussians=max_gaussians,
            name=buffer_name  # Attach to existing
        )

    @classmethod
    def _handle_sync_shared_gaussians(cls, start_idx: int, count: int):
        """
        Handle gaussian sync notification from main process.
        Data is already in shared memory - just read it.
        """
        import torch

        # Zero-copy read from shared memory
        gaussians_np = cls._shared_buffer.read(start_idx, count)

        # Convert to PyTorch tensor (this does copy to GPU)
        gaussians_tensor = torch.from_numpy(gaussians_np).cuda()

        # Store for computation
        cls._current_gaussians = gaussians_tensor

        return {"status": "synced", "count": count}

    @classmethod
    def compute_deformation_shared(cls, params: dict) -> dict:
        """
        Compute deformation and write results back to SharedMemory.
        """
        import torch
        from npr_core.deformation_gpu import apply_deformation

        # Compute on GPU
        deformed = apply_deformation(
            cls._current_gaussians,
            params
        )

        # Write back to shared memory (GPU → CPU → SharedMemory)
        deformed_np = deformed.cpu().numpy()
        cls._shared_buffer.write(deformed_np, start_idx=0)

        return {
            "status": "complete",
            "count": deformed_np.shape[0]
        }
```

#### 5.5 데이터 흐름 다이어그램 (SharedMemory 버전)

```
┌─────────────────────────────────────────────────────────────┐
│                    Blender Main Process                     │
│  ┌─────────────┐    ┌──────────────────┐                   │
│  │ Paint Op    │───▶│ GaussianShared   │                   │
│  │ (UI Thread) │    │ Buffer (write)   │                   │
│  └─────────────┘    └────────┬─────────┘                   │
│                              │ zero-copy                    │
└──────────────────────────────┼──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   SharedMemory      │  ← OS-managed
                    │   (2.3MB for 10k)   │     memory region
                    └──────────┬──────────┘
                               │
┌──────────────────────────────┼──────────────────────────────┐
│                    PyTorch Subprocess                       │
│                              │ zero-copy                    │
│  ┌──────────────────┐   ┌────▼─────────┐   ┌─────────────┐ │
│  │ Queue (notify)   │   │ SharedBuffer │──▶│ CUDA Tensor │ │
│  │ {"action":       │   │ (read)       │   │ (GPU copy)  │ │
│  │  "sync", idx, n} │   └──────────────┘   └──────┬──────┘ │
│  └──────────────────┘                             │        │
│                                                   ▼        │
│                                          ┌───────────────┐ │
│                                          │ Deformation   │ │
│                                          │ (GPU compute) │ │
│                                          └───────┬───────┘ │
│                                                  │         │
│                              ┌───────────────────▼───────┐ │
│                              │ Write back to SharedMem   │ │
│                              └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### 5.6 Thread-Safety 고려사항

```python
# 동시 접근 방지를 위한 Lock 패턴

import threading
from contextlib import contextmanager

class ThreadSafeSharedBuffer(GaussianSharedBuffer):
    """Thread-safe wrapper for GaussianSharedBuffer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    @contextmanager
    def locked(self):
        """Context manager for exclusive access."""
        self._lock.acquire()
        try:
            yield self
        finally:
            self._lock.release()

    def write_locked(self, data: np.ndarray, start_idx: int = 0) -> int:
        """Thread-safe write."""
        with self._lock:
            return self.write(data, start_idx)

    def read_locked(self, start_idx: int = 0, count: int = None) -> np.ndarray:
        """Thread-safe read (returns copy)."""
        with self._lock:
            return self.read_copy(start_idx, count)
```

#### 5.7 에러 처리 및 Fallback

```python
# SharedMemory 실패 시 Queue fallback

class HybridIPCManager:
    """Manages hybrid Queue + SharedMemory IPC with fallback."""

    def __init__(self, actor: Actor):
        self.actor = actor
        self.shared_buffer = None
        self.use_shared_memory = True

    def send_gaussians(self, data: np.ndarray, start_idx: int = 0) -> Future:
        """
        Send gaussians with automatic fallback.

        - Try SharedMemory first (fast)
        - Fall back to Queue if SharedMemory fails (slow but reliable)
        """
        if self.use_shared_memory:
            try:
                return self.actor.send_gaussians_shared(data, start_idx)
            except Exception as e:
                print(f"[IPC] SharedMemory failed: {e}, falling back to Queue")
                self.use_shared_memory = False

        # Fallback: send via Queue (pickle serialization)
        return self.actor._send(
            action="sync_gaussians_queue",
            data=data.tolist(),  # Convert to Python list for pickle
            start_idx=start_idx
        )
```

---

> **📄 브러시 스트로크 파이프라인**은 별도 문서로 분리되었습니다:  
> **`docs/phase4.1_stroke_pipeline.md`** 참조
>
> Phase 4.1에서 다루는 내용:
>
> -   StrokePainter 라이프사이클 (start/update/finish)
> -   Arc-length 기반 균일 스탬프 배치
> -   2계층 브러시 아키텍처 (Pattern/Instance)
> -   3단계 배치 전략 (place_at, place_at_batch, place_at_batch_arrays)
> -   StrokeSpline 구성 및 프레임 계산
> -   GPU 배치 변형 최적화

---

## 🧪 테스트 및 검증

### Hybrid 페인팅 통합 테스트

```python
# tests/test_hybrid_painting.py

def test_hybrid_painting():
    """Test full painting pipeline with Hybrid architecture."""
    import bpy
    from npr_core.brush import Brush
    from npr_core.scene_data import SceneData

    # 1. Initialize components
    scene_data = SceneData()
    brush = Brush.from_image("path/to/brush.png")

    # 2. Simulate stroke
    stroke_points = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
    ]

    # 3. Generate stamps
    stamps = []
    for point in stroke_points:
        stamp = brush.place_at(
            position=np.array(point),
            normal=np.array([0, 0, 1]),
            size_multiplier=1.0,
            opacity_multiplier=0.5
        )
        stamps.append(stamp)
        scene_data.add_stamp(stamp)

    # 4. Update viewport (GLSL)
    viewport_renderer = bpy.context.scene.gaussian_viewport_renderer
    viewport_renderer.update_gaussians(scene_data)

    # 5. Apply deformation (gsplat)
    from npr_core.deformation_gpu import DeformationGPU
    deformer = DeformationGPU()

    deformed_scene = deformer.apply_to_scene(scene_data, stamps)

    # 6. Sync back to viewport
    viewport_renderer.update_gaussians(deformed_scene)

    print("✓ Hybrid painting test passed")


test_hybrid_painting()
```

### SharedMemory IPC 벤치마크

```python
def test_shared_memory_ipc():
    """Benchmark SharedMemory vs Queue IPC performance."""
    import time
    import numpy as np
    from generator_process.shared_buffer import GaussianSharedBuffer
    from multiprocessing import Queue
    import pickle

    # Test data: 10k gaussians (59 floats each)
    n_gaussians = 10000
    data = np.random.randn(n_gaussians, 59).astype(np.float32)
    data_size_mb = data.nbytes / (1024 * 1024)

    print(f"Test data: {n_gaussians} gaussians, {data_size_mb:.2f} MB")

    # Benchmark Queue (pickle)
    queue = Queue()
    start = time.perf_counter()
    for _ in range(10):
        queue.put(data)
        _ = queue.get()
    queue_time = (time.perf_counter() - start) / 10 * 1000

    # Benchmark SharedMemory
    with GaussianSharedBuffer(n_gaussians) as shm:
        start = time.perf_counter()
        for _ in range(10):
            shm.write(data)
            _ = shm.read_copy()
        shm_time = (time.perf_counter() - start) / 10 * 1000

    print(f"Queue (pickle): {queue_time:.2f} ms")
    print(f"SharedMemory:   {shm_time:.2f} ms")
    print(f"Speedup:        {queue_time / shm_time:.1f}x")

    # Verify performance target
    assert shm_time < 5, f"SharedMemory too slow: {shm_time:.2f}ms > 5ms target"
    print("✓ SharedMemory IPC benchmark passed")


test_shared_memory_ipc()
```

### 성능 목표

-   ✓ Stroke latency < 50ms (mouse → viewport)
-   ✓ Deformation time < 1초 (100 stamps)
-   ✓ Viewport FPS > 20 during painting
-   ✓ Memory overhead < 100MB (sync buffers)
-   ✓ **SharedMemory IPC < 1ms** (10k gaussians)
-   ✓ **SharedMemory IPC < 5ms** (100k gaussians)

---

## 📚 참고 자료

-   npr_core deformation_gpu.py implementation
-   Blender Modal Operator docs
-   PyTorch tensor operations guide
-   [Python multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
-   Dream Textures realtime_viewport.py (SharedMemory reference implementation)
    viewport_renderer.update_gaussians(scene_data)

    # 5. Apply deformation (gsplat)

    from npr_core.deformation_gpu import DeformationGPU
    deformer = DeformationGPU()

    deformed_scene = deformer.apply_to_scene(scene_data, stamps)

    # 6. Sync back to viewport

    viewport_renderer.update_gaussians(deformed_scene)

    print("✓ Hybrid painting test passed")

test_hybrid_painting()

````

### SharedMemory IPC 벤치마크

```python
def test_shared_memory_ipc():
    """Benchmark SharedMemory vs Queue IPC performance."""
    import time
    import numpy as np
    from generator_process.shared_buffer import GaussianSharedBuffer
    from multiprocessing import Queue
    import pickle

    # Test data: 10k gaussians (59 floats each)
    n_gaussians = 10000
    data = np.random.randn(n_gaussians, 59).astype(np.float32)
    data_size_mb = data.nbytes / (1024 * 1024)

    print(f"Test data: {n_gaussians} gaussians, {data_size_mb:.2f} MB")

    # Benchmark Queue (pickle)
    queue = Queue()
    start = time.perf_counter()
    for _ in range(10):
        queue.put(data)
        _ = queue.get()
    queue_time = (time.perf_counter() - start) / 10 * 1000

    # Benchmark SharedMemory
    with GaussianSharedBuffer(n_gaussians) as shm:
        start = time.perf_counter()
        for _ in range(10):
            shm.write(data)
            _ = shm.read_copy()
        shm_time = (time.perf_counter() - start) / 10 * 1000

    print(f"Queue (pickle): {queue_time:.2f} ms")
    print(f"SharedMemory:   {shm_time:.2f} ms")
    print(f"Speedup:        {queue_time / shm_time:.1f}x")

    # Verify performance target
    assert shm_time < 5, f"SharedMemory too slow: {shm_time:.2f}ms > 5ms target"
    print("✓ SharedMemory IPC benchmark passed")

test_shared_memory_ipc()
````

### 성능 목표

-   ✓ Stroke latency < 50ms (mouse → viewport)
-   ✓ Deformation time < 1초 (100 stamps)
-   ✓ Viewport FPS > 20 during painting
-   ✓ Memory overhead < 100MB (sync buffers)
-   ✓ **SharedMemory IPC < 1ms** (10k gaussians)
-   ✓ **SharedMemory IPC < 5ms** (100k gaussians)

---

## 📚 참고 자료

-   npr_core deformation_gpu.py implementation
-   Blender Modal Operator docs
-   PyTorch tensor operations guide
-   [Python multiprocessing.shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
-   Dream Textures realtime_viewport.py (SharedMemory reference implementation)
