# Phase 4: 페인팅 인터랙션 구현 (Painting Interaction + gsplat Integration)

**기간**: 3주  
**목표**: Real-time painting + Hybrid 데이터 동기화 + gsplat Deformation 통합

---

## 📋 작업 개요

본 Phase는 Hybrid 아키텍처의 **양방향 통합**을 구현합니다:
- ✓ GLSL Viewport (실시간 페인팅 표시)
- ✓ gsplat Computation (Deformation 계산)
- ✓ 데이터 동기화 (NumPy ↔ PyTorch ↔ GLSL)

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

## 🧪 테스트 및 검증

### 통합 테스트

```python
# Test script

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

### 성능 목표

- ✓ Stroke latency < 50ms (mouse → viewport)
- ✓ Deformation time < 1초 (100 stamps)
- ✓ Viewport FPS > 20 during painting
- ✓ Memory overhead < 100MB (sync buffers)

---

## 📚 참고 자료

- npr_core deformation_gpu.py implementation
- Blender Modal Operator docs
- PyTorch tensor operations guide
