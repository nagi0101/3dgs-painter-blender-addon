# Phase 1: 핵심 구조 리팩터링 (Core Refactoring for Blender)

**기간**: 2주  
**목표**: 웹 기반 프로토타입을 Blender 환경에 맞게 변환

---

## 📋 작업 개요

본 Phase는 프로토타입의 **동기적(sync) Blender 통합**을 위한 핵심 변환 작업입니다:
- ✓ WebSocket/FastAPI 제거 (비동기 → 동기)
- ✓ GPU 컨텍스트 관리 (Blender 통합)
- ✓ 파일 시스템 구조 변경 (addon 규격)
- ✓ npr_core 모듈 독립성 확보

---

## 🎯 핵심 작업

### 1. WebSocket/FastAPI 제거

#### 1.1 현재 구조 분석

**제거 대상**:
```python
# backend/main.py (REMOVE)
from fastapi import FastAPI
from fastapi.websockets import WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # ...

# backend/api/websocket.py (REMOVE)
class WebSocketManager:
    async def send_json(self, data):
        await self.websocket.send_json(data)
```

**유지 대상** (동기화 변환 필요):
```python
# backend/core/*.py
# - brush_converter.py
# - deformation_gpu.py
# - gaussian.py
# - scene_data.py
# - spline.py
# ... etc
```

#### 1.2 동기적 API 설계

```python
# npr_core/api.py (NEW)

class NPRCoreAPI:
    """
    Synchronous API for Blender integration.
    Replaces async WebSocket communication.
    """
    
    def __init__(self):
        self.scene_data = SceneData()
        self.brush_manager = BrushManager()
        self.deformation_engine = DeformationGPU()
    
    def load_brush(self, filepath):
        """
        Load brush from file (sync).
        
        Args:
            filepath: str, path to brush JSON
        
        Returns:
            Brush object
        """
        brush = self.brush_manager.load(filepath)
        return brush
    
    def place_stamp(self, brush, position, normal, size=1.0, opacity=1.0):
        """
        Place brush stamp at position (sync).
        
        Args:
            brush: Brush object
            position: np.ndarray [3]
            normal: np.ndarray [3]
            size: float
            opacity: float
        
        Returns:
            BrushStamp object
        """
        stamp = brush.place_at(
            position=position,
            normal=normal,
            size_multiplier=size,
            opacity_multiplier=opacity
        )
        
        self.scene_data.add_stamp(stamp)
        return stamp
    
    def apply_deformation(self, stamps, spline_params):
        """
        Apply deformation to stamps (sync).
        
        Args:
            stamps: List of BrushStamp objects
            spline_params: dict with spline configuration
        
        Returns:
            Updated scene_data
        """
        deformed_scene = self.deformation_engine.apply(
            self.scene_data,
            stamps,
            spline_params
        )
        
        self.scene_data = deformed_scene
        return self.scene_data
    
    def get_scene_data(self):
        """
        Get current scene data (sync).
        
        Returns:
            SceneData object
        """
        return self.scene_data
    
    def clear_scene(self):
        """Clear all gaussians from scene (sync)."""
        self.scene_data = SceneData()
```

#### 1.3 변환 체크리스트

- [ ] **FastAPI 의존성 제거**
  - `backend/main.py` 삭제
  - `backend/api/websocket.py` 삭제
  - `backend/api/upload.py` 변환 (파일 업로드 → 직접 파일 읽기)

- [ ] **비동기 코드 동기화**
  ```python
  # Before (async)
  async def process_stroke(self, stroke_data):
      result = await self.compute_deformation(stroke_data)
      await self.send_update(result)
  
  # After (sync)
  def process_stroke(self, stroke_data):
      result = self.compute_deformation(stroke_data)
      return result
  ```

- [ ] **JSON 통신 → 직접 객체 전달**
  ```python
  # Before (WebSocket JSON)
  data = {
      "type": "place_stamp",
      "position": [x, y, z],
      "brush_id": "..."
  }
  await websocket.send_json(data)
  
  # After (Direct call)
  stamp = npr_api.place_stamp(
      brush=current_brush,
      position=np.array([x, y, z]),
      normal=np.array([0, 0, 1])
  )
  ```

---

### 2. GPU 컨텍스트 관리

#### 2.1 문제점

**프로토타입**: FastAPI 프로세스가 독립적으로 GPU 컨텍스트 소유
**Blender**: Blender 프로세스가 이미 GPU 컨텍스트 소유 (OpenGL/CUDA 공유 필요)

#### 2.2 해결 방안

```python
# npr_core/gpu_context.py (NEW)

import torch
import bpy

class BlenderGPUContext:
    """
    Manage PyTorch CUDA context within Blender.
    """
    
    def __init__(self):
        self.device = None
        self.initialized = False
    
    def initialize(self):
        """
        Initialize PyTorch CUDA context.
        Must be called after Blender starts.
        """
        if self.initialized:
            return
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # Set device
        self.device = torch.device('cuda:0')
        
        # Warm-up (allocate small tensor to initialize context)
        dummy = torch.zeros(1, device=self.device)
        del dummy
        torch.cuda.synchronize()
        
        self.initialized = True
        print(f"✓ PyTorch CUDA context initialized on {torch.cuda.get_device_name(0)}")
    
    def get_device(self):
        """Get current CUDA device."""
        if not self.initialized:
            self.initialize()
        return self.device
    
    def synchronize(self):
        """Synchronize CUDA operations."""
        if self.initialized:
            torch.cuda.synchronize()
    
    def clear_cache(self):
        """Clear CUDA cache to free memory."""
        if self.initialized:
            torch.cuda.empty_cache()

# Global instance
_gpu_context = BlenderGPUContext()

def get_gpu_context():
    """Get global GPU context."""
    return _gpu_context
```

#### 2.3 Deformation 엔진 통합

```python
# npr_core/deformation_gpu.py (MODIFIED)

import torch
from .gpu_context import get_gpu_context

class DeformationGPU:
    def __init__(self):
        self.gpu_context = get_gpu_context()
        self.device = None
    
    def initialize(self):
        """Initialize GPU resources."""
        self.gpu_context.initialize()
        self.device = self.gpu_context.get_device()
    
    def apply(self, scene_data, stamps, spline_params):
        """Apply deformation (GPU)."""
        if self.device is None:
            self.initialize()
        
        # Convert to tensors
        positions = torch.tensor(
            [g.position for g in scene_data.gaussians],
            device=self.device, dtype=torch.float32
        )
        
        # ... deformation computation ...
        
        # Synchronize before returning
        self.gpu_context.synchronize()
        
        return deformed_scene
```

---

### 3. 파일 시스템 구조 변경

#### 3.1 Addon 구조

```
npr_gaussian_painter/
    __init__.py          # Addon registration
    ui.py                # UI panels
    operators.py         # Operators
    properties.py        # Properties
    viewport_renderer.py # GLSL viewport
    npr_core/            # Core module (refactored from backend)
        __init__.py
        api.py           # Synchronous API
        gpu_context.py   # GPU management
        brush.py
        brush_manager.py
        deformation_gpu.py
        gaussian.py
        scene_data.py
        spline.py
        ... (other core modules)
    shaders/
        gaussian_vert.glsl
        gaussian_frag.glsl
    data/
        brushes/
            library.json
            brushes/*.json
```

#### 3.2 파일 이동 매핑

```bash
# Move backend/core/* to npr_core/
backend/core/brush.py              → npr_core/brush.py
backend/core/brush_manager.py      → npr_core/brush_manager.py
backend/core/deformation_gpu.py    → npr_core/deformation_gpu.py
backend/core/gaussian.py           → npr_core/gaussian.py
backend/core/scene_data.py         → npr_core/scene_data.py
backend/core/spline.py             → npr_core/spline.py

# Move data/ to addon
data/brushes/                      → npr_gaussian_painter/data/brushes/

# Remove frontend (not needed)
frontend/                          → (DELETE)

# Remove backend/api (WebSocket)
backend/api/                       → (DELETE)
backend/main.py                    → (DELETE)
```

#### 3.3 Import 경로 수정

```python
# Before (backend)
from backend.core.brush import Brush
from backend.core.scene_data import SceneData

# After (addon)
from .npr_core.brush import Brush
from .npr_core.scene_data import SceneData
```

---

### 4. npr_core 모듈 독립성

#### 4.1 의존성 최소화

**목표**: npr_core가 Blender에 의존하지 않도록 (순수 Python/NumPy/PyTorch)

```python
# npr_core/brush.py (GOOD - No Blender dependency)

import numpy as np
from PIL import Image

class Brush:
    def __init__(self, image_data):
        self.image_data = image_data  # NumPy array
        self.width = image_data.shape[1]
        self.height = image_data.shape[0]
    
    @classmethod
    def from_file(cls, filepath):
        """Load from image file (PNG/JPG)."""
        img = Image.open(filepath)
        image_data = np.array(img)
        return cls(image_data)
```

```python
# BAD EXAMPLE (avoid this in npr_core/)
import bpy  # ← NO! npr_core should not import bpy

class Brush:
    def __init__(self, blender_image):  # ← NO! Use NumPy arrays
        self.image = blender_image
```

#### 4.2 인터페이스 계층

```
┌─────────────────────────────────┐
│  Blender Addon (operators.py)  │  ← Blender-aware
│  - bpy imports allowed          │
│  - UI interaction               │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  npr_core.api.py                │  ← Adapter layer
│  - Convert bpy types → NumPy    │
│  - No bpy imports               │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  npr_core/* (core modules)      │  ← Pure Python
│  - NumPy/PyTorch only           │
│  - No bpy imports               │
└─────────────────────────────────┘
```

---

## 🧪 테스트 및 검증

### 단위 테스트

```python
# tests/test_npr_core.py

def test_brush_loading():
    """Test brush loading without Blender."""
    from npr_core.brush import Brush
    
    brush = Brush.from_file("data/brushes/test_brush.png")
    assert brush.width > 0
    assert brush.height > 0
    print("✓ Brush loading test passed")

def test_stamp_placement():
    """Test stamp placement."""
    from npr_core.brush import Brush
    from npr_core.scene_data import SceneData
    import numpy as np
    
    brush = Brush.from_file("data/brushes/test_brush.png")
    scene_data = SceneData()
    
    stamp = brush.place_at(
        position=np.array([0, 0, 0]),
        normal=np.array([0, 0, 1]),
        size_multiplier=1.0
    )
    
    scene_data.add_stamp(stamp)
    assert len(scene_data.gaussians) > 0
    print("✓ Stamp placement test passed")

def test_gpu_context():
    """Test GPU context initialization."""
    from npr_core.gpu_context import get_gpu_context
    
    ctx = get_gpu_context()
    ctx.initialize()
    
    import torch
    device = ctx.get_device()
    assert device.type == 'cuda'
    print("✓ GPU context test passed")
```

### 통합 테스트 (Blender)

```python
# Run inside Blender Python console

import sys
sys.path.append("path/to/npr_gaussian_painter")

from npr_core.api import NPRCoreAPI

# Initialize
api = NPRCoreAPI()

# Load brush
brush = api.load_brush("data/brushes/test_brush.json")

# Place stamp
stamp = api.place_stamp(
    brush=brush,
    position=np.array([0, 0, 0]),
    normal=np.array([0, 0, 1])
)

# Check scene data
scene_data = api.get_scene_data()
print(f"Total gaussians: {len(scene_data.gaussians)}")
```

---

## 📚 변환 체크리스트

### 파일 구조
- [ ] `backend/core/*` → `npr_core/*` 이동
- [ ] `data/brushes/` → `npr_gaussian_painter/data/brushes/` 이동
- [ ] `backend/api/`, `backend/main.py`, `frontend/` 삭제
- [ ] Addon 구조 생성 (`__init__.py`, `ui.py`, `operators.py`, etc.)

### 코드 변환
- [ ] 모든 `async`/`await` 제거
- [ ] WebSocket 통신 → 직접 함수 호출
- [ ] JSON 직렬화 → 객체 직접 전달
- [ ] Import 경로 수정 (`backend.core` → `.npr_core`)

### GPU 관리
- [ ] `BlenderGPUContext` 구현
- [ ] `DeformationGPU`에 컨텍스트 통합
- [ ] CUDA 초기화 테스트

### 독립성 검증
- [ ] npr_core 모듈에서 `import bpy` 제거
- [ ] 순수 Python/NumPy/PyTorch로만 동작 확인
- [ ] 단위 테스트 (Blender 없이 실행)

---

## 🎯 완료 기준

- ✓ npr_core 모듈이 Blender 없이 독립적으로 테스트 가능
- ✓ 모든 비동기 코드 제거 완료
- ✓ GPU 컨텍스트가 Blender 내에서 정상 동작
- ✓ Addon 구조로 파일 시스템 재구성 완료
