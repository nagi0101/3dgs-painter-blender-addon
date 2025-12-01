# Phase 0: 실행 가능성 검증 (Feasibility Study)

**기간**: 1주  
**목표**: Hybrid 아키텍처(GLSL + gsplat)의 기술적 검증

---

## 📋 작업 체크리스트

### 1. 블렌더 Python 환경 구축

#### 1.1 pip 설치
```python
import ensurepip
ensurepip.bootstrap()
```

#### 1.2 PyTorch 설치 테스트
```bash
# 블렌더 Python 경로 확인
import sys
print(sys.executable)

# pip로 PyTorch 설치
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

#### 1.3 CUDA 검증
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**성공 기준**:
- ✓ `torch.cuda.is_available()` == True
- ✓ CUDA version >= 11.8
- ✓ VRAM >= 4GB

---

### 2. GLSL Viewport 프로토타입

#### 2.1 최소 Shader 작성

**`minimal_gaussian_vert.glsl`**:
```glsl
#version 330

uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

// Per-instance data (100 gaussians)
uniform sampler2D gaussian_data;  // 100×4 texture (pos.xyz, opacity)

out vec4 v_color;

void main() {
    int gaussian_id = gl_InstanceID;
    
    // Fetch gaussian data
    vec4 data = texelFetch(gaussian_data, ivec2(gaussian_id, 0), 0);
    vec3 pos = data.xyz;
    float opacity = data.w;
    
    // Billboard quad vertices
    vec2 quad_offsets[4] = vec2[](
        vec2(-0.1, -0.1), vec2(0.1, -0.1),
        vec2(-0.1, 0.1), vec2(0.1, 0.1)
    );
    vec2 offset = quad_offsets[gl_VertexID % 4];
    
    // Transform to screen space
    vec4 view_pos = ViewMatrix * vec4(pos, 1.0);
    view_pos.xy += offset;
    gl_Position = ProjectionMatrix * view_pos;
    
    v_color = vec4(1.0, 0.5, 0.2, opacity);
}
```

**`minimal_gaussian_frag.glsl`**:
```glsl
#version 330

in vec4 v_color;
out vec4 FragColor;

void main() {
    // Simple circular falloff
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = length(coord);
    if (dist > 1.0) discard;
    
    float alpha = v_color.a * exp(-dist * dist);
    FragColor = vec4(v_color.rgb, alpha);
}
```

#### 2.2 Draw Handler 등록

```python
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

class MinimalGaussianRenderer:
    def __init__(self):
        self.shader = gpu.types.GPUShader(
            vertexcode=open("minimal_gaussian_vert.glsl").read(),
            fragcode=open("minimal_gaussian_frag.glsl").read()
        )
        
        # Test data: 100 gaussians
        positions = np.random.randn(100, 3).astype(np.float32)
        opacities = np.ones(100, dtype=np.float32) * 0.5
        self.data = np.column_stack([positions, opacities])
        
        # Upload to GPU texture
        self.texture = gpu.types.GPUTexture((100, 1), format='RGBA32F', data=self.data)
        
    def draw(self):
        self.shader.bind()
        self.shader.uniform_sampler("gaussian_data", self.texture)
        
        # Instanced draw (4 vertices × 100 instances)
        gpu.state.blend_set('ALPHA')
        batch = batch_for_shader(
            self.shader, 'TRI_STRIP',
            {"position": [(0, 0), (1, 0), (0, 1), (1, 1)]},
        )
        batch.draw(self.shader, instances=100)

# Register draw handler
renderer = MinimalGaussianRenderer()
handle = bpy.types.SpaceView3D.draw_handler_add(
    renderer.draw, (), 'WINDOW', 'POST_VIEW'
)
```

#### 2.3 FPS 측정

```python
import time

frame_times = []
for i in range(100):
    start = time.time()
    bpy.context.area.tag_redraw()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN', iterations=1)
    frame_times.append(time.time() - start)

avg_fps = 1.0 / np.mean(frame_times)
print(f"Average FPS: {avg_fps:.1f}")
```

**성공 기준**:
- ✓ 100 gaussians @ 60 FPS
- ✓ 1,000 gaussians @ 60 FPS
- ✓ 10,000 gaussians @ 30+ FPS

---

### 3. gsplat Computation 프로토타입

#### 3.1 gsplat 설치
```bash
pip install gsplat
```

#### 3.2 기본 동작 테스트

```python
import torch
from gsplat import rasterization

# Test data
means = torch.randn(100, 3, device='cuda')
quats = torch.randn(100, 4, device='cuda')
scales = torch.ones(100, 3, device='cuda') * 0.1
opacities = torch.ones(100, device='cuda') * 0.5
colors = torch.rand(100, 3, device='cuda')

# Camera parameters
viewmat = torch.eye(4, device='cuda')
K = torch.tensor([
    [500, 0, 256],
    [0, 500, 256],
    [0, 0, 1]
], device='cuda', dtype=torch.float32)

# Render
img_height, img_width = 512, 512
render_colors, render_alphas, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmat[None],
    Ks=K[None],
    width=img_width,
    height=img_height,
    render_mode="RGB",
)

print(f"Output shape: {render_colors.shape}")  # [1, 512, 512, 3]
print(f"Render time: {info['time']:.3f}ms")
```

**성공 기준**:
- ✓ gsplat import 성공
- ✓ rasterization() 정상 실행
- ✓ 출력 텐서 shape 확인

---

### 4. Hybrid 데이터 동기화 검증

#### 4.1 NumPy ↔ PyTorch 변환

```python
# NumPy → PyTorch
numpy_data = np.random.randn(10000, 7).astype(np.float32)
torch_tensor = torch.from_numpy(numpy_data).cuda()

# PyTorch → NumPy
result_tensor = torch_tensor * 2.0  # Some computation
result_numpy = result_tensor.cpu().numpy()

print(f"NumPy shape: {numpy_data.shape}")
print(f"Torch shape: {torch_tensor.shape}")
print(f"Result shape: {result_numpy.shape}")
```

#### 4.2 GPU Texture 업로드 시간 측정

```python
import time

data_sizes = [1000, 10000, 100000]
for size in data_sizes:
    data = np.random.randn(size, 4).astype(np.float32)
    
    start = time.time()
    texture = gpu.types.GPUTexture((size, 1), format='RGBA32F', data=data)
    upload_time = (time.time() - start) * 1000  # ms
    
    print(f"{size} gaussians: {upload_time:.2f}ms")
```

**성공 기준**:
- ✓ 10k gaussians 업로드 < 5ms
- ✓ NumPy ↔ PyTorch 변환 overhead < 1ms

#### 4.3 전체 Roundtrip Latency

```python
# Simulate full pipeline
def test_hybrid_pipeline():
    # 1. Generate data (npr_core)
    numpy_data = np.random.randn(10000, 7).astype(np.float32)
    
    # 2. Upload to GLSL viewport
    start = time.time()
    glsl_texture = gpu.types.GPUTexture((10000, 1), format='RGBA32F', data=numpy_data[:, :4])
    glsl_time = (time.time() - start) * 1000
    
    # 3. Convert to PyTorch for computation
    start = time.time()
    torch_tensor = torch.from_numpy(numpy_data).cuda()
    torch_time = (time.time() - start) * 1000
    
    # 4. Simulate gsplat computation
    start = time.time()
    result_tensor = torch_tensor * 2.0 + 1.0  # Dummy operation
    torch.cuda.synchronize()
    compute_time = (time.time() - start) * 1000
    
    # 5. Convert back to NumPy
    start = time.time()
    result_numpy = result_tensor.cpu().numpy()
    back_time = (time.time() - start) * 1000
    
    # 6. Update GLSL texture
    start = time.time()
    glsl_texture = gpu.types.GPUTexture((10000, 1), format='RGBA32F', data=result_numpy[:, :4])
    update_time = (time.time() - start) * 1000
    
    total = glsl_time + torch_time + compute_time + back_time + update_time
    print(f"GLSL upload: {glsl_time:.2f}ms")
    print(f"To PyTorch: {torch_time:.2f}ms")
    print(f"Computation: {compute_time:.2f}ms")
    print(f"To NumPy: {back_time:.2f}ms")
    print(f"GLSL update: {update_time:.2f}ms")
    print(f"Total: {total:.2f}ms")
    return total

latency = test_hybrid_pipeline()
```

**성공 기준**:
- ✓ Total roundtrip < 20ms
- ✓ GLSL viewport rendering 영향 최소화

---

## 🎯 Decision Point

### 성공 조건 (모두 만족 시 Phase 1 진행)

1. **GLSL Viewport**:
   - [x] 10k gaussians @ 30+ FPS
   - [x] Depth buffer integration 동작
   - [x] Blender 3D 객체와 occlusion 정상

2. **gsplat Computation**:
   - [x] Import 및 rasterization() 성공
   - [x] CUDA 정상 동작
   - [x] 기본 연산 속도 확인

3. **Hybrid 동기화**:
   - [x] 데이터 변환 overhead < 5ms
   - [x] 전체 roundtrip < 20ms
   - [x] Viewport FPS 저하 없음

### 실패 시 대안

- GLSL 성능 미달 → Geometry Nodes 방식 검토
- gsplat 동작 불가 → CPU fallback (NumPy/PyTorch)
- 동기화 overhead 심각 → 단일 파이프라인 재검토

---

## 📊 예상 결과

**성공 시나리오**:
```
✓ GLSL viewport: 10k @ 45 FPS
✓ gsplat computation: 정상 동작
✓ Hybrid roundtrip: 12ms
→ Phase 1 진행 승인
```

**Risk 시나리오**:
```
✗ GLSL viewport: 10k @ 18 FPS (목표 미달)
→ 원인 분석: Texture size? Shader complexity?
→ 최적화 시도 or 대안 검토
```

---

## 🔧 디버깅 팁

### GLSL Shader 디버깅
```python
# Shader compile error 확인
try:
    shader = gpu.types.GPUShader(vertexcode=vert, fragcode=frag)
except Exception as e:
    print(f"Shader error: {e}")
```

### PyTorch CUDA 문제
```python
# OOM 에러 시
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
```

### 성능 프로파일링
```python
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()
profiler.print_stats(sort='cumtime')
```
