# Blender VR 커스텀 렌더링 - 기술 컨텍스트

> **목적**: 검색 에이전트용 코드 참조 문서

---

## 1. 현재 GLSL Gaussian 렌더러 핵심 구조

### Draw Handler 등록 (viewport_renderer.py)

```python
# 문제: 이 handler가 VR에서 호출되지 않음
self.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
    self._draw_callback,
    (),
    'WINDOW',
    'POST_VIEW'  # ← VR offscreen loop에 미포함
)
```

### 셰이더 Uniform 전달 (PC에서 작동)

```python
def _draw_callback(self):
    # gpu.matrix 사용 (VR 호환 시도)
    view_matrix = gpu.matrix.get_model_view_matrix()
    projection_matrix = gpu.matrix.get_projection_matrix()

    self.shader.bind()
    self.shader.uniform_float("viewProjectionMatrix", proj @ view)
    self.shader.uniform_float("viewMatrix", view_matrix)
    # ... 더 많은 uniforms

    self.batch.draw(self.shader)
```

---

## 2. GLSL 셰이더 구조 (Gaussian Splatting)

### Vertex Shader 핵심

```glsl
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform vec4 camPosAndFocalX;
uniform vec4 viewportAndFocalY;
uniform sampler2D gaussianData;  // Texture로 Gaussian 데이터 전달

void main() {
    // 1. Texture에서 Gaussian 파라미터 읽기
    vec3 position = texelFetch(gaussianData, ivec2(offset, 0), 0).xyz;
    vec4 rotation = texelFetch(...);
    vec3 scale = texelFetch(...);

    // 2. 3D Covariance 계산
    mat3 cov3D = computeCov3D(scale, rotation);

    // 3. 2D 투영 (Jacobian)
    vec3 cov2D = computeCov2D(position, cov3D, focal);

    // 4. Billboard quad 생성
    gl_Position = viewProjectionMatrix * vec4(worldPos, 1.0);
}
```

### Fragment Shader 핵심

```glsl
void main() {
    // Gaussian 평가: exp(-0.5 * x^T * Σ^-1 * x)
    float power = -0.5 * (
        conic.x * coord.x * coord.x +
        conic.z * coord.y * coord.y +
        2.0 * conic.y * coord.x * coord.y
    );
    float alpha = opacity * exp(power);
    fragColor = vec4(color * alpha, alpha);
}
```

---

## 3. VR 세션 상태 접근

### 현재 작동하는 코드

```python
def get_vr_head_position(context):
    xr = context.window_manager.xr_session_state
    if xr and xr.is_running(context):
        return xr.viewer_pose_location
    return None

def get_controller_position(context, hand=1):
    xr = context.window_manager.xr_session_state
    return Vector(xr.controller_grip_location_get(context, hand))
```

---

## 4. 브러시 시스템 구조

### BrushStamp (학습된 브러시 포함)

```python
class BrushStamp:
    gaussians: List[Gaussian2D]  # 템플릿 Gaussian들

    def place_at(self, position, normal) -> List[Gaussian2D]:
        # 변환 적용하여 새 위치에 배치
        pass

    def place_at_batch_arrays(self, positions, normals) -> np.ndarray:
        # 배치 처리 (40-80x 빠름)
        pass
```

### StrokePainter (스트로크 처리)

```python
painter = StrokePainter(brush, scene_data)
painter.start_stroke(position, normal, pressure)
painter.update_stroke(position, normal, pressure)
stamps = painter.finish_stroke()  # Deformation + Inpainting 적용
```

---

## 5. 필요한 정보

### VR에서 GLSL을 직접 렌더링하려면:

1. **RenderEngine 접근법**

   - `bpy.types.RenderEngine`이 VR 세션에서 어떻게 작동하는가?
   - `view_draw()` 메서드가 VR per-eye 렌더링에서 호출되는가?

2. **Blender VR 소스 구조**

   - `source/blender/windowmanager/xr/wm_xr_draw.c`
   - `GHOST_IXrGraphicsBinding` 클래스
   - Python callback 주입 지점

3. **OpenXR Layer 접근**

   - Blender 외부에서 OpenXR composition layer 추가
   - Python ctypes로 OpenXR 함수 호출

4. **GPU Context 공유**
   - GPUOffScreen texture를 VR swapchain에 직접 사용
   - OpenGL texture handle 공유

---

## 6. 프로젝트 파일 구조

```
src/
├── viewport/
│   ├── viewport_renderer.py    # ← 핵심: GLSL 렌더러
│   └── gaussian_data.py        # GPU 데이터 관리
├── npr_core/
│   ├── brush.py                # BrushStamp, StrokePainter
│   ├── brush_converter.py      # 이미지→Gaussian
│   ├── spline.py               # Arc-length 스플라인
│   └── deformation_gpu.py      # PyTorch GPU 변형
├── vr/
│   ├── vr_freehand_paint.py    # VR 페인팅 (현재 구현)
│   ├── vr_session.py           # VR 세션 관리
│   └── vr_input.py             # 컨트롤러 입력
└── generator_process/          # Subprocess (PyTorch)
```
