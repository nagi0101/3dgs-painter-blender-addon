# 검색 에이전트용 기술 컨텍스트 - 코드 발췌

> **목적**: VR Gaussian Splatting 문제 해결을 위한 핵심 코드 컨텍스트

---

## 1. 현재 GLSL 렌더러 구조 (viewport_renderer.py 발췌)

### Draw Handler 등록 방식

```python
# Blender에 draw callback 등록
self.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
    self._draw_callback,
    (),
    'WINDOW',
    'POST_VIEW'  # VR에서 호출되지 않는 것으로 확인됨
)
```

### VR 호환을 위해 시도한 수정

```python
def _get_camera_params(self, context):
    # gpu.matrix 사용 (VR per-eye 행렬 지원 시도)
    view_matrix = gpu.matrix.get_model_view_matrix()
    projection_matrix = gpu.matrix.get_projection_matrix()

    # 카메라 위치 추출
    view_inv = view_matrix.inverted()
    camera_position = view_inv.translation.copy()

    return view_matrix, projection_matrix, camera_position, ...
```

### VR Context 검사 우회 시도

```python
def _draw_callback(self):
    context = bpy.context

    # VR 모드 감지
    is_vr_active = (
        hasattr(context.window_manager, 'xr_session_state') and
        context.window_manager.xr_session_state is not None
    )

    # VR에서는 area/region 검사 건너뛰기
    if not is_vr_active:
        if context.area is None or context.area.type != 'VIEW_3D':
            return
```

**결과**: PC 뷰포트에서만 작동. VR 헤드셋에서는 draw_callback 자체가 호출되지 않음.

---

## 2. GLSL Gaussian Splatting 셰이더 (핵심 부분)

### Vertex Shader 요약

```glsl
// Uniforms
uniform mat4 viewProjectionMatrix;  // Combined VP matrix
uniform mat4 viewMatrix;             // View matrix for covariance
uniform vec4 camPosAndFocalX;        // Camera position + focal
uniform vec4 viewportAndFocalY;      // Viewport size + focal
uniform int gaussianCount;
uniform sampler2D gaussianData;      // Gaussian 데이터 텍스처

// 3D Covariance 계산
mat3 computeCov3D(vec3 scale, vec4 rot) {
    mat3 R = quatToMat(rot);
    mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    mat3 RS = R * S;
    return RS * transpose(RS);
}

// 2D Covariance 투영 (Jacobian 사용)
vec3 computeCov2D(vec3 mean, mat3 cov3D, float focalX, float focalY) {
    mat3 J = mat3(
        focalX / z, 0.0, 0.0,
        0.0, focalY / z, 0.0,
        -focalX * mean.x / z2, -focalY * mean.y / z2, 0.0
    );
    return J * cov3D * transpose(J);
}
```

### Fragment Shader 요약

```glsl
void main() {
    // 2D Gaussian 평가: exp(-0.5 * x^T * conic * x)
    float power = -0.5 * (
        vConic.x * vCoordXY.x * vCoordXY.x +
        vConic.z * vCoordXY.y * vCoordXY.y +
        2.0 * vConic.y * vCoordXY.x * vCoordXY.y
    );

    float alpha = vColor.a * exp(power);
    fragColor = vec4(vColor.rgb * alpha, alpha);
}
```

---

## 3. VR 컨트롤러 추적 (작동 확인됨)

```python
def get_controller_tip(context, hand_index=1):
    """VR 컨트롤러 끝 위치 반환 - 이 부분은 정상 작동"""
    xr = context.window_manager.xr_session_state

    # 위치 및 회전 가져오기 (tuple → Vector/Quaternion 변환)
    grip_pos = Vector(xr.controller_grip_location_get(context, hand_index))
    aim_rot = Quaternion(xr.controller_aim_rotation_get(context, hand_index))

    # 컨트롤러 팁 오프셋 적용
    TIP_OFFSET = 0.08  # 8cm
    tip_pos = grip_pos + aim_rot @ Vector((0, 0, -TIP_OFFSET))

    return tip_pos, aim_rot
```

---

## 4. Mesh 기반 대체 (VR에서 작동 확인)

```python
# 실제 3D Mesh 오브젝트로 Gaussian 표현 - VR에서 보임
def add_gaussian(self, position, rotation, scale, color, opacity):
    # Icosphere 메시 생성
    obj = bpy.data.objects.new("VR_Gauss", template_mesh)

    # Transform 설정
    obj.location = position
    obj.rotation_quaternion = rotation
    obj.scale = scale

    # Scene에 추가
    collection.objects.link(obj)
```

**결과**: VR에서 보임 ✅ (하지만 진짜 Gaussian Splatting이 아님)

---

## 5. 핵심 문제 요약

```
[Blender VR Rendering Architecture]

PC Mode:
  SpaceView3D → draw_handler_add() → GLSL Shader ✅

VR Mode:
  OpenXR → GHOST_IXrGraphicsBinding → Offscreen Loop
                                        ↓
                            draw_handler 미포함 ❌
```

### 확인된 사실

1. `draw_handler_add(POST_VIEW)`는 VR offscreen 렌더링에 포함되지 않음
2. `gpu.matrix.get_*` 함수는 VR에서 호출되면 per-eye 행렬을 반환하도록 설계됨
3. 하지만 draw_callback 자체가 VR 렌더링 루프에서 호출되지 않음
4. Blender의 VR 렌더링은 C++ 레벨 (`GHOST_IXrGraphicsBinding`)에서 처리됨
5. Python API로는 이 렌더링 파이프라인에 직접 접근 불가

---

## 6. 필요한 정보

- VR offscreen rendering loop에 custom draw를 주입하는 방법
- OpenXR layer를 통한 커스텀 렌더링 가능성
- Blender 외부에서 OpenXR swapchain에 접근하는 방법
- 다른 3D 소프트웨어에서 유사한 문제를 해결한 사례
