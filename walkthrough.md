# VR Gaussian Painting - Development Summary

> **목적**: 외부 에이전트 리뷰 및 재설계를 위한 개발 현황 정리  
> **날짜**: 2025-12-07

---

## 1. 프로젝트 개요

### 1.1 목표

**Quest 3 VR 컨트롤러로 Blender에서 Gaussian Splatting 페인팅**

- VR 헤드셋을 쓰고 3D 공간에서 직접 Gaussian 페인팅
- 컨트롤러 버튼(B)으로 페인팅 트리거
- 레이저 포인터로 조준점 시각화 (VR 헤드셋 내에서)
- 실시간 Gaussian 생성 및 렌더링

### 1.2 기존 프로젝트 기반

**3DGS Painter Blender Addon** - 마우스/태블릿으로 Gaussian 페인팅하는 기존 애드온에 VR 확장

---

## 2. 구현된 항목

### 2.1 파일 구조

```
src/vr/
├── __init__.py          # VR 모듈 초기화
├── vr_session.py        # VR 세션 시작/종료 관리
├── vr_input.py          # 컨트롤러 위치/회전 추적
├── vr_operators.py      # VR 페인팅 오퍼레이터
├── vr_panels.py         # UI 패널
├── action_maps.py       # OpenXR 액션 바인딩 시도
└── vr_ray_renderer.py   # 레이저 시각화 (PC 뷰포트 전용)
```

### 2.2 작동하는 기능 ✅

| 기능                | 파일                 | API                                        |
| ------------------- | -------------------- | ------------------------------------------ |
| VR 세션 시작/종료   | `vr_session.py`      | `bpy.ops.wm.xr_session_toggle()`           |
| 컨트롤러 위치 추적  | `vr_input.py`        | `controller_grip_location_get(ctx, index)` |
| 컨트롤러 회전 추적  | `vr_input.py`        | `controller_aim_rotation_get(ctx, index)`  |
| PC 화면 레이저 표시 | `vr_ray_renderer.py` | `SpaceView3D.draw_handler`                 |
| OpenXR 연결         | Blender 내장         | Oculus Runtime 연결 확인                   |

### 2.3 확인된 데이터 형식

```python
# 컨트롤러 위치 (작동)
xr.controller_grip_location_get(bpy.context, 1)  # 1=오른손
# → Vector (-59.7862, 50.9772, -0.2976)

# 컨트롤러 회전 (작동)
xr.controller_aim_rotation_get(bpy.context, 1)
# → Quaternion (0.63, 0.53, 0.42, -0.37)

# 페인트 액션 등록 (등록됨)
am.actionmap_items.get("threegds_paint")  # → True
```

---

## 3. 발견된 기술적 한계

### 3.1 🔴 VR 헤드셋에서 커스텀 렌더링 불가

**문제**: `bpy.gpu` 드로잉이 VR 헤드셋에서 **보이지 않음**

- `SpaceView3D.draw_handler`는 PC 뷰포트 전용
- VR 헤드셋은 별도 렌더링 파이프라인 사용
- Python에서 VR 렌더링 파이프라인에 접근할 방법 없음

**영향**: 레이저 포인터, 브러시 미리보기 등 VR에서 표시 불가

### 3.2 🔴 VR 버튼 입력 등록의 어려움

**시도한 방법들**:

1. 런타임에 `actionmaps`에 액션 추가 → 등록되나 작동 안 함
2. `action_state_get()`으로 버튼 상태 조회 → 값 안 옴
3. `defaults.py` 수정 (Blender 시스템 파일) → 시도 중

**원인 추정**:

- OpenXR 액션은 세션 시작 전에 등록되어야 함
- 세션 시작 후에는 불변(immutable)
- Blender VR Scene Inspection이 먼저 액션 등록

### 3.3 🟡 Blender VR Scene Inspection 설계 의도

**확인된 사실**: VR Scene Inspection은 **뷰어 전용**

- 공식 문서: "sculpting, painting, drawing은 지원 안 함"
- 탐색(텔레포트, 그랩), 시점 확인 목적

---

## 4. 시도했지만 실패한 접근법

| 접근법                          | 결과           | 실패 원인               |
| ------------------------------- | -------------- | ----------------------- |
| 런타임 ActionMap 추가           | 등록됨, 작동 X | 세션 시작 후 등록       |
| `defaults.py` 패치 함수         | 패치됨, 순서 X | VR 애드온이 먼저 로드   |
| `action_state_get()`            | (0, 0) 반환    | POSE 액션은 다르게 동작 |
| `controller_aim_location_get()` | (0,0,0) 고정   | API 버그 또는 설정 문제 |

---

## 5. 발견된 가능한 해결 방향

### 5.1 3D Mesh 오버레이 (유력)

레이저를 Python GPU가 아닌 실제 3D 오브젝트(Cylinder)로 생성

```python
# 예시 개념
bpy.ops.mesh.primitive_cylinder_add()
cylinder.location = controller_pos
cylinder.rotation_euler = controller_dir.to_track_quat('Z', 'Y').to_euler()
```

### 5.2 Blender 시스템 파일 직접 수정

`viewport_vr_preview/defaults.py`에 페인트 액션 직접 추가

- 사용자가 관리자 권한으로 수동 수정
- 또는 설치 스크립트로 자동화

### 5.3 외부 VR 페인팅 솔루션 연구

- **FreebirdXR**: Blender VR 모델링 플러그인
- **Shapelab**: VR 스컬핑 앱 (Blender 연동)
- 이들의 접근 방식 분석

---

## 6. 현재 코드 상태

### 6.1 테스트 명령어

```python
# VR 시작
bpy.ops.threegds.start_vr_session()

# 레이 트래킹 (PC 화면에서만 보임)
bpy.ops.threegds.vr_ray_track('INVOKE_DEFAULT')

# 컨트롤러 위치 확인
xr = bpy.context.window_manager.xr_session_state
print(xr.controller_grip_location_get(bpy.context, 1))
```

### 6.2 콘솔 로그 예시

```
[3DGS Painter VR] VR module registered
Connected to OpenXR runtime: Oculus (Version 1.113.0)
[3DGS VR] Paint action added to actionmap (B button)
[VR Ray] Ray renderer registered
```

---

## 7. 핵심 질문 (외부 리뷰용)

1. **Blender VR 헤드셋 렌더링에 Python으로 접근하는 방법이 있는가?**
2. **OpenXR 액션을 런타임에 동적 등록하는 방법은?**
3. **VR Scene Inspection 대신 사용할 수 있는 Blender VR 프레임워크가 있는가?**
4. **3D Mesh 오버레이 방식이 실시간 VR에서 성능 문제 없이 작동할 수 있는가?**

---

## 8. 파일 위치

| 유형               | 경로                                                        |
| ------------------ | ----------------------------------------------------------- |
| 프로젝트           | `c:\Users\LEE\Documents\GitHub\3dgs-painter-blender-addon\` |
| VR 모듈            | `src/vr/`                                                   |
| 수정된 defaults.py | `defaults.py` (프로젝트 루트에 복사본)                      |
