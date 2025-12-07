# VR Gaussian Splatting 렌더링 문제 - 연구 조사 요청서

> **작성일**: 2025-12-07  
> **목적**: 해결 방안 탐색을 위한 논문/기술문서 조사

---

## 1. 문제 요약 (Problem Statement)

### 핵심 문제

**Blender에서 Python `draw_handler`를 통해 그린 GLSL 커스텀 셰이더가 VR 헤드셋(Quest 3)에서 보이지 않음.**

- PC 뷰포트: ✅ 정상 렌더링
- VR 헤드셋: ❌ 커스텀 GLSL 렌더링 **안보임**

### 목표

Quest 3 VR 환경에서 **3D Gaussian Splatting**을 실시간으로 렌더링하며 페인팅하고 싶음.

---

## 2. 기술 스택 및 환경 (Technical Context)

| 항목            | 값                                            |
| --------------- | --------------------------------------------- |
| **소프트웨어**  | Blender 5.0                                   |
| **VR 헤드셋**   | Meta Quest 3 (Oculus Link)                    |
| **VR API**      | OpenXR                                        |
| **렌더링 기술** | 3D Gaussian Splatting (SIGGRAPH 2023)         |
| **구현 언어**   | Python (Blender Addon), GLSL                  |
| **GPU 모듈**    | Blender `bpy.gpu`, `gpu.shader`, `gpu.matrix` |

---

## 3. 현재 아키텍처 (Current Architecture)

```
┌─────────────────────────────────────────────────┐
│  Blender Main Process                           │
│  ├── SpaceView3D.draw_handler_add()             │
│  │      ├── POST_VIEW callback                  │
│  │      └── Custom GLSL Shader (Gaussian)       │
│  │                                              │
│  └── VR Scene Inspection (OpenXR)               │
│         └── Offscreen Rendering Loop ← 문제!    │
│              (draw_handler 미포함)              │
└─────────────────────────────────────────────────┘
```

### 문제의 원인

Blender의 VR Scene Inspection은 **offscreen draw loop**을 사용하며, Python `draw_handler_add()`로 등록된 콜백은 이 렌더링 경로에 **포함되지 않음**.

---

## 4. 시도한 접근법 (Attempted Solutions)

### 4.1 `gpu.matrix` 사용으로 VR 호환 시도

```python
# 시도한 수정
view_matrix = gpu.matrix.get_model_view_matrix()
projection_matrix = gpu.matrix.get_projection_matrix()
```

**결과**: PC에서만 작동, VR에서는 여전히 안보임

### 4.2 VR Context 검사 우회

```python
# VR 모드일 때 area/region 검사 건너뛰기
is_vr_active = wm.xr_session_state is not None
if not is_vr_active:
    if context.area is None: return
```

**결과**: 효과 없음

### 4.3 3D Mesh 오브젝트로 대체

- Gaussian을 Icosphere 메시로 변환하여 Scene에 추가
- **결과**: VR에서 보임 ✅ (하지만 진짜 Gaussian Splatting 아님)

---

## 5. 검색할 주제 키워드 (Research Keywords)

### 영어

- "3D Gaussian Splatting VR rendering"
- "OpenXR custom shader rendering"
- "Blender VR draw handler limitation"
- "VRSplat stereo rendering"
- "Gaussian Splatting real-time VR headset"
- "GLSL shader OpenXR integration"
- "Python VR rendering pipeline"
- "Foveated Gaussian Splatting"

### 논문/프로젝트 참고

- VRSplat (arXiv 2025) - Foveated 3DGS for VR
- VR-GS (SIGGRAPH 2024) - Interactive GS in VR
- GaussianShopVR - VR editing of 3DGS
- "Painting with 3D Gaussian Splat Brushes" (SIGGRAPH 2025)

---

## 6. 탐색하고 싶은 해결 방향 (Research Questions)

### 방향 1: Blender 내부 해결

- Blender VR 렌더링 파이프라인에 커스텀 셰이더를 주입하는 방법이 있는가?
- VR Scene Inspection 애드온을 수정하여 draw_handler를 포함시킬 수 있는가?
- Blender C++ 소스 코드 수준에서 OpenXR 렌더링에 개입하는 방법은?

### 방향 2: 외부 솔루션 연동

- Blender에서 편집하고 외부 VR 뷰어로 실시간 동기화하는 사례가 있는가?
- Unity/Unreal에서 Gaussian Splatting VR 렌더링 구현체 중 참고할 만한 것은?
- SuperSplat, PostShot 등의 VR 모드 기술 구현 방식은?

### 방향 3: 대안적 렌더링 방식

- Geometry Nodes를 활용하여 Gaussian을 Point Cloud로 표현하는 방법은?
- GPUOffScreen 렌더링 결과를 Texture로 변환하여 VR Scene에 표시하는 패턴은?
- EEVEE Render Engine 확장을 통한 커스텀 렌더링 가능성은?

---

## 7. 성공 기준 (Success Criteria)

1. **VR 헤드셋에서 3D Gaussian 보임** (실제 Splatting 또는 근사)
2. **실시간 페인팅 가능** (72+ FPS @ VR)
3. **양안 스테레오 렌더링 지원**
4. **Blender 워크플로우와 통합** (외부 앱 필요 최소화)

---

## 8. 참고 자료 (References)

### 공식 문서

- [Blender VR Scene Inspection](https://docs.blender.org/manual/en/latest/addons/3d_view/vr_scene_inspection.html)
- [Blender GPU Module](https://docs.blender.org/api/current/gpu.html)
- [OpenXR Specification](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html)

### 관련 논문

- "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- "Painting with 3D Gaussian Splat Brushes" (SIGGRAPH 2025)
- "VRSplat: High-Fidelity 3D Gaussian Splatting for VR" (arXiv 2025)

### GitHub 프로젝트

- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- [KIRI-Innovation/3dgs-render-blender-addon](https://github.com/KIRI-Innovation/3dgs-render-blender-addon)

---

## 9. 요청 사항 (Request)

위 문제를 해결할 수 있는 **논문, 기술 문서, 오픈소스 프로젝트, 커뮤니티 토론**을 탐색해주세요.

특히:

1. **Blender VR + Custom Rendering** 관련 선행 사례
2. **3D Gaussian Splatting + VR Headset** 실시간 렌더링 구현체
3. **OpenXR + Custom GLSL Shader** 통합 방법
4. **Python 한계를 우회하는 창의적 접근법**

에 대한 정보가 필요합니다.
