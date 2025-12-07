# Custom VR GLSL Rendering Pipeline 연구 요청서

> **작성일**: 2025-12-07  
> **목적**: Blender에서 커스텀 GLSL 셰이더를 VR 헤드셋에 직접 렌더링하는 방법 조사

---

## 1. 프로젝트 개요

### 프로젝트명

**3DGS Painter for Blender** - SIGGRAPH 2025 논문 "Painting with 3D Gaussian Splat Brushes" 기반 Blender 애드온

### 목표

VR 환경(Quest 3)에서 **3D Gaussian Splatting**을 실시간으로 렌더링하며 페인팅

### 현재 아키텍처

```
┌─────────────────────────────────────────────────┐
│  Blender Main Process                           │
│  ├── GLSL Viewport Renderer (60 FPS) ✅ PC만    │
│  ├── BrushStamp / StrokePainter (브러시 시스템) │
│  ├── Subprocess (PyTorch + gsplat 최적화)       │
│  └── VR Input (컨트롤러 추적) ✅ 작동           │
└─────────────────────────────────────────────────┘
```

---

## 2. 핵심 문제 (이미 확인됨)

### 문제

Python `draw_handler_add()`로 등록한 GLSL 셰이더가 **VR 헤드셋에서 보이지 않음**

### 원인 (확정)

- Blender VR은 **offscreen draw loop** 사용 (C++ 레벨)
- Python draw_handler는 **window framebuffer**에 그림
- VR은 **OpenXR swapchain**에 렌더링
- 두 버퍼는 **완전히 분리된 GPU 메모리**

### 시도한 접근 (실패)

1. `gpu.matrix.get_projection_matrix()` 사용 → VR에서 draw_callback 자체가 호출 안됨
2. VR context 검사 우회 → 효과 없음
3. Custom Overlays 활성화 → 효과 없음

---

## 3. 탐색하고자 하는 방향: Custom VR Rendering Pipeline

### 목표

Python API 한계를 우회하여 **커스텀 GLSL 셰이더를 VR에 직접 렌더링**

### 조사할 접근법

#### A. Blender Custom RenderEngine 확장

```python
class NPRGaussianRenderEngine(bpy.types.RenderEngine):
    def view_draw(self, context, depsgraph):
        # VR 뷰포트에서도 호출되는가?
        pass
```

- `bpy.types.RenderEngine`이 VR 세션에서 어떻게 동작하는지?
- EEVEE를 상속/확장하여 커스텀 pass 추가 가능성?

#### B. OpenXR Composition Layer 주입

- Blender 외부에서 OpenXR swapchain에 직접 접근
- Python ctypes로 DLL 호출하여 그래픽 컨텍스트 공유
- Blender의 `GHOST_IXrGraphicsBinding` 확장 가능성

#### C. Blender VR 소스 수정 (C++ 레벨)

- `source/blender/windowmanager/xr/` 디렉토리 분석
- `wm_xr_draw_view()` 함수에 Python callback 추가
- draw_handler를 VR offscreen loop에 포함시키기

#### D. GPU Texture 공유

- GPUOffScreen으로 렌더링 후 VR Scene에 Texture로 주입
- GPU→CPU 전송 없이 직접 texture handle 공유
- OpenGL interop / Vulkan memory sharing

---

## 4. 기술 환경

| 항목        | 값                                     |
| ----------- | -------------------------------------- |
| Blender     | 5.0                                    |
| VR 헤드셋   | Meta Quest 3 (Oculus Link)             |
| VR API      | OpenXR                                 |
| 그래픽 API  | OpenGL (Blender), DirectX (Quest Link) |
| 현재 렌더러 | Custom GLSL (3D Gaussian Splatting)    |
| Python      | 3.11 (Blender 내장)                    |

---

## 5. 검색 키워드

### 영어

- "Blender custom render engine VR OpenXR"
- "OpenXR composition layer custom rendering"
- "Blender GHOST XR graphics binding extension"
- "OpenGL OpenXR swapchain shared context"
- "Blender VR offscreen draw callback injection"
- "bpy.types.RenderEngine VR stereo rendering"
- "Python ctypes OpenXR graphics binding"
- "Vulkan OpenXR texture sharing zero copy"

### 참고할 프로젝트

- BlenderXR (MARUI-PlugIn)
- Freebird VR (Blender VR 플러그인)
- Dream Textures (Subprocess IPC 패턴)
- VRSplat (CUDA VR 렌더러)

---

## 6. 질문 목록

### 구현 가능성

1. `bpy.types.RenderEngine`의 `view_draw()`가 VR 세션에서 호출되는가?
2. EEVEE를 상속하여 커스텀 rendering pass를 추가할 수 있는가?
3. Python에서 Blender의 OpenGL context를 VR swapchain과 공유할 수 있는가?

### C++ 수정 시

4. Blender VR 렌더링 파이프라인의 진입점은 어디인가?
5. `wm_xr_draw_view()`에서 Python callback을 호출하려면 무엇을 수정해야 하는가?
6. Blender를 포크하지 않고 C++ 확장만으로 가능한가?

### 대안

7. OpenXR Overlay Layer로 별도 렌더링 레이어를 추가할 수 있는가?
8. Vulkan VR 렌더러(NVIDIA vk_gaussian_splatting 등)를 Blender와 통합할 수 있는가?

---

## 7. 성공 기준

1. **커스텀 GLSL 셰이더가 VR 헤드셋에서 보임**
2. **양안 스테레오 렌더링 지원**
3. **72Hz 이상 프레임레이트 유지**
4. **기존 Gaussian 페인팅 시스템과 통합**

---

## 8. 기존 프로젝트 구조 (참고용)

```
3dgs-painter-blender-addon/
├── src/
│   ├── viewport/
│   │   ├── viewport_renderer.py   # GLSL Gaussian 렌더러 (PC 작동)
│   │   └── gaussian_data.py       # GPU 데이터 관리
│   ├── npr_core/
│   │   ├── brush.py               # BrushStamp, StrokePainter
│   │   ├── brush_converter.py     # 이미지→Gaussian 변환
│   │   └── deformation_gpu.py     # GPU 변형
│   ├── vr/
│   │   ├── vr_freehand_paint.py   # VR 페인팅 오퍼레이터
│   │   ├── vr_session.py          # VR 세션 관리
│   │   └── vr_input.py            # 컨트롤러 입력
│   └── generator_process/         # Subprocess (PyTorch)
└── docs/
    ├── PROJECT_PLAN.md            # 전체 아키텍처
    └── phase3_viewport_rendering.md # GLSL 렌더러 설계
```

---

## 9. 요청 사항

위 문제를 해결할 수 있는 **논문, 기술 문서, 오픈소스 프로젝트, Blender 개발자 토론**을 탐색해주세요.

특히:

1. **Blender RenderEngine + VR** 통합 사례
2. **OpenXR custom layer rendering** 구현 방법
3. **GPU context sharing** (OpenGL/Vulkan)
4. **Blender 소스 수정 없이 가능한 방법**

에 대한 정보가 필요합니다.
