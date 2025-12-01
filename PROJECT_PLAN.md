# 3DGS Painter for Blender - Project Plan

**Based on**: "Painting with 3D Gaussian Splat Brushes" (SIGGRAPH 2025)  
**Status**: Architectural Design & Refactoring Phase  
**Last Updated**: 2025-12-01

---

## 📚 문서 구조 (Documentation Structure)

본 프로젝트는 모듈화된 개발을 위해 다음과 같이 문서를 분리합니다:

### **공통 문서** (이 문서)
- 프로젝트 전체 개요 및 아키텍처 결정 사항
- 렌더링 전략 (Hybrid: GLSL + gsplat)
- 의존성 관리 및 배포 전략
- 전체 로드맵 및 성공 지표

### **모듈별 작업 문서** (별도 파일)
각 Phase/모듈 작업 시 해당 문서를 에이전트에게 제공:

1. **`docs/phase0_feasibility.md`** - Phase 0 실행 가능성 검증
2. **`docs/phase1_core_refactoring.md`** - Core 라이브러리 리팩토링
3. **`docs/phase2_dependency_management.md`** - 의존성 설치 시스템
4. **`docs/phase3_viewport_rendering.md`** - GLSL Viewport 렌더링
5. **`docs/phase4_painting_interaction.md`** - 페인팅 인터랙션
6. **`docs/phase5_advanced_features.md`** - 고급 기능 (gsplat 활용)
7. **`docs/technical_considerations.md`** - 기술적 고려사항 상세

**사용 방법**: 각 Phase 시작 시 "이 문서 + 해당 Phase 문서"를 에이전트에게 제공

---

## 1. 개요 (Overview)

본 프로젝트는 기존 웹 기반의 2D 프로토타입(`npr-gaussian-2d-prototype`)을 **블렌더(Blender) 네이티브 애드온**으로 전환하여 확장하는 것을 목표로 합니다.

기존의 **Server-Client (FastAPI + WebSocket)** 아키텍처를 폐기하고, **블렌더 프로세스 내장형(Embedded)** 구조로 전환합니다. 이는 데이터 전송 오버헤드를 제거하고, 블렌더의 강력한 3D 뷰포트, 레이어 시스템, 렌더링 파이프라인(Occlusion, Depth 등)을 활용하기 위함입니다.

### 1.1 기존 아키텍처 분석

**현재 웹 프로토타입의 특징**:
- **3가지 렌더러**: CPU (NumPy), GPU (PyTorch), CUDA (gsplat) - Factory Pattern으로 자동 선택
- **배치 처리 최적화**: 40-80× 성능 향상 (Vectorized operations, GPU batch processing)
- **복잡한 WebSocket 통신**: 18+ 메시지 타입, 실시간 렌더 업데이트 (20 FPS throttling)
- **Deformation System**: CPU + GPU 버전, Spline 기반 커브 변형
- **Inpainting**: Opacity 기반 블렌딩, Anisotropic distance metrics
- **총 의존성 용량**: ~4-5GB (PyTorch + CUDA), 10GB+ (Diffusion models 포함 시)

**제거할 요소**:
- FastAPI + WebSocket 서버-클라이언트 통신 레이어
- 비동기 I/O (async/await) → 동기식 Modal Operator로 전환
- WebSocket 세션 관리 → Blender PropertyGroup으로 전환

**보존할 요소**:
- 배치 처리 로직 (vectorized numpy/torch operations)
- GPU 가속 Deformation
- Inpainting 알고리즘
- 브러시 관리 시스템

---

## 2. 아키텍처 결정 사항

### 2.1 완전 임베디드 방식 채택

**검토된 대안들**:
1. **완전 임베디드** (채택 ✓): 모든 것을 블렌더 프로세스 내에서 실행
2. **하이브리드**: FastAPI 서버를 localhost에서 함께 실행
3. **단계적 접근**: MVP는 하이브리드, 이후 임베디드로 전환

**선택 근거**:
- 단일 프로세스로 배포 및 사용자 경험 단순화
- 블렌더 API와의 긴밀한 통합 (Undo, Preferences, Scene 관리)
- 프로세스 간 통신 오버헤드 제거
- KIRI Innovation 애드온의 성공 사례 (완전 임베디드로 60 FPS 달성)

**트레이드오프**:
- 리팩토링 복잡도 증가 (async → sync 변환)
- Modal Operator blocking 위험 (→ Incremental processing으로 해결)

---

### 2.2 렌더링 방식: Hybrid (GLSL + gsplat)

**검토된 방식들**:
1. **GLSL Only**: Viewport 고성능, 하지만 computation과 분리됨
2. **gsplat Only**: 단일 파이프라인, 하지만 viewport 성능 불확실
3. **Hybrid (GLSL + gsplat)**: ✓ **최종 채택**

**Hybrid 방식 상세**:
```
┌─────────────────────────────────────┐
│   GLSL Viewport (Real-time)        │  ← 60 FPS, 검증된 성능
│   - Instanced rendering            │
│   - Native depth integration       │
│   - KIRI Innovation 방식 참고      │
└─────────────────────────────────────┘
              ↕ (데이터 동기화)
┌─────────────────────────────────────┐
│   gsplat Computation (Heavy ops)   │  ← Differentiable
│   - Deformation calculation        │
│   - Inpainting optimization        │
│   - Final render (optional)        │
└─────────────────────────────────────┘
```

**선택 근거**:
- **성능**: GLSL viewport 60 FPS 검증됨 (KIRI)
- **유연성**: gsplat으로 복잡한 연산 처리
- **Risk 최소화**: 각 영역에서 검증된 기술 사용
- **데이터 중복**: 7MB 수준으로 무시 가능
- **구현 복잡도**: 각 모듈 독립적으로 개발 가능

**기술 스택**:
- **Viewport**: GLSL Shaders (vert/frag), Instanced rendering, 3D Texture
- **Computation**: PyTorch tensors, gsplat CUDA kernels
- **데이터 동기화**: NumPy arrays (중간 형식)

---

### 2.3 의존성 배포: Dream Textures 방식

**검토된 방식들**:
1. **Target Directory 설치** (Dream Textures) ✓ **채택**
2. **Pre-compiled Bundle**: 3GB+ 다운로드, 라이선스 문제
3. **Conda 환경**: 복잡도 증가, Blender Python과 충돌 위험

**구현 계획**:
```
requirements/
├── win-cuda.txt    # PyTorch 2.3.1 + CUDA 11.8
├── win-cpu.txt     # PyTorch 2.3.1 CPU
├── mac-mps.txt     # Apple Silicon
└── linux-cuda.txt  # Linux CUDA

.python_dependencies/  # pip install --target
```

**예상 용량**: CUDA 3GB, CPU 200MB

---

## 3. 핵심 아키텍처: "Embedded Core with Data Sync"

서버를 두지 않고, 무거운 연산(CUDA) 라이브러리를 블렌더의 Python 환경에 직접 로드하여 사용합니다.

### 3.1 구조적 분리
| 모듈 | 역할 (Role) | 주요 기술 스택 |
| :--- | :--- | :--- |
| **`npr_core`** | **연산 엔진 (Logic)**<br>- 프레임워크 독립적인 순수 로직<br>- 브러시 생성, 스트로크 변형(Deformation), 최적화(Optimization)<br>- Inpainting용 Diffusion Model 구동 | Python, PyTorch, CUDA<br>NumPy, gsplat, Diffusers |
| **`blender_addon`** | **UI 및 시각화 (Presentation)**<br>- 사용자 입력(마우스/타블렛) 처리<br>- 3D 뷰포트 가우시안 렌더링<br>- 블렌더 데이터 블록(Mesh, Image) 관리 | Blender Python API (`bpy`)<br>Blender GPU Module (`gpu`) |

### 3.2 렌더링 전략 (Hybrid: GLSL + gsplat)

**아키텍처 개요**:
본 프로젝트는 **Hybrid 렌더링 방식**을 채택하여, 각 영역에 최적화된 기술을 사용합니다.

**역할 분담**:
| 영역 | 기술 | 목적 | 성능 목표 |
|------|------|------|----------|
| **Viewport Rendering** | GLSL Shaders | Real-time visualization | 60 FPS @ 10k gaussians |
| **Computation** | gsplat (PyTorch/CUDA) | Deformation, Optimization | < 1초 @ 100 stamps |
| **Final Render** | gsplat (optional) | High-quality output | F12 key support |

**데이터 흐름**:
```python
# Painting stroke
1. User input → npr_core.brush.generate_stamp()
2. NumPy arrays (shared format)
3. ├─→ GLSL: Upload to GPU texture (viewport)
   └─→ gsplat: Keep in PyTorch tensor (computation)

# Deformation (heavy operation)
1. PyTorch tensor → gsplat.deform()
2. Result → NumPy array
3. → GLSL texture update (viewport sync)
```

#### 3.2.1 Viewport Renderer (Real-time Visualization)

**구현 방식**: Custom Draw Handler + Instanced GLSL Rendering

**기술 참고**: KIRI Innovation/3dgs-render-blender-addon (Apache 2.0)  
**성능 검증**: 10k gaussians @ 60 FPS (실측치)

**데이터 구조** (KIRI 방식):
```python
# Gaussian 데이터를 GPU 3D Texture로 저장
# Stride: 59 floats per Gaussian
# Layout:
# [0-2]:   position (vec3)
# [3-6]:   rotation quaternion (vec4)
# [7-9]:   scale (vec3)
# [10]:    opacity (float)
# [11-58]: spherical harmonics coefficients (16 bands × 3 = 48 floats)
```

**렌더링 파이프라인**:
1. **Vertex Shader** (`gaussian_vert.glsl`):
   - `texelFetch()`로 3D 텍스처에서 Gaussian 데이터 로드
   - 3D Covariance 계산: `Quaternion → Rotation Matrix → Σ = R·S·S^T·R^T`
   - View space 변환 후 2D Covariance 투영
   - Billboard Quad 생성 (3-sigma 범위, instanced rendering)
   - Spherical Harmonics 평가 (view-dependent color, 간소화 가능)

2. **Fragment Shader** (`gaussian_frag.glsl`):
   - Blender depth buffer 샘플링: `texture(blender_depth, screen_coord)`
   - Depth test: `if (v_depth > sampled_depth) discard;`
   - Gaussian splat 평가: `opacity = alpha * exp(-0.5 * r^T * Σ^-1 * r)`
   - Alpha blending

**Depth Integration**:
- Blender의 depth buffer를 texture로 전달
- Fragment shader에서 깊이 비교하여 다른 3D 객체와 자연스러운 occlusion 처리
- `gpu.state.depth_test_set('LESS_EQUAL')` 사용

**성능 목표**:
- 10,000 Gaussians @ 60 FPS (KIRI 실측치 기준)
- View frustum culling 적용 시 100,000+ Gaussians 처리 가능

#### 3.2.2 Internal Rasterizer (for Computation)

**목적**: 무거운 연산 처리 (Viewport와 독립)  
**구현**: `gsplat` / `npr_core`  
**특징**: 
- Off-screen rendering (화면 출력 없음)
- PyTorch tensor 기반 연산
- Differentiable (gradient 계산 가능)

**사용 사례**:
```python
# Deformation (Phase 4)
deformed_gaussians = deformation_gpu.apply(
    gaussians_tensor,  # PyTorch tensor
    spline_params
)

# Inpainting Optimization (Phase 5)
optimized_gaussians = inpainting.optimize(
    gaussians_tensor,
    target_image,
    iterations=100
)

# Final Render (Phase 5, Optional)
render_image = gsplat.render(
    gaussians_tensor,
    camera_params,
    render_mode="RGB+D",  # High quality
    resolution=(1920, 1080)
)
```

**데이터 동기화**:
```python
# Computation 완료 후 Viewport 업데이트
result_numpy = optimized_gaussians.cpu().numpy()
viewport_renderer.update_texture(result_numpy)
```

#### 3.2.3 Final Render Engine (Optional, Phase 5+)

**구현**: `bpy.types.RenderEngine` 상속
```python
class NPRGaussianRenderEngine(bpy.types.RenderEngine):
    bl_idname = "NPR_GAUSSIAN"
    bl_label = "NPR Gaussian Painter"
    
    def render(self, depsgraph):
        # F12 렌더링: gsplat으로 고품질 이미지 생성
        pass
    
    def view_draw(self, context, depsgraph):
        # Viewport rendering (Optional, 현재는 Draw Handler 사용)
        pass
```

**장점**:
- F12 키로 최종 렌더 출력
- 애니메이션 렌더 지원
- Blender의 렌더 설정(해상도, 샘플링 등) 자동 통합

---

---

## 4. 디렉토리 구조 (Proposed Structure)

```
project_root/
├── npr_core/                      # [Library] Core Logic (No bpy dependency)
│   ├── __init__.py
│   ├── data.py                    # Gaussian Data Structure (Numpy/Torch)
│   ├── brush.py                   # Brush generation logic
│   ├── brush_manager.py           # Brush library management
│   ├── deformation.py             # Spline-based deformation (CPU)
│   ├── deformation_gpu.py         # GPU-accelerated deformation (CUDA)
│   ├── optimization.py            # Optimization loop using gsplat
│   ├── inpainting.py              # Opacity-based blending for overlaps
│   ├── renderer.py                # gsplat wrapper (tensor operations)
│   └── scene_data.py              # High-performance SceneData class
│
├── blender_addon/                 # [Addon] Blender Integration
│   ├── __init__.py                # Addon registration
│   ├── install_deps.py            # Dependency installation (Dream Textures 방식)
│   ├── operators.py               # User input handling (Modal Operators)
│   ├── panels.py                  # Sidebar UI
│   ├── preferences.py             # Addon preferences with install UI
│   ├── gaussian_data.py           # Texture-based GPU data management
│   ├── viewport_renderer.py       # KIRI-style GLSL rendering
│   ├── render_engine.py           # Optional: bpy.types.RenderEngine
│   ├── properties.py              # Blender PropertyGroup (session state)
│   └── shaders/                   # GLSL Shaders
│       ├── gaussian_vert.glsl     # Vertex shader (KIRI 방식 기반)
│       ├── gaussian_frag.glsl     # Fragment shader with depth test
│       └── composite.glsl         # Post-processing (optional)
│
├── requirements/                  # Platform-specific dependencies
│   ├── win-cuda.txt               # Windows + NVIDIA CUDA
│   ├── win-cpu.txt                # Windows CPU-only
│   ├── mac-mps.txt                # macOS Apple Silicon
│   └── linux-cuda.txt             # Linux + CUDA
│
└── .python_dependencies/          # Created during installation
    └── (PyTorch, NumPy, etc.)
```

---

---

## 5. 개발 로드맵 (Development Roadmap)

**전체 예상 기간**: 12주

### Phase 0: 실행 가능성 검증 (Feasibility Study) - 1주

**목표**: Hybrid 아키텍처(GLSL + gsplat)의 기술적 검증

**📄 상세 문서**: `docs/phase0_feasibility.md`

**핵심 검증 항목**:
- Blender Python 환경에서 PyTorch + CUDA 동작 확인
- GLSL viewport prototype (100 gaussians @ 30+ FPS)
- gsplat computation 동작 확인
- Hybrid 데이터 동기화 latency 측정 (< 5ms 목표)

**Decision Point**: 모든 검증 통과 시 Phase 1 진행

---

### Phase 1: 코어 라이브러리 리팩토링 (Core Refactoring) - 2주

**목표**: 웹 프로토타입 → Blender 임베디드 라이브러리 변환

**📄 상세 문서**: `docs/phase1_core_refactoring.md`

**핵심 작업**:
- WebSocket/FastAPI 제거, 비동기 → 동기 변환
- `backend/core/*` → `npr_core/*` 이동
- GPU 컨텍스트 관리 (BlenderGPUContext 구현)
- npr_core 독립성 확보 (bpy 의존성 제거)
- 단위 테스트 (Blender 없이 실행 가능)

---

### Phase 2: 의존성 설치 시스템 구축 (Dependency Management) - 1주

**목표**: Dream Textures 방식 pip 설치 시스템 구현

**📄 상세 문서**: `docs/phase2_dependency_management.md`

**핵심 작업**:
- Platform-specific requirements 파일 (Windows/macOS/Linux)
- Preferences UI에서 원클릭 설치
- Progress feedback + Error handling
- CUDA detection 및 fallback
- 예상 용량: CUDA 3GB, CPU 200MB

---

### Phase 3: 뷰포트 렌더링 구현 (Viewport Rendering) - 2주

**목표**: GLSL Instanced Rendering (Hybrid의 Viewport 부분)

**📄 상세 문서**: `docs/phase3_viewport_rendering.md`

**핵심 작업**:
- GLSL Shaders (vertex + fragment, 완전한 코드 포함)
- 59-float stride texture 관리 (GaussianDataManager)
- Draw handler 등록 (viewport integration)
- Blender depth buffer 통합 (occlusion)
- 성능 목표: 10k gaussians @ 60 FPS

---

### Phase 4: 인터랙션 구현 (Painting Interaction) - 3주

**목표**: Real-time painting + Hybrid 데이터 동기화

**📄 상세 문서**: `docs/phase4_painting_interaction.md`

**핵심 작업**:
- Raycasting (마우스 → 3D 위치)
- Modal Operator (painting mode)
- Incremental deformation (gsplat computation)
- Hybrid 데이터 흐름 (NumPy ↔ PyTorch ↔ GLSL)
- Brush system + Undo/Redo
- 성능 목표: 연속 스트로크 20+ FPS

---

### Phase 5: 고급 기능 및 최적화 (Advanced Features) - 2주

**목표**: gsplat 기반 최적화 기능

**📄 상세 문서**: `docs/phase5_advanced_features.md`

**핵심 작업**:
- Inpainting optimization (gsplat differentiable rendering)
- Viewport real-time preview (optimization 진행 상황)
- VRAM 관리 (OOM handling)
- 데이터 영속성 (.blend 파일 저장)
- Export (PLY, Image, Video)
- Final Render Engine (F12 support, optional)

---

---

## 6. 주요 기술적 고려사항 (Technical Considerations)

**📄 상세 문서**: `docs/technical_considerations.md`

이 섹션은 모든 Phase에서 공통적으로 고려해야 할 횡단적(cross-cutting) 기술 이슈를 다룹니다. 상세한 구현 전략과 코드 예제는 별도 문서를 참조하세요.

### 6.1 GPU Context & Compatibility
- OpenGL(Blender) + CUDA(PyTorch) 동시 사용 전략
- 권장 환경: NVIDIA GTX 1060+, VRAM 4GB+, CUDA 11.8+

### 6.2 VRAM Management
- Hybrid 방식 VRAM 예산: Viewport 2.5-4GB, Computation 3.5-6.5GB
- 데이터 중복: 7MB (negligible)
- Lazy loading, 동적 Gaussian 수 조절, OOM 처리

### 6.3 Performance Optimization
- 목표: 10k gaussians @ 60 FPS (viewport)
- 주요 최적화: Partial texture update, Vectorized operations, Frustum culling, Spatial hashing

### 6.4 Modal Operator Blocking
- Incremental processing (점진적 처리)
- Progress indication + Cancel 기능
- Phase 5+ Background processing 검토

### 6.5 의존성 배포
- Dream Textures 방식: Target directory 설치
- 예상 용량: CUDA 3GB, CPU 200MB
- 플랫폼별 requirements 자동 선택

### 6.6 Undo/Redo 통합
- Operator 기반 undo 시스템
- 각 스트로크 = 1 undo step
- `.blend` 파일 영속성

---

## 7. 리스크 및 완화 전략 (Risk Mitigation)

### 7.1 High-Risk: Modal Operator Blocking

**리스크**: Deformation/Optimization 시 UI 정지 (3초+)

**완화책**:
- Incremental processing 구현 (Phase 4)
- Progress indication + Cancel 기능
- 최악의 경우: "Processing..." 모달 다이얼로그 표시

**검증**: Phase 3에서 성능 측정, 목표 미달 시 조기 대응

---

### 7.2 High-Risk: 의존성 설치 실패

**리스크**: 사용자 환경에서 PyTorch 설치 실패 (40% 예상)

**완화책**:
- Phase 2에서 상세한 에러 처리 구현
- 플랫폼별 설치 가이드 문서화
- CPU 전용 fallback 제공
- Discord/GitHub Issues로 지원 채널 운영

---

### 7.3 Medium-Risk: Shader 호환성

**리스크**: AMD/Intel GPU에서 GLSL shader 동작 안 함

**완화책**:
- Phase 3에서 다양한 GPU 테스트 (NVIDIA/AMD/Intel)
- Fallback rendering (단순 point sprites)
- 사용자 리포트 수집 후 hotfix

---

### 7.4 Medium-Risk: VRAM 부족

**리스크**: 사용자 GPU에서 OOM 크래시

**완화책**:
- Phase 0에서 VRAM 체크 기능 구현
- 동적 Gaussian 수 제한
- 경고 메시지: "Large scenes may require 8GB+ VRAM"

---

### 7.5 Low-Risk: 성능 목표 미달

**리스크**: 10k Gaussians @ 30 FPS 미달성

**완화책**:
- Phase 0에서 조기 검증
- KIRI 방식 입증됨 (60 FPS 실측)
- 최악의 경우: Gaussian 수 제한 (5k) 또는 Geometry Nodes 방식 전환

---

## 8. 참고 자료 및 영감 (References)

### 8.1 기존 블렌더 3DGS 애드온

1. **KIRI Innovation/3dgs-render-blender-addon** ✓ 주요 참고
   - GitHub: https://github.com/Kiri-Innovation/3dgs-render-blender-addon
   - 특징: Instanced rendering, 60 FPS, Blender 4.3+
   - 코드: GLSL shaders (vert.glsl, frag.glsl), depth integration

2. **ReshotAI/gaussian-splatting-blender-addon**
   - GitHub: https://github.com/reshotai/gaussian-splatting-blender-addon
   - 특징: Geometry Nodes 방식, Cycles/EEVEE 호환
   - 성능: 10k @30 FPS

### 8.2 의존성 배포 참고

1. **Dream Textures** (Stable Diffusion for Blender)
   - GitHub: https://github.com/carson-katri/dream-textures
   - 학습 내용: Target directory 설치, 플랫폼별 requirements, Windows DLL 처리

### 8.3 기술 문서

1. **Blender Python API**:
   - `bpy.types.RenderEngine`: Custom render engine
   - `bpy.types.SpaceView3D.draw_handler_add()`: Custom viewport drawing
   - `gpu` module: GLSL shader, texture management

2. **gsplat Library**:
   - GitHub: https://github.com/nerfstudio-project/gsplat
   - 2D Gaussian Splatting rasterization

3. **3D Gaussian Splatting Paper**:
   - "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
   - "Painting with 3D Gaussian Splat Brushes" (SIGGRAPH 2025) ✓ 본 프로젝트 기반

---

## 9. 다음 단계 (Next Steps)

### 즉시 실행 가능한 작업

**Phase 0 시작** (우선순위: 최고):
- [ ] `docs/phase0_feasibility.md` 참조하여 검증 테스트 실행
- [ ] Blender 3.6+ 환경 구축
- [ ] PyTorch + CUDA 설치 테스트
- [ ] 간단한 GLSL prototype (100 gaussians)

**기존 코드 분석**:
- [ ] `npr-gaussian-2d-prototype/backend/core/` 리뷰
- [ ] WebSocket 의존성 목록화
- [ ] 배치 처리 로직 이해

**KIRI 애드온 연구**:
- [ ] GitHub 클론 및 GLSL shader 분석
- [ ] 59-float texture layout 파악

### 의사결정 필요 사항

1. **Spherical Harmonics**: SH degree 0-1 권장 (성능/품질 균형)
2. **MVP 범위**: Phase 0-4 (Deformation 포함)
3. **배포**: GitHub Releases (무료) → 이후 Blender Market 검토

---

## 10. 프로젝트 메트릭 (Success Metrics)

### 기술적 성공 기준

- [ ] 10,000 Gaussians @ 30+ FPS (뷰포트)
- [ ] 의존성 설치 성공률 > 80%
- [ ] VRAM 사용량 < 4GB (10k Gaussians)
- [ ] 100+ stamps 스트로크 처리 < 1초

### 사용자 경험 기준

- [ ] 설치 시간 < 15분 (평균)
- [ ] 첫 페인팅까지 < 5분 (튜토리얼 포함)
- [ ] 크래시율 < 5% (100 세션 기준)

### 프로젝트 완성 기준

- [ ] Phase 0-4 완료 (MVP)
- [ ] 3+ 플랫폼 테스트 (Windows/Mac/Linux)
- [ ] 문서화 완료 (설치, 사용법, 트러블슈팅)
- [ ] 10+ 베타 테스터 피드백 수집

