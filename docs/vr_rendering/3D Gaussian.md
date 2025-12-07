<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I'm developing a Blender addon for real-time 3D Gaussian Splatting painting in VR (Quest 3).

**Problem**: Python `draw_handler_add()` custom GLSL shaders work on PC viewport but do NOT render in VR headset. Blender VR uses an offscreen rendering loop that excludes Python draw handlers.

**Goal**: Find solutions to render custom GLSL shaders (3D Gaussian Splatting) in VR headset.

**Search for**:

1. Papers/implementations on "3D Gaussian Splatting VR rendering" (VRSplat, VR-GS, etc.)
2. Blender VR custom rendering workarounds or C++ level solutions
3. OpenXR custom shader injection techniques
4. How other software (Unity, Unreal) solved similar problems
5. Alternative approaches: GPUOffScreen → Texture, Geometry Nodes

**Technical context**:

- Blender 5.0, OpenXR, Quest 3 (Oculus Link)
- gpu.matrix.get_projection_matrix() used but draw_callback not invoked in VR
- Mesh-based fallback works in VR but is not true Gaussian Splatting

Please find academic papers, GitHub projects, Blender developer discussions, and technical documents that address this rendering pipeline limitation.

완벽한! 이제 충분히 깊이 있는 정보를 수집했습니다. 종합적인 보고서를 작성하겠습니다.

***

## 📋 **VR Gaussian Splatting Rendering 문제: 종합 기술 조사 보고서**

당신의 **Blender Quest 3 VR 페인팅 애드온**을 위한 GLSL 셰이더 VR 렌더링 문제에 대해 **논문, GitHub 프로젝트, 커뮤니티 토론**을 기반으로 한 종합적인 솔루션들을 제시합니다.

***

## 1️⃣ **핵심 문제의 기술적 근원**

### 문제 아키텍처

```
Blender VR 렌더링 파이프라인
├─ PC 뷰포트: SpaceView3D → draw_handler_add() → GLSL ✅
└─ VR 헤드셋: OpenXR → GHOST_IXrGraphicsBinding 
              → Offscreen Loop (C++ 레벨)
              → draw_handler 미포함 ❌
```

**근본 원인**[^1]

- Blender의 `draw_handler_add()`는 **표준 viewport 렌더링 루프**에만 포함
- VR 렌더링은 **별도의 C++ offscreen loop**에서 실행 (`wm_xr_draw_controllers()` 레벨)
- Python API는 이 offscreen loop에 **직접 접근 불가**

***

## 2️⃣ **VR에서 Gaussian Splatting을 성공한 사례들**

### **A. 학술 및 상용 VR-GS 구현**

#### **1. VR-Splatting (i3D 2025)**[^2][^3]

**저자**: Linus Franke, Marc Stamminger (FAU Erlangen-Nürnberg)

- **아키텍처**: Foveated 렌더링 + 하이브리드 점 표현
- **핵심 기술**:
    - 주변부(periphery): 저밀도 3D Gaussians (부드러운 렌더링)
    - 중심부(fovea): Neural point splatting (세밀한 디테일)
    - Eye-tracking 기반 동적 해상도 조정
- **성능**:
    - **2016×2240 per eye @ 90Hz** (SteamVR native resolution)
    - 사용자 연구: 76% 선호도 (vs 기본 GS)
    - Per-pixel 정렬 불필요 (popping artifact 최소화)

**관련 기술**:

- TRIPS (Trilinear Point Splatting) 참고
- Gaze-tracked foveated rendering
- Edge-aware blending masks

***

#### **2. VR-GS (SIGGRAPH 2024)**[^4]

**저자**: Y Jiang et al.

- **특징**: Physics-aware interactive Gaussian Splatting
- **구현**:
    - 2-level deformation embedding (local + global)
    - XPBD (Extended Position-Based Dynamics)
    - Tetrahedral mesh cage 구조
    - Real-time 물리 시뮬레이션 + rendering 통합
- **성능 지표**:
    - Mesh resolution: 10K–30K vertices (성능↔품질 트레이드오프)
    - Real-time deformation @ 몇십 FPS
    - Collision detection + shadow mapping 내장

***

#### **3. Fov-GS (2025)**[^5]

**특징**: Dynamic scene foveated rendering

- 동적 씬에 특화 (기존 3DGS는 static scene만)
- 3D Gaussian forest representation
- 11.33× speedup (vs SOTA)
- HVS(Human Visual System) 모델 기반 최적화

***

#### **4. GaussianShopVR (UIST 2025)**[^6]

**저자**: CIS Lab HKUST (hk.ust-gz.edu.cn)

- **목적**: VR에서 3DGS의 fine-grained editing
- **VR 상호작용**:
    - 직관적 점 선택 (VR 조종기)
    - Drawing-based 객체 생성
    - Real-time 색상 조정
    - 객체 splitting 기능

**GitHub**: https://github.com/CISLab-HKUST/GaussianShopVR
**사용자 연구**: 18명 point selection, 20명 generation, 10명 usability 테스트

***

### **B. 게임 엔진 구현**

#### **Unity (권장)**[^7]

- **이유**: OpenXR native support + Meta XR SDK 통합
- **구현 방식**:
    - Universal Render Pipeline (URP)
    - Shader Graph + Visual Effect Graph
    - GPU-accelerated particle systems
    - Asset streaming + partitioning

**참고 프로젝트**: UnityGaussianSplatting (커뮤니티)

#### **Unreal Engine**

- **XVERSE 3D-GS 플러그인** (커뮤니티)
- Custom shader 기반
- 조명 통합 제한적

***

## 3️⃣ **Blender VR 제약과 Python의 한계**

### **왜 Python draw_handler가 VR에서 안 되는가?**

| 영역 | 상태 | 기술 구현 |
| :-- | :-- | :-- |
| **Viewport 렌더링** | ✅ | Python draw_handler + GLSL |
| **VR 렌더링** | ❌ | C++ GHOST_IXrGraphicsBinding (Python 접근 불가) |
| **VR Context** | ⚠️ | `wm.xr_session_state` 읽기만 가능, 렌더링 수정 불가 |
| **OpenXR swapchain** | ❌ | Blender 내부 관리, 직접 주입 불가능 |

### **Blender 개발자 커뮤니티의 답변**[^8][^1]

**DevTalk Thread**: "XR controller support" (2021)

```
문제: "Custom draw_handler_add()가 VR offscreen draw loop에서 호출되지 않음"

답변 (개발자):
"이는 context가 없는 offscreen draw loop 때문이다.
더 큰 변경 없이 해결하려면, 
viewport를 3번째 'eye'로 재렌더링하는 방법뿐이다.
이는 성능상 문제가 있다."
```


***

## 4️⃣ **현재 가능한 해결책 (5가지)**

### **방향 1️⃣: EEVEE Render Engine 확장 (★★★★☆)**

**난이도**: 중간 | **성능**: 좋음 | **유지보수**: 복잡

```python
# Blender bpy.types.RenderEngine 상속
class NPRGaussianRenderEngine(bpy.types.RenderEngine):
    bl_idname = "NPR_GAUSSIAN"
    bl_label = "NPR Gaussian Painter"
    
    def view_draw(self, context, depsgraph):
        # VR viewport rendering (view_draw는 offscreen 아님)
        # EEVEE의 렌더링 파이프라인 확장
        pass
```

**장점**:

- Blender의 공식 rendering API
- F12 final render 지원
- 애니메이션 rendering 가능

**단점**:

- EEVEE 커스터마이제이션 복잡
- C++ 레벨 지식 필요
- VR mirror 성능 이슈 (재렌더링 필요)[^8]

**참고**: Blender DevTalk에서 "EEVEE 상속" 방식은 거부됨 (C++ 아키텍처 문제)

***

### **방향 2️⃣: Geometry Nodes + Point Cloud 렌더링 (★★★☆☆)**

**난이도**: 중간 | **성능**: 중간 | **품질**: 근사

```python
# Gaussian을 Geometry Nodes point cloud로 변환
# VR에서 VR Scene Inspection이 네이티브로 렌더링

# 스텝:
1. Gaussian → Points geometry node
2. Instance object on points (icosphere)
3. Material node로 색상/투명도 설정
4. Cycles 또는 EEVEE로 VR rendering
```

**장점**:

- ✅ VR에서 기본 Blender 메시 렌더링 사용
- ✅ 구현 간단
- ✅ 음영/조명 자동 처리

**단점**:

- ❌ 진정한 Gaussian Splatting 아님 (meshed approximation)
- ❌ 수천 개 인스턴스 → 성능 저하
- ❌ 복잡한 변형 어려움

**참고 자료**:[^9][^10][^11]

- Blender Geometry Nodes point cloud import/processing
- Instance on points workflow

***

### **방향 3️⃣: 외부 VR 뷰어 + 실시간 동기화 (★★★★☆)**

**난이도**: 어려움 | **성능**: 우수 | **유지보수**: 중간

**패턴**: Blender (편집) ↔ WebGL/Unity 뷰어 (VR 표시)

#### **A. PlayCanvas + SuperSplat**[^12][^13]

- **SuperSplat 2.0**: 웹 기반 3DGS 편집기
- **특징**:
    - PLY 파일 직접 로드
    - 브라우저 기반 VR 지원 (WebXR)
    - Camera flythrough 타임라인
    - Gallery 공유 기능

```workflow
Blender → numpy array → PLY export
         → SuperSplat (browser) → VR headset (WebXR)
         ↑                        ↓
         ← real-time sync ←
```

**장점**:

- ✅ 전문 GS 편집 UI
- ✅ 웹 기반 VR (Quest native support)
- ✅ 오프라인 작동 가능

**단점**:

- ❌ 별도 앱 필요
- ❌ 지연 시간 (수백ms)
- ❌ Blender ↔ SuperSplat 동기화 스크립팅 필요

**구현**:

```python
# Blender addon에서:
1. Gaussian data → numpy
2. Export as PLY (with color, opacity, covariance)
3. HTTP POST to local PlayCanvas instance
4. WebXR 뷰어 자동 업데이트
```


***

#### **B. Unity WebGL 빌드**[^14]

- Partitioning + asset streaming
- 로딩: 13초 → 1.5초 (최적화)
- WebGPU 기반 렌더링

```workflow
Blender GS → FBX/custom format → Unity scene
           → WebGL build → Browser VR (WebXR)
```


***

### **방향 4️⃣: FFmpeg + Screen Capture 스트리밍 (★★☆☆☆)**

**난이도**: 쉬움 | **성능**: 낮음 | **지연**: 높음

```python
# Blender viewport를 실시간 비디오로 캡처
# VR headset에서 큰 스크린으로 표시

# 문제:
- 지연: 100-500ms (VR에서 역멀미 유발)
- 해상도: 낮음 (대역폭 제약)
- 상호작용: 일방향만 (보기만 가능)
```

**사용**: 프로토타입/데모 용도만

***

### **방향 5️⃣: Blender C++ 소스 수정 (★★☆☆☆)**

**난이도**: 매우 어려움 | **성능**: 우수 | **유지보수**: 악몽

**위치**: `source/blender/makesrna/intern/rna_xr.c` + `GHOST_IXrGraphicsBinding`

```cpp
// Blender offscreen draw loop에 callback 추가
// 현재는 불가능 (공식적으로 지원되지 않음)

// 가능한 수정:
1. draw_handler_add() 확장 (XR context 지원)
2. VR render callback 등록
3. Per-eye matrix 전달
```

**문제**:

- ❌ Blender 자체 빌드 필요
- ❌ 업데이트마다 유지보수 필요
- ❌ 커뮤니티 지원 없음
- ❌ Windows/Mac/Linux별 분리 구현

**참고**: BlenderXR 프로젝트[^15]

```cpp
// BlenderXR/src/vr_openxr.cpp
// Direct OpenXR graphics binding 구현
// D3D11 + OpenGL 지원

// 하지만 이는 **별도 포크**이며,
// 공식 Blender와 통합되지 않음
```


***

## 5️⃣ **권장 아키텍처: 3단계 하이브리드**

### **당신의 프로젝트에 최적화된 솔루션**

```
┌─────────────────────────────────────────────────────┐
│ Phase 1: PC Viewport (현재 상태) ✅                  │
│ └─ GLSL draw_handler + 60 FPS                        │
│    (이미 작동 중)                                    │
├─────────────────────────────────────────────────────┤
│ Phase 2: VR Fallback (geometry-based)               │
│ └─ Geometry Nodes point cloud → VR rendering        │
│    (빠른 구현, 품질 낮음)                           │
├─────────────────────────────────────────────────────┤
│ Phase 3: External VR Viewer (최종 목표)             │
│ └─ PlayCanvas WebGL → WebXR (Quest native)         │
│    (고품질, 네이티브 VR 경험)                       │
└─────────────────────────────────────────────────────┘
```


### **구현 순서**

**1단계 (2주)**: Geometry Nodes fallback

```python
# VR에서 최소한 무언가 보이게

def paint_to_geometry_nodes(gaussians):
    # Gaussian 데이터 → Points mesh → Instance objects
    # Cycles로 렌더링 (VR에서 기본 메시 지원)
    pass
```

**2단계 (3주)**: PlayCanvas 연동

```python
# PC viewport는 GLSL, VR는 WebGL

# 실시간 동기화:
- Gaussian 변경 → numpy
- numpy → PLY
- PLY → HTTP POST to PlayCanvas
- PlayCanvas → WebXR update
```

**3단계 (향후)**: Custom Blender build (선택사항)

- C++ 수정 + GHOST_IXrGraphicsBinding 확장
- 완전한 통합 (but 높은 유지보수 비용)

***

## 6️⃣ **즉시 적용 가능한 코드 패턴**

### **패턴 A: Geometry Nodes → VR**

```python
# addon/operators.py

def create_gaussian_point_cloud(gaussians_numpy):
    """Gaussian을 point cloud로 변환 (VR 호환)"""
    
    # 1. Point cloud mesh 생성
    mesh = bpy.data.meshes.new("GaussianPointCloud")
    verts = gaussians_numpy[:, :3]  # positions
    mesh.from_pydata(verts, [], [])
    
    # 2. Geometry Nodes 설정
    obj = bpy.data.objects.new("GaussianPC", mesh)
    bpy.context.collection.objects.link(obj)
    
    # 3. Instance on Points
    gn_modifier = obj.modifiers.new("GaussianInstance", 'GEOMETRY_NODES')
    # ... node tree 생성
    
    # 4. VR에서 자동 렌더링
    return obj
```


***

### **패턴 B: PlayCanvas 실시간 동기화**

```python
# addon/vr_sync.py

import json
import requests
import numpy as np
from pathlib import Path

class PlayCanvasSync:
    def __init__(self, playcanvas_url="http://localhost:8080"):
        self.url = playcanvas_url
        self.session_id = None
    
    def update_gaussians(self, gaussians_tensor):
        """PyTorch tensor → PLY → PlayCanvas"""
        
        # 1. PyTorch → NumPy
        gaussian_np = gaussians_tensor.cpu().numpy()
        
        # 2. PLY 생성
        ply_data = self._create_ply(gaussian_np)
        ply_path = Path("/tmp/gaussian.ply")
        ply_path.write_bytes(ply_data)
        
        # 3. PlayCanvas로 전송
        with open(ply_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.url}/upload",
                files=files,
                json={'session_id': self.session_id}
            )
        
        return response.json()
    
    def _create_ply(self, gaussian_np):
        """NumPy → PLY format"""
        # 59-float layout (KIRI 방식):
        # [0-2]: position
        # [3-6]: rotation (quaternion)
        # [7-9]: scale
        # [^10]: opacity
        # [11-58]: SH coefficients
        
        ply_header = f"""ply
format binary_little_endian 1.0
element vertex {len(gaussian_np)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
"""
        # ... PLY 바이너리 생성
        return ply_header.encode() + b''  # 실제 구현
```


***

## 7️⃣ **성능 목표 및 권장사항**

| 항목 | PC 목표 | VR 목표 | 달성 방법 |
| :-- | :-- | :-- | :-- |
| **FPS** | 60+ | 72-90 | Foveated rendering |
| **Gaussian 수** | 10,000+ | 1,000-5,000 | Decimation + LOD |
| **Latency** | <20ms | <11ms | Streaming 최적화 |
| **VRAM** | 2-4GB | 1-2GB | Memory pooling |

**VR-Splatting 참고**: 10K gaussians를 foveated rendering으로 90Hz 달성[^3]

***

## 8️⃣ **추가 자료 및 커뮤니티**

### **관련 GitHub 프로젝트**

1. **gsplat** (nerfstudio-project)
    - https://github.com/nerfstudio-project/gsplat
    - Python/CUDA 3DGS rasterizer
2. **GaussianShopVR** (HKUST)
    - https://github.com/CISLab-HKUST/GaussianShopVR
    - VR-based 3DGS editing (완전 구현 참고)
3. **3dgs-render-blender-addon** (KIRI)
    - https://github.com/Kiri-Innovation/3dgs-render-blender-addon
    - PC viewport GLSL 구현 (현재 작업 기반)
4. **Dream Textures**
    - Subprocess + SharedMemory IPC 패턴 (PyTorch DLL 충돌 회피)

### **핵심 학술 자료**

| 제목 | 저자 | 연도 | 링크 |
| :-- | :-- | :-- | :-- |
| VR-Splatting | Franke et al. | i3D 2025 | [^3] |
| VR-GS | Jiang et al. | SIGGRAPH 2024 | [^4] |
| Fov-GS | - | 2025 | [^5] |
| GaussianShopVR | Shen et al. | UIST 2025 | [^6] |
| A3FR | - | 2025 | [^16] |
| VR Radiance Fields Survey | - | 2025 | [^17] |


***

## 9️⃣ **최종 권장사항**

### **단기 (1-2주)**

✅ **Geometry Nodes fallback 구현**

- VR에서 최소 기능성 확보
- Quest에서 무언가 보이게 함
- 품질은 낮지만 빠른 반복 가능


### **중기 (3-4주)**

✅ **PlayCanvas 동기화 프로토타입**

- PC viewport: GLSL high-quality
- VR headset: WebGL streaming
- WebXR native support


### **장기 (2-3개월)**

✅ **GaussianShopVR 참고 + 커스터마이제이션**

- VR에서 인터랙티브 editing
- Physics-aware manipulation (VR-GS 참고)
- Foveated rendering (VR-Splatting 기술)


### **주의**

❌ **Blender C++ 소스 수정은 피하세요**

- 높은 유지보수 비용
- 커뮤니티 지원 부족
- 멀티 플랫폼 복잡성

***

## 🎯 **결론**

**draw_handler가 VR에서 작동하지 않는 것은 Blender 아키텍처의 근본적인 제약**입니다. 하지만 **5가지 대안 경로**를 통해 해결 가능합니다:

1. **Geometry Nodes** (빠름, 품질 낮음) ← **권장 1차**
2. **외부 WebGL 뷰어** (고품질, 통합 필요) ← **권장 최종**
3. EEVEE 확장 (복잡)
4. Blender 소스 수정 (불권장)
5. 스트리밍 (지연 높음)

**당신의 SIGGRAPH 2025 페인팅 시스템**을 위해서는 **Phase 1(PC GLSL) + Phase 2/3(VR fallback/external)** 조합이 최적입니다. 🚀
<span style="display:none">[^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96]</span>

<div align="center">⁂</div>

[^1]: https://devtalk.blender.org/t/xr-controller-support/19317

[^2]: https://arxiv.org/html/2410.17932

[^3]: https://lfranke.github.io/vr_splatting/

[^4]: https://par.nsf.gov/servlets/purl/10535783

[^5]: https://ieeexplore.ieee.org/document/10932755/

[^6]: https://cislab.hkust-gz.edu.cn/projects/gaussianshopvr/

[^7]: https://thesis.unipd.it/retrieve/d9b4c987-bfac-410b-a37a-6525455e4fe8/Rahimzadeh_Sina.pdf

[^8]: https://devtalk.blender.org/t/vr-scene-inspection-feedback/13043

[^9]: https://www.youtube.com/watch?v=DCkFhHNeSc0

[^10]: https://www.youtube.com/watch?v=kG-rO73bP9Q

[^11]: https://www.youtube.com/watch?v=iXvcFQdND-c

[^12]: https://www.cgchannel.com/2025/02/supersplat-2-0-lets-you-create-flythroughs-of-3dgs-scans/

[^13]: https://www.youtube.com/watch?v=motXibqJWa8

[^14]: https://www.reddit.com/r/GaussianSplatting/comments/1jumaqm/virtual_tour_with_3d_gaussian_splatting_in_unity/

[^15]: https://github.com/MARUI-PlugIn/BlenderXR/blob/master/src/vr_openxr.cpp

[^16]: VR_RENDERING_RESEARCH_REQUEST.md

[^17]: https://www.arxiv.org/pdf/2508.04326.pdf

[^18]: VR_TECHNICAL_CONTEXT.md

[^19]: PROJECT_PLAN.md

[^20]: https://dl.acm.org/doi/10.1145/3728302

[^21]: https://dl.acm.org/doi/10.1145/3721145.3735112

[^22]: https://ieeexplore.ieee.org/document/11092302/

[^23]: https://ieeexplore.ieee.org/document/10536453/

[^24]: https://dl.acm.org/doi/10.1145/3641234.3671044

[^25]: https://onlinelibrary.wiley.com/doi/10.1111/cgf.70069

[^26]: https://ieeexplore.ieee.org/document/11236153/

[^27]: https://ieeexplore.ieee.org/document/11125628/

[^28]: https://ieeexplore.ieee.org/document/11011170/

[^29]: http://arxiv.org/pdf/2409.15959.pdf

[^30]: https://arxiv.org/html/2405.12218v1

[^31]: https://arxiv.org/html/2503.23625v1

[^32]: https://arxiv.org/html/2409.08353v1

[^33]: https://arxiv.org/html/2312.05941

[^34]: https://arxiv.org/pdf/2403.20309v1.pdf

[^35]: https://arxiv.org/html/2402.00525v3

[^36]: https://www.reddit.com/r/GaussianSplatting/comments/1iyz4si/realtime_gaussian_splatting/

[^37]: https://arxiv.org/abs/2410.17932

[^38]: https://www.youtube.com/watch?v=fovZlYSMhAI

[^39]: https://developer.nvidia.com/blog/real-time-gpu-accelerated-gaussian-splatting-with-nvidia-designworks-sample-vk_gaussian_splatting/

[^40]: https://www.themoonlight.io/en/review/vrsplat-fast-and-robust-gaussian-splatting-for-virtual-reality

[^41]: https://kimjy99.github.io/논문리뷰/vr-gs/

[^42]: https://www.arxiv.org/abs/2511.12930

[^43]: https://dl.acm.org/doi/10.1145/3728311

[^44]: https://arxiv.org/html/2401.05750v2

[^45]: https://joss.theoj.org/papers/10.21105/joss.04901.pdf

[^46]: https://arxiv.org/html/2407.12486v1

[^47]: https://arxiv.org/pdf/2412.09008.pdf

[^48]: https://dl.acm.org/doi/pdf/10.1145/3610548.3618139

[^49]: http://arxiv.org/pdf/2502.17078.pdf

[^50]: http://arxiv.org/pdf/2310.02881.pdf

[^51]: https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B4-2020/567/2020/isprs-archives-XLIII-B4-2020-567-2020.pdf

[^52]: https://stackoverflow.com/questions/12157646/how-to-render-offscreen-on-opengl

[^53]: https://openxr-tutorial.com/linux/opengl/3-graphics.html

[^54]: https://www.youtube.com/watch?v=ZrXAEsYiIyE

[^55]: https://devtalk.blender.org/t/drawing-to-gpuoffscreen-from-within-an-operator-seems-to-freeze-blender-until-the-3d-view-is-redrawn/14459

[^56]: https://community.khronos.org/t/is-openxrs-swapchain-fake/110028

[^57]: https://varjo.com/blog/how-to-view-blender-content-with-varjo-headsets-a-step-by-step-guide

[^58]: https://devtalk.blender.org/t/rendering-text-in-opengl-off-screen/13533

[^59]: https://stackoverflow.com/questions/79489881/how-to-fix-msaa-performance-issue-with-vulkan-openxr-custom-game-engine

[^60]: https://www.reddit.com/r/virtualreality/comments/gvytf5/blender_has_vr_scene_inspection_now_first/

[^61]: https://arxiv.org/html/2410.17858v1

[^62]: https://ijaers.com/uploads/issue_files/12%20IJAERS-DEC-2017-17-Updating%20and%20Rendering%20Content.pdf

[^63]: https://arxiv.org/abs/2303.15666

[^64]: https://arxiv.org/pdf/2210.04847.pdf

[^65]: https://arxiv.org/pdf/2001.03537.pdf

[^66]: http://arxiv.org/pdf/2312.06575.pdf

[^67]: https://www.reddit.com/r/blenderhelp/comments/10q512n/how_to_change_render_viewport_to_gpu_blender_34/

[^68]: https://yelzkizi.org/what-is-gaussian-splatting/

[^69]: https://stackoverflow.com/questions/77971943/blender-bpy-module-ignores-gpu-configuration-for-rendering

[^70]: https://blenderartists.org/t/gpu-not-being-used-when-using-blender-as-a-python-module/1463774

[^71]: https://docs.blender.org/api/current/gpu.html

[^72]: https://www.semanticscholar.org/paper/1b5e98483e56b4790f962fcedf442874e8248eba

[^73]: https://dl.acm.org/doi/10.1145/2947688.2947699

[^74]: https://lib.dr.iastate.edu/etd/12419/

[^75]: http://ieeexplore.ieee.org/document/6743722/

[^76]: https://arxiv.org/pdf/2402.13724.pdf

[^77]: https://arxiv.org/html/2312.11729v1

[^78]: http://arxiv.org/pdf/2404.09833.pdf

[^79]: https://devtalk.blender.org/t/custom-render-engine-extending-eevee/10156

[^80]: https://docs.blender.org/manual/en/latest/render/eevee/introduction.html

[^81]: https://www.linkedin.com/pulse/tools-workflows-optimisations-gaussian-splatting-gabriele-romagnoli-usfyc

[^82]: http://ieeevr.org/2025/program/posters/

[^83]: https://www.youtube.com/watch?v=R76o8dYN_GI

[^84]: https://blenderartists.org/t/vr-viewer-add-on-for-blender-using-cycle-or-eevee-render/1516282

[^85]: https://www.semanticscholar.org/paper/fb880f073ba77d6c39b58c749a148fc97c55b2a6

[^86]: https://arxiv.org/abs/2103.14507

[^87]: https://arxiv.org/html/2409.13926v1

[^88]: https://arxiv.org/html/2407.10707v1

[^89]: https://arxiv.org/html/2401.08398v2

[^90]: https://devtalk.blender.org/t/enable-viewport-bvh-build-for-final-render/9650

[^91]: https://docs.blender.org/manual/en/latest/addons/3d_view/vr_scene_inspection.html

[^92]: https://github.com/mikeroyal/Blender-Guide

[^93]: https://devtalk.blender.org/t/optmization-for-viewport-refresh-cycles-and-render-part-1-4/20245

[^94]: https://www.youtube.com/watch?v=nh_vSi0tzg0

[^95]: https://devtalk.blender.org/t/gsoc-2019-vr-support-through-openxr-weekly-reports/7665

[^96]: https://www.youtube.com/watch?v=quXJbxULMTg

