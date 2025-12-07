# VR Rendering 개발 문서

> **최종 업데이트**: 2025-12-08  
> **목표**: Blender에서 3D Gaussian Splatting을 VR 헤드셋에 렌더링

---

## 📋 현재 상태

| 항목             | 상태                       |
| ---------------- | -------------------------- |
| PC GLSL Viewport | ✅ 작동                    |
| VR 컨트롤러 추적 | ✅ 작동                    |
| VR GLSL 렌더링   | ❌ **draw_handler 미지원** |
| VR Mesh 대체     | ✅ 작동 (임시)             |

---

## 📁 문서 구조

### 연구 보고서 (Gemini Research)

| 파일                                                             | 내용                             |
| ---------------------------------------------------------------- | -------------------------------- |
| `3D Gaussian.md`                                                 | 1차 조사 - 기본 VR 렌더링 방법   |
| `Blender VR Gaussian Splatting Rendering.md`                     | 2차 조사 - Geometry Nodes 접근   |
| `Blender VR GLSL 렌더링 커스텀 파이프라인 - 종합 기술 리포트.md` | 3차 조사 - 5가지 솔루션 비교     |
| `Blender VR Custom Shader Rendering.md`                          | 4차 조사 - OpenXR API Layer 상세 |

### 연구 요청 문서

| 파일                               | 내용                        |
| ---------------------------------- | --------------------------- |
| `VR_RENDERING_RESEARCH_REQUEST.md` | 초기 연구 요청서            |
| `VR_CUSTOM_PIPELINE_RESEARCH.md`   | 커스텀 파이프라인 연구 요청 |
| `VR_CUSTOM_PIPELINE_CONTEXT.md`    | 기술 컨텍스트 코드 발췌     |
| `VR_TECHNICAL_CONTEXT.md`          | 기술 상세                   |

---

## 🎯 권장 개발 로드맵

### Phase 1: gpu.offscreen + Plane (1주)

- 목표: VR에서 2D 텍스처로 Gaussian 표시
- 난이도: ⭐⭐
- Stereo: ❌

### Phase 2: Custom RenderEngine (2주)

- 목표: `view_draw()` VR 호환성 테스트
- 난이도: ⭐⭐⭐
- Stereo: 테스트 필요

### Phase 3: OpenXR API Layer (2-3개월)

- 목표: `xrEndFrame` 후킹하여 Composition Layer 주입
- 난이도: ⭐⭐⭐⭐⭐
- Stereo: ✅

---

## 🔑 핵심 발견

1. **draw_handler가 VR에서 안 되는 이유**: `wm_xr_draw.c`에서 overlay pass 건너뜀
2. **Custom RenderEngine**: VR에서 호출되지만 전체 렌더러 대체
3. **OpenXR API Layer**: 최종 솔루션 (C++ 개발 필요)

---

## 📚 핵심 참고 자료

- [OpenXR-API-Layer-Template](https://github.com/Ybalrid/OpenXR-API-Layer-Template)
- [BlenderXR](https://github.com/MARUI-PlugIn/BlenderXR)
- [VRSplat Paper](https://arxiv.org/abs/2505.10144)
- [Blender VR Source](https://fossies.org/dox/blender-4.5.1/wm__xr__draw_8cc_source.html)
