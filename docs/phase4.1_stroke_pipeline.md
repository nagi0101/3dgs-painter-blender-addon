# Phase 4.1: 브러시 스트로크 파이프라인 (Brush Stroke Pipeline)

**기간**: 1주  
**목표**: 사용자 입력을 Gaussian 데이터로 변환하는 스트로크 생성 파이프라인 구현  
**선행 조건**: Phase 4 (페인팅 인터랙션 기반 인프라)

---

## ⚠️ 기존 코드 활용 안내

> **본 Phase의 핵심 로직은 `npr_core/` 모듈에 이미 구현되어 있습니다.**
>
> 프로토타입(`npr-gaussian-2d-prototype`)에서 복사/리팩토링된 코드를 최대한 활용합니다.
> 새로운 구현보다는 **블렌더 통합** 및 **3D 확장**에 집중합니다.

### 기존 구현 현황

| 모듈              | 파일                          | 구현 상태    | 핵심 API                                  |
| ----------------- | ----------------------------- | ------------ | ----------------------------------------- |
| **StrokePainter** | `npr_core/brush.py`           | ✅ 완전 구현 | `start/update/finish_stroke()`            |
| **BrushStamp**    | `npr_core/brush.py`           | ✅ 완전 구현 | `create_*()`, `place_at_*()`              |
| **StrokeSpline**  | `npr_core/spline.py`          | ✅ 완전 구현 | `add_point()`, `evaluate_at_arc_length()` |
| **SceneData**     | `npr_core/scene_data.py`      | ✅ 완전 구현 | Array-based storage (40-80× faster)       |
| **Deformation**   | `npr_core/deformation_gpu.py` | ✅ 완전 구현 | `deform_all_stamps_batch_gpu()`           |

---

## 📋 작업 개요

**기존 코드를 블렌더 환경에 통합**하는 것이 주 목표:

```
User Input → Spline Construction → Arc-Length Sampling → Stamp Placement → Deformation
```

| 작업                  | 상태      | 접근 방식              |
| --------------------- | --------- | ---------------------- |
| 스트로크 라이프사이클 | ✅ 구현됨 | `StrokePainter` 활용   |
| Arc-length 균일 배치  | ✅ 구현됨 | `StrokeSpline` 활용    |
| 2계층 브러시 아키텍처 | ✅ 구현됨 | `BrushStamp` 활용      |
| GPU 배치 변형         | ✅ 구현됨 | `deformation_gpu` 활용 |
| **블렌더 3D 통합**    | ⚡ 신규   | Phase 4 IPC 연동       |

---

## 🎯 기존 모듈 활용 가이드

### 1. StrokePainter (`npr_core/brush.py`)

스트로크 라이프사이클 관리:

```python
painter = StrokePainter(brush, scene_data)
painter.start_stroke(position, normal, pressure)
painter.update_stroke(position, normal, pressure)  # N times
stamps = painter.finish_stroke()  # Deformation + Inpainting 자동 적용
```

### 2. BrushStamp (`npr_core/brush.py`)

**2계층 아키텍처**: Pattern (템플릿) + Instance (런타임 파라미터)

```python
# 프로그래매틱 생성
brush = BrushStamp.create_circular(num_gaussians=20, radius=0.15)
brush = BrushStamp.create_line(num_gaussians=10, length=0.3)
brush = BrushStamp.create_grid(rows=5, cols=5, spacing=0.1)

# 3단계 배치 전략 (성능별 선택)
stamp = brush.place_at(position, normal)                    # 단일, UI 미리보기
stamps = brush.place_at_batch(positions, normals)          # 3-10개, 10-20× 빠름
arrays = brush.place_at_batch_arrays(positions, normals)   # 10+개, 40-80× 빠름
```

### 3. StrokeSpline (`npr_core/spline.py`)

Arc-length 파라미터화된 Cubic spline:

```python
spline = StrokeSpline(force_2d=False)  # 3D 모드
spline.add_point(position, normal, threshold=0.01)

pos = spline.evaluate_at_arc_length(arc_length)
tangent, normal, binormal = spline.get_frame_at_arc_length(arc_length)
```

**핵심 기능**: 입력 필터링, arc-length 샘플링, Slerp 노멀 보간

### 4. GPU Deformation (`npr_core/deformation_gpu.py`)

```python
deform_all_stamps_batch_gpu(
    scene_data, spline, stamp_placements,
    start_idx, end_idx, sparse_threshold=0.5
)
```

**최적화**: GPU 스플라인 캐시 (5-10×), 희소 변형 (30-50% 추가)

---

## 🔧 블렌더 통합 작업 (신규)

### 5.1 3D 좌표계 적응

-   `StrokeSpline`: `force_2d=False` 사용
-   Surface normal: Phase 4 Raycasting에서 얻은 실제 법선 사용

### 5.2 Viewport 동기화

Phase 4의 SharedMemory IPC와 연동하여 실시간 렌더링 업데이트

---

## 📊 아키텍처 참고

### 아크 길이 기반 스탬프 간격 (왜 중요한가)

| 방식           | 빠른 이동 시 | 느린 이동 시 | 커브 구간 |
| -------------- | ------------ | ------------ | --------- |
| 시간 기반 (dt) | 간격 큼      | 과밀         | 불균일    |
| 포인트 기반    | 불균일       | 불균일       | 코너 밀집 |
| **아크 길이**  | **균일**     | **균일**     | **균일**  |

| 조건      | 전략                          | 상대 성능  | 용도              |
| --------- | ----------------------------- | ---------- | ----------------- |
| 단일      | `place_at()`                  | 1×         | UI 미리보기       |
| 3-10개    | `place_at_batch()`            | 10-20×     | 짧은 스트로크     |
| **10+개** | **`place_at_batch_arrays()`** | **40-80×** | **실시간 페인팅** |

### 색상/투명도 틴팅 시스템 (구현됨)

-   **Pattern Layer**: Luminance 정보 저장 (grayscale-like)
-   **Instance Layer**: Tint 색상 제공
-   **공식**: `final_color = tint_color × pattern_luminance`

---

## 📊 성능 목표

| 연산                       | 목표   | 허용    | 위험    |
| -------------------------- | ------ | ------- | ------- |
| 스탬프 생성 (단일)         | <5ms   | <10ms   | >20ms   |
| 스탬프 배치 (10개)         | <1ms   | <2ms    | >5ms    |
| 스플라인 구성 (100 포인트) | <2ms   | <5ms    | >10ms   |
| 변형 (100 스탬프, GPU)     | <500ms | <1000ms | >2000ms |
| 변형 (100 스탬프, sparse)  | <300ms | <600ms  | >1000ms |

---

## 🧪 테스트 및 검증

`tests/test_stroke_pipeline.py`에 다음 테스트 케이스 구현 (참조용):

| 테스트                          | 검증 항목                             |
| ------------------------------- | ------------------------------------- |
| `test_spline_arc_length`        | Arc-length 파라미터화 정확도          |
| `test_stamp_placement_batch`    | 배치 배치 성능 (<10ms for 100 stamps) |
| `test_stroke_painter_lifecycle` | start/update/finish 라이프사이클      |
| `test_frame_computation`        | Frenet-Serret 프레임 직교성           |

> **참고**: 기존 `npr_core/` 모듈의 단위 테스트도 함께 활용

---

## 📚 참고 자료

-   `npr-gaussian-2d-prototype` StrokePainter 구현
-   `src/npr_core/brush.py`, `spline.py`, `deformation_gpu.py`
-   SciPy CubicSpline documentation
-   Frenet-Serret formulas (differential geometry)
