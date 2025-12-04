# Phase 4.5: 브러시 생성 (Brush Creation & Conversion)

**기간**: 1주  
**목표**: 프로그래매틱 브러시 생성 + Image-to-Brush 변환 파이프라인

---

## ⚠️ 기존 코드 활용 안내

> **프로그래매틱 브러시 및 BrushManager는 이미 구현되어 있습니다.**
>
> 본 Phase에서는 **Image-to-Brush 변환 (BrushConverter)**과 **블렌더 UI 통합**만 신규 구현합니다.

### 기존 구현 현황

| 모듈                      | 파일                        | 구현 상태    | 주요 API                                |
| ------------------------- | --------------------------- | ------------ | --------------------------------------- |
| **BrushStamp.create\_\*** | `npr_core/brush.py`         | ✅ 완전 구현 | `create_circular/line/grid()`           |
| **BrushManager**          | `npr_core/brush_manager.py` | ✅ 완전 구현 | `get_brush()`, `save_brush()`, LRU 캐시 |
| **BrushSerializer**       | `npr_core/brush_manager.py` | ✅ 완전 구현 | JSON 직렬화                             |
| **Default Library**       | `npr_core/brush_manager.py` | ✅ 완전 구현 | Soft/Hard Round, Pencil, Marker         |
| **BrushConverter**        | -                           | ❌ 미구현    | **본 Phase에서 구현**                   |

---

## 📋 작업 개요

| 작업                     | 상태      | 접근 방식                  |
| ------------------------ | --------- | -------------------------- |
| 프로그래매틱 브러시 생성 | ✅ 구현됨 | `BrushStamp.create_*` 활용 |
| 브러시 라이브러리 관리   | ✅ 구현됨 | `BrushManager` 활용        |
| JSON 직렬화/역직렬화     | ✅ 구현됨 | `BrushSerializer` 활용     |
| **Image-to-Brush 변환**  | ⚡ 신규   | `BrushConverter` 구현      |
| **블렌더 UI 통합**       | ⚡ 신규   | 패널 + 오퍼레이터          |

---

## 🎯 기존 모듈 활용 가이드

### 1. 프로그래매틱 브러시 (`npr_core/brush.py`)

```python
brush = BrushStamp.create_circular(num_gaussians=20, radius=0.15)
brush = BrushStamp.create_line(num_gaussians=10, length=0.3)
brush = BrushStamp.create_grid(rows=5, cols=5, spacing=0.1)
```

### 2. BrushManager (`npr_core/brush_manager.py`)

```python
manager = BrushManager.get_instance()
manager.create_default_brushes()  # Soft/Hard Round, Pencil, Marker
brush = manager.load_brush(brush_id)
manager.save_brush(brush, "My Brush", brush_type="circular")
```

**기본 브러시**: Soft Round, Hard Round, Pencil, Marker, Airbrush

---

## 🔧 신규 구현: BrushConverter

### 변환 파이프라인

```
Input Image → Alpha Mask → Feature 추출 → Depth 계산 → Point Sampling → Gaussian 초기화
```

### 핵심 알고리즘: Skeleton + Thickness 기반 Depth

MiDaS 대신 구조적 특성 기반 depth 추정:

```
depth = skeleton_weight × skeleton_proximity + thickness_weight × thickness_normalized
```

-   **Skeleton proximity**: 중심선에 가까울수록 높음
-   **Thickness**: 두꺼울수록 높음

**Depth Profile 옵션**:

| Profile | 특징                      |
| ------- | ------------------------- |
| flat    | 평평한 브러시             |
| convex  | 볼록 (skeleton 중심 높음) |
| concave | 오목 (skeleton 중심 낮음) |
| ridge   | Sharp ridge on skeleton   |

### BrushConversionConfig 주요 파라미터

| 파라미터                 | 기본값       | 설명                            |
| ------------------------ | ------------ | ------------------------------- |
| `num_gaussians`          | 100          | Gaussian 개수                   |
| `sampling_method`        | "importance" | importance / uniform / skeleton |
| `depth_profile`          | "convex"     | flat / convex / concave / ridge |
| `skeleton_depth_weight`  | 0.7          | Skeleton 가중치                 |
| `thickness_depth_weight` | 0.3          | Thickness 가중치                |
| `enable_elongation`      | True         | 방향성 elongation               |

### API 사용 예시

```python
from npr_core.brush_converter import BrushConverter, BrushConversionConfig

config = BrushConversionConfig(
    num_gaussians=50,
    depth_profile="convex",
    skeleton_depth_weight=0.7
)

converter = BrushConverter(config)
brush = converter.convert("brush_stroke.png")
```

---

## 🔧 신규 구현: 블렌더 UI 통합

### 패널 구조

```
NPR Gaussian > Brush Creation
├── Programmatic Brushes: [Circular] [Line] [Grid]
├── Image to Brush
│   ├── Image, num_gaussians, sampling_method
│   ├── depth_profile, skeleton/thickness weights
│   └── [Convert to Brush]
└── Preview
```

### 오퍼레이터

| Operator                          | 기능             |
| --------------------------------- | ---------------- |
| `gaussian.create_brush_circular`  | 원형 브러시 생성 |
| `gaussian.create_brush_line`      | 선형 브러시 생성 |
| `gaussian.create_brush_grid`      | 격자 브러시 생성 |
| `gaussian.convert_image_to_brush` | 이미지 변환      |

---

## 📊 성능 목표

| 작업                     | 목표   | 허용   |
| ------------------------ | ------ | ------ |
| 프로그래매틱 브러시 생성 | <10ms  | <50ms  |
| Image-to-Brush 변환      | <300ms | <500ms |
| 브러시 저장/로드 (JSON)  | <50ms  | <100ms |

---

## 🧪 테스트 및 검증

| 테스트                      | 검증 항목                           |
| --------------------------- | ----------------------------------- |
| `test_programmatic_brushes` | create_circular/line/grid 정상 동작 |
| `test_image_to_brush`       | 변환 파이프라인, depth profile      |
| `test_brush_serialization`  | JSON save/load 일관성               |

---

## 📚 참고 자료

-   `src/npr_core/brush.py`, `brush_manager.py`
-   scikit-image skeletonize
-   scipy distance_transform_edt

---

## 🔗 Phase 연계

-   **Phase 4**: 생성된 브러시를 사용하여 페인팅
-   **Phase 5**: Appearance Optimization (gsplat differentiable rendering)
