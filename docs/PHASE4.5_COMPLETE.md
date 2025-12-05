# Phase 4.5: Brush Creation & Conversion - 완료 보고서

**완료일**: 2024년 12월 4일  
**상태**: ✅ 완료

## 개요

Phase 4.5에서는 이미지 기반 브러시 변환 시스템과 프로그래매틱 브러시 생성 기능을 구현했습니다. 기존 프로토타입의 `BrushConverter`를 Blender 애드온에 맞게 이식하고, MiDaS 기반 깊이 추정을 제거하여 thickness 기반 접근법으로 단순화했습니다.

## 구현 내용

### 1. BrushConverter 클래스 (`src/npr_core/brush_converter.py`)

**새 파일 생성** - 약 787줄

#### BrushConversionConfig (데이터클래스)

```python
@dataclass
class BrushConversionConfig:
    num_gaussians: int = 256        # 생성할 Gaussian 수
    sampling_method: str = "skeleton"  # skeleton, uniform, random
    depth_profile: str = "convex"   # flat, convex, concave, ridge
    skeleton_depth_weight: float = 0.5  # 스켈레톤 기반 깊이 가중치
    thickness_depth_weight: float = 0.5  # 두께 기반 깊이 가중치
    enable_elongation: bool = True  # Gaussian 신장 활성화
```

#### 주요 메서드

| 메서드                                                              | 설명                                       |
| ------------------------------------------------------------------- | ------------------------------------------ |
| `convert_image(image, brush_name, config)`                          | 이미지 → 브러시 변환 메인 파이프라인       |
| `_extract_alpha_mask(image)`                                        | 알파 채널 추출 또는 적응형 임계값으로 생성 |
| `_extract_features(image, alpha_mask)`                              | 스켈레톤, 두께맵, 그라디언트 플로우 추출   |
| `_estimate_depth(features, config)`                                 | 두께 기반 깊이 추정 (MiDaS 제거)           |
| `_importance_sampling(alpha_mask, features, config)`                | 중요도 기반 포인트 샘플링                  |
| `_initialize_gaussians(points, image, depth_map, features, config)` | 샘플링된 포인트에서 Gaussian 초기화        |
| `_refine_gaussians(gaussians, alpha_mask, features)`                | 프로시저럴 브러시 정제                     |
| `create_circular_brush(name, num_gaussians, radius)`                | 원형 브러시 생성                           |
| `create_line_brush(name, num_gaussians, length, width)`             | 선형 브러시 생성                           |
| `create_grid_brush(name, rows, cols, spacing)`                      | 그리드 브러시 생성                         |

#### 깊이 프로파일 옵션

-   **flat**: 균일한 깊이 (z=0)
-   **convex**: 중심이 튀어나옴 (스켈레톤/두께 가중 합)
-   **concave**: 중심이 들어감 (convex의 역)
-   **ridge**: 스켈레톤을 따라 릿지 형성

### 2. 브러시 생성 오퍼레이터 (`src/operators.py`)

**추가된 오퍼레이터**:

| 오퍼레이터                        | bl_idname                         | 설명                       |
| --------------------------------- | --------------------------------- | -------------------------- |
| `THREEGDS_OT_CreateBrushCircular` | `threegds.create_brush_circular`  | 원형 프로그래매틱 브러시   |
| `THREEGDS_OT_CreateBrushLine`     | `threegds.create_brush_line`      | 선형 프로그래매틱 브러시   |
| `THREEGDS_OT_CreateBrushGrid`     | `threegds.create_brush_grid`      | 그리드 프로그래매틱 브러시 |
| `THREEGDS_OT_ConvertImageToBrush` | `threegds.convert_image_to_brush` | 이미지 파일 → 브러시 변환  |

### 3. UI 패널 (`src/viewport/panels.py`)

**새 패널**: `NPR_PT_BrushCreationPanel`

#### Scene 프로퍼티

```python
bpy.types.Scene.npr_conversion_num_gaussians      # 128-2048, 기본 256
bpy.types.Scene.npr_conversion_depth_profile      # flat/convex/concave/ridge
bpy.types.Scene.npr_conversion_skeleton_weight    # 0.0-1.0, 기본 0.5
bpy.types.Scene.npr_conversion_thickness_weight   # 0.0-1.0, 기본 0.5
bpy.types.Scene.npr_conversion_enable_elongation  # bool, 기본 True
```

#### UI 레이아웃

```
┌─────────────────────────────────────┐
│ NPR Brush Creation                  │
├─────────────────────────────────────┤
│ Programmatic Brushes:               │
│  [Circular] [Line] [Grid]           │
├─────────────────────────────────────┤
│ Image to Brush:                     │
│  [Convert Image to Brush]           │
│                                     │
│  Conversion Settings:               │
│   Num Gaussians: [====256====]      │
│   Depth Profile: [convex    ▼]      │
│   Skeleton Weight: [===0.5===]      │
│   Thickness Weight: [===0.5===]     │
│   [✓] Enable Elongation             │
└─────────────────────────────────────┘
```

### 4. 의존성 업데이트

**수정된 파일**: `src/requirements/*.txt` (4개 파일 모두)

추가된 패키지:

-   `opencv-python>=4.8.0`
-   `scikit-image>=0.21.0`

### 5. 의존성 검사 및 설치 개선

**수정된 파일**:

-   `src/npr_core/dependencies.py` - REQUIRED_PACKAGES에 opencv-python, scikit-image 추가
-   `src/npr_core/installer.py` - 이미 설치된 패키지 건너뛰기 로직 추가

#### dependencies.py 변경사항

```python
REQUIRED_PACKAGES: List[DependencyInfo] = [
    # ... 기존 패키지 ...
    DependencyInfo("opencv-python", ">=4.8.0", import_name="cv2"),
    DependencyInfo("scikit-image", ">=0.21.0", import_name="skimage"),
    DependencyInfo("gsplat", ">=0.1.0", import_name="gsplat", optional=True),
]
```

#### installer.py 변경사항

-   `install_all()` 메서드가 이제 `get_missing_packages()`를 호출하여 누락된 패키지만 확인
-   PyTorch가 이미 설치되어 있으면 건너뜀
-   gsplat이 이미 설치되어 있으면 건너뜀
-   설치 진행 시 "✓ PyTorch already installed - skipping" 등의 메시지 표시

## 설계 결정사항

### MiDaS 제거

-   기존 프로토타입에서 MiDaS 기반 깊이 추정은 큰 효과가 없었음
-   대신 thickness map과 skeleton proximity를 활용한 깊이 추정으로 대체
-   결과적으로 의존성 감소 및 변환 속도 향상

### 런타임 의존성 처리

-   `cv2`, `scipy`, `skimage`는 메서드 내부에서 lazy import
-   `_check_dependencies()` 메서드로 런타임에 의존성 검증
-   Blender Python 환경에 패키지가 설치되지 않은 경우 명확한 에러 메시지 제공

## 테스트 방법

### 사전 요구사항

1. Phase 2의 의존성 설치가 완료되어 있어야 함
2. Preferences에서 "Install Dependencies" 실행 필요 (opencv-python, scikit-image 포함)

### 테스트 1: 프로그래매틱 브러시 생성

```python
# Blender Python 콘솔에서 실행
import bpy

# 1. 원형 브러시 생성
bpy.ops.threegds.create_brush_circular()

# 2. 선형 브러시 생성
bpy.ops.threegds.create_brush_line()

# 3. 그리드 브러시 생성
bpy.ops.threegds.create_brush_grid()

# 4. BrushManager에서 생성된 브러시 확인
from npr_core.brush_manager import BrushManager
manager = BrushManager.get_instance()
print(f"등록된 브러시: {list(manager._brushes.keys())}")
```

**예상 결과**:

-   각 오퍼레이터 실행 시 INFO 메시지로 브러시 생성 확인
-   BrushManager에 "Circular", "Line", "Grid" 브러시 등록됨

### 테스트 2: 이미지 → 브러시 변환

```python
# Blender Python 콘솔에서 실행
import bpy

# 파일 브라우저를 통한 변환
bpy.ops.threegds.convert_image_to_brush('INVOKE_DEFAULT')
```

**수동 테스트 절차**:

1. 3D Viewport 사이드바 (N) → NPR Gaussian Painter 탭 → Brush Creation 패널 열기
2. "Convert Image to Brush" 버튼 클릭
3. PNG/JPG 브러시 이미지 선택 (알파 채널 있는 이미지 권장)
4. 변환 완료 후 INFO 메시지 확인

### 테스트 3: BrushConverter 직접 테스트

```python
# Blender Python 콘솔에서 실행
import numpy as np
from npr_core.brush_converter import BrushConverter, BrushConversionConfig

# 테스트 이미지 생성 (128x128 원형)
size = 128
y, x = np.ogrid[:size, :size]
center = size // 2
radius = size // 3
mask = ((x - center)**2 + (y - center)**2 <= radius**2).astype(np.uint8) * 255

# RGBA 이미지로 변환
test_image = np.zeros((size, size, 4), dtype=np.uint8)
test_image[:, :, 0] = 128  # R
test_image[:, :, 1] = 64   # G
test_image[:, :, 2] = 192  # B
test_image[:, :, 3] = mask # A

# 변환 설정
config = BrushConversionConfig(
    num_gaussians=64,
    depth_profile="convex",
    sampling_method="skeleton"
)

# 변환 실행
converter = BrushConverter()
brush = converter.convert_image(test_image, "TestBrush", config)

# 결과 확인
print(f"브러시 이름: {brush.name}")
print(f"Gaussian 수: {len(brush.gaussians)}")
print(f"첫 번째 Gaussian 위치: {brush.gaussians[0].position}")
```

**예상 결과**:

-   `TestBrush` 이름의 브러시 생성
-   약 64개의 Gaussian이 원형 영역에 분포
-   각 Gaussian에 위치, 색상, 스케일, 회전 값 할당됨

### 테스트 4: 깊이 프로파일 비교

```python
import numpy as np
from npr_core.brush_converter import BrushConverter, BrushConversionConfig

# 테스트 이미지 (위와 동일)
# ... (생략)

profiles = ["flat", "convex", "concave", "ridge"]
converter = BrushConverter()

for profile in profiles:
    config = BrushConversionConfig(
        num_gaussians=32,
        depth_profile=profile
    )
    brush = converter.convert_image(test_image, f"Test_{profile}", config)

    # Z 값 통계
    z_values = [g.position[2] for g in brush.gaussians]
    print(f"{profile}: z_min={min(z_values):.3f}, z_max={max(z_values):.3f}, z_mean={np.mean(z_values):.3f}")
```

**예상 결과**:

-   `flat`: 모든 z 값이 0에 가까움
-   `convex`: 중심부 z 값이 양수 (튀어나옴)
-   `concave`: 중심부 z 값이 음수 (들어감)
-   `ridge`: 스켈레톤 근처 z 값이 높음

### 테스트 5: UI 패널 검증

1. Blender 실행 후 애드온 활성화
2. 3D Viewport에서 N 키로 사이드바 열기
3. "NPR Gaussian Painter" 탭 선택
4. "NPR Brush Creation" 패널 확인
5. 각 설정값 조절 후 브러시 생성

**확인 항목**:

-   [ ] Num Gaussians 슬라이더 작동 (128-2048)
-   [ ] Depth Profile 드롭다운 작동
-   [ ] Skeleton/Thickness Weight 슬라이더 작동
-   [ ] Enable Elongation 체크박스 작동
-   [ ] Programmatic 브러시 버튼 3개 표시
-   [ ] Convert Image to Brush 버튼 표시

## 알려진 제한사항

1. **대용량 이미지**: 2048x2048 이상의 이미지는 변환 시간이 오래 걸릴 수 있음
2. **복잡한 형태**: 매우 복잡한 브러시 형태는 스켈레톤 추출이 정확하지 않을 수 있음
3. **투명도**: 반투명 영역은 이진화되어 처리됨 (50% 임계값)

## 다음 단계

-   Phase 5: 고급 기능 (브러시 라이브러리 관리, 프리셋 시스템 등)
-   브러시 미리보기 렌더링 기능
-   브러시 내보내기/가져오기 기능

## 파일 변경 요약

| 파일                              | 상태 | 설명                                                 |
| --------------------------------- | ---- | ---------------------------------------------------- |
| `src/npr_core/brush_converter.py` | 신규 | BrushConverter 클래스                                |
| `src/npr_core/dependencies.py`    | 수정 | REQUIRED_PACKAGES에 opencv-python, scikit-image 추가 |
| `src/npr_core/installer.py`       | 수정 | 이미 설치된 패키지 건너뛰기 로직                     |
| `src/operators.py`                | 수정 | 4개 오퍼레이터 추가                                  |
| `src/viewport/panels.py`          | 수정 | UI 패널 및 Scene 프로퍼티 추가                       |
| `src/requirements/win_cuda.txt`   | 수정 | opencv-python, scikit-image 추가                     |
| `src/requirements/win_cpu.txt`    | 수정 | opencv-python, scikit-image 추가                     |
| `src/requirements/linux_cuda.txt` | 수정 | opencv-python, scikit-image 추가                     |
| `src/requirements/mac_mps.txt`    | 수정 | opencv-python, scikit-image 추가                     |
