# Wall Gaussian Rotation Bug Fix

## 문제 설명 (Problem Description)

### 현상

수직 벽면(vertical walls)에 Gaussian을 페인팅할 때, Gaussian들이 벽면에 평평하게 누워있어야 하는데 대신 **세로로 서있는 선(vertical lines)**처럼 보이는 현상이 발생했습니다.

### 영향받는 케이스

- Y-facing 벽 (XZ 평면, normal = [0, ±1, 0])을 Y축 방향에서 수직으로 바라볼 때
- 기울여서 바라보면 정상적으로 보이고, 정면에서 보면 Gaussian이 세로 선으로 표시됨

### 원인 분석

문제는 **GLSL 셰이더의 `quatToMat` 함수**에서 quaternion을 rotation matrix로 변환할 때 **column-major 순서를 고려하지 않아** 발생했습니다.

---

## 근본 원인 (Root Cause)

### GLSL의 Column-Major 행렬 저장 방식

GLSL에서 `mat3(a, b, c, d, e, f, g, h, i)` 생성자는 값을 **column-major** 순서로 배치합니다:

```
| a  d  g |    ← Column 0: [a,b,c]
| b  e  h |    ← Column 1: [d,e,f]
| c  f  i |    ← Column 2: [g,h,i]
```

### 잘못된 기존 코드

기존 `quatToMat` 함수는 row-major처럼 값을 배치했습니다:

```glsl
// 잘못된 코드 (row-major 스타일)
return mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),  // 이것이 Column 0이 됨!
    2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),  // 이것이 Column 1이 됨!
    2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)   // 이것이 Column 2가 됨!
);
```

**결과**: 회전 행렬이 **전치(transpose)**되어 생성됨 → Gaussian의 방향이 잘못 계산됨

---

## 해결 방법 (Solution)

### 수정된 코드

`quatToMat` 함수를 column-major 순서에 맞게 재작성:

```glsl
mat3 quatToMat(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;

    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;

    return mat3(
        // Column 0
        1.0 - 2.0*(yy + zz),
        2.0*(xy + wz),
        2.0*(xz - wy),
        // Column 1
        2.0*(xy - wz),
        1.0 - 2.0*(xx + zz),
        2.0*(yz + wx),
        // Column 2
        2.0*(xz + wy),
        2.0*(yz - wx),
        1.0 - 2.0*(xx + yy)
    );
}
```

### 추가 수정 사항

1. **`computeCov2D` Jacobian 행렬** (이미 Phase 3에서 문제 있었음)

   - Jacobian 행렬도 column-major 순서로 수정됨

2. **Covariance의 View Space 변환** (이번 디버깅 중 추가)
   - World space covariance를 view space로 변환: `cov3D_view = V * cov3D_world * V^T`

---

## 수정된 파일

| 파일                                | 변경 내용                                                                           |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `src/viewport/viewport_renderer.py` | `quatToMat`, `computeCov2D` column-major 순서 수정, covariance view-space 변환 추가 |

---

## 교훈 (Lessons Learned)

1. **GLSL은 Column-Major**: GLSL의 모든 행렬은 column-major 저장 순서를 사용합니다. 행렬 생성자와 인덱싱 시 항상 이를 고려해야 합니다.

2. **디버깅 시 축별 테스트**: 특정 축(X, Y, Z)에서만 문제가 발생하면 행렬 전치 문제일 가능성이 높습니다.

3. **3DGS 파이프라인 이해**:
   - Quaternion → Rotation Matrix → Scale → 3D Covariance
   - World Space → View Space → 2D Projection (Jacobian)
   - 각 단계에서 행렬 convention을 일관되게 유지해야 합니다.
