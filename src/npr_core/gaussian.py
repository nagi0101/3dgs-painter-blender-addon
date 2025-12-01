"""
Gaussian2D: 2D 평면에 제한된 3D Gaussian Splat
z=0 평면에서 동작하도록 3DGS를 단순화
"""

import numpy as np
from typing import Optional, Tuple
import copy


class Gaussian2D:
    """
    2D Gaussian Splat representation

    3D Gaussian Splatting을 z=0 평면에 제한하여 2D 회화 효과 구현
    각 Gaussian은 위치, 크기, 회전, 투명도, 색상으로 정의됨
    """

    def __init__(
        self,
        position: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        opacity: float,
        color: np.ndarray,
        sh_coeffs: Optional[np.ndarray] = None,
    ):
        """
        Args:
            position: 3D position (x, y, z) - z는 0으로 고정됨
            scale: Scale in 3 axes (sx, sy, sz) - sz는 최소값으로 설정
            rotation: Quaternion (x, y, z, w) for orientation
            opacity: Opacity value [0, 1]
            color: RGB color [0, 1]
            sh_coeffs: Optional Spherical Harmonics coefficients
        """
        # Position: z=0으로 강제
        self.position = np.array([position[0], position[1], 0.0], dtype=np.float32)

        # Scale: z축은 매우 작게 (2D처럼 보이도록)
        self.scale = np.array([scale[0], scale[1], 1e-4], dtype=np.float32)

        # Rotation: quaternion (x, y, z, w)
        self.rotation = np.array(rotation, dtype=np.float32)
        self._normalize_quaternion()

        # Opacity
        self.opacity = np.clip(opacity, 0.0, 1.0)

        # Color: RGB
        self.color = np.array(color, dtype=np.float32)

        # Spherical Harmonics (optional, for view-dependent effects)
        self.sh_coeffs = sh_coeffs if sh_coeffs is not None else None

    def _normalize_quaternion(self):
        """Quaternion 정규화"""
        norm = np.linalg.norm(self.rotation)
        if norm > 1e-8:
            self.rotation /= norm
        else:
            # Default to identity rotation
            self.rotation = np.array([0, 0, 0, 1], dtype=np.float32)

    def to_dict(self) -> dict:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        data = {
            "position": self.position.tolist(),
            "scale": self.scale.tolist(),
            "rotation": self.rotation.tolist(),
            "opacity": float(self.opacity),
            "color": self.color.tolist(),
        }
        if self.sh_coeffs is not None:
            data["sh_coeffs"] = self.sh_coeffs.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Gaussian2D":
        """딕셔너리로부터 Gaussian2D 생성"""
        return cls(
            position=np.array(data["position"]),
            scale=np.array(data["scale"]),
            rotation=np.array(data["rotation"]),
            opacity=data["opacity"],
            color=np.array(data["color"]),
            sh_coeffs=np.array(data["sh_coeffs"]) if "sh_coeffs" in data else None,
        )

    def copy(self) -> "Gaussian2D":
        """Deep copy of the Gaussian"""
        return Gaussian2D(
            position=self.position.copy(),
            scale=self.scale.copy(),
            rotation=self.rotation.copy(),
            opacity=self.opacity,
            color=self.color.copy(),
            sh_coeffs=self.sh_coeffs.copy() if self.sh_coeffs is not None else None,
        )

    def transform(self, transform_matrix: np.ndarray) -> "Gaussian2D":
        """
        4x4 변환 행렬을 적용한 새 Gaussian 반환

        Args:
            transform_matrix: 4x4 homogeneous transformation matrix

        Returns:
            Transformed Gaussian2D
        """
        # Position 변환
        pos_homo = np.append(self.position, 1.0)
        new_pos = (transform_matrix @ pos_homo)[:3]

        # Rotation 변환 (3x3 rotation part)
        R = transform_matrix[:3, :3]
        new_rotation = self._apply_rotation_matrix_to_quaternion(R, self.rotation)

        # Scale은 유지 (rigid transform 가정)
        return Gaussian2D(
            position=new_pos,
            scale=self.scale.copy(),
            rotation=new_rotation,
            opacity=self.opacity,
            color=self.color.copy(),
            sh_coeffs=self.sh_coeffs.copy() if self.sh_coeffs is not None else None,
        )

    def _apply_rotation_matrix_to_quaternion(
        self, R: np.ndarray, q: np.ndarray
    ) -> np.ndarray:
        """
        회전 행렬을 quaternion에 적용

        Args:
            R: 3x3 rotation matrix
            q: quaternion (x, y, z, w)

        Returns:
            New quaternion
        """
        # Quaternion to rotation matrix
        R_q = self._quaternion_to_matrix(q)

        # Combine rotations
        R_new = R @ R_q

        # Matrix to quaternion
        return self._matrix_to_quaternion(R_new)

    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        """
        Quaternion to 3x3 rotation matrix
        q = (x, y, z, w)
        """
        x, y, z, w = q

        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        3x3 rotation matrix to quaternion
        Returns (x, y, z, w)
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([x, y, z, w], dtype=np.float32)

    def compute_covariance_2d(
        self, view_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        2D covariance matrix 계산 (렌더링용)

        3DGS 논문의 2D projection 공식을 단순화:
        Σ' = J Σ J^T

        여기서는 orthographic projection 가정

        Args:
            view_matrix: Optional 4x4 view matrix (사용 안함, orthographic이므로)

        Returns:
            2x2 covariance matrix
        """
        # 3D covariance matrix 계산
        # Σ = R S S^T R^T
        R = self._quaternion_to_matrix(self.rotation)
        S = np.diag(self.scale)

        Sigma_3d = R @ S @ S.T @ R.T

        # Orthographic projection: z축 제거
        # 단순히 상위 2x2 부분만 사용
        Sigma_2d = Sigma_3d[:2, :2]

        return Sigma_2d

    def get_ellipse_parameters(self) -> Tuple[float, float, float]:
        """
        2D ellipse 파라미터 계산 (시각화용)

        Returns:
            (semi_major_axis, semi_minor_axis, rotation_angle_radians)
        """
        cov = self.compute_covariance_2d()

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Semi-axes (2σ for 95% confidence)
        semi_major = 2.0 * np.sqrt(eigenvalues[0])
        semi_minor = 2.0 * np.sqrt(eigenvalues[1])

        # Rotation angle
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        return semi_major, semi_minor, angle

    def __repr__(self) -> str:
        return (
            f"Gaussian2D(pos={self.position[:2]}, "
            f"scale={self.scale[:2]}, "
            f"opacity={self.opacity:.2f}, "
            f"color={self.color})"
        )


def create_test_gaussian(x: float = 0.0, y: float = 0.0) -> Gaussian2D:
    """테스트용 Gaussian 생성"""
    return Gaussian2D(
        position=np.array([x, y, 0.0]),
        scale=np.array([0.1, 0.1, 1e-4]),
        rotation=np.array([0, 0, 0, 1]),
        opacity=0.8,
        color=np.array([0.5, 0.5, 0.5]),
    )
