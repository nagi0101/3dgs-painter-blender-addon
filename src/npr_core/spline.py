"""
StrokeSpline: 3D cubic spline with arc-length parameterization

마우스 입력 포인트를 smooth한 spline으로 변환하고,
브러시 스탬프를 등간격으로 배치하기 위한 arc-length 파라미터화 제공
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.integrate import cumtrapz


class StrokeSpline:
    """
    3D cubic spline for brush strokes

    Features:
    - Smooth interpolation of input points
    - Arc-length parameterization for uniform stamp spacing
    - Tangent and curvature computation for deformation
    - Optional 2D constraint for canvas painting
    """

    def __init__(self, force_2d: bool = True):
        """
        Initialize empty spline

        Args:
            force_2d: If True, constrain spline to XY plane (Z=0) with upward normal
        """
        self.control_points: List[np.ndarray] = []  # 3D points
        self.normals: List[np.ndarray] = []  # Surface normals at each point
        self.spline: Optional[CubicSpline] = None
        self.spline_x: Optional[object] = None  # Parametric spline (t -> x)
        self.spline_y: Optional[object] = None  # Parametric spline (t -> y)
        self.spline_z: Optional[object] = None  # Parametric spline (t -> z)
        self.arc_lengths: Optional[np.ndarray] = None
        self.total_arc_length: float = 0.0
        self.force_2d: bool = force_2d  # 2D painting mode
        self.control_point_arc_lengths: Optional[np.ndarray] = (
            None  # Arc lengths at control points
        )

    def add_point(
        self, point: np.ndarray, normal: np.ndarray, threshold: float = 0.01
    ) -> bool:
        """
        Add a point to the spline

        Args:
            point: 3D position (x, y, z) - will be constrained to 2D if force_2d=True
            normal: Surface normal at this point - will be forced to [0,0,1] if force_2d=True
            threshold: Minimum distance from previous point

        Returns:
            True if point was added, False if too close to previous
        """
        # Apply 2D constraints if enabled
        if self.force_2d:
            point = np.array(point, dtype=np.float32)
            point[2] = 0.0  # Force Z=0 (XY plane)
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Force upward normal

        if len(self.control_points) == 0:
            self.control_points.append(np.array(point, dtype=np.float32))
            self.normals.append(np.array(normal, dtype=np.float32))
            return True

        # Check distance threshold (use 2D distance if force_2d)
        if self.force_2d:
            distance = np.linalg.norm(point[:2] - self.control_points[-1][:2])
        else:
            distance = np.linalg.norm(point - self.control_points[-1])

        if distance < threshold:
            return False

        self.control_points.append(np.array(point, dtype=np.float32))
        self.normals.append(np.array(normal, dtype=np.float32))

        # Refit spline
        self._refit_spline()

        return True

    def _refit_spline(self):
        """Recompute the spline from control points"""
        if len(self.control_points) < 2:
            self.spline = None
            return

        points = np.array(self.control_points)

        if len(points) == 2:
            # Linear interpolation for 2 points
            t = np.array([0, 1])
            self.spline_x = lambda s: np.interp(s, t, points[:, 0])
            self.spline_y = lambda s: np.interp(s, t, points[:, 1])
            self.spline_z = lambda s: np.interp(s, t, points[:, 2])
        elif len(points) == 3:
            # Quadratic for 3 points
            t = np.linspace(0, 1, len(points))
            self.spline_x = CubicSpline(t, points[:, 0], bc_type="natural")
            self.spline_y = CubicSpline(t, points[:, 1], bc_type="natural")
            self.spline_z = CubicSpline(t, points[:, 2], bc_type="natural")
        else:
            # Cubic spline for 4+ points
            t = np.linspace(0, 1, len(points))
            self.spline_x = CubicSpline(t, points[:, 0], bc_type="natural")
            self.spline_y = CubicSpline(t, points[:, 1], bc_type="natural")
            self.spline_z = CubicSpline(t, points[:, 2], bc_type="natural")

        # Compute arc-length parameterization
        self._compute_arc_length()

    def _compute_arc_length(self):
        """
        Compute arc-length parameterization

        Arc length: L(t) = ∫₀ᵗ ||ds/du|| du

        저장:
        - arc_lengths: t 파라미터에 대응하는 arc length 배열
        - total_arc_length: 전체 spline의 arc length
        - control_point_arc_lengths: Arc lengths at control points
        """
        if self.spline_x is None:
            return

        # Sample points along spline
        t_samples = np.linspace(0, 1, 100)
        positions = np.array(
            [[self.spline_x(t), self.spline_y(t), self.spline_z(t)] for t in t_samples]
        )

        # Compute segment lengths
        segment_vectors = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)

        # Cumulative arc length
        self.arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        self.total_arc_length = self.arc_lengths[-1]

        # Store t_samples for inverse lookup
        self.t_samples = t_samples

        # Compute arc lengths at control points
        num_control = len(self.control_points)
        if num_control > 0:
            # Find t values that correspond to control points
            control_t_values = np.linspace(0, 1, num_control)
            self.control_point_arc_lengths = np.zeros(num_control)

            for i, t in enumerate(control_t_values):
                # Interpolate arc length at this t value
                self.control_point_arc_lengths[i] = np.interp(
                    t, t_samples, self.arc_lengths
                )

        # Invalidate GPU cache when spline geometry changes
        # This ensures deformation uses fresh spline data after refit
        self._gpu_cache = None

    def evaluate_at_t(self, t: float) -> np.ndarray:
        """
        Evaluate spline at parameter t

        Args:
            t: Parameter in [0, 1]

        Returns:
            3D position
        """
        if self.spline_x is None:
            if len(self.control_points) > 0:
                return self.control_points[0]
            return np.zeros(3)

        t = np.clip(t, 0.0, 1.0)

        return np.array(
            [self.spline_x(t), self.spline_y(t), self.spline_z(t)], dtype=np.float32
        )

    def evaluate_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Evaluate spline at given arc length

        Args:
            arc_length: Arc length from start

        Returns:
            3D position
        """
        if self.arc_lengths is None or self.total_arc_length == 0:
            return self.evaluate_at_t(0.0)

        # Clamp arc length
        arc_length = np.clip(arc_length, 0.0, self.total_arc_length)

        # Find t corresponding to this arc length
        t = np.interp(arc_length, self.arc_lengths, self.t_samples)

        return self.evaluate_at_t(t)

    def get_tangent_at_t(self, t: float) -> np.ndarray:
        """
        Get tangent vector at parameter t

        Args:
            t: Parameter in [0, 1]

        Returns:
            Normalized tangent vector
        """
        if self.spline_x is None:
            return np.array([1, 0, 0], dtype=np.float32)

        t = np.clip(t, 0.0, 1.0)

        # Derivative
        if isinstance(self.spline_x, CubicSpline):
            dx = self.spline_x.derivative()(t)
            dy = self.spline_y.derivative()(t)
            dz = self.spline_z.derivative()(t)
        else:
            # Numerical derivative for simple interpolation
            eps = 0.001
            t_plus = min(t + eps, 1.0)
            t_minus = max(t - eps, 0.0)

            pos_plus = self.evaluate_at_t(t_plus)
            pos_minus = self.evaluate_at_t(t_minus)

            derivative = (pos_plus - pos_minus) / (t_plus - t_minus)
            return derivative / (np.linalg.norm(derivative) + 1e-8)

        tangent = np.array([dx, dy, dz], dtype=np.float32)
        norm = np.linalg.norm(tangent)

        if norm < 1e-8:
            return np.array([1, 0, 0], dtype=np.float32)

        return tangent / norm

    def get_tangent_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Get tangent at given arc length

        Args:
            arc_length: Arc length from start

        Returns:
            Normalized tangent vector
        """
        if self.arc_lengths is None or self.total_arc_length == 0:
            return self.get_tangent_at_t(0.0)

        arc_length = np.clip(arc_length, 0.0, self.total_arc_length)
        t = np.interp(arc_length, self.arc_lengths, self.t_samples)

        return self.get_tangent_at_t(t)

    def get_normal_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Get surface normal at given arc length with SMOOTH INTERPOLATION

        Args:
            arc_length: Arc length from start

        Returns:
            Normalized normal vector (smoothly interpolated from control points)
        """
        if len(self.normals) == 0:
            return np.array([0, 0, 1], dtype=np.float32)

        if self.total_arc_length == 0:
            return self.normals[0]

        if len(self.normals) == 1:
            return self.normals[0]

        # Clip arc length to valid range
        arc_length = np.clip(arc_length, 0.0, self.total_arc_length)

        # Use actual arc lengths at control points (not uniform distribution)
        if self.control_point_arc_lengths is None or len(
            self.control_point_arc_lengths
        ) != len(self.normals):
            # Fallback to uniform if not computed properly
            num_points = len(self.normals)
            control_arc_lengths = np.linspace(0, self.total_arc_length, num_points)
        else:
            control_arc_lengths = self.control_point_arc_lengths

        # Find surrounding control points for interpolation
        idx_after = np.searchsorted(control_arc_lengths, arc_length)

        # Boundary cases
        if idx_after == 0:
            return self.normals[0]
        if idx_after >= len(self.normals):
            return self.normals[-1]

        # Linear interpolation between two nearest normals
        idx_before = idx_after - 1
        arc_before = control_arc_lengths[idx_before]
        arc_after = control_arc_lengths[idx_after]

        # Interpolation factor
        t = (arc_length - arc_before) / (arc_after - arc_before + 1e-10)
        t = np.clip(t, 0.0, 1.0)

        # Get normals
        n1 = self.normals[idx_before]
        n2 = self.normals[idx_after]

        # Use spherical linear interpolation for better results
        # This preserves the normal's magnitude better than linear interpolation
        dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)

        # If normals are nearly identical, use linear interpolation
        if dot_product > 0.9995:
            normal = (1.0 - t) * n1 + t * n2
        else:
            # Spherical linear interpolation
            theta = np.arccos(dot_product)
            sin_theta = np.sin(theta)
            if sin_theta > 1e-6:
                a = np.sin((1.0 - t) * theta) / sin_theta
                b = np.sin(t * theta) / sin_theta
                normal = a * n1 + b * n2
            else:
                # Fallback to linear if theta is too small
                normal = (1.0 - t) * n1 + t * n2

        # Normalize
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            # Fallback to first normal if degenerate
            return n1

        return normal / norm

    def get_binormal_at_arc_length(self, arc_length: float) -> np.ndarray:
        """
        Get binormal vector at given arc length with improved stability

        binormal = normal × tangent

        Args:
            arc_length: Arc length from start

        Returns:
            Normalized binormal vector
        """
        tangent = self.get_tangent_at_arc_length(arc_length)
        normal = self.get_normal_at_arc_length(arc_length)

        # Ensure tangent and normal are normalized
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        # Compute binormal
        binormal = np.cross(normal, tangent)
        norm = np.linalg.norm(binormal)

        if norm < 1e-6:
            # Degenerate case: normal and tangent are (nearly) parallel
            # This typically happens in 2D painting where normal is always [0,0,1]
            # and tangent is in XY plane

            # Check if this is a 2D case (normal pointing up, tangent in XY plane)
            if abs(normal[2]) > 0.9 and abs(tangent[2]) < 0.1:
                # 2D case: rotate tangent 90° counterclockwise in XY plane
                binormal = np.array([-tangent[1], tangent[0], 0.0], dtype=np.float32)
                norm = np.linalg.norm(binormal)
                if norm < 1e-8:
                    # Tangent is zero or vertical - use default
                    binormal = np.array([0, 1, 0], dtype=np.float32)
                    return binormal
            else:
                # 3D case or unknown: find any perpendicular vector
                # Use Gram-Schmidt to find perpendicular to both
                if abs(normal[0]) < 0.9:
                    temp = np.array([1, 0, 0], dtype=np.float32)
                else:
                    temp = np.array([0, 1, 0], dtype=np.float32)

                binormal = temp - np.dot(temp, normal) * normal
                binormal = binormal - np.dot(binormal, tangent) * tangent
                norm = np.linalg.norm(binormal)
                if norm < 1e-8:
                    binormal = np.array([0, 1, 0], dtype=np.float32)
                    return binormal

        return binormal / norm

    def get_frame_at_arc_length(
        self, arc_length: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get orientation frame (tangent, normal, binormal) at arc length

        Args:
            arc_length: Arc length from start

        Returns:
            (tangent, normal, binormal) - orthonormal frame
        """
        tangent = self.get_tangent_at_arc_length(arc_length)
        normal = self.get_normal_at_arc_length(arc_length)
        binormal = self.get_binormal_at_arc_length(arc_length)

        return tangent, normal, binormal

    def get_curvature_at_t(self, t: float) -> float:
        """
        Get curvature at parameter t

        κ = ||ds/dt × d²s/dt²|| / ||ds/dt||³

        Args:
            t: Parameter in [0, 1]

        Returns:
            Curvature value
        """
        if self.spline_x is None or not isinstance(self.spline_x, CubicSpline):
            return 0.0

        t = np.clip(t, 0.0, 1.0)

        # First derivative
        dx1 = self.spline_x.derivative(1)(t)
        dy1 = self.spline_y.derivative(1)(t)
        dz1 = self.spline_z.derivative(1)(t)
        first_deriv = np.array([dx1, dy1, dz1])

        # Second derivative
        dx2 = self.spline_x.derivative(2)(t)
        dy2 = self.spline_y.derivative(2)(t)
        dz2 = self.spline_z.derivative(2)(t)
        second_deriv = np.array([dx2, dy2, dz2])

        # Curvature formula
        cross = np.cross(first_deriv, second_deriv)
        numerator = np.linalg.norm(cross)
        denominator = np.linalg.norm(first_deriv) ** 3

        if denominator < 1e-8:
            return 0.0

        return numerator / denominator

    def sample_by_arc_length(
        self, spacing: float
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample points along spline at uniform arc-length intervals

        Args:
            spacing: Arc length spacing between samples

        Returns:
            List of (position, tangent, normal) tuples
        """
        if self.total_arc_length == 0 or spacing <= 0:
            return []

        samples = []
        arc_length = 0.0

        while arc_length <= self.total_arc_length:
            position = self.evaluate_at_arc_length(arc_length)
            tangent = self.get_tangent_at_arc_length(arc_length)
            normal = self.get_normal_at_arc_length(arc_length)

            samples.append((position, tangent, normal))

            arc_length += spacing

        return samples

    def get_num_points(self) -> int:
        """Get number of control points"""
        return len(self.control_points)

    def get_last_point(self) -> Optional[np.ndarray]:
        """Get last control point"""
        if len(self.control_points) == 0:
            return None
        return self.control_points[-1]

    def clear(self):
        """Clear all data"""
        self.control_points = []
        self.normals = []
        self.spline = None
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.arc_lengths = None
        self.total_arc_length = 0.0

    def __repr__(self) -> str:
        return (
            f"StrokeSpline(points={len(self.control_points)}, "
            f"length={self.total_arc_length:.3f})"
        )


def test_spline():
    """Test spline functionality"""
    spline = StrokeSpline()

    # Add some test points
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.3, 0.2, 0.0]),
        np.array([0.6, 0.1, 0.0]),
        np.array([1.0, 0.3, 0.0]),
    ]

    normal = np.array([0, 0, 1])

    for p in points:
        spline.add_point(p, normal, threshold=0.0)

    print(f"Spline: {spline}")
    print(f"Total arc length: {spline.total_arc_length:.4f}")

    # Sample
    samples = spline.sample_by_arc_length(spacing=0.1)
    print(f"Number of samples with spacing 0.1: {len(samples)}")

    # Test evaluation
    for i, (pos, tangent, normal) in enumerate(samples[:3]):
        print(f"Sample {i}: pos={pos}, tangent={tangent}")


if __name__ == "__main__":
    test_spline()
