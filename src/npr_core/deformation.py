"""
Non-rigid deformation for brush stamps

논문 Section 3.3 구현:
Per-Gaussian 좌표 변환 및 방향 조정을 통해
spline 곡률을 따라 자연스러운 변형 적용
"""

import numpy as np
from typing import List, Tuple
from .gaussian import Gaussian2D
from .spline import StrokeSpline


def compute_rotation_matrix_from_frames(
    from_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    to_frame: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Compute rotation matrix to transform from one frame to another

    Args:
        from_frame: (tangent, normal, binormal) source frame
        to_frame: (tangent, normal, binormal) target frame

    Returns:
        3x3 rotation matrix
    """
    # Build source matrix (column vectors)
    t_from, n_from, b_from = from_frame
    R_from = np.column_stack([t_from, n_from, b_from])

    # Build target matrix
    t_to, n_to, b_to = to_frame
    R_to = np.column_stack([t_to, n_to, b_to])

    # Rotation: R = R_to @ R_from^T
    R = R_to @ R_from.T

    return R


def deform_stamp_along_spline(
    stamp_gaussians: List[Gaussian2D],
    stamp_center: np.ndarray,
    stamp_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    spline: StrokeSpline,
    arc_length_param: float,
    placement_position: np.ndarray = None,
    placement_tangent: np.ndarray = None,
    placement_normal: np.ndarray = None
) -> List[Gaussian2D]:
    """
    Apply non-rigid deformation to stamp along spline

    논문의 "Per-Gaussian Coordinate Transform" 및
    "Orientation Adjustment" 알고리즘 구현

    Args:
        stamp_gaussians: List of Gaussians in stamp (brush-local coordinates)
        stamp_center: Stamp center position (in brush frame)
        stamp_frame: (tangent, normal, binormal) of stamp (brush frame)
        spline: Stroke spline
        arc_length_param: Arc length where stamp is placed
        placement_position: Position on spline where stamp is placed
        placement_tangent: Tangent at placement position
        placement_normal: Normal at placement position

    Returns:
        List of deformed Gaussians
    """
    if len(stamp_gaussians) == 0:
        return []

    t_B, n_B, b_B = stamp_frame

    # If placement info not provided, get from spline
    if placement_position is None:
        placement_position = spline.evaluate_at_arc_length(arc_length_param)
    if placement_tangent is None:
        placement_tangent = spline.get_tangent_at_arc_length(arc_length_param)
    if placement_normal is None:
        placement_normal = spline.get_normal_at_arc_length(arc_length_param)

    deformed = []

    for g in stamp_gaussians:
        # 1. Get local coordinates in brush frame (g.position is already in brush-local)
        local_pos = g.position - stamp_center

        # Project onto brush axes
        x = np.dot(local_pos, t_B)  # Along tangent (longitudinal offset)
        y = np.dot(local_pos, b_B)  # Along binormal (lateral offset)
        z = np.dot(local_pos, n_B)  # Along normal (height offset)

        # 2. Convert tangential offset to arc length offset
        # This is an approximation - assumes locally linear spline
        # For more accuracy, could use numerical search
        arc_length_offset = x  # Simplified: assume unit speed parameterization

        # 3. Compute spline frame at deformed location
        # Move along spline by the tangential offset
        a_new = arc_length_param + arc_length_offset

        # Clamp to valid range
        a_new = np.clip(a_new, 0.0, spline.total_arc_length)

        # Get position and frame at new arc length
        pos_on_spline = spline.evaluate_at_arc_length(a_new)
        t_s, n_s, b_s = spline.get_frame_at_arc_length(a_new)

        # 4. Compute deformed position
        # Position on spline + offsets along binormal and normal
        new_pos = pos_on_spline + y * b_s + z * n_s

        # 5. Compute rotation for this Gaussian
        # Rotate from brush frame to spline frame at this location
        R = compute_rotation_matrix_from_frames(
            from_frame=(t_B, n_B, b_B),
            to_frame=(t_s, n_s, b_s)
        )

        # Apply rotation to Gaussian's quaternion
        new_rotation = apply_rotation_matrix_to_quaternion(R, g.rotation)

        # 6. Optional: Apply curvature-based scaling
        # Could adjust scale based on local curvature here
        new_scale = g.scale.copy()

        # 7. Create deformed Gaussian
        g_new = Gaussian2D(
            position=new_pos,
            scale=new_scale,
            rotation=new_rotation,
            opacity=g.opacity,
            color=g.color.copy(),
            sh_coeffs=g.sh_coeffs.copy() if g.sh_coeffs is not None else None
        )

        deformed.append(g_new)

    return deformed


def apply_rotation_matrix_to_quaternion(
    R: np.ndarray,
    q: np.ndarray
) -> np.ndarray:
    """
    Apply rotation matrix to quaternion

    Args:
        R: 3x3 rotation matrix
        q: quaternion (x, y, z, w)

    Returns:
        New quaternion
    """
    # Convert quaternion to rotation matrix
    R_q = quaternion_to_matrix(q)

    # Combine rotations
    R_new = R @ R_q

    # Convert back to quaternion
    return matrix_to_quaternion(R_new)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix

    Args:
        q: quaternion (x, y, z, w)

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion

    Args:
        R: 3x3 rotation matrix

    Returns:
        quaternion (x, y, z, w)
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


def apply_deformation_to_stroke(
    placed_stamps: List[List[Gaussian2D]],
    stamp_center: np.ndarray,
    stamp_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    spline: StrokeSpline,
    spacing: float
) -> List[List[Gaussian2D]]:
    """
    Apply deformation to all stamps in a stroke

    Args:
        placed_stamps: List of stamp Gaussian lists
        stamp_center: Stamp center
        stamp_frame: Stamp orientation frame
        spline: Stroke spline
        spacing: Stamp spacing

    Returns:
        List of deformed stamp Gaussian lists
    """
    deformed_stamps = []

    for i, stamp in enumerate(placed_stamps):
        arc_length = i * spacing

        deformed = deform_stamp_along_spline(
            stamp,
            stamp_center,
            stamp_frame,
            spline,
            arc_length
        )

        deformed_stamps.append(deformed)

    return deformed_stamps


def test_deformation():
    """Test deformation functions"""
    from .brush import BrushStamp
    from .spline import StrokeSpline

    print("Testing deformation...")

    # Create brush
    brush = BrushStamp()
    brush.create_circular_pattern(num_gaussians=10)

    # Create spline
    spline = StrokeSpline()
    spline.add_point(np.array([0, 0, 0]), np.array([0, 0, 1]), threshold=0.0)
    spline.add_point(np.array([0.5, 0.5, 0]), np.array([0, 0, 1]), threshold=0.0)
    spline.add_point(np.array([1, 0, 0]), np.array([0, 0, 1]), threshold=0.0)

    print(f"Spline: {spline}")

    # Place stamp at start
    stamp_frame = (brush.tangent, brush.normal, brush.binormal)
    stamp = brush.place_at(
        np.array([0, 0, 0]),
        brush.tangent,
        brush.normal
    )

    print(f"Original stamp: {len(stamp)} Gaussians")

    # Deform
    deformed = deform_stamp_along_spline(
        stamp,
        brush.center,
        stamp_frame,
        spline,
        arc_length_param=0.0
    )

    print(f"Deformed stamp: {len(deformed)} Gaussians")

    # Check positions changed
    for i, (g_orig, g_def) in enumerate(zip(stamp, deformed)):
        dist = np.linalg.norm(g_def.position - g_orig.position)
        if i < 3:
            print(f"  Gaussian {i}: moved {dist:.4f}")


if __name__ == "__main__":
    test_deformation()
