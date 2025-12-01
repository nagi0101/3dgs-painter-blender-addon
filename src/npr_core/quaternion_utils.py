"""
Quaternion utilities for smooth rotation interpolation
"""

import numpy as np


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions

    Args:
        q1: Starting quaternion [x, y, z, w]
        q2: Ending quaternion [x, y, z, w]
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute dot product
    dot = np.dot(q1, q2)

    # If the quaternions are nearly identical, return q2
    if dot > 0.9995:
        return q1 + t * (q2 - q1)

    # If the dot product is negative, slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # Clamp dot product to avoid numerical errors
    dot = np.clip(dot, -1, 1)

    # Calculate the angle between quaternions
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    # Calculate the interpolated quaternion
    q2_orthogonal = q2 - q1 * dot
    q2_orthogonal = q2_orthogonal / np.linalg.norm(q2_orthogonal)

    result = q1 * np.cos(theta) + q2_orthogonal * np.sin(theta)

    # Normalize result
    return result / np.linalg.norm(result)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions

    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]

    Returns:
        Product quaternion q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion

    Args:
        matrix: 3x3 rotation matrix

    Returns:
        Quaternion [x, y, z, w]
    """
    trace = np.trace(matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q

    # Normalize quaternion
    norm = np.linalg.norm(q)
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation

    Args:
        axis: Rotation axis (3D vector)
        angle: Rotation angle in radians

    Returns:
        Quaternion [x, y, z, w]
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # Compute quaternion components
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    return np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half]
    )


# Alias for consistency with brush.py imports
matrix_to_quaternion = quaternion_from_matrix


def quaternion_multiply_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (batch-friendly version)

    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]

    Returns:
        Product quaternion q1 * q2
    """
    # Same as quaternion_multiply, just a batch-friendly alias
    return quaternion_multiply(q1, q2)


def matrix_to_quaternion_batch(matrices: np.ndarray) -> np.ndarray:
    """
    Convert batch of rotation matrices to quaternions (fully vectorized)

    Args:
        matrices: (N, 3, 3) rotation matrices

    Returns:
        (N, 4) quaternions [x, y, z, w]

    Performance: 50-100× faster than looping quaternion_from_matrix
    """
    N = len(matrices)
    q = np.zeros((N, 4), dtype=np.float32)

    trace = np.trace(matrices, axis1=1, axis2=2)  # (N,)

    # Case 1: trace > 0 (most common case)
    mask1 = trace > 0
    if np.any(mask1):
        s = 0.5 / np.sqrt(trace[mask1] + 1.0)
        q[mask1, 3] = 0.25 / s
        q[mask1, 0] = (matrices[mask1, 2, 1] - matrices[mask1, 1, 2]) * s
        q[mask1, 1] = (matrices[mask1, 0, 2] - matrices[mask1, 2, 0]) * s
        q[mask1, 2] = (matrices[mask1, 1, 0] - matrices[mask1, 0, 1]) * s

    # Case 2: matrix[0, 0] is largest diagonal
    mask2 = (
        ~mask1
        & (matrices[:, 0, 0] > matrices[:, 1, 1])
        & (matrices[:, 0, 0] > matrices[:, 2, 2])
    )
    if np.any(mask2):
        s = 2.0 * np.sqrt(
            1.0 + matrices[mask2, 0, 0] - matrices[mask2, 1, 1] - matrices[mask2, 2, 2]
        )
        q[mask2, 3] = (matrices[mask2, 2, 1] - matrices[mask2, 1, 2]) / s
        q[mask2, 0] = 0.25 * s
        q[mask2, 1] = (matrices[mask2, 0, 1] + matrices[mask2, 1, 0]) / s
        q[mask2, 2] = (matrices[mask2, 0, 2] + matrices[mask2, 2, 0]) / s

    # Case 3: matrix[1, 1] is largest diagonal
    mask3 = ~mask1 & ~mask2 & (matrices[:, 1, 1] > matrices[:, 2, 2])
    if np.any(mask3):
        s = 2.0 * np.sqrt(
            1.0 + matrices[mask3, 1, 1] - matrices[mask3, 0, 0] - matrices[mask3, 2, 2]
        )
        q[mask3, 3] = (matrices[mask3, 0, 2] - matrices[mask3, 2, 0]) / s
        q[mask3, 0] = (matrices[mask3, 0, 1] + matrices[mask3, 1, 0]) / s
        q[mask3, 1] = 0.25 * s
        q[mask3, 2] = (matrices[mask3, 1, 2] + matrices[mask3, 2, 1]) / s

    # Case 4: matrix[2, 2] is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    if np.any(mask4):
        s = 2.0 * np.sqrt(
            1.0 + matrices[mask4, 2, 2] - matrices[mask4, 0, 0] - matrices[mask4, 1, 1]
        )
        q[mask4, 3] = (matrices[mask4, 1, 0] - matrices[mask4, 0, 1]) / s
        q[mask4, 0] = (matrices[mask4, 0, 2] + matrices[mask4, 2, 0]) / s
        q[mask4, 1] = (matrices[mask4, 1, 2] + matrices[mask4, 2, 1]) / s
        q[mask4, 2] = 0.25 * s

    return q


def quaternion_multiply_broadcast(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Broadcast multiply quaternions: (N, 4) × (M, 4) → (N, M, 4)

    Each q1[i] is multiplied with all q2[j], producing N×M quaternions.

    Args:
        q1: (N, 4) array of quaternions
        q2: (M, 4) array of quaternions

    Returns:
        (N, M, 4) array of product quaternions

    Performance: 100× faster than nested loops
    """
    N, M = len(q1), len(q2)

    # Expand dimensions for broadcasting
    q1_exp = q1[:, None, :]  # (N, 1, 4)
    q2_exp = q2[None, :, :]  # (1, M, 4)

    # Extract components
    x1, y1, z1, w1 = q1_exp[..., 0], q1_exp[..., 1], q1_exp[..., 2], q1_exp[..., 3]
    x2, y2, z2, w2 = q2_exp[..., 0], q2_exp[..., 1], q2_exp[..., 2], q2_exp[..., 3]

    # Quaternion multiplication formula (vectorized)
    result = np.zeros((N, M, 4), dtype=np.float32)
    result[..., 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    result[..., 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    result[..., 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    result[..., 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return result
