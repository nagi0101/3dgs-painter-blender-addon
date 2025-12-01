"""
Inpainting for overlapping brush stamps

Phase 3-3: Simple opacity-based blending for overlapping regions
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .gaussian import Gaussian2D
import time


# ============================================================================
# Performance Optimized Functions
# ============================================================================

def compute_stamp_bounding_box(stamp: List[Gaussian2D]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box for a stamp

    Args:
        stamp: List of Gaussians in the stamp

    Returns:
        min_corner: (2,) array of minimum x, y coordinates
        max_corner: (2,) array of maximum x, y coordinates
    """
    if len(stamp) == 0:
        return np.array([0, 0]), np.array([0, 0])

    # Get all positions and scales
    positions = np.array([g.position[:2] for g in stamp])
    scales = np.array([g.scale[:2] for g in stamp])

    # Compute extent for each Gaussian (3 sigma covers 99.7% of distribution)
    extents = scales * 3.0

    # Compute min/max including extents
    min_corner = np.min(positions - extents, axis=0)
    max_corner = np.max(positions + extents, axis=0)

    return min_corner, max_corner


def compute_stamp_bounding_box_array(stamp_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bounding box for array-based stamp

    Args:
        stamp_data: Dict with 'positions' and 'scales' arrays

    Returns:
        min_corner: (2,) array
        max_corner: (2,) array
    """
    if len(stamp_data['positions']) == 0:
        return np.array([0, 0]), np.array([0, 0])

    positions = stamp_data['positions'][:, :2]
    scales = stamp_data['scales'][:, :2] if 'scales' in stamp_data else np.ones_like(positions) * 0.05

    # 3 sigma extent
    extents = scales * 3.0

    min_corner = np.min(positions - extents, axis=0)
    max_corner = np.max(positions + extents, axis=0)

    return min_corner, max_corner


def bounding_boxes_overlap(
    min1: np.ndarray, max1: np.ndarray,
    min2: np.ndarray, max2: np.ndarray,
    margin: float = 0.0
) -> bool:
    """
    Check if two axis-aligned bounding boxes overlap

    Args:
        min1, max1: First box corners
        min2, max2: Second box corners
        margin: Additional margin to add

    Returns:
        True if boxes overlap
    """
    # Add margin
    min1_m = min1 - margin
    max1_m = max1 + margin
    min2_m = min2 - margin
    max2_m = max2 + margin

    # Check overlap in each dimension
    return not (max1_m[0] < min2_m[0] or max2_m[0] < min1_m[0] or
                max1_m[1] < min2_m[1] or max2_m[1] < min1_m[1])


def find_overlapping_gaussians_optimized(
    stamp1: List[Gaussian2D],
    stamp2: List[Gaussian2D],
    threshold: float = 0.1,
    use_anisotropic: bool = False,
    max_overlaps: int = 1000
) -> List[Tuple[int, int, float]]:
    """
    Optimized version with bounding box pre-check and early termination

    Args:
        stamp1, stamp2: Stamp Gaussians
        threshold: Distance threshold
        use_anisotropic: Whether to use anisotropic distance
        max_overlaps: Maximum overlaps to detect (for early termination)

    Returns:
        List of (index1, index2, distance) tuples
    """
    if len(stamp1) == 0 or len(stamp2) == 0:
        return []

    # Compute stamp bounding boxes
    min1, max1 = compute_stamp_bounding_box(stamp1)
    min2, max2 = compute_stamp_bounding_box(stamp2)

    # Quick rejection test
    if not bounding_boxes_overlap(min1, max1, min2, max2, threshold):
        return []

    overlaps = []

    # Precompute positions for vectorization potential
    pos1 = np.array([g.position[:2] for g in stamp1])
    pos2 = np.array([g.position[:2] for g in stamp2])

    # Use spatial grid if stamps are large
    if len(stamp1) * len(stamp2) > 10000:  # Threshold for using spatial acceleration
        # Build spatial grid for stamp2
        grid_size = threshold * 2
        grid = {}

        for j, g2 in enumerate(stamp2):
            grid_x = int(g2.position[0] / grid_size)
            grid_y = int(g2.position[1] / grid_size)
            key = (grid_x, grid_y)

            if key not in grid:
                grid[key] = []
            grid[key].append((j, g2))

        # Query grid for each Gaussian in stamp1
        for i, g1 in enumerate(stamp1):
            grid_x = int(g1.position[0] / grid_size)
            grid_y = int(g1.position[1] / grid_size)

            # Check neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (grid_x + dx, grid_y + dy)
                    if key in grid:
                        for j, g2 in grid[key]:
                            if use_anisotropic:
                                dist = compute_anisotropic_distance(g1, g2)
                            else:
                                dist = np.linalg.norm(pos1[i] - pos2[j])

                            if dist < threshold:
                                overlaps.append((i, j, dist))

                                # Early termination
                                if len(overlaps) >= max_overlaps:
                                    return overlaps
    else:
        # Original nested loop for small stamps
        for i, g1 in enumerate(stamp1):
            for j, g2 in enumerate(stamp2):
                if use_anisotropic:
                    dist = compute_anisotropic_distance(g1, g2)
                else:
                    dist = np.linalg.norm(pos1[i] - pos2[j])

                if dist < threshold:
                    overlaps.append((i, j, dist))

                    # Early termination
                    if len(overlaps) >= max_overlaps:
                        return overlaps

    return overlaps


def find_overlapping_gaussians_vectorized(
    pos1: np.ndarray,
    pos2: np.ndarray,
    threshold: float = 0.1,
    chunk_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully vectorized overlap detection with chunking to avoid memory explosion

    Args:
        pos1: (M, 2) positions of first stamp
        pos2: (N, 2) positions of second stamp
        threshold: Distance threshold
        chunk_size: Process in chunks to limit memory usage

    Returns:
        indices1, indices2, distances arrays
    """
    M, N = len(pos1), len(pos2)

    # If small enough, do it all at once
    if M * N < 1000000:  # 1M comparisons ~8MB memory
        diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        mask = distances < threshold
        indices = np.where(mask)
        return indices[0], indices[1], distances[mask]

    # Otherwise, process in chunks
    all_idx1, all_idx2, all_dists = [], [], []

    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        chunk1 = pos1[i_start:i_end]

        # Compute distances for this chunk
        diff = chunk1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        mask = distances < threshold

        # Get indices
        chunk_idx = np.where(mask)
        if len(chunk_idx[0]) > 0:
            all_idx1.append(chunk_idx[0] + i_start)
            all_idx2.append(chunk_idx[1])
            all_dists.append(distances[mask])

    if len(all_idx1) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.concatenate(all_idx1), np.concatenate(all_idx2), np.concatenate(all_dists)


# ============================================================================
# Original Functions (keeping for compatibility)
# ============================================================================

def find_overlapping_gaussians(
    stamp1: List[Gaussian2D],
    stamp2: List[Gaussian2D],
    threshold: float = 0.1,
    use_anisotropic: bool = False
) -> List[Tuple[int, int, float]]:
    """
    Find overlapping Gaussian pairs between two stamps

    Args:
        stamp1: First stamp Gaussians
        stamp2: Second stamp Gaussians
        threshold: Distance threshold for overlap detection
        use_anisotropic: Whether to use anisotropic (elliptical) distance

    Returns:
        List of (index1, index2, distance) tuples for overlapping pairs
    """
    overlaps = []

    for i, g1 in enumerate(stamp1):
        for j, g2 in enumerate(stamp2):
            if use_anisotropic:
                # Anisotropic distance considering Gaussian shape
                dist = compute_anisotropic_distance(g1, g2)
            else:
                # Simple Euclidean distance (ignore z)
                dist = np.linalg.norm(g1.position[:2] - g2.position[:2])

            if dist < threshold:
                overlaps.append((i, j, dist))

    return overlaps


def compute_anisotropic_distance(g1: Gaussian2D, g2: Gaussian2D) -> float:
    """
    Compute anisotropic distance between two Gaussians considering their elliptical shapes

    Uses a Mahalanobis-like distance metric that accounts for scale and rotation.
    This gives more accurate overlap detection for elongated or rotated Gaussians.

    Args:
        g1, g2: Gaussian2D objects

    Returns:
        Anisotropic distance (smaller = more overlap)
    """
    # Get 2D positions
    pos1 = g1.position[:2]
    pos2 = g2.position[:2]
    diff = pos2 - pos1

    # Get 2D scales (ignore Z)
    scale1 = g1.scale[:2]
    scale2 = g2.scale[:2]

    # Average scales for symmetric distance
    avg_scale = (scale1 + scale2) / 2.0

    # For simplicity, we'll use the average scale as an ellipse
    # and compute the normalized distance
    # This is a simplified anisotropic distance
    # Full implementation would involve rotation matrices from quaternions

    # Normalize difference by average scale
    normalized_diff = diff / (avg_scale + 1e-6)

    # Compute the distance in normalized space
    anisotropic_dist = np.linalg.norm(normalized_diff)

    # Scale back to world units (approximate)
    # We want the distance to be comparable to the isotropic case
    return anisotropic_dist * np.mean(avg_scale)


def compute_overlap_factor(distance: float, threshold: float, mode: str = 'linear') -> float:
    """
    Compute overlap factor based on distance

    Args:
        distance: Distance between Gaussians
        threshold: Overlap threshold
        mode: Blending mode ('linear', 'smoothstep', 'gaussian')

    Returns:
        Overlap factor in [0, 1], where 1 = complete overlap
    """
    if distance >= threshold:
        return 0.0

    normalized = distance / threshold  # 0 at center, 1 at threshold

    if mode == 'linear':
        # Linear falloff: 1.0 at distance=0, 0.0 at distance=threshold
        return max(0.0, 1.0 - normalized)
    elif mode == 'smoothstep':
        # Smooth S-curve for more natural blending
        x = 1.0 - normalized  # Invert so 1 at center, 0 at edge
        return x * x * (3.0 - 2.0 * x)
    elif mode == 'gaussian':
        # Gaussian-like falloff
        # Using exp(-x^2) where x is scaled to give good falloff
        return np.exp(-4.0 * normalized * normalized)
    else:
        # Default to linear
        return max(0.0, 1.0 - normalized)


def blend_overlapping_stamps(
    stamps: List[List[Gaussian2D]],
    overlap_threshold: float = 0.1,
    blend_strength: float = 0.3,
    blend_mode: str = 'linear',
    enable_color_blending: bool = False,
    use_anisotropic: bool = False,
    use_optimized: bool = True,
    verbose: bool = False
) -> None:
    """
    Blend overlapping regions between consecutive stamps

    Modifies Gaussians in-place by reducing opacity and optionally blending colors

    Now handles multiple overlaps correctly: uses maximum reduction instead of
    compounding multiple reductions.

    Args:
        stamps: List of stamp Gaussian lists
        overlap_threshold: Distance threshold for overlap detection
        blend_strength: Maximum opacity reduction factor (0.0-1.0)
        blend_mode: Blending falloff mode ('linear', 'smoothstep', 'gaussian')
        enable_color_blending: Whether to blend colors as well as opacity
        use_anisotropic: Whether to use anisotropic (elliptical) distance
    """
    if len(stamps) < 2:
        return

    print(f"[Inpainting] Blending {len(stamps)} stamps with threshold={overlap_threshold:.3f}")

    start_time = time.time() if verbose else None
    total_overlaps = 0
    skipped_pairs = 0

    # Track opacity reductions for each Gaussian (stamp_idx, gaussian_idx) -> reduction
    opacity_reductions = {}
    # Track color blends for each Gaussian (stamp_idx, gaussian_idx) -> (color, weight)
    color_blends = {}

    # Process consecutive stamp pairs
    for i in range(len(stamps) - 1):
        stamp1 = stamps[i]
        stamp2 = stamps[i + 1]

        # Find overlapping Gaussians
        if use_optimized:
            overlaps = find_overlapping_gaussians_optimized(
                stamp1, stamp2, overlap_threshold, use_anisotropic, max_overlaps=1000
            )
        else:
            overlaps = find_overlapping_gaussians(stamp1, stamp2, overlap_threshold, use_anisotropic)

        if len(overlaps) == 0:
            continue

        total_overlaps += len(overlaps)

        # Collect reduction factors (don't apply yet)
        for idx1, idx2, dist in overlaps:
            # Compute overlap factor
            overlap_factor = compute_overlap_factor(dist, overlap_threshold, blend_mode)

            # Calculate reduction: opacity *= (1 - blend_strength * overlap_factor)
            # At complete overlap (factor=1.0), reduce by blend_strength
            # At threshold (factor=0.0), no reduction
            reduction = 1.0 - blend_strength * overlap_factor

            # Track maximum reduction for each Gaussian (avoid compound reduction)
            key1 = (i, idx1)
            key2 = (i + 1, idx2)
            opacity_reductions[key1] = min(opacity_reductions.get(key1, 1.0), reduction)
            opacity_reductions[key2] = min(opacity_reductions.get(key2, 1.0), reduction)

            # Optional: Track color blending
            if enable_color_blending and overlap_factor > 0:
                # Weight for color blending (less aggressive than opacity)
                color_weight = overlap_factor * blend_strength * 0.5

                # Store color blend information for later averaging
                g1_color = stamp1[idx1].color
                g2_color = stamp2[idx2].color

                # Accumulate blended colors (will average later)
                if key1 not in color_blends:
                    color_blends[key1] = []
                if key2 not in color_blends:
                    color_blends[key2] = []

                color_blends[key1].append((g2_color, color_weight))
                color_blends[key2].append((g1_color, color_weight))

    # Apply opacity reductions
    for (stamp_idx, gaussian_idx), reduction in opacity_reductions.items():
        stamps[stamp_idx][gaussian_idx].opacity *= reduction

    # Apply color blending (average all color contributions)
    if enable_color_blending:
        for (stamp_idx, gaussian_idx), blend_list in color_blends.items():
            if len(blend_list) > 0:
                # Get original color
                original_color = stamps[stamp_idx][gaussian_idx].color

                # Average all color contributions
                total_weight = 0
                blended_color = np.zeros(3)
                for color, weight in blend_list:
                    blended_color += color * weight
                    total_weight += weight

                # Normalize and combine with original
                if total_weight > 0:
                    avg_weight = min(total_weight / len(blend_list), 1.0)  # Cap at 1.0
                    final_color = original_color * (1 - avg_weight) + blended_color / len(blend_list)
                    stamps[stamp_idx][gaussian_idx].color = final_color

    if verbose and start_time is not None:
        elapsed = time.time() - start_time
        total_gaussians = sum(len(s) for s in stamps)
        print(f"[Inpainting] ✓ Blending complete: {total_overlaps} overlapping pairs processed")
        print(f"[Inpainting]   Time: {elapsed:.3f}s, Gaussians: {total_gaussians}, Mode: {'optimized' if use_optimized else 'original'}")
    else:
        print(f"[Inpainting] ✓ Blending complete: {total_overlaps} overlapping pairs processed")


def inpaint_stroke(
    stamps: List[List[Gaussian2D]],
    overlap_threshold: float = 0.1,
    blend_strength: float = 0.3
) -> List[List[Gaussian2D]]:
    """
    Apply inpainting to a complete stroke

    This is a convenience function that applies blending to stamps
    and returns them (though blending is done in-place)

    Args:
        stamps: List of stamp Gaussian lists
        overlap_threshold: Distance threshold for overlap detection
        blend_strength: Opacity reduction strength (0.0-1.0)

    Returns:
        Same stamps list (modified in-place)
    """
    blend_overlapping_stamps(stamps, overlap_threshold, blend_strength)
    return stamps


def test_inpainting():
    """Test inpainting functions"""
    print("Testing inpainting...")

    # Create two simple stamps
    stamp1 = [
        Gaussian2D(
            position=np.array([0.0, 0.0, 0.0]),
            scale=np.array([0.05, 0.05, 1e-4]),
            rotation=np.array([0, 0, 0, 1]),
            opacity=0.8,
            color=np.array([1.0, 0.0, 0.0])
        )
    ]

    stamp2 = [
        Gaussian2D(
            position=np.array([0.05, 0.0, 0.0]),  # 0.05 units away
            scale=np.array([0.05, 0.05, 1e-4]),
            rotation=np.array([0, 0, 0, 1]),
            opacity=0.8,
            color=np.array([0.0, 1.0, 0.0])
        )
    ]

    print(f"Stamp 1 opacity before: {stamp1[0].opacity}")
    print(f"Stamp 2 opacity before: {stamp2[0].opacity}")

    # Find overlaps
    overlaps = find_overlapping_gaussians(stamp1, stamp2, threshold=0.1)
    print(f"Found {len(overlaps)} overlaps")

    # Blend
    blend_overlapping_stamps([stamp1, stamp2], overlap_threshold=0.1)

    print(f"Stamp 1 opacity after: {stamp1[0].opacity}")
    print(f"Stamp 2 opacity after: {stamp2[0].opacity}")


def blend_overlapping_stamps_arrays(
    stamp_arrays: List[Dict[str, np.ndarray]],
    overlap_threshold: float = 0.1,
    blend_strength: float = 0.3,
    blend_mode: str = 'linear',
    enable_color_blending: bool = False
) -> None:
    """
    Blend overlapping regions between consecutive stamps (array-based version)

    Optimized for SceneData format using vectorized numpy operations.
    Modifies opacity arrays in-place, and optionally colors.

    Now handles multiple overlaps correctly: uses maximum reduction instead of
    compounding multiple reductions.

    Args:
        stamp_arrays: List of stamp dicts with keys:
            - 'positions': (M, 3) array
            - 'opacities': (M,) array
            - 'colors': (M, 3) array (optional, for color blending)
            - other keys preserved but not modified
        overlap_threshold: Distance threshold for overlap detection
        blend_strength: Maximum opacity reduction factor (0.0-1.0)
        blend_mode: Blending falloff mode ('linear', 'smoothstep', 'gaussian')
        enable_color_blending: Whether to blend colors as well as opacity
    """
    if len(stamp_arrays) < 2:
        return

    print(f"[Inpainting] Blending {len(stamp_arrays)} array stamps with threshold={overlap_threshold:.3f}")

    total_overlaps = 0

    # Track minimum opacity multipliers for each Gaussian
    opacity_multipliers = []
    for stamp in stamp_arrays:
        opacity_multipliers.append(np.ones(len(stamp['opacities'])))

    # Track color blend accumulations if needed
    if enable_color_blending:
        color_accums = []
        color_counts = []
        for stamp in stamp_arrays:
            if 'colors' in stamp:
                color_accums.append(np.zeros_like(stamp['colors']))
                color_counts.append(np.zeros(len(stamp['colors'])))
            else:
                color_accums.append(None)
                color_counts.append(None)

    # Process consecutive stamp pairs
    for i in range(len(stamp_arrays) - 1):
        stamp1 = stamp_arrays[i]
        stamp2 = stamp_arrays[i + 1]

        # Check bounding boxes first for quick rejection
        min1, max1 = compute_stamp_bounding_box_array(stamp1)
        min2, max2 = compute_stamp_bounding_box_array(stamp2)

        if not bounding_boxes_overlap(min1, max1, min2, max2, overlap_threshold):
            continue

        # Extract positions (2D only, ignore Z)
        pos1 = stamp1['positions'][:, :2]  # (M, 2)
        pos2 = stamp2['positions'][:, :2]  # (N, 2)

        # Use optimized vectorized overlap detection with chunking
        indices1, indices2, overlap_dists = find_overlapping_gaussians_vectorized(
            pos1, pos2, overlap_threshold, chunk_size=1000
        )

        if len(indices1) == 0:
            continue

        total_overlaps += len(indices1)

        # Vectorized overlap factor computation
        # overlap_dists already computed by find_overlapping_gaussians_vectorized
        normalized = overlap_dists / overlap_threshold

        # Apply blend mode
        if blend_mode == 'smoothstep':
            x = 1.0 - normalized
            overlap_factors = x * x * (3.0 - 2.0 * x)
        elif blend_mode == 'gaussian':
            overlap_factors = np.exp(-4.0 * normalized * normalized)
        else:  # linear
            overlap_factors = np.maximum(0.0, 1.0 - normalized)

        # Compute reductions
        reductions = 1.0 - blend_strength * overlap_factors

        # Track minimum multipliers (avoid compound reduction)
        # Group by unique indices and take minimum reduction
        for idx, red in zip(indices1, reductions):
            opacity_multipliers[i][idx] = min(opacity_multipliers[i][idx], red)
        for idx, red in zip(indices2, reductions):
            opacity_multipliers[i + 1][idx] = min(opacity_multipliers[i + 1][idx], red)

        # Optional: Accumulate color blending information
        if enable_color_blending and 'colors' in stamp1 and 'colors' in stamp2:
            color_weights = overlap_factors * blend_strength * 0.5

            # Accumulate color contributions
            colors1 = stamp1['colors'][indices1]  # (K, 3)
            colors2 = stamp2['colors'][indices2]  # (K, 3)

            # Add contributions for averaging later
            np.add.at(color_accums[i], indices1, colors2 * color_weights[:, np.newaxis])
            np.add.at(color_counts[i], indices1, color_weights)

            np.add.at(color_accums[i + 1], indices2, colors1 * color_weights[:, np.newaxis])
            np.add.at(color_counts[i + 1], indices2, color_weights)

    # Apply opacity reductions
    for i, stamp in enumerate(stamp_arrays):
        stamp['opacities'] *= opacity_multipliers[i]

    # Apply averaged color blending
    if enable_color_blending:
        for i, stamp in enumerate(stamp_arrays):
            if 'colors' in stamp and color_counts[i] is not None:
                # Only blend where we have contributions
                mask = color_counts[i] > 0
                if np.any(mask):
                    # Average the accumulated colors
                    avg_colors = np.zeros_like(stamp['colors'])
                    avg_colors[mask] = color_accums[i][mask] / color_counts[i][mask, np.newaxis]

                    # Blend with original colors
                    weights = np.minimum(color_counts[i], 1.0)  # Cap weights at 1.0
                    stamp['colors'] = stamp['colors'] * (1 - weights[:, np.newaxis]) + avg_colors * weights[:, np.newaxis]

    print(f"[Inpainting] ✓ Array blending complete: {total_overlaps} overlapping pairs processed")


def blend_all_overlapping_stamps(
    stamps: List[List[Gaussian2D]],
    overlap_threshold: float = 0.1,
    blend_strength: float = 0.3,
    blend_mode: str = 'linear',
    enable_color_blending: bool = False,
    use_anisotropic: bool = False,
    use_spatial_culling: bool = True
) -> None:
    """
    Blend ALL overlapping stamp pairs (not just consecutive)

    Useful for complex strokes with self-intersections or multiple overlapping strokes.
    Uses spatial culling to avoid O(N^2) Gaussian comparisons.

    Now handles multiple overlaps correctly: uses maximum reduction instead of
    compounding multiple reductions.

    Args:
        stamps: List of stamp Gaussian lists
        overlap_threshold: Distance threshold for overlap detection
        blend_strength: Opacity reduction strength (0.0-1.0)
        blend_mode: Blending falloff mode ('linear', 'smoothstep', 'gaussian')
        enable_color_blending: Whether to blend colors as well as opacity
        use_anisotropic: Whether to use anisotropic (elliptical) distance
        use_spatial_culling: Use stamp center distance to skip distant pairs
    """
    if len(stamps) < 2:
        return

    print(f"[Inpainting] Global blending: {len(stamps)} stamps")

    # Compute stamp centers for spatial culling
    stamp_centers = []
    if use_spatial_culling:
        for stamp in stamps:
            if len(stamp) > 0:
                positions = np.array([g.position[:2] for g in stamp])
                center = positions.mean(axis=0)
                stamp_centers.append(center)
            else:
                stamp_centers.append(np.array([0.0, 0.0]))

    total_overlaps = 0
    pairs_checked = 0

    # Track opacity reductions for each Gaussian (stamp_idx, gaussian_idx) -> reduction
    opacity_reductions = {}
    # Track color blends for each Gaussian (stamp_idx, gaussian_idx) -> (color, weight)
    color_blends = {}

    # Check all pairs (i < j)
    for i in range(len(stamps)):
        for j in range(i + 1, len(stamps)):
            stamp1 = stamps[i]
            stamp2 = stamps[j]

            if len(stamp1) == 0 or len(stamp2) == 0:
                continue

            # Spatial culling: skip if stamp centers are too far
            if use_spatial_culling:
                center_dist = np.linalg.norm(stamp_centers[i] - stamp_centers[j])
                # Heuristic: max_radius = threshold * average_stamp_size
                # Assuming average stamp radius ~10 Gaussians, threshold=0.1
                max_radius = overlap_threshold * 5
                if center_dist > max_radius:
                    continue

            pairs_checked += 1

            # Find overlapping Gaussians
            overlaps = find_overlapping_gaussians(stamp1, stamp2, overlap_threshold, use_anisotropic)

            if len(overlaps) == 0:
                continue

            total_overlaps += len(overlaps)

            # Collect reduction factors (don't apply yet)
            for idx1, idx2, dist in overlaps:
                overlap_factor = compute_overlap_factor(dist, overlap_threshold, blend_mode)
                reduction = 1.0 - blend_strength * overlap_factor

                # Track maximum reduction for each Gaussian
                key1 = (i, idx1)
                key2 = (j, idx2)
                opacity_reductions[key1] = min(opacity_reductions.get(key1, 1.0), reduction)
                opacity_reductions[key2] = min(opacity_reductions.get(key2, 1.0), reduction)

                # Optional: Track color blending
                if enable_color_blending and overlap_factor > 0:
                    color_weight = overlap_factor * blend_strength * 0.5

                    g1_color = stamp1[idx1].color
                    g2_color = stamp2[idx2].color

                    if key1 not in color_blends:
                        color_blends[key1] = []
                    if key2 not in color_blends:
                        color_blends[key2] = []

                    color_blends[key1].append((g2_color, color_weight))
                    color_blends[key2].append((g1_color, color_weight))

    # Apply opacity reductions
    for (stamp_idx, gaussian_idx), reduction in opacity_reductions.items():
        stamps[stamp_idx][gaussian_idx].opacity *= reduction

    # Apply color blending
    if enable_color_blending:
        for (stamp_idx, gaussian_idx), blend_list in color_blends.items():
            if len(blend_list) > 0:
                original_color = stamps[stamp_idx][gaussian_idx].color

                total_weight = 0
                blended_color = np.zeros(3)
                for color, weight in blend_list:
                    blended_color += color * weight
                    total_weight += weight

                if total_weight > 0:
                    avg_weight = min(total_weight / len(blend_list), 1.0)
                    final_color = original_color * (1 - avg_weight) + blended_color / len(blend_list)
                    stamps[stamp_idx][gaussian_idx].color = final_color

    print(f"[Inpainting] ✓ Global blending: {pairs_checked}/{len(stamps)*(len(stamps)-1)//2} pairs checked, {total_overlaps} overlaps")


def blend_overlapping_stamps_auto(
    stamps,
    overlap_threshold: float = 0.1,
    blend_strength: float = 0.3,
    global_mode: bool = False,
    blend_mode: str = 'linear',
    enable_color_blending: bool = False,
    use_anisotropic: bool = False
) -> None:
    """
    Auto-dispatch blending based on stamp format

    Detects whether stamps are List[List[Gaussian2D]] or List[Dict[str, np.ndarray]]
    and calls the appropriate blending function.

    Args:
        stamps: Either object-based or array-based stamps
        overlap_threshold: Distance threshold
        blend_strength: Opacity reduction factor
        global_mode: If True, blend all pairs (not just consecutive)
        blend_mode: Blending falloff mode ('linear', 'smoothstep', 'gaussian')
        enable_color_blending: Whether to blend colors as well as opacity
        use_anisotropic: Whether to use anisotropic (elliptical) distance
    """
    if len(stamps) == 0:
        return

    # Check format of first stamp
    if isinstance(stamps[0], dict):
        # Array-based format (SceneData)
        # TODO: Implement anisotropic for arrays (more complex)
        # For now, arrays use isotropic distance only
        blend_overlapping_stamps_arrays(stamps, overlap_threshold, blend_strength,
                                       blend_mode, enable_color_blending)
    else:
        # Object-based format (List[Gaussian2D])
        if global_mode:
            # Now passes all parameters to global blending
            blend_all_overlapping_stamps(stamps, overlap_threshold, blend_strength,
                                        blend_mode, enable_color_blending, use_anisotropic)
        else:
            blend_overlapping_stamps(stamps, overlap_threshold, blend_strength,
                                    blend_mode, enable_color_blending, use_anisotropic)


if __name__ == "__main__":
    test_inpainting()
