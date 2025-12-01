"""
BrushStamp and StrokePainter: Brush system for 3DGS painting

BrushStamp: Collection of Gaussians forming a brush pattern
StrokePainter: Applies brush along a stroke spline
"""

import numpy as np
from typing import List, Optional, Tuple
from .gaussian import Gaussian2D
from .spline import StrokeSpline
from .deformation import apply_deformation_to_stroke
from .inpainting import blend_overlapping_stamps
import copy


class BrushStamp:
    """
    Brush stamp: Collection of Gaussians with metadata

    브러시는 여러 Gaussian의 집합으로, 상대 위치를 유지하면서
    stroke를 따라 반복 배치됨
    """

    def __init__(self):
        """Initialize empty brush"""
        # Working gaussians (with runtime parameters applied)
        self.gaussians: List[Gaussian2D] = []

        # Base pattern (original, immutable)
        self.base_gaussians: List[Gaussian2D] = []

        self.center: np.ndarray = np.zeros(3, dtype=np.float32)

        # Orientation frame
        self.tangent: np.ndarray = np.array([1, 0, 0], dtype=np.float32)
        self.normal: np.ndarray = np.array([0, 0, 1], dtype=np.float32)
        self.binormal: np.ndarray = np.array([0, 1, 0], dtype=np.float32)

        # Runtime parameters (can be changed without modifying pattern)
        self.current_color: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.current_size_multiplier: float = 1.0
        self.current_global_opacity: float = 1.0  # Multiplies pattern opacity
        self.spacing: float = 0.3  # Arc length spacing between stamps

        # Metadata for brush library
        self.metadata: dict = {}

        # Bounding box diagonal (spatial extent of brush)
        self.size: float = 0.1  # Default for empty brush

    def create_circular_pattern(
        self,
        num_gaussians: int = 20,
        radius: float = 0.5,
        gaussian_scale: float = 0.05,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None  # Kept for backward compatibility, ignored
    ):
        """
        Create circular brush pattern

        Args:
            num_gaussians: Number of Gaussians in circle
            radius: Circle radius
            gaussian_scale: Individual Gaussian scale
            opacity: Pattern opacity (can have gradient/variation)
            color: Ignored (patterns are neutral, color applied at runtime)
        """
        # Clear existing pattern
        self.base_gaussians = []

        # Use neutral gray for pattern
        neutral_color = np.array([0.5, 0.5, 0.5])

        for i in range(num_gaussians):
            angle = 2 * np.pi * i / num_gaussians
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            g = Gaussian2D(
                position=np.array([x, y, 0.0]),
                scale=np.array([gaussian_scale, gaussian_scale, 1e-4]),
                rotation=np.array([0, 0, 0, 1]),
                opacity=opacity,
                color=neutral_color.copy()
            )
            self.base_gaussians.append(g)

        self._update_center()

        # Compute size from bounding box
        self.size = self._compute_size()

        # Apply default parameters to create working gaussians
        self.apply_parameters()

    def create_line_pattern(
        self,
        num_gaussians: int = 10,
        length: float = 1.0,
        thickness: float = 0.05,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None  # Kept for backward compatibility, ignored
    ):
        """
        Create line brush pattern (for stroke-like brushes)

        Args:
            num_gaussians: Number of Gaussians
            length: Line length
            thickness: Gaussian thickness
            opacity: Pattern opacity (can have gradient/variation)
            color: Ignored (patterns are neutral, color applied at runtime)
        """
        # Clear existing pattern
        self.base_gaussians = []

        # Use neutral gray for pattern
        neutral_color = np.array([0.5, 0.5, 0.5])

        for i in range(num_gaussians):
            # Distribute along line
            t = -0.5 + i / (num_gaussians - 1) if num_gaussians > 1 else 0
            x = t * length

            g = Gaussian2D(
                position=np.array([x, 0.0, 0.0]),
                scale=np.array([thickness, thickness, 1e-4]),
                rotation=np.array([0, 0, 0, 1]),
                opacity=opacity,
                color=neutral_color.copy()
            )
            self.base_gaussians.append(g)

        self._update_center()

        # Compute size from bounding box
        self.size = self._compute_size()

        # Apply default parameters to create working gaussians
        self.apply_parameters()

    def create_grid_pattern(
        self,
        grid_size: int = 5,
        spacing: float = 0.1,
        gaussian_scale: float = 0.04,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None  # Kept for backward compatibility, ignored
    ):
        """
        Create grid brush pattern

        Args:
            grid_size: NxN grid size
            spacing: Spacing between Gaussians
            gaussian_scale: Individual Gaussian scale
            opacity: Pattern opacity (can have gradient/variation)
            color: Ignored (patterns are neutral, color applied at runtime)
        """
        # Clear existing pattern
        self.base_gaussians = []

        # Use neutral gray for pattern
        neutral_color = np.array([0.5, 0.5, 0.5])

        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size // 2) * spacing
                y = (j - grid_size // 2) * spacing

                g = Gaussian2D(
                    position=np.array([x, y, 0.0]),
                    scale=np.array([gaussian_scale, gaussian_scale, 1e-4]),
                    rotation=np.array([0, 0, 0, 1]),
                    opacity=opacity,
                    color=neutral_color.copy()
                )
                self.base_gaussians.append(g)

        self._update_center()

        # Compute size from bounding box
        self.size = self._compute_size()

        # Apply default parameters to create working gaussians
        self.apply_parameters()

    def _update_center(self):
        """Update brush center as mean of Gaussian positions"""
        # Use base_gaussians if available, otherwise use working gaussians
        source_gaussians = self.base_gaussians if self.base_gaussians else self.gaussians

        if len(source_gaussians) == 0:
            self.center = np.zeros(3, dtype=np.float32)
            return

        positions = np.array([g.position for g in source_gaussians])
        self.center = np.mean(positions, axis=0)

    def place_at(
        self,
        position: np.ndarray,
        tangent: np.ndarray,
        normal: np.ndarray
    ) -> List[Gaussian2D]:
        """
        Place stamp at given position and orientation (rigid transform)

        Args:
            position: 3D world position
            tangent: Tangent direction (t)
            normal: Normal direction (n)

        Returns:
            List of transformed Gaussians
        """
        # Compute binormal: b = n × t
        binormal = np.cross(normal, tangent)
        binormal_norm = np.linalg.norm(binormal)
        if binormal_norm > 1e-8:
            binormal = binormal / binormal_norm
        else:
            binormal = self.binormal

        # Build rotation matrix: brush frame → world frame
        # Brush frame: (tangent_B, normal_B, binormal_B)
        # World frame: (tangent, normal, binormal)

        # Source frame (brush)
        R_src = np.column_stack([self.tangent, self.binormal, self.normal])

        # Target frame (world)
        R_tgt = np.column_stack([tangent, binormal, normal])

        # Rotation: R = R_tgt @ R_src^T
        R = R_tgt @ R_src.T

        # Build 4x4 transform matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = position - (R @ self.center)

        # Transform all Gaussians
        placed_gaussians = []
        for g in self.gaussians:
            g_new = g.transform(T)
            placed_gaussians.append(g_new)

        return placed_gaussians

    def place_at_batch(
        self,
        positions: np.ndarray,
        tangents: np.ndarray,
        normals: np.ndarray
    ) -> List[List[Gaussian2D]]:
        """
        Place stamp at multiple positions in batch (vectorized, 10-20× faster)

        Args:
            positions: (N, 3) array of 3D world positions
            tangents: (N, 3) array of tangent directions
            normals: (N, 3) array of normal directions

        Returns:
            List of N lists, each containing transformed Gaussians
        """
        N = len(positions)
        M = len(self.gaussians)

        if N == 0 or M == 0:
            return []

        # Compute binormals: b = n × t (vectorized)
        binormals = np.cross(normals, tangents)
        binormal_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
        # Normalize, fallback to default if too small
        mask = (binormal_norms[:, 0] > 1e-8)
        binormals[mask] = binormals[mask] / binormal_norms[mask]
        binormals[~mask] = self.binormal

        # Build rotation matrices for all placements: (N, 3, 3)
        R_src = np.column_stack([self.tangent, self.binormal, self.normal])  # (3, 3)

        # Target frames: (N, 3, 3) - vectorized construction
        R_tgt = np.stack([tangents, binormals, normals], axis=2)  # (N, 3, 3)

        # Batch rotation: R[i] = R_tgt[i] @ R_src.T
        R_batch = R_tgt @ R_src.T  # (N, 3, 3)

        # Extract Gaussian positions, rotations, etc. as arrays
        gaussian_positions = np.array([g.position for g in self.gaussians])  # (M, 3)
        gaussian_rotations = np.array([g.rotation for g in self.gaussians])  # (M, 4)
        gaussian_scales = np.array([g.scale for g in self.gaussians])  # (M, 3)
        gaussian_colors = np.array([g.color for g in self.gaussians])  # (M, 3)
        gaussian_opacities = np.array([g.opacity for g in self.gaussians])  # (M,)

        # Transform positions: (N, M, 3)
        # For each placement i: new_pos[i, j] = R[i] @ (pos[j] - center) + position[i]
        centered_positions = gaussian_positions - self.center  # (M, 3)
        # Einsum: (N, 3, 3) @ (M, 3) → (N, M, 3)
        rotated_positions = np.einsum('nij,mj->nmi', R_batch, centered_positions)
        transformed_positions = rotated_positions + positions[:, None, :]  # (N, M, 3)

        # Transform rotations (quaternions): (N, M, 4)
        # Convert rotation matrices to quaternions (batch) and compose (broadcast)
        from .quaternion_utils import matrix_to_quaternion_batch, quaternion_multiply_broadcast

        # Batch convert rotation matrices to quaternions
        R_quats = matrix_to_quaternion_batch(R_batch)  # (N, 4)

        # Broadcast multiply: (N, 4) × (M, 4) → (N, M, 4)
        transformed_rotations = quaternion_multiply_broadcast(R_quats, gaussian_rotations)  # (N, M, 4)

        # Create Gaussians batch (N × M Gaussians)
        # Use list comprehension instead of nested loops for better performance
        all_gaussians = [
            [
                Gaussian2D(
                    position=transformed_positions[i, j],
                    scale=gaussian_scales[j].copy(),
                    rotation=transformed_rotations[i, j],
                    opacity=gaussian_opacities[j],
                    color=gaussian_colors[j].copy()
                )
                for j in range(M)
            ]
            for i in range(N)
        ]

        return all_gaussians

    def place_at_batch_arrays(
        self,
        positions: np.ndarray,
        tangents: np.ndarray,
        normals: np.ndarray
    ) -> dict:
        """
        Place stamp at multiple positions and return arrays (no object creation)

        Args:
            positions: (N, 3) array of 3D world positions
            tangents: (N, 3) array of tangent directions
            normals: (N, 3) array of normal directions

        Returns:
            dict with keys 'positions', 'rotations', 'scales', 'colors', 'opacities'
            Each value is (N, M, ...) array where N=stamps, M=gaussians_per_stamp

        Performance: 40-80× faster than place_at_batch (no Gaussian2D object creation)
        """
        N = len(positions)
        M = len(self.gaussians)

        if N == 0 or M == 0:
            return {
                'positions': np.empty((0, 0, 3), dtype=np.float32),
                'rotations': np.empty((0, 0, 4), dtype=np.float32),
                'scales': np.empty((0, 0, 3), dtype=np.float32),
                'colors': np.empty((0, 0, 3), dtype=np.float32),
                'opacities': np.empty((0, 0), dtype=np.float32)
            }

        # Compute binormals: b = n × t (vectorized)
        binormals = np.cross(normals, tangents)
        binormal_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
        mask = (binormal_norms[:, 0] > 1e-8)
        binormals[mask] = binormals[mask] / binormal_norms[mask]
        binormals[~mask] = self.binormal

        # Build rotation matrices for all placements: (N, 3, 3)
        R_src = np.column_stack([self.tangent, self.binormal, self.normal])

        # Target frames: (N, 3, 3) - vectorized
        R_tgt = np.stack([tangents, binormals, normals], axis=2)

        # Batch rotation
        R_batch = R_tgt @ R_src.T

        # Extract Gaussian data as arrays
        gaussian_positions = np.array([g.position for g in self.gaussians])
        gaussian_rotations = np.array([g.rotation for g in self.gaussians])
        gaussian_scales = np.array([g.scale for g in self.gaussians])
        gaussian_colors = np.array([g.color for g in self.gaussians])
        gaussian_opacities = np.array([g.opacity for g in self.gaussians])

        # Transform positions: (N, M, 3)
        centered_positions = gaussian_positions - self.center
        rotated_positions = np.einsum('nij,mj->nmi', R_batch, centered_positions)
        transformed_positions = rotated_positions + positions[:, None, :]

        # Transform rotations: (N, M, 4)
        from .quaternion_utils import matrix_to_quaternion_batch, quaternion_multiply_broadcast

        R_quats = matrix_to_quaternion_batch(R_batch)
        transformed_rotations = quaternion_multiply_broadcast(R_quats, gaussian_rotations)

        # Broadcast scales, colors, opacities to (N, M, ...)
        transformed_scales = np.broadcast_to(gaussian_scales, (N, M, 3)).copy()
        transformed_colors = np.broadcast_to(gaussian_colors, (N, M, 3)).copy()
        transformed_opacities = np.broadcast_to(gaussian_opacities, (N, M)).copy()

        return {
            'positions': transformed_positions,
            'rotations': transformed_rotations,
            'scales': transformed_scales,
            'colors': transformed_colors,
            'opacities': transformed_opacities
        }

    def add_gaussian(self, gaussian: Gaussian2D):
        """Add a Gaussian to the brush pattern"""
        # Add to base pattern
        self.base_gaussians.append(gaussian)
        self._update_center()
        # Recreate working gaussians with current parameters
        self.apply_parameters()

    def apply_parameters(self, color: Optional[np.ndarray] = None,
                         size_multiplier: Optional[float] = None,
                         global_opacity: Optional[float] = None,
                         spacing: Optional[float] = None):
        """
        Apply runtime parameters to base pattern.
        Creates working gaussians from base pattern with parameters applied.

        Args:
            color: RGB color to apply (None keeps current)
            size_multiplier: Scale multiplier (None keeps current)
            global_opacity: Global opacity multiplier (None keeps current)
            spacing: Stamp spacing (None keeps current)
        """
        # Update current parameters if provided
        if color is not None:
            self.current_color = np.array(color, dtype=np.float32)
        if size_multiplier is not None:
            self.current_size_multiplier = size_multiplier
        if global_opacity is not None:
            self.current_global_opacity = global_opacity
        if spacing is not None:
            self.spacing = spacing

        # Apply parameters to create working gaussians
        self.gaussians = []
        for base_g in self.base_gaussians:
            g = base_g.copy()

            # Apply color with luminance-preserving tinting
            # base_g.color stores grayscale luminance pattern [lum, lum, lum]
            # Extract luminance (since R=G=B in grayscale, just use first channel)
            pattern_luminance = base_g.color[0]
            # Tint: runtime color × pattern luminance (preserves texture brightness variations)
            g.color = self.current_color * pattern_luminance

            # Apply size multiplier
            g.scale = base_g.scale * self.current_size_multiplier
            # Apply global opacity (multiply with pattern opacity)
            g.opacity = base_g.opacity * self.current_global_opacity
            self.gaussians.append(g)

    def set_color(self, color: np.ndarray):
        """Set color for all Gaussians (legacy method, uses apply_parameters)"""
        self.apply_parameters(color=color)

    def set_opacity(self, opacity: float):
        """Set global opacity multiplier (legacy method, uses apply_parameters)"""
        self.apply_parameters(global_opacity=opacity)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of brush

        Returns:
            (min_pos, max_pos) - 3D bounds
        """
        if len(self.gaussians) == 0:
            return np.zeros(3), np.zeros(3)

        positions = np.array([g.position for g in self.gaussians])
        return positions.min(axis=0), positions.max(axis=0)

    def _compute_size(self) -> float:
        """
        Compute bounding box diagonal from current base_gaussians

        Returns:
            Diagonal length of bounding box (0.1 if empty)
        """
        if len(self.base_gaussians) == 0:
            return 0.1

        positions = np.array([g.position for g in self.base_gaussians])
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        return float(np.linalg.norm(max_pos - min_pos))

    def copy(self) -> 'BrushStamp':
        """Deep copy of brush"""
        brush_copy = BrushStamp()
        brush_copy.gaussians = [g.copy() for g in self.gaussians]
        brush_copy.base_gaussians = [g.copy() for g in self.base_gaussians]
        brush_copy.center = self.center.copy()
        brush_copy.tangent = self.tangent.copy()
        brush_copy.normal = self.normal.copy()
        brush_copy.binormal = self.binormal.copy()
        brush_copy.current_color = self.current_color.copy()
        brush_copy.current_size_multiplier = self.current_size_multiplier
        brush_copy.current_global_opacity = self.current_global_opacity
        brush_copy.spacing = self.spacing
        brush_copy.metadata = self.metadata.copy()
        brush_copy.size = self.size
        return brush_copy

    def __len__(self) -> int:
        return len(self.gaussians)

    def __repr__(self) -> str:
        return f"BrushStamp(gaussians={len(self.gaussians)}, spacing={self.spacing})"

    def to_dict(self) -> dict:
        """Convert brush to dictionary for serialization"""
        from .brush_manager import BrushSerializer
        return BrushSerializer.brush_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'BrushStamp':
        """Create brush from dictionary"""
        from .brush_manager import BrushSerializer
        return BrushSerializer.dict_to_brush(data)


class StrokePainter:
    """
    Stroke painting logic

    브러시를 사용하여 spline을 따라 stamp를 배치
    """

    def __init__(
        self,
        brush: BrushStamp,
        scene_gaussians=None
    ):
        """
        Initialize painter

        Args:
            brush: Brush stamp to use
            scene_gaussians: Scene (SceneData or List[Gaussian2D] for compatibility)
        """
        self.brush = brush

        # Support both SceneData (fast path) and List[Gaussian2D] (legacy)
        if scene_gaussians is None:
            from .scene_data import SceneData
            self.scene = SceneData()
            self.use_arrays = True
        elif hasattr(scene_gaussians, 'add_gaussians_batch'):
            # SceneData object
            self.scene = scene_gaussians
            self.use_arrays = True
        else:
            # Legacy List[Gaussian2D]
            self.scene = scene_gaussians
            self.use_arrays = False

        self.current_stroke: Optional[StrokeSpline] = None
        # Store (gaussians, arc_length) tuples to preserve placement info
        self.placed_stamps: List[Tuple[List[Gaussian2D], float]] = []
        # Store original stamps for deformation (position, tangent, normal, arc_length)
        self.stamp_placements: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self.last_stamp_arc_length: float = 0.0
        self.current_stroke_start_index: int = 0
        # Store deformation flag for current stroke to ensure consistency
        self.enable_deformation_for_current_stroke: bool = False

    def start_stroke(self, position: np.ndarray, normal: np.ndarray, enable_deformation: bool = None):
        """
        Start a new stroke

        Args:
            position: 3D starting position
            normal: Surface normal
            enable_deformation: Enable deformation for this stroke (if None, uses config)
        """
        # Determine and store deformation flag for this stroke
        # This ensures consistency between stamp placement and finish_stroke
        if enable_deformation is None:
            from backend.config import config
            enable_deformation = config.ENABLE_DEFORMATION

        self.enable_deformation_for_current_stroke = enable_deformation

        # Create spline with 2D constraint for canvas painting
        self.current_stroke = StrokeSpline(force_2d=True)
        self.current_stroke.add_point(position, normal, threshold=0.0)
        self.placed_stamps = []
        self.stamp_placements = []
        self.last_stamp_arc_length = 0.0
        self.current_stroke_start_index = len(self.scene)

    def update_stroke(self, position: np.ndarray, normal: np.ndarray):
        """
        Update stroke with new point

        Args:
            position: 3D position
            normal: Surface normal
        """
        if self.current_stroke is None:
            self.start_stroke(position, normal)
            return

        # Add point to spline
        added = self.current_stroke.add_point(position, normal, threshold=self.brush.spacing * 0.1)

        if not added:
            return

        # Place new stamps along spline
        self._place_new_stamps()

    def _place_new_stamps(self):
        """Place stamps from last_stamp_arc_length to current end (batch-optimized)"""
        if self.current_stroke is None:
            return

        from backend.config import config

        total_length = self.current_stroke.total_arc_length
        spacing = self.brush.spacing

        # Calculate all stamp positions first
        arc_lengths = []
        arc_length = self.last_stamp_arc_length

        while arc_length <= total_length:
            arc_lengths.append(arc_length)
            arc_length += spacing

        # No new stamps to place
        if len(arc_lengths) == 0:
            return

        # Get positions, tangents, normals for all stamps (vectorized)
        positions = np.array([self.current_stroke.evaluate_at_arc_length(al) for al in arc_lengths])
        tangents = np.array([self.current_stroke.get_tangent_at_arc_length(al) for al in arc_lengths])
        normals = np.array([self.current_stroke.get_normal_at_arc_length(al) for al in arc_lengths])

        if self.enable_deformation_for_current_stroke:
            # Store placement info for deformation phase
            for i, al in enumerate(arc_lengths):
                self.stamp_placements.append((positions[i], tangents[i], normals[i], al))

                # Create placeholder stamps (original brush-local coordinates)
                placeholder_stamps = [g.copy() for g in self.brush.gaussians]
                self.placed_stamps.append((placeholder_stamps, al))

            # For immediate visualization, place them rigidly (batch)
            # This will be replaced in finish_stroke if deformation is enabled
            if self.use_arrays:
                # Fast path: Use arrays
                arrays = self.brush.place_at_batch_arrays(positions, tangents, normals)
                self.scene.add_gaussians_batch(
                    arrays['positions'],
                    arrays['rotations'],
                    arrays['scales'],
                    arrays['colors'],
                    arrays['opacities']
                )
            else:
                # Legacy path: Create objects
                temp_stamps_batch = self.brush.place_at_batch(positions, tangents, normals)
                for temp_stamps in temp_stamps_batch:
                    self.scene.extend(temp_stamps)
        else:
            # Normal rigid placement when deformation is disabled (batch)
            if self.use_arrays:
                # Fast path: Use arrays (40-80× faster)
                arrays = self.brush.place_at_batch_arrays(positions, tangents, normals)
                self.scene.add_gaussians_batch(
                    arrays['positions'],
                    arrays['rotations'],
                    arrays['scales'],
                    arrays['colors'],
                    arrays['opacities']
                )

                # Store per-stamp arrays in placed_stamps for potential later use
                M = len(self.brush.gaussians)
                for i, al in enumerate(arc_lengths):
                    start_idx = i * M
                    end_idx = (i + 1) * M
                    stamp_arrays = {
                        'positions': arrays['positions'][start_idx:end_idx].copy(),
                        'rotations': arrays['rotations'][start_idx:end_idx].copy(),
                        'scales': arrays['scales'][start_idx:end_idx].copy(),
                        'colors': arrays['colors'][start_idx:end_idx].copy(),
                        'opacities': arrays['opacities'][start_idx:end_idx].copy()
                    }
                    self.placed_stamps.append((stamp_arrays, al))
            else:
                # Legacy path: Create Gaussian2D objects
                stamps_batch = self.brush.place_at_batch(positions, tangents, normals)
                for i, stamp_gaussians in enumerate(stamps_batch):
                    self.placed_stamps.append((stamp_gaussians, arc_lengths[i]))
                    self.scene.extend(stamp_gaussians)

        # Update last stamp position
        self.last_stamp_arc_length = arc_length - spacing

    def finish_stroke(self, enable_deformation: bool = False, enable_inpainting: bool = False,
                     blend_strength: float = 0.3, global_inpainting: bool = False,
                     blend_mode: str = 'linear', color_blending: bool = False,
                     use_anisotropic: bool = False):
        """
        Finish current stroke

        Phase 3: Apply deformation and/or inpainting

        Args:
            enable_deformation: Apply non-rigid deformation
            enable_inpainting: Apply overlap inpainting
            blend_strength: Opacity reduction strength for inpainting (0.0-1.0)
            global_inpainting: Blend all overlapping pairs (not just consecutive)
            blend_mode: Blending falloff mode ('linear', 'smoothstep', 'gaussian')
            color_blending: Whether to blend colors as well as opacity
            use_anisotropic: Whether to use anisotropic (elliptical) distance
        """
        if self.current_stroke is None:
            return

        # Phase 3-2: Apply non-rigid deformation
        if enable_deformation and len(self.placed_stamps) > 0:
            print(f"[Deformation] Applying deformation to {len(self.placed_stamps)} stamps")

            # Remove temporarily placed rigid stamps from scene
            del self.scene[self.current_stroke_start_index:]

            # Apply deformation using original brush coordinates and placement info
            stamp_frame = (self.brush.tangent, self.brush.normal, self.brush.binormal)

            deformed_stamps = []

            # Use stamp_placements for deformation if available (when ENABLE_DEFORMATION was true during placement)
            if len(self.stamp_placements) > 0:
                # Extract original brush-local stamps and placement info
                stamps_only = [gaussians for gaussians, _ in self.placed_stamps]

                # GPU-accelerated batch deformation (50-100× faster)
                try:
                    import torch
                    if torch.cuda.is_available():
                        from .deformation_gpu import deform_all_stamps_batch_gpu

                        print(f"[Deformation] Using GPU batch deformation for {len(stamps_only)} stamps")
                        deformed_stamps = deform_all_stamps_batch_gpu(
                            all_stamps=stamps_only,
                            stamp_center=self.brush.center,
                            stamp_frame=stamp_frame,
                            spline=self.current_stroke,
                            stamp_placements=self.stamp_placements,
                            device='cuda'
                        )
                    else:
                        raise RuntimeError("CUDA not available")
                except Exception as e:
                    # Fallback to CPU deformation if GPU fails
                    print(f"[Deformation] GPU deformation failed ({e}), falling back to CPU")

                    for i, (stamp, (position, tangent, normal, arc_length)) in enumerate(zip(stamps_only, self.stamp_placements)):
                        from .deformation import deform_stamp_along_spline

                        # Apply deformation to original brush-local coordinates
                        deformed = deform_stamp_along_spline(
                            stamp,  # Original brush-local coordinates
                            self.brush.center,
                            stamp_frame,
                            self.current_stroke,
                            arc_length,
                            position,
                            tangent,
                            normal
                        )

                        deformed_stamps.append(deformed)
            else:
                # Fallback: Use placed stamps (old behavior, less accurate)
                print("[Deformation] Warning: Using fallback deformation (less accurate)")
                stamps_only = [gaussians for gaussians, _ in self.placed_stamps]
                arc_lengths = [arc_len for _, arc_len in self.placed_stamps]

                for i, (stamp, arc_length) in enumerate(zip(stamps_only, arc_lengths)):
                    from .deformation import deform_stamp_along_spline

                    # Get placement info from spline
                    position = self.current_stroke.evaluate_at_arc_length(arc_length)
                    tangent = self.current_stroke.get_tangent_at_arc_length(arc_length)
                    normal = self.current_stroke.get_normal_at_arc_length(arc_length)

                    # Apply deformation (less accurate as stamps may already be transformed)
                    deformed = deform_stamp_along_spline(
                        stamp,
                        self.brush.center,
                        stamp_frame,
                        self.current_stroke,
                        arc_length,
                        position,
                        tangent,
                        normal
                    )

                    deformed_stamps.append(deformed)

            print(f"[Deformation] ✓ Deformation applied, {len(deformed_stamps)} stamps deformed")

        # Phase 3-3: Apply inpainting
        if enable_inpainting and len(self.placed_stamps) > 1:
            mode_str = "global" if global_inpainting else "consecutive"
            print(f"[Inpainting] Applying {mode_str} inpainting to {len(self.placed_stamps)} stamps (strength={blend_strength:.2f})")

            from backend.config import config
            from .inpainting import blend_overlapping_stamps_auto
            overlap_threshold = config.INPAINT_OVERLAP_THRESHOLD
            use_anisotropic = config.INPAINT_ANISOTROPIC if hasattr(config, 'INPAINT_ANISOTROPIC') else False

            if enable_deformation and len(deformed_stamps) > 0:
                # Deformation was applied - blend deformed_stamps before adding to scene
                print(f"[Inpainting] Blending deformed stamps (mode={blend_mode}, color={color_blending}, anisotropic={use_anisotropic})")
                blend_overlapping_stamps_auto(deformed_stamps, overlap_threshold, blend_strength,
                                            global_inpainting, blend_mode, color_blending, use_anisotropic)
                stamps_to_add = deformed_stamps
            else:
                # No deformation - blend the placed_stamps directly
                stamps_only = [gaussians for gaussians, _ in self.placed_stamps]
                blend_overlapping_stamps_auto(stamps_only, overlap_threshold, blend_strength,
                                            global_inpainting, blend_mode, color_blending, use_anisotropic)
                stamps_to_add = stamps_only

            print(f"[Inpainting] ✓ Inpainting applied")
        elif enable_deformation and len(deformed_stamps) > 0:
            # Deformation without inpainting - just use deformed stamps
            stamps_to_add = deformed_stamps
        else:
            # No deformation, no inpainting - nothing to add (stamps already in scene from rigid placement)
            stamps_to_add = []

        # Add processed stamps to scene
        if len(stamps_to_add) > 0:
            total_gaussians = 0
            for stamp in stamps_to_add:
                self.scene.extend(stamp)
                total_gaussians += len(stamp)
            print(f"[Stroke] Added {total_gaussians} Gaussians to scene")

        # Reset
        self.current_stroke = None
        self.placed_stamps = []
        self.stamp_placements = []
        self.last_stamp_arc_length = 0.0

    def get_stroke_gaussians(self) -> List[Gaussian2D]:
        """
        Get all Gaussians from current stroke

        Returns:
            List of Gaussians in current stroke
        """
        result = []
        for gaussians, _ in self.placed_stamps:
            result.extend(gaussians)
        return result

    def clear_scene(self):
        """Clear all scene Gaussians"""
        self.scene.clear()
        self.placed_stamps = []


def test_brush():
    """Test brush functionality"""
    # Create circular brush
    brush = BrushStamp()
    brush.create_circular_pattern(num_gaussians=20, radius=0.3)
    print(f"Created brush: {brush}")

    # Test placement
    position = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    placed = brush.place_at(position, tangent, normal)
    print(f"Placed {len(placed)} Gaussians")

    # Test painter
    painter = StrokePainter(brush)
    painter.start_stroke(np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1]))
    painter.update_stroke(np.array([0.5, 0.2, 0.0]), np.array([0, 0, 1]))
    painter.update_stroke(np.array([1.0, 0.1, 0.0]), np.array([0, 0, 1]))
    painter.finish_stroke()

    print(f"Total scene Gaussians: {len(painter.scene)}")


if __name__ == "__main__":
    test_brush()
