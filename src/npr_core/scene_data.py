"""
SceneData: Lightweight scene representation using NumPy arrays

Replaces List[Gaussian2D] with pure arrays for 40-80× better performance.
No object creation overhead during stroke placement.
"""

import numpy as np
from typing import List, Optional


class SceneData:
    """
    Scene representation using NumPy arrays (no Gaussian2D objects)

    Stores Gaussians as parallel arrays for efficient batch operations.
    40-80× faster than creating Gaussian2D objects.
    """

    def __init__(self):
        """Initialize empty scene"""
        self.positions = np.empty((0, 3), dtype=np.float32)
        self.rotations = np.empty((0, 4), dtype=np.float32)
        self.scales = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.float32)
        self.opacities = np.empty((0,), dtype=np.float32)
        self.count = 0

    def add_gaussians_batch(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        scales: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ):
        """
        Add batch of Gaussians from arrays

        Args:
            positions: (N, M, 3) or (K, 3) array of positions
            rotations: (N, M, 4) or (K, 4) array of quaternions
            scales: (N, M, 3) or (K, 3) array of scales
            colors: (N, M, 3) or (K, 3) array of colors
            opacities: (N, M) or (K,) array of opacities
        """
        # Flatten to (K, ...) if needed
        positions_flat = positions.reshape(-1, 3)
        rotations_flat = rotations.reshape(-1, 4)
        scales_flat = scales.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        opacities_flat = opacities.reshape(-1)

        # Append to existing arrays
        if self.count == 0:
            self.positions = positions_flat
            self.rotations = rotations_flat
            self.scales = scales_flat
            self.colors = colors_flat
            self.opacities = opacities_flat
        else:
            self.positions = np.vstack([self.positions, positions_flat])
            self.rotations = np.vstack([self.rotations, rotations_flat])
            self.scales = np.vstack([self.scales, scales_flat])
            self.colors = np.vstack([self.colors, colors_flat])
            self.opacities = np.concatenate([self.opacities, opacities_flat])

        self.count = len(self.positions)

    def clear(self):
        """Clear all Gaussians"""
        self.__init__()

    def extend(self, gaussians: List):
        """
        Add list of Gaussian2D objects (for backward compatibility)

        Args:
            gaussians: List of Gaussian2D objects

        Note: This is slower than add_gaussians_batch() as it requires
        object-to-array conversion. Use add_gaussians_batch() directly
        when possible for better performance.
        """
        if len(gaussians) == 0:
            return

        # Convert Gaussian2D objects to arrays
        from .gaussian import Gaussian2D

        positions = np.array([g.position for g in gaussians], dtype=np.float32)
        rotations = np.array([g.rotation for g in gaussians], dtype=np.float32)
        scales = np.array([g.scale for g in gaussians], dtype=np.float32)
        colors = np.array([g.color for g in gaussians], dtype=np.float32)
        opacities = np.array([g.opacity for g in gaussians], dtype=np.float32)

        # Add to scene
        self.add_gaussians_batch(positions, rotations, scales, colors, opacities)

    def to_gaussian_list(self):
        """
        Convert to List[Gaussian2D] for compatibility

        WARNING: This creates objects and is slow (defeats the purpose).
        Only use for legacy code that requires Gaussian2D objects.

        Returns:
            List of Gaussian2D objects
        """
        from .gaussian import Gaussian2D

        return [
            Gaussian2D(
                position=self.positions[i],
                rotation=self.rotations[i],
                scale=self.scales[i],
                opacity=self.opacities[i],
                color=self.colors[i],
            )
            for i in range(self.count)
        ]

    def __len__(self):
        """Return number of Gaussians"""
        return self.count

    def __delitem__(self, key):
        """
        Delete Gaussians by index or slice

        Args:
            key: int index or slice object

        Examples:
            del scene[10:]  # Delete from index 10 to end
            del scene[5:10]  # Delete indices 5-9
        """
        if isinstance(key, slice):
            # Convert slice to range of indices to keep
            start, stop, step = key.indices(self.count)

            # For deletion, we want to keep everything NOT in the slice
            if start == 0 and stop == self.count:
                # Delete everything
                self.clear()
            elif start == 0:
                # Delete from beginning: keep [stop:]
                self.positions = self.positions[stop:].copy()
                self.rotations = self.rotations[stop:].copy()
                self.scales = self.scales[stop:].copy()
                self.colors = self.colors[stop:].copy()
                self.opacities = self.opacities[stop:].copy()
                self.count = len(self.positions)
            elif stop == self.count:
                # Delete from start to end: keep [:start]
                self.positions = self.positions[:start].copy()
                self.rotations = self.rotations[:start].copy()
                self.scales = self.scales[:start].copy()
                self.colors = self.colors[:start].copy()
                self.opacities = self.opacities[:start].copy()
                self.count = len(self.positions)
            else:
                # Delete middle section: keep [:start] + [stop:]
                self.positions = np.vstack(
                    [self.positions[:start], self.positions[stop:]]
                )
                self.rotations = np.vstack(
                    [self.rotations[:start], self.rotations[stop:]]
                )
                self.scales = np.vstack([self.scales[:start], self.scales[stop:]])
                self.colors = np.vstack([self.colors[:start], self.colors[stop:]])
                self.opacities = np.concatenate(
                    [self.opacities[:start], self.opacities[stop:]]
                )
                self.count = len(self.positions)
        elif isinstance(key, int):
            # Delete single index
            if key < 0:
                key = self.count + key
            if key < 0 or key >= self.count:
                raise IndexError(
                    f"Index {key} out of range for SceneData with {self.count} Gaussians"
                )

            # Keep everything except index key
            mask = np.ones(self.count, dtype=bool)
            mask[key] = False

            self.positions = self.positions[mask].copy()
            self.rotations = self.rotations[mask].copy()
            self.scales = self.scales[mask].copy()
            self.colors = self.colors[mask].copy()
            self.opacities = self.opacities[mask].copy()
            self.count = len(self.positions)
        else:
            raise TypeError(
                f"SceneData indices must be integers or slices, not {type(key).__name__}"
            )

    def __repr__(self):
        return f"SceneData(gaussians={self.count})"


# ============================================
# Global Active SceneData Accessor
# ============================================
_active_scene_data: Optional[SceneData] = None


def get_active_scene_data() -> Optional[SceneData]:
    """Get the currently active SceneData instance (used by VR timer)."""
    return _active_scene_data


def set_active_scene_data(scene_data: Optional[SceneData]) -> None:
    """Set the active SceneData instance (called when painting starts)."""
    global _active_scene_data
    _active_scene_data = scene_data

