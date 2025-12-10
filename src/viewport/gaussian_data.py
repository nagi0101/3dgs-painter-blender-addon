# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 3DGS Painter Project

"""
GPU data management for Gaussian splatting viewport rendering.

Manages Gaussian data in GPU texture format (59 floats per gaussian).

Data layout:
[0-2]:   position (vec3)
[3-6]:   rotation quaternion (vec4, w,x,y,z)
[7-9]:   scale (vec3)
[10]:    opacity (float)
[11-58]: spherical harmonics (16 bands × 3 = 48 floats)
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..npr_core.scene_data import SceneData
    from ..npr_core.gaussian import Gaussian2D

# Constants
FLOATS_PER_GAUSSIAN = 59
SH_COEFFS_COUNT = 48  # 16 bands × 3 channels


class GaussianDataManager:
    """
    Manages Gaussian data in GPU 3D Texture format (59 floats per gaussian).
    
    Data layout per Gaussian (59 floats total):
    [0-2]:   position (vec3)
    [3-6]:   rotation quaternion (vec4: w, x, y, z)
    [7-9]:   scale (vec3)
    [10]:    opacity (float)
    [11-58]: spherical harmonics coefficients (16 bands × 3 = 48 floats)
    
    Attributes:
        texture: GPU texture object (created lazily)
        gaussian_count: Number of gaussians currently stored
        data_buffer: NumPy array cache for partial updates
        texture_dimensions: (width, height, depth) of the 3D texture
    """
    
    def __init__(self):
        self.texture = None  # Will be gpu.types.GPUTexture
        self.gaussian_count: int = 0
        self.data_buffer: Optional[np.ndarray] = None
        self.texture_dimensions: Tuple[int, int, int] = (0, 0, 0)
        self._max_texture_width: int = 2048  # Max texture dimension
    
    def update_from_scene_data(self, scene_data: "SceneData") -> bool:
        """
        Update GPU texture from SceneData.
        
        Args:
            scene_data: npr_core.scene_data.SceneData instance
            
        Returns:
            True if update successful, False otherwise
        """
        if scene_data.count == 0:
            self.clear()
            return True
        
        # Convert to 59-float stride format
        data = self._pack_from_scene_data(scene_data)
        self.gaussian_count = scene_data.count
        
        # Upload to GPU
        return self._upload_to_texture(data)
    
    def update_from_gaussians(self, gaussians: list) -> bool:
        """
        Update GPU texture from list of Gaussian2D objects.
        
        Args:
            gaussians: List of Gaussian2D instances
            
        Returns:
            True if update successful, False otherwise
        """
        if len(gaussians) == 0:
            self.clear()
            return True
        
        # Convert to 59-float stride format
        data = self._pack_from_gaussians(gaussians)
        self.gaussian_count = len(gaussians)
        
        # Upload to GPU
        return self._upload_to_texture(data)
    
    def _pack_from_scene_data(self, scene_data: "SceneData") -> np.ndarray:
        """
        Pack SceneData into 59-float stride format.
        
        Args:
            scene_data: SceneData with arrays (positions, rotations, scales, colors, opacities)
            
        Returns:
            np.ndarray: Shape (N, 59), dtype float32
        """
        N = scene_data.count
        data = np.zeros((N, FLOATS_PER_GAUSSIAN), dtype=np.float32)
        
        # Position [0-2]
        data[:, 0:3] = scene_data.positions
        
        # Rotation [3-6] (quaternion w, x, y, z)
        # SceneData stores as (x, y, z, w), need to reorder to (w, x, y, z)
        data[:, 3] = scene_data.rotations[:, 3]  # w
        data[:, 4:7] = scene_data.rotations[:, 0:3]  # x, y, z
        
        # Scale [7-9]
        data[:, 7:10] = scene_data.scales
        
        # Opacity [10]
        data[:, 10] = scene_data.opacities
        
        # Spherical Harmonics [11-58]
        # For degree 0: Use color as base SH coefficient
        # SH_C0 = 0.28209479177387814, so color ≈ SH_C0 * sh_coeff + 0.5
        # Inverse: sh_coeff = (color - 0.5) / SH_C0
        SH_C0 = 0.28209479177387814
        data[:, 11:14] = (scene_data.colors - 0.5) / SH_C0
        # Remaining SH coefficients [14-58] are zero (initialized above)
        
        return data
    
    def _pack_from_gaussians(self, gaussians: list) -> np.ndarray:
        """
        Pack list of Gaussian2D objects into 59-float stride format.
        
        Args:
            gaussians: List of Gaussian2D instances
            
        Returns:
            np.ndarray: Shape (N, 59), dtype float32
        """
        N = len(gaussians)
        data = np.zeros((N, FLOATS_PER_GAUSSIAN), dtype=np.float32)
        
        SH_C0 = 0.28209479177387814
        
        for i, g in enumerate(gaussians):
            # Position [0-2]
            data[i, 0:3] = g.position
            
            # Rotation [3-6] (quaternion w, x, y, z)
            # Gaussian2D stores as (x, y, z, w), reorder to (w, x, y, z)
            data[i, 3] = g.rotation[3]  # w
            data[i, 4:7] = g.rotation[0:3]  # x, y, z
            
            # Scale [7-9]
            data[i, 7:10] = g.scale
            
            # Opacity [10]
            data[i, 10] = g.opacity
            
            # Spherical Harmonics [11-58]
            if g.sh_coeffs is not None:
                # Use provided SH coefficients
                sh_len = min(len(g.sh_coeffs), SH_COEFFS_COUNT)
                data[i, 11:11+sh_len] = g.sh_coeffs[:sh_len]
            else:
                # Convert color to degree 0 SH
                data[i, 11:14] = (g.color - 0.5) / SH_C0
        
        return data
    
    def _upload_to_texture(self, data: np.ndarray) -> bool:
        """
        Upload packed data to GPU texture.
        
        Uses a 2D texture approach for better compatibility:
        - Width: up to max_texture_width
        - Height: ceil(N * 59 / width)
        
        Args:
            data: np.ndarray, shape (N, 59)
            
        Returns:
            True if upload successful
        """
        try:
            import gpu
            from gpu.types import GPUTexture, Buffer
        except ImportError:
            # Not in Blender context
            self.data_buffer = data
            return False
        
        N = data.shape[0]
        total_floats = N * FLOATS_PER_GAUSSIAN
        
        # Calculate texture dimensions
        # Use 2D texture for better compatibility
        width = min(total_floats, self._max_texture_width)
        height = (total_floats + width - 1) // width
        
        # Flatten and pad
        flat_data = data.flatten().astype(np.float32)
        required_size = width * height
        if len(flat_data) < required_size:
            flat_data = np.pad(flat_data, (0, required_size - len(flat_data)))
        else:
            flat_data = flat_data[:required_size]
        
        # Create GPU texture
        # Using R32F format (single-channel float)
        try:
            # Convert to buffer for GPU upload
            # Buffer expects: format, dimensions, data
            buffer = Buffer('FLOAT', required_size, flat_data.tolist())
            
            # GPUTexture expects: size (tuple), format, data
            self.texture = GPUTexture(
                (width, height),
                format='R32F',
                data=buffer
            )
            
            self.texture_dimensions = (width, height, 1)
            self.data_buffer = data
            
            print(f"[GaussianDataManager] Texture created: {width}x{height}, {N} gaussians")
            return True
            
        except Exception as e:
            print(f"[GaussianDataManager] Texture upload failed: {e}")
            import traceback
            traceback.print_exc()
            self.data_buffer = data
            return False
    
    def update_partial(self, start_idx: int, end_idx: int, 
                       scene_data: Optional["SceneData"] = None,
                       gaussians: Optional[list] = None) -> bool:
        """
        Update a subset of gaussians (for incremental painting).
        
        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            scene_data: SceneData with updated values, or
            gaussians: List of Gaussian2D objects
            
        Returns:
            True if update successful
        """
        if self.data_buffer is None:
            raise RuntimeError("Must call update_from_scene_data or update_from_gaussians first")
        
        if scene_data is not None:
            new_data = self._pack_from_scene_data(scene_data)
        elif gaussians is not None:
            new_data = self._pack_from_gaussians(gaussians)
        else:
            raise ValueError("Must provide either scene_data or gaussians")
        
        # Update cache
        count = end_idx - start_idx
        if count != new_data.shape[0]:
            raise ValueError(f"Data size mismatch: expected {count}, got {new_data.shape[0]}")
        
        self.data_buffer[start_idx:end_idx] = new_data
        
        # Re-upload entire texture
        # (Partial texture update is complex in Blender GPU API)
        return self._upload_to_texture(self.data_buffer)
    
    def append_gaussians(self, scene_data: Optional["SceneData"] = None,
                        gaussians: Optional[list] = None) -> bool:
        """
        Append new gaussians to existing data.
        
        Args:
            scene_data: SceneData with new gaussians, or
            gaussians: List of new Gaussian2D objects
            
        Returns:
            True if append successful
        """
        if scene_data is not None:
            new_data = self._pack_from_scene_data(scene_data)
        elif gaussians is not None:
            new_data = self._pack_from_gaussians(gaussians)
        else:
            raise ValueError("Must provide either scene_data or gaussians")
        
        if self.data_buffer is None or self.gaussian_count == 0:
            self.gaussian_count = new_data.shape[0]
            return self._upload_to_texture(new_data)
        
        # Concatenate with existing data
        self.data_buffer = np.vstack([self.data_buffer, new_data])
        self.gaussian_count = self.data_buffer.shape[0]
        
        return self._upload_to_texture(self.data_buffer)
    
    def clear(self):
        """Clear all gaussian data."""
        self.texture = None
        self.gaussian_count = 0
        self.data_buffer = None
        self.texture_dimensions = (0, 0, 0)
    
    def get_texture(self):
        """
        Get the GPU texture for shader binding.
        
        Returns:
            GPU texture object or None if not created
        """
        return self.texture
    
    def get_texture_info(self) -> dict:
        """
        Get texture information for shader uniforms.
        
        Returns:
            dict with texture_dimensions, gaussian_count, floats_per_gaussian
        """
        return {
            "texture_width": self.texture_dimensions[0],
            "texture_height": self.texture_dimensions[1],
            "gaussian_count": self.gaussian_count,
            "floats_per_gaussian": FLOATS_PER_GAUSSIAN,
        }

    def sort_and_update_texture(self, view_matrix: "Matrix"):
        """
        Sorts gaussians back-to-front based on the view matrix and updates the GPU texture.

        Args:
            view_matrix: 4x4 view matrix from the current viewport.

        Returns:
            The updated GPU texture object.
        """
        if self.data_buffer is None or self.gaussian_count == 0:
            return self.texture

        try:
            # Extract positions for sorting
            positions = self.data_buffer[:, 0:3]
            count = self.gaussian_count

            # Homogenize world positions (add w=1)
            pos_world_h = np.hstack((positions, np.ones((count, 1), dtype=np.float32)))

            # Transform to view space.
            # Blender's Matrix is column-major in memory layout.
            # When converted to numpy, it becomes a transposed representation of
            # what one might expect from a row-major perspective.
            # The operation `pos @ view_matrix.T` is correct here.
            pos_view_h = pos_world_h @ view_matrix.transposed()

            # Get view-space depth (Z coordinate)
            depths = pos_view_h[:, 2]

            # Get sorting indices. In OpenGL's right-handed view space, the camera
            # looks down the -Z axis, so farther objects have a more negative Z.
            # Sorting in ascending order gives the correct back-to-front sequence.
            sort_indices = np.argsort(depths)

            # Apply sorting to the entire data buffer
            sorted_data = self.data_buffer[sort_indices]
            
            # Re-upload the now-sorted data to the texture
            self._upload_to_texture(sorted_data)

        except Exception as e:
            print(f"[GaussianDataManager Sort] Sorting failed: {e}")
            # If sorting fails, we might still try to upload the unsorted data
            # to avoid a complete crash, though visuals will be incorrect.
            self._upload_to_texture(self.data_buffer)

        return self.texture
    
    @property
    def is_valid(self) -> bool:
        """Check if data manager has valid data."""
        return self.texture is not None and self.gaussian_count > 0




# Utility functions for standalone testing
def create_test_data(num_gaussians: int = 100) -> np.ndarray:
    """
    Create random test data for debugging.
    
    Args:
        num_gaussians: Number of gaussians to generate
        
    Returns:
        np.ndarray: Shape (N, 59)
    """
    data = np.zeros((num_gaussians, FLOATS_PER_GAUSSIAN), dtype=np.float32)
    
    # Random positions in [-1, 1]
    data[:, 0:3] = np.random.uniform(-1, 1, (num_gaussians, 3))
    
    # Identity rotation (w=1, x=y=z=0)
    data[:, 3] = 1.0  # w
    data[:, 4:7] = 0.0  # x, y, z
    
    # Random scales [0.01, 0.1]
    data[:, 7:10] = np.random.uniform(0.01, 0.1, (num_gaussians, 3))
    
    # Random opacity [0.5, 1.0]
    data[:, 10] = np.random.uniform(0.5, 1.0, num_gaussians)
    
    # Random SH degree 0 (color)
    data[:, 11:14] = np.random.uniform(-0.5, 0.5, (num_gaussians, 3))
    
    return data
