"""
Shared Memory Writer for OpenXR Layer Communication

This module writes Gaussian data to Windows Named Shared Memory,
which is read by the OpenXR API Layer DLL to render in VR.

Data format matches gaussian_data.h:
- SharedMemoryHeader: 180 bytes
- Index Array: 400,000 bytes (uint32[100,000])
- GaussianPrimitive: 56 bytes each × 100,000
"""

import mmap
import struct
import ctypes
from typing import List, Optional, Tuple
import numpy as np

# Constants matching C++ header
SHARED_MEMORY_NAME = "Local\\3DGS_Gaussian_Data"
MAX_GAUSSIANS = 100000
MAGIC_NUMBER = 0x33444753  # "3DGS" in little-endian

# Struct sizes - must match C++ SharedMemoryHeader
# magic(4) + version(4) + frame_id(4) + count(4) + flags(4) + view(64) + proj(64) + camera_rot(16) + camera_pos(12) + padding(4)
HEADER_SIZE = 4 + 4 + 4 + 4 + 4 + 64 + 64 + 16 + 12 + 4  # 180 bytes
GAUSSIAN_SIZE = 56  # 12 + 16 + 12 + 16 bytes

# Index array layout
INDEX_ARRAY_SIZE = MAX_GAUSSIANS * 4  # uint32 array (4 bytes each)
INDEX_ARRAY_OFFSET = HEADER_SIZE  # Indices start after header
GAUSSIAN_DATA_OFFSET = HEADER_SIZE + INDEX_ARRAY_SIZE  # Data starts after indices
BUFFER_SIZE = HEADER_SIZE + INDEX_ARRAY_SIZE + (MAX_GAUSSIANS * GAUSSIAN_SIZE)

# Flag constants
SHMEM_FLAG_INDICES_VALID = 0x00000001


class GaussianPrimitive:
    """Single Gaussian data (56 bytes total)"""
    FORMAT = '<3f 4f 3f 4f'  # position(3) + color(4) + scale(3) + rotation(4)
    SIZE = struct.calcsize(FORMAT)
    
    def __init__(self,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 color: Tuple[float, float, float, float] = (1, 1, 1, 1),
                 scale: Tuple[float, float, float] = (0.1, 0.1, 0.1),
                 rotation: Tuple[float, float, float, float] = (1, 0, 0, 0)):
        self.position = position
        self.color = color
        self.scale = scale
        self.rotation = rotation
    
    def pack(self) -> bytes:
        return struct.pack(self.FORMAT,
            self.position[0], self.position[1], self.position[2],
            self.color[0], self.color[1], self.color[2], self.color[3],
            self.scale[0], self.scale[1], self.scale[2],
            self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])


class SharedMemoryWriter:
    """
    Writes Gaussian data to Windows Named Shared Memory.
    
    Usage:
        writer = SharedMemoryWriter()
        if writer.create():
            writer.write_gaussians(gaussians, view_matrix, proj_matrix)
            # ... later
            writer.close()
    """
    
    def __init__(self):
        self._handle = None
        self._mmap: Optional[mmap.mmap] = None
        self._frame_id = 0
        self._created = False
        
    def create(self) -> bool:
        """Create shared memory. Returns True on success."""
        if self._created:
            return True
            
        try:
            # Windows Named Shared Memory via mmap
            # mmap.mmap with tagname creates a named file mapping
            self._mmap = mmap.mmap(-1, BUFFER_SIZE, tagname=SHARED_MEMORY_NAME)
            
            # Initialize header with magic number
            self._write_header(0, None, None)
            self._created = True
            print(f"[SharedMemory] Created: {SHARED_MEMORY_NAME} ({BUFFER_SIZE} bytes)")
            return True
            
        except Exception as e:
            print(f"[SharedMemory] Create failed: {e}")
            return False
    
    def close(self):
        """Close shared memory."""
        if self._mmap:
            try:
                self._mmap.close()
            except:
                pass
            self._mmap = None
        self._created = False
        print("[SharedMemory] Closed")
    
    def is_open(self) -> bool:
        return self._created and self._mmap is not None
    
    def update_matrices(self,
                        view_matrix: np.ndarray,
                        proj_matrix: np.ndarray,
                        camera_rotation: Optional[Tuple[float, float, float, float]] = None,
                        camera_position: Optional[Tuple[float, float, float]] = None):
        """
        Update only the view/projection matrices in shared memory header.
        
        This is called every frame to update 3D projection for head tracking,
        without modifying the Gaussian data.
        
        Args:
            view_matrix: Flat array of 16 floats (column-major)
            proj_matrix: Flat array of 16 floats (column-major)
            camera_rotation: Camera rotation quaternion (w, x, y, z)
        """
        if not self._mmap:
            return
        
        self._frame_id += 1
        
        # Read current gaussian count from header (offset 12 = magic(4) + version(4) + frame_id(4))
        self._mmap.seek(12)
        count_bytes = self._mmap.read(4)
        gaussian_count = struct.unpack('<I', count_bytes)[0]
        
        # Rewrite entire header with updated matrices but same gaussian count
        header_data = struct.pack('<5I',
            MAGIC_NUMBER,
            1,  # version
            self._frame_id,
            gaussian_count,
            0   # flags
        )
        
        # View matrix (16 floats = 64 bytes)
        if view_matrix is not None and len(view_matrix) >= 16:
            header_data += struct.pack('<16f', *view_matrix[:16])
        else:
            header_data += struct.pack('<16f', *([0.0] * 16))
        
        # Projection matrix (16 floats = 64 bytes)
        if proj_matrix is not None and len(proj_matrix) >= 16:
            header_data += struct.pack('<16f', *proj_matrix[:16])
        else:
            header_data += struct.pack('<16f', *([0.0] * 16))
        
        # Camera rotation quaternion (4 floats = 16 bytes)
        if camera_rotation is not None:
            header_data += struct.pack('<4f', *camera_rotation)
        else:
            header_data += struct.pack('<4f', 1.0, 0.0, 0.0, 0.0)  # Identity
        
        # Camera position (3 floats = 12 bytes) + padding (4 bytes)
        if camera_position is not None:
            header_data += struct.pack('<3f', *camera_position)
        else:
            header_data += struct.pack('<3f', 0.0, 0.0, 0.0)
        header_data += struct.pack('<f', 0.0)  # padding
        
        self._mmap.seek(0)
        self._mmap.write(header_data)
    
    def _write_header(self,
                      gaussian_count: int,
                      view_matrix: Optional[np.ndarray],
                      proj_matrix: Optional[np.ndarray],
                      camera_rotation: Optional[Tuple[float, float, float, float]] = None,
                      camera_position: Optional[Tuple[float, float, float]] = None,
                      flags: int = 0):
        """Write header to shared memory."""
        if not self._mmap:
            return
            
        self._frame_id += 1
        
        # Pack header: magic(4) + version(4) + frame_id(4) + count(4) + flags(4) + view(64) + proj(64) + camera_rot(16)
        header_data = struct.pack('<5I',
            MAGIC_NUMBER,
            1,  # version
            self._frame_id,
            gaussian_count,
            flags
        )
        
        # View matrix (16 floats = 64 bytes)
        if view_matrix is not None and len(view_matrix) >= 16:
            header_data += struct.pack('<16f', *view_matrix[:16])
        else:
            header_data += struct.pack('<16f', *([0.0] * 16))
        
        # Projection matrix (16 floats = 64 bytes)
        if proj_matrix is not None and len(proj_matrix) >= 16:
            header_data += struct.pack('<16f', *proj_matrix[:16])
        else:
            header_data += struct.pack('<16f', *([0.0] * 16))
        
        # Camera rotation quaternion (4 floats = 16 bytes) - w, x, y, z
        if camera_rotation is not None:
            header_data += struct.pack('<4f', *camera_rotation)
        else:
            header_data += struct.pack('<4f', 1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        
        # Camera position (3 floats = 12 bytes) + padding (4 bytes)
        if camera_position is not None:
            header_data += struct.pack('<3f', *camera_position)
        else:
            header_data += struct.pack('<3f', 0.0, 0.0, 0.0)
        header_data += struct.pack('<f', 0.0)  # padding
        
        self._mmap.seek(0)
        self._mmap.write(header_data)
    
    def write_gaussians(self,
                        gaussians: List[GaussianPrimitive],
                        view_matrix: Optional[np.ndarray] = None,
                        proj_matrix: Optional[np.ndarray] = None,
                        camera_rotation: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Write Gaussian data to shared memory.
        
        Args:
            gaussians: List of GaussianPrimitive objects
            view_matrix: Optional 4x4 view matrix as flat array (16 floats)
            proj_matrix: Optional 4x4 projection matrix as flat array (16 floats)
            camera_rotation: Optional camera rotation quaternion (w,x,y,z)
        
        Returns:
            True on success
        """
        if not self._mmap:
            return False
        
        count = min(len(gaussians), MAX_GAUSSIANS)
        
        # Write header first
        self._write_header(count, view_matrix, proj_matrix, camera_rotation)
        
        # Write gaussian data
        if count > 0:
            self._mmap.seek(GAUSSIAN_DATA_OFFSET)
            for i, g in enumerate(gaussians[:count]):
                self._mmap.write(g.pack())
        
        return True
    
    def write_gaussians_numpy(self,
                              positions: np.ndarray,
                              colors: np.ndarray,
                              scales: np.ndarray,
                              rotations: np.ndarray,
                              view_matrix: Optional[np.ndarray] = None,
                              proj_matrix: Optional[np.ndarray] = None,
                              camera_rotation: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Write Gaussian data from numpy arrays (faster for large datasets).
        This includes back-to-front sorting for correct alpha blending.
        
        Args:
            positions: Nx3 float32 array
            colors: Nx4 float32 array (RGBA)
            scales: Nx3 float32 array
            rotations: Nx4 float32 array (quaternion wxyz)
            view_matrix: Optional 16-element float array (column-major)
            proj_matrix: Optional 16-element float array (column-major)
            camera_rotation: Optional camera rotation quaternion (w,x,y,z)
        
        Returns:
            True on success
        """
        if not self._mmap:
            return False
        
        count = min(len(positions), MAX_GAUSSIANS)
        
        # Write header (indices NOT valid - will be set by VR timer)
        self._write_header(count, view_matrix, proj_matrix, camera_rotation, flags=0)
        
        if count > 0:
            # Write UNSORTED data - indices will be used by C++ to read in correct order
            # Each gaussian: pos(3) + color(4) + scale(3) + rotation(4) = 14 floats = 56 bytes
            data = np.zeros((count, 14), dtype=np.float32)
            data[:, 0:3] = positions[:count]
            data[:, 3:7] = colors[:count]
            data[:, 7:10] = scales[:count]
            data[:, 10:14] = rotations[:count]
            
            self._mmap.seek(GAUSSIAN_DATA_OFFSET)
            self._mmap.write(data.tobytes())
        
        return True
    
    def write_sorted_indices(self,
                             positions: np.ndarray,
                             view_matrix: np.ndarray,
                             gaussian_count: int) -> bool:
        """
        Compute back-to-front sorted indices and write to shared memory.
        Called by VR timer every frame to update sort order based on headset position.
        
        Args:
            positions: Nx3 float32 array (world positions)
            view_matrix: 16-element float array (column-major, from VR headset)
            gaussian_count: Number of valid gaussians
            
        Returns:
            True on success
        """
        if not self._mmap:
            return False
        
        count = min(gaussian_count, MAX_GAUSSIANS)
        if count == 0:
            return True
        
        try:
            # Reshape view matrix to 4x4
            # Note: view_matrix is stored column-major (transposed for OpenGL)
            # For row-vector multiplication pos @ M, we need to use it directly
            view_mat_4x4 = np.asarray(view_matrix, dtype=np.float32).reshape(4, 4)
            
            # Homogenize world positions (add w=1)
            pos_world_h = np.hstack((positions[:count], np.ones((count, 1), dtype=np.float32)))
            
            # Transform to view space
            # Since view_matrix is column-major (transposed), use it directly for row-vector mult
            pos_view_h = pos_world_h @ view_mat_4x4
            
            # Get view-space depth (Z coordinate)
            # In Blender/OpenGL view space, -Z is forward, so more negative = farther
            depths = pos_view_h[:, 2]
            
            # Sort back-to-front (ascending Z = back to front for right-handed -Z forward)
            sort_indices = np.argsort(depths).astype(np.uint32)
            
        except Exception as e:
            print(f"[VR SHM] Sorting failed: {e}. Using identity indices.")
            sort_indices = np.arange(count, dtype=np.uint32)
        
        # Write sorted indices
        self._mmap.seek(INDEX_ARRAY_OFFSET)
        self._mmap.write(sort_indices.tobytes())
        
        # Update flags to indicate indices are valid
        # Read current header, update flags, write back
        self._mmap.seek(16)  # Offset to flags field (magic + version + frame_id + count)
        self._mmap.write(struct.pack('<I', SHMEM_FLAG_INDICES_VALID))
        
        return True


# Global instance for easy access
_shared_memory_writer: Optional[SharedMemoryWriter] = None


def get_shared_memory_writer() -> SharedMemoryWriter:
    """Get or create the global SharedMemoryWriter instance."""
    global _shared_memory_writer
    if _shared_memory_writer is None:
        _shared_memory_writer = SharedMemoryWriter()
    return _shared_memory_writer


def init_shared_memory() -> bool:
    """Initialize shared memory. Call once at addon startup."""
    writer = get_shared_memory_writer()
    return writer.create()


def shutdown_shared_memory():
    """Close shared memory. Call at addon shutdown."""
    global _shared_memory_writer
    if _shared_memory_writer:
        _shared_memory_writer.close()
        _shared_memory_writer = None


def write_gaussians_to_vr(gaussians_data: dict) -> bool:
    """
    Write Gaussian data from the addon's internal format to VR shared memory.
    
    Args:
        gaussians_data: Dictionary with keys:
            - 'positions': Nx3 float array
            - 'colors': Nx4 float array
            - 'scales': Nx3 float array
            - 'rotations': Nx4 float array
            - 'view_matrix': Optional 16-element array
            - 'proj_matrix': Optional 16-element array
            - 'camera_rotation': Optional quaternion (w,x,y,z)
    
    Returns:
        True on success
    """
    writer = get_shared_memory_writer()
    
    if not writer.is_open():
        if not writer.create():
            return False
    
    return writer.write_gaussians_numpy(
        positions=np.asarray(gaussians_data.get('positions', []), dtype=np.float32),
        colors=np.asarray(gaussians_data.get('colors', []), dtype=np.float32),
        scales=np.asarray(gaussians_data.get('scales', []), dtype=np.float32),
        rotations=np.asarray(gaussians_data.get('rotations', []), dtype=np.float32),
        view_matrix=gaussians_data.get('view_matrix'),
        proj_matrix=gaussians_data.get('proj_matrix'),
        camera_rotation=gaussians_data.get('camera_rotation')
    )
