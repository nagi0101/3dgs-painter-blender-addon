"""
Brush Management System

Provides unified handling of programmatic and image-converted brushes,
including persistence, serialization, and library management.
"""

import json
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np

from .brush import BrushStamp
from .gaussian import Gaussian2D


class BrushSerializer:
    """Handles serialization/deserialization of BrushStamp objects"""

    @staticmethod
    def brush_to_dict(brush: BrushStamp) -> Dict[str, Any]:
        """Convert BrushStamp to dictionary for JSON serialization"""
        # Serialize base pattern (base_gaussians)
        pattern_data = []
        base_gaussians = brush.base_gaussians if hasattr(brush, 'base_gaussians') else brush.gaussians
        for g in base_gaussians:
            pattern_data.append({
                'position': g.position.tolist() if hasattr(g.position, 'tolist') else g.position,
                'scale': g.scale.tolist() if hasattr(g.scale, 'tolist') else g.scale,
                'rotation': g.rotation.tolist() if hasattr(g.rotation, 'tolist') else g.rotation,
                'opacity': float(g.opacity),  # Pattern opacity preserved
                'color': g.color.tolist() if hasattr(g.color, 'tolist') else g.color  # Preserve luminance texture
            })

        # Create brush data dictionary
        brush_data = {
            'pattern': pattern_data,  # Base pattern
            'center': brush.center.tolist() if hasattr(brush.center, 'tolist') else brush.center,
            'tangent': brush.tangent.tolist() if hasattr(brush.tangent, 'tolist') else brush.tangent,
            'normal': brush.normal.tolist() if hasattr(brush.normal, 'tolist') else brush.normal,
            'binormal': brush.binormal.tolist() if hasattr(brush.binormal, 'tolist') else brush.binormal,
            'size': float(brush.size) if hasattr(brush, 'size') else 0.1,
            'spacing': float(brush.spacing),
            'default_params': {
                'color': brush.current_color.tolist() if hasattr(brush, 'current_color') else [0.5, 0.5, 0.5],
                'size_multiplier': brush.current_size_multiplier if hasattr(brush, 'current_size_multiplier') else 1.0,
                'global_opacity': brush.current_global_opacity if hasattr(brush, 'current_global_opacity') else 1.0
            },
            'metadata': brush.metadata if hasattr(brush, 'metadata') else {}
        }

        return brush_data

    @staticmethod
    def dict_to_brush(brush_data: Dict[str, Any]) -> BrushStamp:
        """Create BrushStamp from dictionary"""
        brush = BrushStamp()

        # Restore base pattern
        brush.base_gaussians = []
        # Support both old format (gaussians) and new format (pattern)
        pattern_key = 'pattern' if 'pattern' in brush_data else 'gaussians'
        for g_data in brush_data.get(pattern_key, []):
            g = Gaussian2D(
                position=np.array(g_data['position'], dtype=np.float32),
                scale=np.array(g_data['scale'], dtype=np.float32),
                rotation=np.array(g_data['rotation'], dtype=np.float32),
                opacity=g_data['opacity'],
                color=np.array(g_data.get('color', [0.5, 0.5, 0.5]), dtype=np.float32)  # Preserve luminance or use default
            )
            brush.base_gaussians.append(g)

        # Restore brush properties
        brush.center = np.array(brush_data.get('center', [0, 0, 0]), dtype=np.float32)
        brush.tangent = np.array(brush_data.get('tangent', [1, 0, 0]), dtype=np.float32)
        brush.normal = np.array(brush_data.get('normal', [0, 0, 1]), dtype=np.float32)
        brush.binormal = np.array(brush_data.get('binormal', [0, 1, 0]), dtype=np.float32)
        brush.size = float(brush_data.get('size', 0.1))
        brush.spacing = brush_data.get('spacing', 0.3)
        brush.metadata = brush_data.get('metadata', {})

        # Restore default parameters if present
        default_params = brush_data.get('default_params', {})
        brush.current_color = np.array(default_params.get('color', [0.5, 0.5, 0.5]), dtype=np.float32)
        brush.current_size_multiplier = default_params.get('size_multiplier', 1.0)
        brush.current_global_opacity = default_params.get('global_opacity', 1.0)

        # Apply parameters to create working gaussians
        brush.apply_parameters()

        return brush


class BrushMetadata:
    """Metadata for a brush in the library"""

    def __init__(self, brush_id: str = None, name: str = "Unnamed Brush"):
        self.id = brush_id or str(uuid.uuid4())
        self.name = name
        self.type = "unknown"  # 'programmatic', 'converted', 'imported'
        self.source = None  # 'circular', 'line', 'grid', 'image', etc.
        self.created_at = datetime.now().isoformat()
        self.modified_at = datetime.now().isoformat()
        self.gaussian_count = 0
        self.tags = []
        self.thumbnail_path = None
        self.file_path = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'source': self.source,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'gaussian_count': self.gaussian_count,
            'tags': self.tags,
            'thumbnail_path': self.thumbnail_path,
            'file_path': self.file_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrushMetadata':
        """Create from dictionary"""
        metadata = cls(data.get('id'), data.get('name', 'Unnamed Brush'))
        metadata.type = data.get('type', 'unknown')
        metadata.source = data.get('source')
        metadata.created_at = data.get('created_at', datetime.now().isoformat())
        metadata.modified_at = data.get('modified_at', datetime.now().isoformat())
        metadata.gaussian_count = data.get('gaussian_count', 0)
        metadata.tags = data.get('tags', [])
        metadata.thumbnail_path = data.get('thumbnail_path')
        metadata.file_path = data.get('file_path')
        return metadata


class BrushManager:
    """
    Manages brush library, persistence, and operations.
    Uses JSON file-based storage for simplicity and portability.
    """

    def __init__(self, storage_path: str = None):
        """Initialize brush manager with storage path"""
        if storage_path is None:
            storage_path = os.path.join(os.path.dirname(__file__), '../../data/brushes')

        self.storage_path = Path(storage_path)
        self.brushes_dir = self.storage_path / 'brushes'
        self.thumbnails_dir = self.storage_path / 'thumbnails'
        self.library_file = self.storage_path / 'library.json'

        # Create directories if they don't exist
        self.brushes_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

        # Load or create library index
        self.library = self._load_library()
        self.serializer = BrushSerializer()

        # Cache for loaded brushes (LRU-style)
        self._brush_cache = {}
        self._cache_max_size = 10

    def _load_library(self) -> Dict[str, BrushMetadata]:
        """Load brush library from JSON file"""
        library = {}

        if self.library_file.exists():
            try:
                with open(self.library_file, 'r') as f:
                    data = json.load(f)
                    for brush_id, brush_data in data.items():
                        library[brush_id] = BrushMetadata.from_dict(brush_data)
            except Exception as e:
                print(f"Error loading brush library: {e}")

        return library

    def _save_library(self):
        """Save brush library to JSON file"""
        data = {
            brush_id: metadata.to_dict()
            for brush_id, metadata in self.library.items()
        }

        with open(self.library_file, 'w') as f:
            json.dump(data, f, indent=2)

    def save_brush(self, brush: BrushStamp, name: str = None,
                   brush_type: str = None, source: str = None,
                   thumbnail: np.ndarray = None) -> str:
        """
        Save a brush to the library.

        Args:
            brush: BrushStamp to save
            name: Optional name for the brush
            brush_type: 'programmatic' or 'converted'
            source: Source identifier (e.g., 'circular', 'image', etc.)
            thumbnail: Optional thumbnail image (numpy array)

        Returns:
            Brush ID
        """
        # Generate metadata
        metadata = BrushMetadata(name=name or "Unnamed Brush")
        metadata.type = brush_type or "unknown"
        metadata.source = source
        metadata.gaussian_count = len(brush.gaussians)

        # Save brush data
        brush_data = self.serializer.brush_to_dict(brush)
        brush_file = self.brushes_dir / f"brush_{metadata.id}.json"
        metadata.file_path = str(brush_file.relative_to(self.storage_path))

        with open(brush_file, 'w') as f:
            json.dump(brush_data, f, indent=2)

        # Save thumbnail if provided
        if thumbnail is not None:
            thumbnail_file = self.thumbnails_dir / f"brush_{metadata.id}.png"
            metadata.thumbnail_path = str(thumbnail_file.relative_to(self.storage_path))
            # TODO: Save thumbnail image (requires cv2 or PIL)

        # Add to library and save
        self.library[metadata.id] = metadata
        self._save_library()

        # Add to cache
        self._add_to_cache(metadata.id, brush)

        print(f"Saved brush '{metadata.name}' with ID {metadata.id}")
        return metadata.id

    def load_brush(self, brush_id: str) -> Optional[BrushStamp]:
        """Load a brush from the library by ID"""
        # Check cache first
        if brush_id in self._brush_cache:
            return self._brush_cache[brush_id]

        # Check if brush exists
        if brush_id not in self.library:
            print(f"Brush {brush_id} not found in library")
            return None

        metadata = self.library[brush_id]
        brush_file = self.storage_path / metadata.file_path

        if not brush_file.exists():
            print(f"Brush file {brush_file} not found")
            return None

        try:
            with open(brush_file, 'r') as f:
                brush_data = json.load(f)
                brush = self.serializer.dict_to_brush(brush_data)

                # Add to cache
                self._add_to_cache(brush_id, brush)

                return brush
        except Exception as e:
            print(f"Error loading brush {brush_id}: {e}")
            return None

    def delete_brush(self, brush_id: str) -> bool:
        """Delete a brush from the library"""
        if brush_id not in self.library:
            return False

        metadata = self.library[brush_id]

        # Delete files
        if metadata.file_path:
            brush_file = self.storage_path / metadata.file_path
            if brush_file.exists():
                brush_file.unlink()

        if metadata.thumbnail_path:
            thumbnail_file = self.storage_path / metadata.thumbnail_path
            if thumbnail_file.exists():
                thumbnail_file.unlink()

        # Remove from library and cache
        del self.library[brush_id]
        if brush_id in self._brush_cache:
            del self._brush_cache[brush_id]

        self._save_library()
        return True

    def list_brushes(self) -> List[Dict[str, Any]]:
        """List all brushes in the library"""
        return [
            {
                **metadata.to_dict(),
                'preview_available': metadata.thumbnail_path is not None
            }
            for metadata in self.library.values()
        ]

    def get_brush_info(self, brush_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific brush"""
        if brush_id not in self.library:
            return None

        metadata = self.library[brush_id]
        return metadata.to_dict()

    def update_brush_metadata(self, brush_id: str, **kwargs) -> bool:
        """Update brush metadata (name, tags, etc.)"""
        if brush_id not in self.library:
            return False

        metadata = self.library[brush_id]

        # Update allowed fields
        if 'name' in kwargs:
            metadata.name = kwargs['name']
        if 'tags' in kwargs:
            metadata.tags = kwargs['tags']

        metadata.modified_at = datetime.now().isoformat()
        self._save_library()
        return True

    def clone_brush(self, brush_id: str, new_name: str = None) -> Optional[str]:
        """Clone an existing brush"""
        brush = self.load_brush(brush_id)
        if brush is None:
            return None

        original_metadata = self.library[brush_id]
        clone_name = new_name or f"{original_metadata.name} (Copy)"

        # Save as new brush
        new_id = self.save_brush(
            brush,
            name=clone_name,
            brush_type=original_metadata.type,
            source=original_metadata.source
        )

        return new_id

    def export_brush(self, brush_id: str, export_path: str) -> bool:
        """Export a brush to a standalone file"""
        if brush_id not in self.library:
            return False

        metadata = self.library[brush_id]
        brush_file = self.storage_path / metadata.file_path

        # Create export package (brush data + metadata)
        export_data = {
            'metadata': metadata.to_dict(),
            'brush': None
        }

        with open(brush_file, 'r') as f:
            export_data['brush'] = json.load(f)

        # Save to export path
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return True

    def import_brush(self, import_path: str) -> Optional[str]:
        """Import a brush from an external file"""
        try:
            with open(import_path, 'r') as f:
                export_data = json.load(f)

            # Extract brush and metadata
            brush_data = export_data.get('brush')
            metadata_dict = export_data.get('metadata', {})

            if not brush_data:
                return None

            # Create brush from data
            brush = self.serializer.dict_to_brush(brush_data)

            # Save with imported metadata
            name = metadata_dict.get('name', 'Imported Brush')
            brush_type = metadata_dict.get('type', 'imported')
            source = metadata_dict.get('source', 'import')

            new_id = self.save_brush(brush, name=name,
                                     brush_type=brush_type, source=source)
            return new_id

        except Exception as e:
            print(f"Error importing brush: {e}")
            return None

    def _add_to_cache(self, brush_id: str, brush: BrushStamp):
        """Add brush to cache with LRU eviction"""
        if len(self._brush_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO for now)
            oldest = next(iter(self._brush_cache))
            del self._brush_cache[oldest]

        self._brush_cache[brush_id] = brush

    def create_default_brushes(self):
        """Create a set of default brushes if library is empty"""
        if len(self.library) > 0:
            print("Library already contains brushes, skipping defaults")
            return

        print("Creating default brush library...")

        # Create default programmatic brushes
        default_brushes = [
            {
                'name': 'Soft Round',
                'pattern': 'circular',
                'params': {'num_gaussians': 20, 'radius': 0.15, 'opacity': 0.8}
            },
            {
                'name': 'Hard Round',
                'pattern': 'circular',
                'params': {'num_gaussians': 15, 'radius': 0.1, 'opacity': 1.0}
            },
            {
                'name': 'Pencil',
                'pattern': 'line',
                'params': {'num_gaussians': 10, 'length': 0.2, 'opacity': 0.9}
            },
            {
                'name': 'Marker',
                'pattern': 'grid',
                'params': {'grid_size': 3, 'spacing': 0.05, 'opacity': 0.7}
            }
        ]

        for brush_def in default_brushes:
            brush = BrushStamp()

            if brush_def['pattern'] == 'circular':
                brush.create_circular_pattern(**brush_def['params'])
            elif brush_def['pattern'] == 'line':
                brush.create_line_pattern(**brush_def['params'])
            elif brush_def['pattern'] == 'grid':
                brush.create_grid_pattern(**brush_def['params'])

            self.save_brush(
                brush,
                name=brush_def['name'],
                brush_type='programmatic',
                source=brush_def['pattern']
            )

        print(f"Created {len(default_brushes)} default brushes")


# Global instance (singleton pattern)
_brush_manager_instance = None


def get_brush_manager(storage_path: str = None) -> BrushManager:
    """Get or create the global BrushManager instance"""
    global _brush_manager_instance
    if _brush_manager_instance is None:
        _brush_manager_instance = BrushManager(storage_path)
        # Create default brushes on first initialization
        _brush_manager_instance.create_default_brushes()
    return _brush_manager_instance