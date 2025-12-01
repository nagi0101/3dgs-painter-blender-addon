"""
NPR Core API: Synchronous interface for Blender integration

Replaces async WebSocket communication with direct function calls.
This is the main entry point for Blender addon to interact with NPR Core.
"""

import numpy as np
from typing import Optional, List
import logging

from .gaussian import Gaussian2D
from .scene_data import SceneData
from .gpu_context import get_gpu_context

logger = logging.getLogger(__name__)


class NPRCoreAPI:
    """
    Synchronous API for NPR Core.

    This replaces the async WebSocket-based communication from the prototype.
    All methods are synchronous and return results directly.

    Usage:
        api = NPRCoreAPI()
        api.create_circular_brush(num_gaussians=20, radius=0.15)
        api.set_brush_parameters(color=np.array([1.0, 0.0, 0.0]))
        api.start_stroke(position, normal)
        api.update_stroke(position, normal)
        api.finish_stroke()
    """

    def __init__(self):
        """Initialize API with clean scene."""
        self.scene_data = SceneData()
        self.current_brush: Optional[object] = None  # BrushStamp
        self.stroke_painter: Optional[object] = None  # StrokePainter
        self.gpu_context = get_gpu_context()

        # Try to initialize GPU context (non-blocking)
        try:
            self.gpu_context.initialize()
        except Exception as e:
            logger.warning(f"[API] GPU initialization failed: {e}")

    def create_circular_brush(
        self,
        num_gaussians: int = 20,
        radius: float = 0.15,
        opacity: float = 0.8,
        color: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Create a circular brush pattern.

        Args:
            num_gaussians: Number of Gaussians in circle
            radius: Circle radius
            opacity: Pattern opacity
            color: Optional RGB color [0,1], defaults to white

        Returns:
            True if successful
        """
        try:
            from .brush import BrushStamp

            brush = BrushStamp()
            brush.create_circular_pattern(
                num_gaussians=num_gaussians,
                radius=radius,
                gaussian_scale=radius * 0.25,
                opacity=opacity,
            )

            # Set initial color if provided
            if color is not None:
                brush.apply_parameters(color=color)

            self.current_brush = brush
            logger.info(
                f"[API] Created circular brush: {num_gaussians} gaussians, radius {radius}"
            )
            return True

        except Exception as e:
            logger.error(f"[API] Failed to create brush: {e}")
            return False

    def create_line_brush(
        self,
        num_gaussians: int = 10,
        length: float = 0.2,
        thickness: float = 0.05,
        opacity: float = 0.8,
    ) -> bool:
        """
        Create a line brush pattern.

        Args:
            num_gaussians: Number of Gaussians
            length: Line length
            thickness: Gaussian thickness
            opacity: Pattern opacity

        Returns:
            True if successful
        """
        try:
            from .brush import BrushStamp

            brush = BrushStamp()
            brush.create_line_pattern(
                num_gaussians=num_gaussians,
                length=length,
                thickness=thickness,
                opacity=opacity,
            )

            self.current_brush = brush
            logger.info(f"[API] Created line brush: {num_gaussians} gaussians")
            return True

        except Exception as e:
            logger.error(f"[API] Failed to create line brush: {e}")
            return False

    def set_brush_parameters(
        self,
        color: Optional[np.ndarray] = None,
        size_multiplier: Optional[float] = None,
        opacity: Optional[float] = None,
        spacing: Optional[float] = None,
    ) -> bool:
        """
        Set brush runtime parameters.

        Args:
            color: RGB color [0,1]
            size_multiplier: Scale multiplier
            opacity: Global opacity multiplier
            spacing: Stamp spacing along stroke

        Returns:
            True if successful
        """
        if self.current_brush is None:
            logger.warning("[API] No brush loaded")
            return False

        try:
            self.current_brush.apply_parameters(
                color=color,
                size_multiplier=size_multiplier,
                global_opacity=opacity,
                spacing=spacing,
            )
            return True
        except Exception as e:
            logger.error(f"[API] Failed to set brush parameters: {e}")
            return False

    def start_stroke(
        self, position: np.ndarray, normal: np.ndarray, enable_deformation: bool = False
    ) -> bool:
        """
        Start a new painting stroke.

        Args:
            position: 3D starting position
            normal: Surface normal
            enable_deformation: Enable deformation for this stroke

        Returns:
            True if successful
        """
        if self.current_brush is None:
            logger.warning("[API] No brush loaded, cannot start stroke")
            return False

        try:
            from .brush import StrokePainter

            # Create stroke painter
            self.stroke_painter = StrokePainter(
                brush=self.current_brush, scene_gaussians=self.scene_data
            )

            self.stroke_painter.start_stroke(
                position=position, normal=normal, enable_deformation=enable_deformation
            )

            logger.info(f"[API] Started stroke at {position}")
            return True

        except Exception as e:
            logger.error(f"[API] Failed to start stroke: {e}")
            return False

    def update_stroke(self, position: np.ndarray, normal: np.ndarray) -> bool:
        """
        Update stroke with new point.

        Args:
            position: 3D position
            normal: Surface normal

        Returns:
            True if successful
        """
        if self.stroke_painter is None:
            logger.warning("[API] No active stroke")
            return False

        try:
            self.stroke_painter.update_stroke(position, normal)
            return True
        except Exception as e:
            logger.error(f"[API] Failed to update stroke: {e}")
            return False

    def finish_stroke(
        self,
        enable_deformation: bool = False,
        enable_inpainting: bool = False,
        blend_strength: float = 0.3,
    ) -> bool:
        """
        Finish current stroke with optional processing.

        Args:
            enable_deformation: Apply non-rigid deformation
            enable_inpainting: Apply overlap inpainting
            blend_strength: Opacity blending strength

        Returns:
            True if successful
        """
        if self.stroke_painter is None:
            logger.warning("[API] No active stroke")
            return False

        try:
            self.stroke_painter.finish_stroke(
                enable_deformation=enable_deformation,
                enable_inpainting=enable_inpainting,
                blend_strength=blend_strength,
            )

            logger.info(
                f"[API] Finished stroke, total gaussians: {len(self.scene_data)}"
            )
            self.stroke_painter = None
            return True

        except Exception as e:
            logger.error(f"[API] Failed to finish stroke: {e}")
            return False

    def get_scene_data(self) -> SceneData:
        """
        Get current scene data.

        Returns:
            SceneData object with all gaussians
        """
        return self.scene_data

    def clear_scene(self):
        """Clear all gaussians from scene."""
        self.scene_data.clear()
        logger.info(f"[API] Scene cleared")

    def undo_last_stroke(self):
        """Undo last stroke (simple implementation - removes all gaussians added in last stroke)."""
        # TODO: Implement proper undo stack with stroke boundaries
        logger.warning("[API] Undo not yet implemented")

    def get_gaussian_count(self) -> int:
        """Get total number of gaussians in scene."""
        return len(self.scene_data)

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        return self.gpu_context.get_memory_usage()

    def is_gpu_available(self) -> bool:
        """Check if GPU/CUDA is available."""
        return self.gpu_context.is_cuda_available()
