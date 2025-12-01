"""
GPU Context Management for Blender Integration

Manages PyTorch CUDA context within Blender's process,
ensuring compatibility with Blender's OpenGL context.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BlenderGPUContext:
    """
    Manage PyTorch CUDA context within Blender.
    Ensures CUDA operations don't interfere with Blender's OpenGL.
    """

    def __init__(self):
        self.device: Optional[object] = None  # torch.device
        self.initialized = False
        self._cuda_available = False
        self._torch_available = False

    def initialize(self) -> bool:
        """
        Initialize PyTorch CUDA context.
        Must be called after Blender starts.

        Returns:
            True if CUDA initialized successfully, False otherwise
        """
        if self.initialized:
            return self._cuda_available

        # Check if PyTorch is available
        try:
            import torch

            self._torch_available = True
        except ImportError:
            logger.warning("[NPR Core] PyTorch not installed, GPU features disabled")
            self.initialized = True
            self._cuda_available = False
            return False

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.info("[NPR Core] CUDA not available, using CPU fallback")
            self.device = torch.device("cpu")
            self.initialized = True
            self._cuda_available = False
            return False

        try:
            # Set device
            self.device = torch.device("cuda:0")

            # Warm-up (allocate small tensor to initialize context)
            dummy = torch.zeros(1, device=self.device)
            del dummy
            torch.cuda.synchronize()

            self._cuda_available = True
            self.initialized = True

            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[NPR Core] ✓ PyTorch CUDA context initialized")
            logger.info(f"[NPR Core]   Device: {device_name}")
            logger.info(f"[NPR Core]   VRAM: {vram_gb:.1f} GB")

            return True

        except Exception as e:
            logger.error(f"[NPR Core] Failed to initialize CUDA: {e}")
            logger.info("[NPR Core] Falling back to CPU")
            import torch

            self.device = torch.device("cpu")
            self.initialized = True
            self._cuda_available = False
            return False

    def get_device(self) -> object:
        """Get current device (CUDA or CPU)."""
        if not self.initialized:
            self.initialize()

        if self.device is None:
            # Fallback to CPU if initialization failed
            try:
                import torch

                self.device = torch.device("cpu")
            except ImportError:
                raise RuntimeError("PyTorch not available")

        return self.device

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available and initialized."""
        if not self.initialized:
            self.initialize()
        return self._cuda_available

    def is_torch_available(self) -> bool:
        """Check if PyTorch is available."""
        return self._torch_available

    def synchronize(self):
        """Synchronize CUDA operations."""
        if self.initialized and self._cuda_available:
            try:
                import torch

                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"[NPR Core] CUDA synchronize failed: {e}")

    def clear_cache(self):
        """Clear CUDA cache to free memory."""
        if self.initialized and self._cuda_available:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"[NPR Core] CUDA cache clear failed: {e}")

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        if not self._cuda_available:
            return {"allocated": 0, "reserved": 0, "total": 0, "unit": "GB"}

        try:
            import torch

            return {
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
                "reserved": torch.cuda.memory_reserved(0) / (1024**3),  # GB
                "total": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),  # GB
                "unit": "GB",
            }
        except Exception as e:
            logger.warning(f"[NPR Core] Failed to get memory usage: {e}")
            return {"allocated": 0, "reserved": 0, "total": 0, "unit": "GB"}


# Global singleton instance
_gpu_context = BlenderGPUContext()


def get_gpu_context() -> BlenderGPUContext:
    """Get global GPU context instance."""
    return _gpu_context
