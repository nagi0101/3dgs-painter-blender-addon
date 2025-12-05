"""
PyTorch-based Optimizer for Gaussian Parameters using gsplat

Uses gsplat's differentiable CUDA rasterization for fast optimization
of position, scale, rotation, and opacity.

Reference: npr-gaussian-2d-prototype/backend/core/optimizer_torch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Callable, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Lazy import for gsplat - don't fail at module load time
_gsplat_rasterization = None
_gsplat_available = None


def _ensure_gsplat():
    """
    Lazy import gsplat rasterization function.
    
    Returns:
        gsplat.rasterization function
        
    Raises:
        ImportError: If gsplat is not available
    """
    global _gsplat_rasterization, _gsplat_available
    
    if _gsplat_available is None:
        try:
            from gsplat import rasterization as _rasterization
            _gsplat_rasterization = _rasterization
            _gsplat_available = True
            logger.info("[TorchOptimizer] gsplat loaded successfully")
        except ImportError as e:
            _gsplat_available = False
            logger.error(f"[TorchOptimizer] Failed to import gsplat: {e}")
            raise ImportError(
                f"gsplat is required for PyTorch optimization. "
                f"Please install it with CUDA support. Error: {e}"
            )
    
    if not _gsplat_available:
        raise ImportError("gsplat is not available")
    
    return _gsplat_rasterization


def is_gsplat_available() -> bool:
    """Check if gsplat is available without raising an exception."""
    global _gsplat_available
    
    if _gsplat_available is None:
        try:
            _ensure_gsplat()
        except ImportError:
            pass
    
    return _gsplat_available == True


class TorchGaussianOptimizer:
    """
    PyTorch-based optimizer for 2D Gaussian Splatting parameters.
    
    Uses gsplat's differentiable CUDA rasterization for fast optimization
    of position, scale, rotation, and opacity.
    """
    
    def __init__(
        self,
        gaussians_data: np.ndarray,
        target_image: np.ndarray,
        target_alpha: Optional[np.ndarray] = None,
        render_width: int = 256,
        render_height: int = 256,
        device: Optional[str] = None
    ):
        """
        Initialize PyTorch optimizer.
        
        Args:
            gaussians_data: Gaussian parameters as structured array with fields:
                           position (3,), rotation (4,), scale (3,), opacity, color (3,)
            target_image: Target image (H, W, 3) in range [0, 1]
            target_alpha: Optional alpha mask (H, W) in range [0, 1]
            render_width: Render width for optimization
            render_height: Render height for optimization
            device: 'cuda' or 'cpu', auto-detect if None
        """
        # Ensure gsplat is available before proceeding
        _ensure_gsplat()
        
        # Device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        logger.info(f"[TorchOptimizer] Using device: {self.device}")
        
        self.n_gaussians = len(gaussians_data)
        self.width = render_width
        self.height = render_height
        
        # Extract gaussian parameters from structured array
        # Position: use x, y from position, add z=0
        positions_2d = np.array([g['position'][:2] for g in gaussians_data], dtype=np.float32)
        positions_3d = np.concatenate([
            positions_2d,
            np.zeros((self.n_gaussians, 1), dtype=np.float32)
        ], axis=1)
        
        # Scale: use x, y from scale, add small z
        scales_2d = np.array([g['scale'][:2] for g in gaussians_data], dtype=np.float32)
        scales_3d = np.concatenate([
            scales_2d,
            np.ones((self.n_gaussians, 1), dtype=np.float32) * 1e-4
        ], axis=1)
        
        # Rotation: convert from xyzw to wxyz for gsplat
        rotations = np.array([g['rotation'] for g in gaussians_data], dtype=np.float32)
        # Convert xyzw -> wxyz
        quaternions = np.zeros((self.n_gaussians, 4), dtype=np.float32)
        quaternions[:, 0] = rotations[:, 3]  # w
        quaternions[:, 1] = rotations[:, 0]  # x
        quaternions[:, 2] = rotations[:, 1]  # y
        quaternions[:, 3] = rotations[:, 2]  # z
        
        # Opacity and color
        opacities = np.array([g['opacity'] for g in gaussians_data], dtype=np.float32)
        colors = np.array([g['color'] for g in gaussians_data], dtype=np.float32)
        
        # Create PyTorch parameters
        self.means = nn.Parameter(
            torch.tensor(positions_3d, device=self.device, dtype=torch.float32)
        )
        self.scales = nn.Parameter(
            torch.tensor(scales_3d, device=self.device, dtype=torch.float32)
        )
        self.quats = nn.Parameter(
            torch.tensor(quaternions, device=self.device, dtype=torch.float32)
        )
        self.opacities = nn.Parameter(
            torch.tensor(opacities, device=self.device, dtype=torch.float32)
        )
        self.colors = nn.Parameter(
            torch.tensor(colors, device=self.device, dtype=torch.float32)
        )
        
        # Resize target image to render size
        target_resized = self._resize_image(target_image, render_width, render_height)
        self.target = torch.tensor(target_resized, device=self.device, dtype=torch.float32)
        
        # Alpha mask
        if target_alpha is not None:
            # Ensure target_alpha is 2D (H, W)
            alpha_arr = np.asarray(target_alpha, dtype=np.float32)
            if alpha_arr.ndim == 1:
                # Try to reshape if it's flattened
                total_pixels = alpha_arr.size
                # Assume square if possible, otherwise use target image shape
                if int(np.sqrt(total_pixels)) ** 2 == total_pixels:
                    side = int(np.sqrt(total_pixels))
                    alpha_arr = alpha_arr.reshape(side, side)
                elif target_image is not None and len(target_image.shape) >= 2:
                    alpha_arr = alpha_arr.reshape(target_image.shape[0], target_image.shape[1])
                else:
                    logger.warning(f"[TorchOptimizer] Cannot reshape 1D alpha ({alpha_arr.shape}), creating default mask")
                    alpha_arr = None
            elif alpha_arr.ndim == 3:
                # Take first channel if it's 3D
                alpha_arr = alpha_arr[..., 0]
            
            if alpha_arr is not None and alpha_arr.ndim == 2:
                logger.info(f"[TorchOptimizer] Input alpha shape: {alpha_arr.shape}")
                # Add channel dim for resize, then remove
                alpha_3ch = alpha_arr[..., None]
                alpha_resized = self._resize_image(alpha_3ch, render_width, render_height)
                # Handle case where resize might return 2D or 3D
                if alpha_resized.ndim == 3:
                    alpha_resized = alpha_resized[..., 0]
                self.alpha_mask = torch.tensor(alpha_resized, device=self.device, dtype=torch.float32)
                # Ensure 2D tensor
                if self.alpha_mask.ndim == 1:
                    self.alpha_mask = self.alpha_mask.reshape(render_height, render_width)
                self.alpha_mask = (self.alpha_mask > 0.5).float()
                self.target_alpha = self.alpha_mask.clone()
            else:
                target_alpha = None  # Fall through to auto-generation
        
        if target_alpha is None:
            # Create mask from non-background pixels
            luminance = 0.299 * self.target[..., 0] + 0.587 * self.target[..., 1] + 0.114 * self.target[..., 2]
            self.alpha_mask = ((luminance > 0.05) & (luminance < 0.95)).float()
            self.target_alpha = self.alpha_mask.clone()
        
        # Final shape validation
        if self.alpha_mask.ndim != 2:
            logger.error(f"[TorchOptimizer] alpha_mask has wrong ndim: {self.alpha_mask.ndim}, shape: {self.alpha_mask.shape}")
            raise ValueError(f"alpha_mask must be 2D, got shape {self.alpha_mask.shape}")
        
        logger.info(f"[TorchOptimizer] Alpha mask shape: {self.alpha_mask.shape}, coverage: {self.alpha_mask.mean():.1%}")
        
        # Create camera matrices for orthographic projection
        self._create_camera_matrices()
        
        logger.info(f"[TorchOptimizer] Initialized with {self.n_gaussians} Gaussians")
        logger.info(f"[TorchOptimizer] Render size: {self.width}x{self.height}")
    
    def _resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image using bilinear interpolation."""
        import cv2
        
        # Ensure we have a valid numpy array with at least 2 dimensions
        image = np.asarray(image, dtype=np.float32)
        if image.ndim < 2:
            raise ValueError(f"Cannot resize image with shape {image.shape}")
        
        if image.shape[0] == height and image.shape[1] == width:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def _create_camera_matrices(self):
        """Create orthographic camera matrices for gsplat."""
        # View matrix: identity with camera at z=5
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],  # Flip Y
                [0.0, 0.0, 1.0, 5.0],   # Camera at z=5
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 4, 4]
        
        # Projection matrix for orthographic
        fx = fy = min(self.width, self.height) / 2.0
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        self.K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 3, 3]
    
    def _render_differentiable(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render Gaussians using gsplat's differentiable CUDA rasterization.
        
        Returns:
            Tuple of (rendered_rgb [H, W, 3], rendered_alpha [H, W]) in range [0, 1]
        """
        rasterization = _ensure_gsplat()
        
        render_colors, render_alphas, meta = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=self.viewmat,
            Ks=self.K,
            width=self.width,
            height=self.height,
            packed=False,
            camera_model="ortho",
            near_plane=0.1,
            far_plane=10.0,
            eps2d=0.01,
            render_mode="RGB",
        )
        
        rendered_rgb = render_colors[0]
        rendered_alpha = render_alphas[0].squeeze(-1)
        
        return rendered_rgb, rendered_alpha
    
    def _compute_ssim_torch(
        self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
    ) -> torch.Tensor:
        """Compute SSIM (Structural Similarity Index) in PyTorch."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to grayscale
        gray1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
        gray2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]
        
        # Add batch and channel dimensions
        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.exp(
            -torch.arange(window_size, dtype=torch.float32, device=self.device) ** 2 / (2 * sigma ** 2)
        )
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        
        # Compute local means
        mu1 = F.conv2d(gray1, window, padding=window_size // 2)
        mu2 = F.conv2d(gray2, window, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(gray1 ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(gray2 ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(gray1 * gray2, window, padding=window_size // 2) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def optimize(
        self,
        iterations: int = 50,
        lr_position: float = 0.001,
        lr_scale: float = 0.005,
        lr_rotation: float = 0.001,
        lr_opacity: float = 0.01,
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Optimize Gaussian parameters using PyTorch Adam optimizer with gsplat.
        
        Args:
            iterations: Number of optimization iterations
            lr_position: Learning rate for position
            lr_scale: Learning rate for scale
            lr_rotation: Learning rate for rotation
            lr_opacity: Learning rate for opacity
            progress_callback: Optional callback(iter, total, loss)
        
        Returns:
            Optimized gaussians as structured numpy array
        """
        if self.alpha_mask.numel() == 0:
            logger.warning("[TorchOptimizer] Alpha mask is empty. Skipping optimization.")
            return self._tensors_to_structured_array()

        logger.info(f"[TorchOptimizer] Starting optimization for {iterations} iterations")
        
        # Create Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.means, 'lr': lr_position},
            {'params': self.scales, 'lr': lr_scale},
            {'params': self.quats, 'lr': lr_rotation},
            {'params': self.opacities, 'lr': lr_opacity}
        ])
        
        best_loss = float('inf')
        best_state = None
        final_loss = float('inf')
        
        for iter_idx in range(iterations):
            optimizer.zero_grad()
            
            # Render with gsplat
            rendered_rgb, rendered_alpha = self._render_differentiable()
            
            # Masked RGB loss
            mask_3d = self.alpha_mask.unsqueeze(-1).expand(-1, -1, 3)
            masked_rendered = rendered_rgb * mask_3d
            masked_target = self.target * mask_3d
            
            valid_pixels = self.alpha_mask.sum()
            total_pixels = self.height * self.width
            
            if valid_pixels > 0:
                l1_rgb = F.l1_loss(masked_rendered, masked_target) * (total_pixels / valid_pixels)
                ssim_value = self._compute_ssim_torch(rendered_rgb, self.target)
                ssim_loss = (1.0 - ssim_value) * (valid_pixels / total_pixels)
            else:
                l1_rgb = F.l1_loss(rendered_rgb, self.target)
                ssim_value = self._compute_ssim_torch(rendered_rgb, self.target)
                ssim_loss = 1.0 - ssim_value
            
            # Alpha loss
            l1_alpha = F.l1_loss(rendered_alpha, self.target_alpha)
            
            # Scale regularization
            scale_reg = torch.mean((self.scales[:, :2] - 0.02) ** 2) * 0.01
            
            # Total loss
            total_loss = 0.4 * l1_rgb + 0.4 * l1_alpha + 0.15 * ssim_loss + scale_reg
            final_loss = total_loss.item()
            
            # Backprop
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [self.means, self.scales, self.quats, self.opacities],
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Clamp to valid ranges
            with torch.no_grad():
                self.scales.data = self.scales.clamp(min=0.001, max=0.2)
                self.opacities.data = self.opacities.clamp(min=0.01, max=1.0)
                self.quats.data = F.normalize(self.quats, dim=-1)
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {
                    'means': self.means.clone(),
                    'scales': self.scales.clone(),
                    'quats': self.quats.clone(),
                    'opacities': self.opacities.clone()
                }
            
            # Progress callback
            if progress_callback and iter_idx % 5 == 0:
                progress_callback(iter_idx, iterations, total_loss.item())
            
            # Logging
            if iter_idx % 10 == 0:
                logger.info(
                    f"[TorchOptimizer] Iter {iter_idx}/{iterations}: "
                    f"Loss={total_loss.item():.4f}"
                )
            
            # Early stopping
            if total_loss.item() < 0.01:
                logger.info(f"[TorchOptimizer] Early stopping at iteration {iter_idx}")
                break
        
        # Restore best state
        if best_state and final_loss > best_loss:
            self.means.data = best_state['means']
            self.scales.data = best_state['scales']
            self.quats.data = best_state['quats']
            self.opacities.data = best_state['opacities']
            logger.info(f"[TorchOptimizer] Restored best state (loss={best_loss:.4f})")
        
        return self._tensors_to_structured_array()
    
    def _tensors_to_structured_array(self) -> np.ndarray:
        """Convert optimized tensors back to structured numpy array."""
        with torch.no_grad():
            means_cpu = self.means.cpu().numpy()
            scales_cpu = self.scales.cpu().numpy()
            quats_cpu = self.quats.cpu().numpy()
            opacities_cpu = self.opacities.cpu().numpy()
            colors_cpu = self.colors.cpu().numpy()
        
        # Create structured array
        dtype = np.dtype([
            ('position', np.float32, (3,)),
            ('rotation', np.float32, (4,)),
            ('scale', np.float32, (3,)),
            ('opacity', np.float32),
            ('color', np.float32, (3,))
        ])
        
        result = np.zeros(self.n_gaussians, dtype=dtype)
        
        for i in range(self.n_gaussians):
            # Position: x, y from means, z=0
            result[i]['position'] = [means_cpu[i, 0], means_cpu[i, 1], 0.0]
            
            # Rotation: convert wxyz -> xyzw
            quat_wxyz = quats_cpu[i]
            result[i]['rotation'] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            
            # Scale: x, y from scales, z=small
            result[i]['scale'] = [scales_cpu[i, 0], scales_cpu[i, 1], scales_cpu[i, 2]]
            
            result[i]['opacity'] = opacities_cpu[i]
            result[i]['color'] = colors_cpu[i]
        
        return result


def optimize_gaussians(
    gaussians_data: np.ndarray,
    target_image: np.ndarray,
    target_alpha: Optional[np.ndarray] = None,
    iterations: int = 50,
    render_size: int = 256
) -> np.ndarray:
    """
    Convenience function to optimize gaussians.
    
    Args:
        gaussians_data: Structured array of gaussian parameters
        target_image: Target RGB image (H, W, 3) in [0, 1]
        target_alpha: Optional alpha mask (H, W) in [0, 1]
        iterations: Number of optimization iterations
        render_size: Size for rendering during optimization
    
    Returns:
        Optimized gaussians as structured array
    """
    optimizer = TorchGaussianOptimizer(
        gaussians_data=gaussians_data,
        target_image=target_image,
        target_alpha=target_alpha,
        render_width=render_size,
        render_height=render_size
    )
    
    return optimizer.optimize(iterations=iterations)
