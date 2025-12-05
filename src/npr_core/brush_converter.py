"""
Brush Converter: 2D Image to 3DGS Brush Conversion Pipeline

Converts 2D brush stroke images into 3D Gaussian Splatting brush stamps.
Uses skeleton + thickness based depth estimation (no MiDaS dependency).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Core dependencies
from .gaussian import Gaussian2D
from .brush import BrushStamp
from .quaternion_utils import quaternion_from_axis_angle

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BrushConversionConfig:
    """Configuration for brush conversion pipeline"""
    
    # Gaussian count
    num_gaussians: int = 100
    
    # Sampling method: 'importance', 'uniform', 'skeleton'
    sampling_method: str = 'importance'
    
    # Depth profile: 'flat', 'convex', 'concave', 'ridge'
    depth_profile: str = 'convex'
    
    # Depth weights (must sum to 1.0 for normalized depth)
    skeleton_depth_weight: float = 0.7
    thickness_depth_weight: float = 0.3
    
    # Depth scale (max Z value)
    depth_scale: float = 0.2
    
    # Gaussian initialization
    scale_multiplier: float = 0.6
    z_scale_factor: float = 0.3  # Thickness to Z scale ratio
    
    # Elongation (directional scaling)
    enable_elongation: bool = True
    elongation_ratio: float = 1.5  # Long axis / short axis
    elongation_strength: float = 0.7  # 0 = circular, 1 = full elongation
    
    # Contrast enhancement
    enable_contrast: bool = True
    min_contrast: float = 0.3
    
    # Jitter for organic look
    position_jitter: float = 0.005
    rotation_jitter_deg: float = 2.0
    
    # gsplat optimization (differentiable rendering)
    enable_optimization: bool = True
    optimization_iterations: int = 50
    optimization_render_size: int = 256


def _check_dependencies() -> Tuple[bool, str]:
    """Check if required dependencies are available"""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    try:
        from skimage import morphology
    except ImportError:
        missing.append('scikit-image')
    
    try:
        from scipy.spatial import KDTree
        from scipy import ndimage
    except ImportError:
        missing.append('scipy')
    
    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"
    return True, ""


class BrushConverter:
    """
    Converts 2D brush stroke images to 3D Gaussian Splatting brushes.
    
    Pipeline stages:
    1. Alpha mask extraction (adaptive background detection)
    2. Feature extraction (skeleton, thickness map)
    3. Depth estimation (skeleton + thickness heuristic)
    4. Point sampling (importance-based)
    5. Gaussian initialization (KDTree density, directional elongation)
    6. Procedural refinement (jitter, thickness scaling)
    """
    
    def __init__(self, config: Optional[BrushConversionConfig] = None):
        """
        Initialize brush converter
        
        Args:
            config: Conversion configuration (uses defaults if None)
        """
        # Check dependencies
        deps_ok, deps_msg = _check_dependencies()
        if not deps_ok:
            raise ImportError(deps_msg)
        
        self.config = config or BrushConversionConfig()
        
        # Debug data storage (for visualization)
        self.debug_data: Dict[str, Any] = {}
        
        logger.info(f"[BrushConverter] Initialized with target {self.config.num_gaussians} Gaussians")
    
    def convert(
        self,
        image_path: str,
        brush_name: str = "converted_brush"
    ) -> BrushStamp:
        """
        Convert image file to BrushStamp
        
        Args:
            image_path: Path to image file
            brush_name: Name for the converted brush
            
        Returns:
            BrushStamp containing converted Gaussians
        """
        import cv2
        
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return self.convert_from_array(image, brush_name)
    
    def convert_from_array(
        self,
        image: np.ndarray,
        brush_name: str = "converted_brush"
    ) -> BrushStamp:
        """
        Convert numpy array image to BrushStamp
        
        Args:
            image: Input image (H, W, 3) BGR or (H, W, 4) BGRA
            brush_name: Name for the converted brush
            
        Returns:
            BrushStamp containing converted Gaussians
        """
        logger.info(f"[BrushConverter] Converting '{brush_name}' with {self.config.depth_profile} profile")
        
        # Store debug data
        self.debug_data = {'original_image': image.copy(), 'brush_name': brush_name}
        
        # Step 1: Extract alpha mask
        alpha_mask = self._extract_alpha_mask(image)
        self.debug_data['alpha_mask'] = alpha_mask.copy()
        
        # Step 2: Extract features (skeleton, thickness)
        features = self._extract_features(image, alpha_mask)
        self.debug_data['features'] = features
        
        # Step 3: Estimate depth from features
        depth_map = self._estimate_depth(alpha_mask, features)
        self.debug_data['depth_map'] = depth_map.copy()
        
        # Step 4: Generate point cloud with importance sampling
        points, colors, normals = self._generate_point_cloud(image, depth_map, alpha_mask, features)
        self.debug_data['points'] = points
        self.debug_data['colors'] = colors
        
        # Step 5: Initialize Gaussians
        gaussians = self._initialize_gaussians(points, colors, normals, features, alpha_mask.shape)
        
        # Step 6: Procedural refinement
        gaussians = self._refine_gaussians(gaussians, features, alpha_mask.shape)
        
        # Step 7: Optimize appearance using gsplat (if enabled)
        if self.config.enable_optimization:
            gaussians = self._optimize_appearance(gaussians, image, alpha_mask)
        
        # Step 8: Create BrushStamp
        brush = self._create_brush_stamp(gaussians, brush_name)
        
        logger.info(f"[BrushConverter] ✓ Created brush with {len(gaussians)} Gaussians")
        
        return brush
    
    def _extract_alpha_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract alpha channel or create from intensity with adaptive background detection
        
        Args:
            image: Input image (H, W, 3) or (H, W, 4)
            
        Returns:
            Alpha mask (H, W) in range [0, 1], binary
        """
        import cv2
        
        if len(image.shape) == 3 and image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            min_alpha, max_alpha = np.min(alpha_channel), np.max(alpha_channel)
            
            # Use alpha channel if it has a meaningful value range (i.e., not all black or all white)
            if max_alpha > 0 and min_alpha < 255:
                logger.info(f"[AlphaMask] Using alpha channel (range: {min_alpha}-{max_alpha})")
                normalized_alpha = alpha_channel.astype(np.float32) / 255.0
                # Use a lower threshold to better capture soft brushes
                return (normalized_alpha > 0.1).astype(np.float32)
        
        # Create from grayscale with adaptive thresholding if no meaningful alpha found
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect background from corners
        h, w = gray.shape
        margin = max(min(h, w) // 10, 5)
        corners = [
            gray[0:margin, 0:margin],
            gray[0:margin, w-margin:w],
            gray[h-margin:h, 0:margin],
            gray[h-margin:h, w-margin:w],
        ]
        bg_luminance = np.mean([np.mean(c) for c in corners if c.size > 0])
        
        logger.info(f"[AlphaMask] Creating mask from luminance. Background: {bg_luminance:.1f}")
        
        # Adaptive threshold using Otsu's method
        if bg_luminance > 128:
            # Bright background → keep dark pixels
            _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Dark background → keep bright pixels
            _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        
        # Binary
        alpha = (alpha > 127).astype(np.float32)
        
        return alpha
    
    def _extract_features(
        self,
        image: np.ndarray,
        alpha_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for Gaussian initialization
        
        Args:
            image: Input image
            alpha_mask: Binary alpha mask
            
        Returns:
            Dictionary with 'skeleton', 'thickness', 'flow_x', 'flow_y'
        """
        import cv2
        from skimage import morphology
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Binary mask for morphology
        binary = (alpha_mask > 0.5).astype(np.uint8)
        
        # 1. Skeleton (medial axis)
        skeleton = morphology.skeletonize(binary > 0)
        
        # 2. Distance transform (thickness)
        thickness_map = cv2.distanceTransform(binary * 255, cv2.DIST_L2, 5).astype(np.float32)
        if thickness_map.max() > 0:
            thickness_map = thickness_map / thickness_map.max()
        
        # 3. Gradient flow field
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
        flow_x = grad_x / grad_magnitude
        flow_y = grad_y / grad_magnitude
        
        return {
            'skeleton': skeleton.astype(np.float32),
            'thickness': thickness_map,
            'flow_x': flow_x,
            'flow_y': flow_y,
            'grad_magnitude': grad_magnitude,
            # Alias for compatibility
            'medial_axis': skeleton.astype(np.float32),
            'thickness_map': thickness_map,
        }
    
    def _estimate_depth(
        self,
        alpha_mask: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Estimate depth from skeleton proximity and thickness.
        
        Formula:
            depth = skeleton_weight × skeleton_proximity + thickness_weight × thickness
        
        Args:
            alpha_mask: Binary alpha mask
            features: Extracted features
            
        Returns:
            Depth map (H, W) in range [0, depth_scale]
        """
        from scipy import ndimage
        
        h, w = alpha_mask.shape
        skeleton = features['skeleton']
        thickness = features['thickness']
        
        # Compute skeleton proximity (distance from skeleton, inverted)
        skeleton_dist = ndimage.distance_transform_edt(~skeleton.astype(bool))
        max_dist = skeleton_dist.max() if skeleton_dist.max() > 0 else 1.0
        skeleton_proximity = 1.0 - (skeleton_dist / max_dist)
        skeleton_proximity = np.clip(skeleton_proximity, 0, 1)
        
        # Combine based on profile
        profile = self.config.depth_profile
        sw = self.config.skeleton_depth_weight
        tw = self.config.thickness_depth_weight
        
        if profile == 'flat':
            # Minimal depth variation
            depth_base = np.ones_like(alpha_mask) * 0.5
        elif profile == 'convex':
            # Center (skeleton) bulges outward
            depth_base = sw * skeleton_proximity + tw * thickness
        elif profile == 'concave':
            # Center depressed (invert skeleton proximity)
            depth_base = sw * (1.0 - skeleton_proximity) + tw * thickness
        elif profile == 'ridge':
            # Sharp ridge only on skeleton
            depth_base = skeleton_proximity ** 2  # Sharper falloff
        else:
            depth_base = sw * skeleton_proximity + tw * thickness
        
        # Normalize and scale
        if depth_base.max() > 0:
            depth_base = depth_base / depth_base.max()
        
        depth_map = depth_base * self.config.depth_scale
        
        # Apply alpha mask
        depth_map = depth_map * alpha_mask
        
        return depth_map.astype(np.float32)
    
    def _generate_point_cloud(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha_mask: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D point cloud from image with importance-based sampling
        
        Args:
            image: Input image
            depth_map: Depth values
            alpha_mask: Alpha mask
            features: Extracted features
            
        Returns:
            (points, colors, normals) arrays
        """
        import cv2
        
        h, w = depth_map.shape
        
        # Preserve aspect ratio
        aspect = w / h
        target_size = 0.5  # World space extent
        
        if w > h:
            x_range = target_size
            y_range = target_size / aspect
        else:
            x_range = target_size * aspect
            y_range = target_size
        
        # Create coordinate grids
        xx, yy = np.meshgrid(
            np.linspace(-x_range, x_range, w),
            np.linspace(-y_range, y_range, h),
            indexing='xy'
        )
        
        # Valid mask
        mask = alpha_mask > 0.5
        total_pixels = np.sum(mask)
        
        # Importance sampling if needed
        if total_pixels > self.config.num_gaussians:
            importance_map = self._compute_importance_map(features, mask)
            mask = self._importance_based_sampling(mask, importance_map, self.config.num_gaussians)
        
        # Compute luminance with contrast enhancement
        if len(image.shape) == 3:
            color_bgr = image[:, :, :3] / 255.0
            luminance_map = 0.114 * color_bgr[:, :, 0] + 0.587 * color_bgr[:, :, 1] + 0.299 * color_bgr[:, :, 2]
        else:
            luminance_map = image / 255.0
        
        # Contrast stretching within valid region
        valid_luminance = luminance_map[mask]
        if len(valid_luminance) > 0 and self.config.enable_contrast:
            lum_min = np.percentile(valid_luminance, 2)
            lum_max = np.percentile(valid_luminance, 98)
            
            if lum_max - lum_min < self.config.min_contrast:
                center = (lum_min + lum_max) / 2
                lum_min = max(0.0, center - self.config.min_contrast / 2)
                lum_max = min(1.0, center + self.config.min_contrast / 2)
            
            luminance_map = np.clip((luminance_map - lum_min) / (lum_max - lum_min + 1e-8), 0.0, 1.0)
        
        # Extract points
        valid_indices = np.where(mask)
        
        points = []
        colors = []
        normals = []
        
        for i, j in zip(valid_indices[0], valid_indices[1]):
            x = xx[i, j]
            y = yy[i, j]
            z = depth_map[i, j]
            points.append([x, y, z])
            
            # Luminance as grayscale color
            lum = luminance_map[i, j]
            colors.append([lum, lum, lum])
            
            # Normal from depth gradient
            normal = self._compute_normal_at_point(depth_map, i, j, xx, yy)
            normals.append(normal)
        
        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        
        logger.info(f"[BrushConverter] Generated {len(points)} points")
        
        return points, colors, normals
    
    def _compute_importance_map(
        self,
        features: Dict[str, np.ndarray],
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance map for adaptive sampling.
        Higher values = more Gaussians placed there.
        
        Uses thickness + skeleton proximity (no gradient to avoid luminance bias).
        """
        from scipy import ndimage
        
        h, w = mask.shape
        
        # Thickness importance
        thickness = features['thickness']
        thickness_norm = thickness / (thickness.max() + 1e-8)
        
        # Skeleton proximity importance
        skeleton = features['skeleton']
        skeleton_dist = ndimage.distance_transform_edt(~skeleton.astype(bool))
        max_dist = skeleton_dist[mask].max() if np.any(mask) else 1.0
        skeleton_importance = 1.0 - (skeleton_dist / (max_dist + 1e-8))
        skeleton_importance = np.clip(skeleton_importance, 0, 1)
        
        # Combine: 50% thickness, 50% skeleton
        importance = 0.5 * thickness_norm + 0.5 * skeleton_importance
        
        # Apply mask
        importance = importance * (mask > 0)
        importance = importance / (importance.max() + 1e-8)
        
        return importance
    
    def _importance_based_sampling(
        self,
        mask: np.ndarray,
        importance_map: np.ndarray,
        target_count: int
    ) -> np.ndarray:
        """Weighted random sampling based on importance map"""
        
        valid_indices = np.where(mask)
        num_valid = len(valid_indices[0])
        
        if num_valid <= target_count:
            return mask
        
        # Extract importance values
        importance_values = importance_map[valid_indices]
        total = np.sum(importance_values)
        
        if total < 1e-8:
            probabilities = np.ones(num_valid) / num_valid
        else:
            probabilities = importance_values / total
        
        # Weighted sampling
        probabilities = probabilities / probabilities.sum()  # Ensure sum = 1
        selected_indices = np.random.choice(num_valid, size=target_count, replace=False, p=probabilities)
        
        # Create new mask
        h, w = mask.shape
        new_mask = np.zeros((h, w), dtype=bool)
        new_mask[valid_indices[0][selected_indices], valid_indices[1][selected_indices]] = True
        
        logger.info(f"[Sampling] {num_valid} → {target_count} points")
        
        return new_mask
    
    def _compute_normal_at_point(
        self,
        depth_map: np.ndarray,
        i: int,
        j: int,
        xx: np.ndarray,
        yy: np.ndarray
    ) -> np.ndarray:
        """Compute surface normal from depth gradient"""
        h, w = depth_map.shape
        
        # Finite differences
        if 0 < j < w - 1:
            dz_dx = (depth_map[i, j + 1] - depth_map[i, j - 1]) / (xx[i, j + 1] - xx[i, j - 1] + 1e-8)
        else:
            dz_dx = 0
        
        if 0 < i < h - 1:
            dz_dy = (depth_map[i + 1, j] - depth_map[i - 1, j]) / (yy[i + 1, j] - yy[i - 1, j] + 1e-8)
        else:
            dz_dy = 0
        
        normal = np.array([-dz_dx, -dz_dy, 1.0])
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        return normal
    
    def _initialize_gaussians(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
        features: Dict[str, np.ndarray],
        image_shape: Tuple[int, int]
    ) -> List[Gaussian2D]:
        """Initialize Gaussians from point cloud"""
        from scipy.spatial import KDTree
        
        gaussians = []
        
        if len(points) == 0:
            logger.warning("[BrushConverter] No points to create Gaussians")
            return gaussians
        
        # Build KD-tree for neighbor queries
        kdtree = KDTree(points)
        h, w = image_shape
        thickness = features['thickness']
        
        for i, (pos, color, normal) in enumerate(zip(points, colors, normals)):
            # Map world position to image space
            x_img = int((pos[0] + 0.5) * w)
            y_img = int((0.5 - pos[1]) * h)
            x_img = np.clip(x_img, 0, w - 1)
            y_img = np.clip(y_img, 0, h - 1)
            
            # Get local thickness
            local_thickness = thickness[y_img, x_img] if 0 <= y_img < h and 0 <= x_img < w else 0.5
            
            # Opacity based on thickness
            opacity = min(1.0, local_thickness * 2.0)
            
            # Scale based on local density (KNN)
            distances, _ = kdtree.query([pos], k=min(8, len(points)))
            if len(distances[0]) > 1:
                avg_distance = np.mean(distances[0][1:])
            else:
                avg_distance = 0.05
            
            # Find skeleton tangent for orientation
            tangent = self._find_skeleton_tangent(pos, features, image_shape)
            has_direction = tangent is not None
            
            if has_direction:
                angle = np.arctan2(tangent[1], tangent[0])
                rotation = quaternion_from_axis_angle(np.array([0, 0, 1]), angle)
            else:
                rotation = np.array([0, 0, 0, 1], dtype=np.float32)
            
            # Scale calculation
            base_scale = avg_distance * self.config.scale_multiplier
            thickness_scale = np.clip(local_thickness * 2.0, 0.5, 2.0)
            base_scale *= thickness_scale
            
            # Elongation along stroke direction
            if has_direction and self.config.enable_elongation:
                long_axis = base_scale * self.config.elongation_ratio
                long_axis = base_scale + (long_axis - base_scale) * self.config.elongation_strength
                scale = np.array([long_axis, base_scale, local_thickness * self.config.z_scale_factor])
            else:
                scale = np.array([base_scale, base_scale, local_thickness * self.config.z_scale_factor])
            
            g = Gaussian2D(
                position=pos.copy(),
                scale=scale.astype(np.float32),
                rotation=rotation.astype(np.float32),
                opacity=float(opacity),
                color=color.copy().astype(np.float32)
            )
            gaussians.append(g)
        
        return gaussians
    
    def _find_skeleton_tangent(
        self,
        position: np.ndarray,
        features: Dict[str, np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Find tangent vector from nearest skeleton point"""
        
        skeleton = features.get('skeleton')
        if skeleton is None:
            return None
        
        h, w = image_shape
        
        # Map world to image space
        x_img = int((position[0] + 0.5) * w)
        y_img = int((0.5 - position[1]) * h)
        
        if not (0 <= x_img < w and 0 <= y_img < h):
            return None
        
        # Find skeleton points
        skeleton_points = np.argwhere(skeleton > 0)
        if len(skeleton_points) == 0:
            return None
        
        # Distance to all skeleton points
        dists = np.sqrt((skeleton_points[:, 0] - y_img)**2 + (skeleton_points[:, 1] - x_img)**2)
        
        # Get nearby points for tangent estimation
        radius = 5
        nearby_mask = dists < radius
        nearby_points = skeleton_points[nearby_mask]
        
        if len(nearby_points) < 2:
            # Use flow field as fallback
            flow_x = features.get('flow_x')
            flow_y = features.get('flow_y')
            if flow_x is not None and flow_y is not None and 0 <= y_img < h and 0 <= x_img < w:
                fx = flow_x[y_img, x_img]
                fy = flow_y[y_img, x_img]
                tangent = np.array([fx, -fy, 0.0])
                norm = np.linalg.norm(tangent)
                if norm > 1e-8:
                    return tangent / norm
            return None
        
        # PCA for direction
        points_centered = nearby_points - nearby_points.mean(axis=0)
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal_idx = np.argmax(eigenvalues)
        tangent_img = eigenvectors[:, principal_idx]  # [dy, dx] in image space
        
        # Convert to world space
        tangent_world = np.array([tangent_img[1], -tangent_img[0], 0.0])
        tangent_world = tangent_world / (np.linalg.norm(tangent_world) + 1e-8)
        
        return tangent_world
    
    def _refine_gaussians(
        self,
        gaussians: List[Gaussian2D],
        features: Dict[str, np.ndarray],
        image_shape: Tuple[int, int]
    ) -> List[Gaussian2D]:
        """Apply procedural refinements to Gaussians"""
        
        h, w = image_shape
        thickness = features['thickness']
        grad_magnitude = features.get('grad_magnitude')
        
        for g in gaussians:
            # Map to image space
            x_img = int((g.position[0] + 0.5) * w)
            y_img = int((0.5 - g.position[1]) * h)
            x_img = np.clip(x_img, 0, w - 1)
            y_img = np.clip(y_img, 0, h - 1)
            
            # 1. Thickness-based scale adjustment
            if 0 <= y_img < h and 0 <= x_img < w:
                t = thickness[y_img, x_img]
                thickness_factor = 0.5 + t  # Range [0.5, 1.5]
                g.scale = (g.scale * thickness_factor).astype(np.float32)
            
            # 2. Position jitter
            jitter = np.random.normal(0, self.config.position_jitter, 3).astype(np.float32)
            g.position = (g.position + jitter).astype(np.float32)
            
            # 3. Rotation jitter
            jitter_angle = np.random.normal(0, np.radians(self.config.rotation_jitter_deg))
            jitter_quat = quaternion_from_axis_angle(np.array([0, 0, 1]), jitter_angle)
            from .quaternion_utils import quaternion_multiply
            g.rotation = quaternion_multiply(g.rotation, jitter_quat)
            
            # 4. Opacity adjustment based on gradient
            if grad_magnitude is not None and 0 <= y_img < h and 0 <= x_img < w:
                grad = grad_magnitude[y_img, x_img]
                opacity_factor = 1.0 - (grad / (grad + 50))
                g.opacity = g.opacity * opacity_factor
        
        return gaussians
    
    def _optimize_appearance(
        self,
        gaussians: List[Gaussian2D],
        image: np.ndarray,
        alpha_mask: np.ndarray
    ) -> List[Gaussian2D]:
        """
        Optimize Gaussian parameters using gsplat differentiable rendering.
        
        Runs in subprocess to avoid TBB DLL conflicts with Blender.
        Falls back to heuristic-only if gsplat is not available.
        
        Args:
            gaussians: Initial Gaussians from heuristic initialization
            image: Original input image (BGR or BGRA)
            alpha_mask: Binary alpha mask (H, W) in [0, 1]
            
        Returns:
            Optimized Gaussians (or original if optimization fails)
        """
        import cv2
        
        logger.info(f"[BrushConverter] Starting gsplat optimization ({self.config.optimization_iterations} iterations)")
        
        try:
            from ..generator_process import get_generator
            
            # Prepare target image (convert BGR to RGB, normalize to [0, 1])
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    target_rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
                else:
                    target_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                target_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            target_rgb = target_rgb.astype(np.float32) / 255.0
            
            # Convert Gaussians to serializable format
            gaussians_data = []
            for g in gaussians:
                gaussians_data.append({
                    'position': g.position.tolist(),
                    'rotation': g.rotation.tolist(),
                    'scale': g.scale.tolist(),
                    'opacity': float(g.opacity),
                    'color': g.color.tolist()
                })
            
            # Call subprocess optimization
            generator = get_generator()
            future = generator.optimize_brush_gaussians(
                gaussians_data=gaussians_data,
                target_image=target_rgb.tolist(),
                target_alpha=alpha_mask.tolist(),
                iterations=self.config.optimization_iterations,
                render_size=self.config.optimization_render_size
            )
            result = future.result() if hasattr(future, 'result') else future
            
            if result.get('success'):
                # Convert back to Gaussian2D objects
                optimized = []
                for g_data in result['gaussians']:
                    g = Gaussian2D(
                        position=np.array(g_data['position'], dtype=np.float32),
                        rotation=np.array(g_data['rotation'], dtype=np.float32),
                        scale=np.array(g_data['scale'], dtype=np.float32),
                        opacity=float(g_data['opacity']),
                        color=np.array(g_data['color'], dtype=np.float32)
                    )
                    optimized.append(g)
                
                logger.info(f"[BrushConverter] ✓ Optimization complete ({result.get('elapsed_ms', 0):.0f}ms)")
                return optimized
            else:
                if result.get('fallback'):
                    error_msg = result.get('error', 'unknown reason')
                    logger.warning(f"[BrushConverter] gsplat not available, using heuristic only. Reason: {error_msg}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"[BrushConverter] Optimization failed: {error_msg}")
                # Log full traceback if available (always, not just debug)
                if result.get('traceback'):
                    logger.warning(f"[BrushConverter] Subprocess traceback:\n{result['traceback']}")
                return gaussians
                
        except ImportError as e:
            logger.warning(f"[BrushConverter] Generator process not available: {e}")
            return gaussians
        except Exception as e:
            import traceback
            logger.warning(f"[BrushConverter] Optimization error: {e}")
            logger.debug(f"[BrushConverter] Full traceback:\n{traceback.format_exc()}")
            return gaussians
    
    def _create_brush_stamp(
        self,
        gaussians: List[Gaussian2D],
        brush_name: str
    ) -> BrushStamp:
        """Create BrushStamp from Gaussians"""
        
        if len(gaussians) == 0:
            logger.warning("[BrushConverter] Creating empty brush")
            center = np.zeros(3, dtype=np.float32)
            size = 0.1
        else:
            positions = np.array([g.position for g in gaussians])
            center = np.mean(positions, axis=0)
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            size = float(np.linalg.norm(max_pos - min_pos))
        
        # Create brush
        brush = BrushStamp()
        brush.base_gaussians = gaussians
        brush.center = center.astype(np.float32)
        brush.size = size
        brush.tangent = np.array([1, 0, 0], dtype=np.float32)
        brush.normal = np.array([0, 0, 1], dtype=np.float32)
        brush.binormal = np.array([0, 1, 0], dtype=np.float32)
        brush.spacing = size * 0.5
        
        # Apply default parameters
        brush.apply_parameters()
        
        # Metadata
        brush.metadata = {
            'name': brush_name,
            'gaussian_count': len(gaussians),
            'type': 'converted',
            'source': 'image',
            'config': {
                'num_gaussians': self.config.num_gaussians,
                'depth_profile': self.config.depth_profile,
                'sampling_method': self.config.sampling_method,
            }
        }
        
        return brush
