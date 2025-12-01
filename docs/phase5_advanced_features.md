# Phase 5: 고급 기능 (Advanced Features with gsplat)

**기간**: 2주  
**목표**: gsplat 기반 최적화 기능 (Inpainting, Final Render)

---

## 📋 작업 개요

본 Phase는 Hybrid 아키텍처에서 **gsplat의 고급 기능**을 활용합니다:
- ✓ Inpainting Optimization (differentiable rendering)
- ✓ Final Render Engine (F12 고품질 출력)
- ✓ Export 기능 (PLY, Image, Video)

---

## 🎯 핵심 작업

### 1. Inpainting Optimization (gsplat)

#### 1.1 개념

**목적**: 스트로크 완료 후 overlapping Gaussians를 최적화하여 자연스러운 블렌딩

**방법**: gsplat의 differentiable rendering으로 loss 계산 → gradient descent

```
Before Optimization:        After Optimization:
┌─────────────┐            ┌─────────────┐
│  ▓▓▓▓       │            │  ████       │
│ ▓▓▓▓▓▓▓     │            │ ███████     │
│  ▓▓?▓▓▓     │  →         │  ██████     │  (smooth blend)
│   ▓▓▓▓      │            │   ████      │
└─────────────┘            └─────────────┘
```

#### 1.2 구현

```python
# inpainting_optimizer.py

import torch
import torch.nn as nn
from gsplat import rasterization

class InpaintingOptimizer:
    """
    gsplat-based inpainting optimizer for overlapping gaussians.
    """
    
    def __init__(self, scene_data, viewport_renderer):
        self.scene_data = scene_data
        self.viewport_renderer = viewport_renderer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def optimize_region(self, region_bounds, iterations=100, lr=0.01):
        """
        Optimize gaussians in specified region.
        
        Args:
            region_bounds: dict with 'min' and 'max' coordinates
            iterations: Number of optimization steps
            lr: Learning rate
        
        Returns:
            Optimized scene_data
        """
        # 1. Extract gaussians in region
        mask = self.get_region_mask(region_bounds)
        regional_gaussians = self.scene_data.gaussians[mask]
        
        # 2. Convert to PyTorch tensors (trainable)
        means = torch.tensor(
            [g.position for g in regional_gaussians],
            device=self.device, dtype=torch.float32, requires_grad=True
        )
        
        quats = torch.tensor(
            [g.rotation_quat for g in regional_gaussians],
            device=self.device, dtype=torch.float32, requires_grad=True
        )
        
        scales = torch.tensor(
            [g.scale for g in regional_gaussians],
            device=self.device, dtype=torch.float32, requires_grad=True
        )
        
        opacities = torch.tensor(
            [g.opacity for g in regional_gaussians],
            device=self.device, dtype=torch.float32, requires_grad=True
        )
        
        colors = torch.tensor(
            [g.color for g in regional_gaussians],
            device=self.device, dtype=torch.float32, requires_grad=True
        )
        
        # 3. Target image (current rendering without optimized region)
        target_image = self.render_without_region(mask)
        target_tensor = torch.from_numpy(target_image).to(self.device)
        
        # 4. Setup optimizer
        optimizer = torch.optim.Adam([
            {'params': means, 'lr': lr},
            {'params': quats, 'lr': lr * 0.1},
            {'params': scales, 'lr': lr},
            {'params': opacities, 'lr': lr},
            {'params': colors, 'lr': lr},
        ])
        
        # 5. Optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Render with current parameters (gsplat)
            rendered = self.render_gsplat(
                means, quats, scales, opacities, colors
            )
            
            # Compute loss
            loss = self.compute_loss(rendered, target_tensor)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Viewport preview every 10 iterations
            if i % 10 == 0:
                self.update_viewport_preview(
                    mask, means, quats, scales, opacities, colors
                )
                print(f"Iteration {i}/{iterations}, Loss: {loss.item():.4f}")
        
        # 6. Update scene_data with optimized parameters
        self.update_scene_data(
            mask, means, quats, scales, opacities, colors
        )
        
        return self.scene_data
    
    def render_gsplat(self, means, quats, scales, opacities, colors):
        """
        Render gaussians using gsplat.
        
        Returns:
            torch.Tensor: Rendered image [H, W, 3]
        """
        # Camera parameters (from viewport)
        viewmat = self.get_view_matrix()
        K = self.get_intrinsics()
        
        # Render
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=self.viewport_width,
            height=self.viewport_height,
            render_mode="RGB",
        )
        
        return render_colors[0]  # [H, W, 3]
    
    def compute_loss(self, rendered, target):
        """
        Compute optimization loss.
        
        Args:
            rendered: torch.Tensor [H, W, 3]
            target: torch.Tensor [H, W, 3]
        
        Returns:
            torch.Tensor: scalar loss
        """
        # MSE loss
        mse_loss = torch.nn.functional.mse_loss(rendered, target)
        
        # Opacity regularization (prevent over-dense regions)
        opacity_reg = torch.mean(torch.abs(opacities - 0.5)) * 0.01
        
        # Total loss
        total_loss = mse_loss + opacity_reg
        
        return total_loss
    
    def update_viewport_preview(self, mask, means, quats, scales, opacities, colors):
        """
        Update GLSL viewport with current optimization state.
        
        Args:
            mask: Boolean mask of gaussians being optimized
            means, quats, scales, opacities, colors: Current parameters
        """
        # Convert to NumPy
        means_np = means.detach().cpu().numpy()
        quats_np = quats.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        opacities_np = opacities.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()
        
        # Update scene_data (temporary)
        indices = np.where(mask)[0]
        for i, idx in enumerate(indices):
            self.scene_data.gaussians[idx].position = means_np[i]
            self.scene_data.gaussians[idx].rotation_quat = quats_np[i]
            self.scene_data.gaussians[idx].scale = scales_np[i]
            self.scene_data.gaussians[idx].opacity = opacities_np[i]
            self.scene_data.gaussians[idx].color = colors_np[i]
        
        # Sync to GLSL viewport
        self.viewport_renderer.data_manager.update_partial(
            indices[0], indices[-1] + 1,
            self.viewport_renderer.data_manager.pack_gaussians_range(
                indices[0], indices[-1] + 1
            )
        )
    
    def get_region_mask(self, bounds):
        """
        Get boolean mask of gaussians within bounds.
        
        Args:
            bounds: dict with 'min' [x,y,z] and 'max' [x,y,z]
        
        Returns:
            np.ndarray: Boolean mask
        """
        positions = np.array([g.position for g in self.scene_data.gaussians])
        
        mask = np.all([
            positions[:, 0] >= bounds['min'][0],
            positions[:, 0] <= bounds['max'][0],
            positions[:, 1] >= bounds['min'][1],
            positions[:, 1] <= bounds['max'][1],
            positions[:, 2] >= bounds['min'][2],
            positions[:, 2] <= bounds['max'][2],
        ], axis=0)
        
        return mask
```

#### 1.3 Operator 통합

```python
# operators.py

class InpaintingOperator(bpy.types.Operator):
    """Optimize overlapping gaussians (Inpainting)"""
    bl_idname = "gaussian.inpaint"
    bl_label = "Inpaint Region"
    bl_options = {'REGISTER', 'UNDO'}
    
    iterations: bpy.props.IntProperty(name="Iterations", default=100, min=10, max=1000)
    learning_rate: bpy.props.FloatProperty(name="Learning Rate", default=0.01, min=0.001, max=0.1)
    
    def invoke(self, context, event):
        # Get selected region bounds (from 3D cursor or selection)
        region_bounds = self.get_selected_region(context)
        
        if region_bounds is None:
            self.report({'ERROR'}, "No region selected")
            return {'CANCELLED'}
        
        # Initialize optimizer
        scene_data = context.scene.gaussian_scene_data
        viewport_renderer = context.scene.gaussian_viewport_renderer
        
        self.optimizer = InpaintingOptimizer(scene_data, viewport_renderer)
        
        # Run optimization (modal for progress display)
        return self.execute(context)
    
    def execute(self, context):
        # Get region bounds
        region_bounds = self.get_selected_region(context)
        
        # Optimize
        optimized_scene = self.optimizer.optimize_region(
            region_bounds,
            iterations=self.iterations,
            lr=self.learning_rate
        )
        
        # Update viewport
        viewport_renderer = context.scene.gaussian_viewport_renderer
        viewport_renderer.update_gaussians(optimized_scene)
        
        self.report({'INFO'}, f"Inpainting completed ({self.iterations} iterations)")
        return {'FINISHED'}
```

---

### 2. Final Render Engine (gsplat)

#### 2.1 구현

```python
# render_engine.py

import bpy
import numpy as np
import torch
from gsplat import rasterization

class NPRGaussianRenderEngine(bpy.types.RenderEngine):
    """
    Custom render engine using gsplat for high-quality output.
    """
    bl_idname = "NPR_GAUSSIAN"
    bl_label = "NPR Gaussian Painter"
    bl_use_preview = False
    
    def render(self, depsgraph):
        """
        Main render function (F12 or animation render).
        
        Args:
            depsgraph: Blender dependency graph
        """
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        
        # Get gaussian data
        scene_data = scene.gaussian_scene_data
        
        # Render with gsplat
        rendered_image = self.render_gsplat_highquality(
            scene_data, width, height, depsgraph
        )
        
        # Write to render result
        result = self.begin_result(0, 0, width, height)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rendered_image.flatten()
        self.end_result(result)
    
    def render_gsplat_highquality(self, scene_data, width, height, depsgraph):
        """
        High-quality gsplat rendering.
        
        Returns:
            np.ndarray: RGBA image [H, W, 4]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert scene_data to tensors
        means = torch.tensor(
            [g.position for g in scene_data.gaussians],
            device=device, dtype=torch.float32
        )
        quats = torch.tensor(
            [g.rotation_quat for g in scene_data.gaussians],
            device=device, dtype=torch.float32
        )
        scales = torch.tensor(
            [g.scale for g in scene_data.gaussians],
            device=device, dtype=torch.float32
        )
        opacities = torch.tensor(
            [g.opacity for g in scene_data.gaussians],
            device=device, dtype=torch.float32
        )
        colors = torch.tensor(
            [g.color for g in scene_data.gaussians],
            device=device, dtype=torch.float32
        )
        
        # Camera parameters from Blender
        camera = depsgraph.scene.camera
        viewmat, K = self.get_camera_matrices(camera, width, height)
        
        # Render
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            render_mode="RGB+D",  # RGB + Depth
        )
        
        # Convert to RGBA
        rgb = render_colors[0].cpu().numpy()  # [H, W, 3]
        alpha = render_alphas[0].cpu().numpy()  # [H, W, 1]
        rgba = np.concatenate([rgb, alpha], axis=-1)
        
        return rgba
    
    def get_camera_matrices(self, camera, width, height):
        """
        Extract camera view and intrinsic matrices.
        
        Returns:
            tuple: (viewmat: torch.Tensor [4, 4], K: torch.Tensor [3, 3])
        """
        # View matrix (world to camera)
        viewmat = np.array(camera.matrix_world.inverted())
        viewmat_tensor = torch.from_numpy(viewmat).float()
        
        # Intrinsic matrix (camera to screen)
        focal_length = camera.data.lens  # mm
        sensor_width = camera.data.sensor_width  # mm
        focal_x = focal_length / sensor_width * width
        focal_y = focal_x  # Assuming square pixels
        
        K = torch.tensor([
            [focal_x, 0, width / 2],
            [0, focal_y, height / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return viewmat_tensor, K
```

#### 2.2 등록

```python
# __init__.py

def register():
    # ... other registrations ...
    bpy.utils.register_class(NPRGaussianRenderEngine)

def unregister():
    bpy.utils.unregister_class(NPRGaussianRenderEngine)
```

---

### 3. Export 기능

#### 3.1 PLY Export

```python
# export_operators.py

class ExportPLYOperator(bpy.types.Operator):
    """Export Gaussians to PLY file"""
    bl_idname = "gaussian.export_ply"
    bl_label = "Export PLY"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        scene_data = context.scene.gaussian_scene_data
        
        # Write PLY
        self.write_ply(scene_data, self.filepath)
        
        self.report({'INFO'}, f"Exported to {self.filepath}")
        return {'FINISHED'}
    
    def write_ply(self, scene_data, filepath):
        """
        Write gaussian splatting PLY file.
        
        Format:
        - Header: vertex count, properties
        - Data: position, normal, sh_coeffs, opacity, scale, rotation
        """
        N = len(scene_data.gaussians)
        
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        
        # Data
        data = np.zeros((N, 17), dtype=np.float32)
        for i, g in enumerate(scene_data.gaussians):
            data[i, 0:3] = g.position
            data[i, 3:6] = g.normal if hasattr(g, 'normal') else [0, 0, 1]
            data[i, 6:9] = g.color  # SH degree 0
            data[i, 9] = g.opacity
            data[i, 10:13] = g.scale
            data[i, 13:17] = g.rotation_quat
        
        # Write
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            data.tofile(f)
```

---

## 🧪 테스트 및 검증

### Inpainting 테스트

```python
def test_inpainting():
    """Test inpainting optimization."""
    import bpy
    
    # Create test scene with overlapping gaussians
    # ...
    
    # Run inpainting
    bpy.ops.gaussian.inpaint('INVOKE_DEFAULT', iterations=100)
    
    # Verify smoothness
    # ...
    
    print("✓ Inpainting test passed")
```

### Final Render 테스트

```python
def test_final_render():
    """Test F12 rendering with gsplat."""
    import bpy
    
    # Set render engine
    bpy.context.scene.render.engine = 'NPR_GAUSSIAN'
    
    # Render
    bpy.ops.render.render(write_still=True)
    
    # Check output
    # ...
    
    print("✓ Final render test passed")
```

### 성능 목표

- ✓ Inpainting: 100 iterations < 10초
- ✓ Final render: 1920×1080 < 30초
- ✓ PLY export: 100k gaussians < 5초

---

## 📚 참고 자료

- gsplat documentation: https://github.com/nerfstudio-project/gsplat
- Blender Render Engine API
- PLY file format specification
