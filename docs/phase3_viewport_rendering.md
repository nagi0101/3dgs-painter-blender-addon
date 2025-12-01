# Phase 3: Viewport 렌더링 구현 (GLSL)

**기간**: 2주  
**목표**: KIRI Innovation 방식 기반 고성능 Viewport 렌더링 (Hybrid의 Viewport 부분)

---

## 📋 작업 개요

본 Phase는 Hybrid 아키텍처의 **Viewport 렌더링** 부분을 구현합니다.
- ✓ GLSL Instanced Rendering
- ✓ Blender Depth Integration
- ✓ 60 FPS @ 10k gaussians 목표

---

## 🎯 핵심 작업

### 1. GLSL Shader 구현

#### 1.1 Vertex Shader (`gaussian_vert.glsl`)

**참고**: KIRI Innovation의 `vert.glsl` 구조 활용

```glsl
#version 330

// Uniforms
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;
uniform vec3 camera_position;
uniform vec2 focal_parameters;  // (focal_x, focal_y)
uniform sampler3D gaussian_data;  // 3D texture: N×1×1, 59 floats per gaussian
uniform usampler2D sorted_indices;  // Sorting order
uniform vec2 texture_dimensions;  // (width, height)

// Outputs to fragment shader
out vec4 v_color;
out float v_alpha;
out vec3 v_conic;  // 2D covariance inverse
out vec2 v_coordxy;  // Pixel offset from center
out float v_depth;

// Spherical Harmonics constants (degree 0-1)
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

// Helper: 1D index → 3D texture coordinate
ivec3 indexTo3D(int index, int width, int height) {
    int z = index / (width * height);
    int remainder = index - z * (width * height);
    int y = remainder / width;
    int x = remainder - y * width;
    return ivec3(x, y, z);
}

// Fetch gaussian data (59-float stride)
vec3 getPosition(int idx, int w, int h) {
    return vec3(
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 0, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 1, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 2, w, h), 0).r
    );
}

vec4 getRotation(int idx, int w, int h) {
    return vec4(
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 3, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 4, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 5, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 6, w, h), 0).r
    );
}

vec3 getScale(int idx, int w, int h) {
    return vec3(
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 7, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 8, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(idx * 59 + 9, w, h), 0).r
    );
}

float getOpacity(int idx, int w, int h) {
    return texelFetch(gaussian_data, indexTo3D(idx * 59 + 10, w, h), 0).r;
}

vec3 getSHCoeff(int idx, int sh_idx, int w, int h) {
    int base = idx * 59 + 11 + sh_idx * 3;
    return vec3(
        texelFetch(gaussian_data, indexTo3D(base + 0, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(base + 1, w, h), 0).r,
        texelFetch(gaussian_data, indexTo3D(base + 2, w, h), 0).r
    );
}

// Compute 3D covariance from scale and rotation
mat3 computeCov3D(vec3 scale, vec4 q) {
    mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    
    float r = q.x, x = q.y, y = q.z, z = q.w;
    mat3 R = mat3(
        1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - r*z), 2.0 * (x*z + r*y),
        2.0 * (x*y + r*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - r*x),
        2.0 * (x*z - r*y), 2.0 * (y*z + r*x), 1.0 - 2.0 * (x*x + y*y)
    );
    
    mat3 M = S * R;
    return transpose(M) * M;
}

// Project 3D covariance to 2D screen space
vec3 computeCov2D(vec4 mean_view, mat3 cov3D, float focal_x, float focal_y) {
    vec4 t = mean_view;
    
    mat3 J = mat3(
        focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z),
        0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0
    );
    
    mat3 W = transpose(mat3(ViewMatrix));
    mat3 T = W * J;
    
    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3;  // Regularization
    cov[1][1] += 0.3;
    
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

// Evaluate Spherical Harmonics (degree 0-1)
vec3 evaluateSH(int idx, vec3 dir, int w, int h) {
    vec3 color = SH_C0 * getSHCoeff(idx, 0, w, h);
    
    // Degree 1
    color += -SH_C1 * dir.y * getSHCoeff(idx, 1, w, h);
    color += SH_C1 * dir.z * getSHCoeff(idx, 2, w, h);
    color += -SH_C1 * dir.x * getSHCoeff(idx, 3, w, h);
    
    return color + 0.5;
}

void main() {
    int instance_id = gl_InstanceID;
    
    // Get sorted gaussian index
    ivec2 dims = ivec2(texture_dimensions);
    int indices_y = instance_id / dims.x;
    int indices_x = instance_id - indices_y * dims.x;
    int gaussian_index = int(texelFetch(sorted_indices, ivec2(indices_x, indices_y), 0).r);
    
    int tex_w = dims.x;
    int tex_h = dims.y;
    
    // Fetch gaussian parameters
    vec3 g_pos = getPosition(gaussian_index, tex_w, tex_h);
    vec4 g_rot = getRotation(gaussian_index, tex_w, tex_h);
    vec3 g_scale = getScale(gaussian_index, tex_w, tex_h);
    float g_opacity = getOpacity(gaussian_index, tex_w, tex_h);
    
    // Transform to view space
    vec4 g_pos_view = ViewMatrix * vec4(g_pos, 1.0);
    vec4 g_pos_screen = ProjectionMatrix * g_pos_view;
    g_pos_screen.xyz /= g_pos_screen.w;
    
    // Frustum culling
    if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3)))) {
        gl_Position = vec4(-100, -100, -100, 1);
        return;
    }
    
    // Compute 2D covariance
    mat3 cov3d = computeCov3D(g_scale, g_rot);
    vec3 cov2d = computeCov2D(g_pos_view, cov3d, focal_parameters.x, focal_parameters.y);
    
    // Inverse covariance (conic)
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.0) {
        gl_Position = vec4(0, 0, 0, 0);
        return;
    }
    float det_inv = 1.0 / det;
    vec3 conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    
    // Billboard quad size (3-sigma)
    vec2 quadwh_scr = vec2(3.0 * sqrt(cov2d.x), 3.0 * sqrt(cov2d.z));
    vec2 wh = 2.0 * focal_parameters * focal_parameters.z;  // Screen size
    vec2 quadwh_ndc = quadwh_scr / wh * 2.0;
    
    // Quad vertex offset
    vec2 quad_coords[4] = vec2[](
        vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1)
    );
    vec2 quad_coord = quad_coords[gl_VertexID % 4];
    
    g_pos_screen.xy += quad_coord * quadwh_ndc;
    vec2 coordxy = quad_coord * quadwh_scr;
    
    gl_Position = g_pos_screen;
    
    // Evaluate color (Spherical Harmonics)
    vec3 view_dir = normalize(g_pos - camera_position);
    v_color = vec4(evaluateSH(gaussian_index, view_dir, tex_w, tex_h), 1.0);
    v_alpha = g_opacity;
    v_conic = conic;
    v_coordxy = coordxy;
    v_depth = (g_pos_screen.z + 1.0) * 0.5;  // Normalized depth
}
```

#### 1.2 Fragment Shader (`gaussian_frag.glsl`)

```glsl
#version 330

// Inputs from vertex shader
in vec4 v_color;
in float v_alpha;
in vec3 v_conic;
in vec2 v_coordxy;
in float v_depth;

// Uniforms
uniform sampler2D blender_depth;  // Blender's depth buffer
uniform vec2 depth_texture_size;  // Screen size

// Output
out vec4 fragColor;

void main() {
    // Depth test with Blender scene
    vec2 screen_coord = gl_FragCoord.xy / depth_texture_size;
    float sampled_depth = texture(blender_depth, screen_coord).r;
    
    if (v_depth > sampled_depth) {
        discard;  // Behind Blender object
    }
    
    // Gaussian splat evaluation
    float power = -0.5 * (v_conic.x * v_coordxy.x * v_coordxy.x + 
                           v_conic.z * v_coordxy.y * v_coordxy.y) - 
                   v_conic.y * v_coordxy.x * v_coordxy.y;
    
    if (power > 0.0) discard;
    
    float opacity = min(0.99, v_alpha * exp(power));
    if (opacity < 0.00392) discard;  // 1/255 threshold
    
    fragColor = vec4(v_color.rgb, opacity);
}
```

---

### 2. Texture 기반 데이터 관리 (`gaussian_data.py`)

```python
import numpy as np
import gpu
from mathutils import Quaternion

class GaussianDataManager:
    """
    Manages Gaussian data in GPU 3D Texture format (59 floats per gaussian).
    
    Data layout:
    [0-2]:   position (vec3)
    [3-6]:   rotation quaternion (vec4, w,x,y,z)
    [7-9]:   scale (vec3)
    [10]:    opacity (float)
    [11-58]: spherical harmonics (16 bands × 3 = 48 floats)
    """
    
    def __init__(self):
        self.texture_3d = None
        self.gaussian_count = 0
        self.data_buffer = None  # NumPy array cache
        
    def update_from_npr_core(self, scene_data):
        """
        Update GPU texture from npr_core SceneData.
        
        Args:
            scene_data: npr_core.scene_data.SceneData instance
        """
        # Convert to 59-float stride format
        data = self.pack_gaussians(scene_data)
        self.gaussian_count = len(scene_data.gaussians)
        
        # Upload to GPU
        self.upload_to_texture(data)
    
    def pack_gaussians(self, scene_data):
        """
        Pack gaussian data into 59-float stride format.
        
        Returns:
            np.ndarray: Shape (N, 59), dtype float32
        """
        N = len(scene_data.gaussians)
        data = np.zeros((N, 59), dtype=np.float32)
        
        for i, g in enumerate(scene_data.gaussians):
            # Position [0-2]
            data[i, 0:3] = g.position
            
            # Rotation [3-6] (quaternion w,x,y,z)
            # Convert from numpy or compute from covariance
            if hasattr(g, 'rotation_quat'):
                data[i, 3:7] = g.rotation_quat  # [w, x, y, z]
            else:
                # Compute from covariance matrix
                quat = self.rotation_from_covariance(g.covariance)
                data[i, 3:7] = quat
            
            # Scale [7-9]
            if hasattr(g, 'scale'):
                data[i, 7:10] = g.scale
            else:
                # Compute from covariance eigenvalues
                scale = self.scale_from_covariance(g.covariance)
                data[i, 7:10] = scale
            
            # Opacity [10]
            data[i, 10] = g.opacity
            
            # Spherical Harmonics [11-58]
            # For simplicity, use degree 0 (constant color)
            if hasattr(g, 'sh_coeffs'):
                data[i, 11:59] = g.sh_coeffs  # 48 floats
            else:
                # Degree 0: RGB color
                data[i, 11:14] = g.color
                # Remaining coefficients = 0
        
        return data
    
    def upload_to_texture(self, data):
        """
        Upload data to GPU 3D texture.
        
        Args:
            data: np.ndarray, shape (N, 59)
        """
        N = data.shape[0]
        
        # 3D texture dimensions: (width, height, depth)
        # For N gaussians × 59 floats, use (width, 1, depth)
        width = min(N, 2048)  # Max texture width
        depth = (N * 59 + width - 1) // width
        
        # Flatten and pad
        flat_data = data.flatten()
        required_size = width * depth
        if len(flat_data) < required_size:
            flat_data = np.pad(flat_data, (0, required_size - len(flat_data)))
        
        # Reshape to (width, 1, depth)
        texture_data = flat_data[:required_size].reshape(depth, 1, width)
        
        # Create/update GPU texture
        if self.texture_3d is None:
            self.texture_3d = gpu.types.GPUTexture(
                (width, 1, depth),
                format='R32F',  # Single-channel float
                data=texture_data
            )
        else:
            # Update existing texture
            self.texture_3d.clear(format='R32F', value=(0,))
            # Note: Blender GPU API may not support direct texture update
            # Recreate texture if needed
            self.texture_3d = gpu.types.GPUTexture(
                (width, 1, depth),
                format='R32F',
                data=texture_data
            )
        
        self.data_buffer = data  # Cache for partial updates
    
    def update_partial(self, start_idx, end_idx, new_data):
        """
        Update a subset of gaussians (for incremental painting).
        
        Args:
            start_idx: int
            end_idx: int
            new_data: np.ndarray, shape (end_idx - start_idx, 59)
        """
        if self.data_buffer is None:
            raise RuntimeError("Must call update_from_npr_core first")
        
        # Update cache
        self.data_buffer[start_idx:end_idx] = new_data
        
        # Re-upload entire texture
        # (Partial texture update not well-supported in Blender GPU API)
        self.upload_to_texture(self.data_buffer)
    
    def rotation_from_covariance(self, cov_matrix):
        """
        Extract rotation quaternion from 2×2 covariance matrix.
        
        Returns:
            np.ndarray: [w, x, y, z]
        """
        # Simplified: assume aligned with axes
        # For full implementation, use eigendecomposition
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def scale_from_covariance(self, cov_matrix):
        """
        Extract scale from 2×2 covariance matrix.
        
        Returns:
            np.ndarray: [sx, sy, sz]
        """
        # Eigenvalues = variance
        eig_vals = np.linalg.eigvalsh(cov_matrix)
        return np.array([np.sqrt(eig_vals[0]), np.sqrt(eig_vals[1]), 0.01], dtype=np.float32)
```

---

### 3. Draw Handler 등록 (`viewport_renderer.py`)

```python
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

class GaussianViewportRenderer:
    """
    GLSL-based viewport renderer for Gaussians (Hybrid architecture).
    """
    
    def __init__(self):
        # Load shaders
        with open("shaders/gaussian_vert.glsl") as f:
            vert_code = f.read()
        with open("shaders/gaussian_frag.glsl") as f:
            frag_code = f.read()
        
        self.shader = gpu.types.GPUShader(vert_code, frag_code)
        self.data_manager = GaussianDataManager()
        self.draw_handle = None
        
    def register(self):
        """Register draw handler."""
        self.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw, (), 'WINDOW', 'POST_VIEW'
        )
    
    def unregister(self):
        """Unregister draw handler."""
        if self.draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handle, 'WINDOW')
            self.draw_handle = None
    
    def draw(self):
        """Main draw call (called every frame)."""
        if self.data_manager.texture_3d is None:
            return  # No data to render
        
        context = bpy.context
        region = context.region
        region_3d = context.space_data.region_3d
        
        # Get camera matrices
        view_matrix = region_3d.view_matrix
        projection_matrix = region_3d.window_matrix
        camera_position = region_3d.view_location
        
        # Get Blender depth buffer
        # Note: Requires custom implementation or offscreen rendering
        depth_texture = self.get_blender_depth_buffer(context)
        
        # Set GPU state
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.blend_set('ALPHA')
        
        # Bind shader and set uniforms
        self.shader.bind()
        self.shader.uniform_float("ViewMatrix", view_matrix)
        self.shader.uniform_float("ProjectionMatrix", projection_matrix)
        self.shader.uniform_float("camera_position", camera_position)
        
        # Focal parameters (simplified)
        focal_x = projection_matrix[0][0] * region.width / 2.0
        focal_y = projection_matrix[1][1] * region.height / 2.0
        self.shader.uniform_float("focal_parameters", (focal_x, focal_y))
        
        # Textures
        self.shader.uniform_sampler("gaussian_data", self.data_manager.texture_3d)
        if depth_texture:
            self.shader.uniform_sampler("blender_depth", depth_texture)
            self.shader.uniform_float("depth_texture_size", (region.width, region.height))
        
        # Texture dimensions
        tex_dims = self.data_manager.texture_3d.size
        self.shader.uniform_float("texture_dimensions", (tex_dims[0], tex_dims[2]))
        
        # Create quad vertices for instanced rendering
        vertices = [
            (0, 0), (1, 0), (0, 1), (1, 1)  # Quad corners
        ]
        batch = batch_for_shader(
            self.shader, 'TRI_STRIP', {"position": vertices}
        )
        
        # Draw instances (one per gaussian)
        batch.draw(self.shader, instances=self.data_manager.gaussian_count)
    
    def get_blender_depth_buffer(self, context):
        """
        Get Blender's depth buffer as texture.
        
        TODO: Implement offscreen rendering or depth buffer readback.
        """
        # Placeholder
        return None
    
    def update_gaussians(self, scene_data):
        """Update gaussian data from npr_core."""
        self.data_manager.update_from_npr_core(scene_data)
```

---

## 🧪 테스트 및 검증

### 성능 벤치마크

```python
# Test script
import time
import numpy as np

def benchmark_viewport():
    renderer = GaussianViewportRenderer()
    renderer.register()
    
    # Generate test data
    from npr_core.scene_data import SceneData
    scene = SceneData()
    for i in range(10000):
        scene.add_gaussian(
            position=np.random.randn(3),
            covariance=np.eye(2) * 0.01,
            opacity=0.5,
            color=np.random.rand(3)
        )
    
    renderer.update_gaussians(scene)
    
    # Measure FPS
    frame_times = []
    for _ in range(100):
        start = time.time()
        bpy.context.area.tag_redraw()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN', iterations=1)
        frame_times.append(time.time() - start)
    
    avg_fps = 1.0 / np.mean(frame_times)
    print(f"10,000 Gaussians: {avg_fps:.1f} FPS")
    
    renderer.unregister()

benchmark_viewport()
```

### 성공 기준

- ✓ 10,000 Gaussians @ 30+ FPS
- ✓ Depth integration with Blender objects
- ✓ Smooth camera navigation
- ✓ Memory usage < 2GB VRAM

---

## 📚 참고 자료

- KIRI Innovation repo: https://github.com/KIRI-Innovation/3dgs-render-blender-addon
- Blender GPU module docs: https://docs.blender.org/api/current/gpu.html
- 3D Gaussian Splatting paper: SIGGRAPH 2023
