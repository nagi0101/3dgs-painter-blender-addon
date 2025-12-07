# **Architectural Paradigms for Real-Time 3D Gaussian Splatting in Blender OpenXR Workflows**

## **1\. Introduction: The Intersection of Volumetric Rendering and Immersive Display Architectures**

The integration of 3D Gaussian Splatting (3DGS) into Virtual Reality (VR) environments represents a convergence of two rapidly evolving fields: real-time neural rendering and immersive display standards. For developers leveraging Blender as a content creation and visualization platform, the transition from traditional polygonal rasterization to splat-based rendering introduces significant architectural friction. This friction is most acute when attempting to bridge Blender's Python-accessible OpenGL context with the rigid, latency-sensitive requirements of the OpenXR specification used by devices such as the Meta Quest 3\.

The user's specific challenge—where custom GLSL shaders injected via draw\_handler\_add function correctly in the desktop viewport but fail to appear in the VR headset—is not merely a bug but a symptom of the structural decoupling between Blender’s windowing system and its OpenXR session management. To resolve this, one must move beyond standard API usage and engage with the lower-level mechanics of the Ghost-XR abstraction layer, the GPU rendering pipeline, and the mathematical foundations of stereoscopic projection. This report provides an exhaustive analysis of these systems, proposing three distinct architectural methodologies to achieve high-fidelity 3DGS painting in VR: a Geometry Node Proxy Pipeline, a Stereo-Aware Offscreen Buffer Injection system, and a theoretical framework for Native Extension.

### **1.1 The Mathematical and Computational Nature of 3D Gaussian Splatting**

To appreciate the rendering challenges, one must first define the computational workload of 3DGS. Unlike mesh-based rendering, which relies on hardware-accelerated triangle rasterization, 3DGS represents a scene as a collection of anisotropic 3D Gaussians. Each Gaussian $G$ is defined by a mean position $\\mu$, a covariance matrix $\\Sigma$, opacity $\\alpha$, and view-dependent color coefficients (Spherical Harmonics).1

The radiance field is rendered by projecting these 3D Gaussians into 2D splats on the image plane. The covariance matrix $\\Sigma$ in 3D is projected to 2D covariance $\\Sigma'$ using the viewing transformation $W$ and the Jacobian of the affine approximation of the projection matrix $J$:

$$\\Sigma' \= J W \\Sigma W^T J^T$$  
This projection determines the extent and orientation of the splat in screen space. Crucially, the rendering process requires $\\alpha$-blending of these overlapping splats. To ensure correct occlusion and transparency, the splats must be sorted by depth (distance from the camera center) for every single frame.3

In a desktop environment, this sorting and rasterization are typically handled by a CUDA or Compute Shader kernel. However, when interfacing with Blender via Python, developers often rely on the gpu module to draw instances or points using custom GLSL shaders. While efficient enough for a single viewport, the VR context imposes a "double-duty" requirement: the scene must be rendered twice (once for each eye) with distinct view and projection matrices, and crucially, the depth sorting must be valid for *both* eye positions simultaneously, or performed twice per frame.5

### **1.2 The OpenXR Rendering Constraint in Blender**

The core failure mode described—shaders appearing on the PC but not the HMD—stems from how Blender isolates its VR render loop. Blender's VR implementation is built upon the OpenXR standard, which abstracts hardware specifics.7

When a VR session is initiated:

1. **Session Creation:** Blender interacts with the OpenXR runtime (e.g., Oculus/Meta Link, SteamVR) via the Ghost-XR library.  
2. **Swapchain Acquisition:** The OpenXR runtime provides a set of swapchain images (textures) for the Left and Right eyes.  
3. **The Render Pass:** Blender's internal engine (Eevee or Workbench) iterates through the scene graph, renders to these swapchain images, and submits them to the runtime for display.7

The critical disconnect lies in the execution context of draw\_handler\_add. These Python callbacks are registered to the SpaceView3D region of the *desktop window*. They execute during the window manager's draw cycle to overlay content onto the *desktop framebuffer*. The VR render loop, however, is often a separate path that does not trigger these window-region callbacks, or if it does, it renders to a specific offscreen context that the Python API does not automatically bind to.9 Consequently, the custom GLSL draw calls issued by the Python script are effectively drawing "invisible ink" onto the desktop overlay, which is never copied to the OpenXR swapchain sent to the headset.

### **1.3 Scope of Solutions**

Addressing this requires establishing a data pathway that persists across these isolated contexts. This report analyzes solutions ranging from high-level geometric abstraction to low-level framebuffer manipulation:

* **The Geometry Nodes Proxy:** Converting splats to native mesh data that Blender's VR-compliant engines can see.11  
* **The Offscreen Injection:** Manually rendering stereo pairs to textures and projecting them into the VR scene.12  
* **Native Engine Integration:** Understanding the limits of Python and where C++ extensions become necessary.13

## ---

**2\. Architectural Analysis: Why the Python Draw Handler Fails in VR**

To engineer a robust solution, we must first dissect the failure mechanism of the standard draw\_handler\_add approach within the OpenXR ecosystem. This analysis reveals why standard viewport techniques are insufficient for HMD rendering.

### **2.1 Context Isolation and the Render Loop**

Blender's drawing architecture is heavily tied to its windowing system (GHOST). When a user registers a draw handler using bpy.types.SpaceView3D.draw\_handler\_add, they are hooking into the redraw cycle of a specific editor region (the 3D Viewport).15

In a standard desktop workflow, the render loop looks conceptually like this:

1. **Input Event:** Mouse movement or timer.  
2. **Region Invalidation:** The 3D View is marked for redraw.  
3. **Engine Render:** Eevee/Workbench renders the scene to the framebuffer.  
4. **Callback Execution:** draw\_handler functions are called, issuing GLSL commands on top of the existing framebuffer.  
5. **Swap Buffers:** The result is displayed on the monitor.

In a VR session using OpenXR, the flow diverges significantly. The OpenXR specification dictates a "pull" model where the runtime dictates timing.17 The loop resembles:

1. **WaitFrame:** Blender waits for the HMD to be ready.  
2. **BeginFrame:** The session starts.  
3. **AcquireSwapchainImage:** Blender gets the texture handle for the Left Eye.  
4. **Render View 0:** Blender's internal C++ engine renders the scene camera (offset for left eye) to this texture.  
5. **Release/Acquire:** Repeat for Right Eye.  
6. **EndFrame:** Submit layers to the compositor.

Crucially, the draw\_handler callbacks are typically tied to the *window* refresh, not the *OpenXR* frame submission. Even if they are triggered, the OpenGL context active during the Python callback execution is likely the *window's* context, not the *VR swapchain's* framebuffer. Drawing commands issued to the window context do not propagate to the VR texture because they are separate memory buffers on the GPU.10

### **2.2 The Stereoscopic Disconnect**

A further complication is stereoscopy. A standard draw\_handler assumes a monoscopic view matrix (the viewport camera). For VR, the rendering must occur twice with two distinct view matrices (offset by the Interpupillary Distance, IPD) and two distinct projection matrices (often asymmetric frustums defined by the headset's lenses).6

If a Python script simply executes batch.draw(shader), it uses the uniform values currently bound to the shader. Unless the script explicitly calculates and updates the Model-View-Projection (MVP) matrix for the *Left* eye, draws, then updates for the *Right* eye, and draws again—all while ensuring the correct target framebuffer is bound—the result will be incorrect. It might render a "cyclops" view (center eye) or effectively render nothing if the clipping planes are mismatched with the VR projection.6 The Python API for draw\_handler exposes bpy.context.region\_data.view\_matrix, which usually reflects the *desktop* viewport camera, not the HMD's instantaneous pose.21

### **2.3 Backend Compatibility: OpenGL vs. Vulkan**

The transition of Blender (and the graphics industry) from OpenGL to Vulkan introduces another layer of potential failure. While the Quest 3 via Link (on Windows) often bridges through DirectX, Linux environments rely on OpenGL or Vulkan via Monado or SteamVR.23

Research snippets indicate that Blender's OpenGL-OpenXR bridge can be fragile, particularly on Linux/Wayland systems, leading to errors like "Failed to create VR session" or context mismatches.9 While the user's issue is specifically about *rendering content* rather than session creation, the underlying issue is related: the resource sharing (texture handles) between the Python-accessible OpenGL context and the OpenXR composition layer is not automatic. In many modern engines (Unity/Unreal), this is handled by "Single Pass Stereo" or "Multiview" extensions, which are exposed to native shaders but not easily to Python scripts.24

### **2.4 Performance Latency and "Popping"**

Even if one successfully injects draw calls into the VR buffer, 3DGS imposes a heavy performance tax. The sorting of Gaussians must happen per frame. Doing this in Python (argsort on millions of particles) introduces massive CPU latency. VR requires a stable 72Hz or 90Hz. If the Python script takes 50ms to sort and upload buffers, the VR view will stutter or "reproject" (using old frames), causing nausea and breaking the illusion.5

This latency also manifests as "popping," where the sorting order lags behind the head rotation, causing splats to flicker in and out of correct occlusion. This is a known artifact in 3DGS implementations that lack "StopThePop" or similar temporal coherence algorithms.13

## ---

**3\. Solution I: The Geometry Nodes Proxy Pipeline (Recommended for Stability)**

Given the architectural barriers to direct GLSL injection, the most robust solution is to bypass the draw\_handler entirely and convert the 3D Gaussian data into native Blender geometry. By representing splats as standard mesh data (quads), we allow Blender's internal VR render pass (which *does* work) to handle the visualization naturally.

### **3.1 Concept and Workflow**

This approach treats each Gaussian as a "billboard"—a square plane that always faces the camera.

1. **Data Ingestion:** The Python script reads the .ply file containing the Gaussian parameters (Position, Scale, Rotation, Opacity, Color/SH).  
2. **Mesh Generation:** Instead of drawing points, the script generates a mesh with vertices at the Gaussian centers. All parameters are stored as **Named Attributes** on this mesh.11  
3. **Geometry Nodes Processing:** A Geometry Nodes (GN) modifier is applied to this mesh. Its primary function is to instance a quad on each vertex and align it to the camera.  
4. **Shader Material:** A standard material uses the Attribute node to retrieve color and opacity, feeding them into the Principled BSDF or Emission shader.

This workflow ensures that as far as Blender's render engine (Eevee) is concerned, the splats are just normal geometry. Consequently, when the VR session requests a render, Eevee renders these quads into the OpenXR swapchain automatically.

### **3.2 Solving View-Dependent Alignment in VR**

The illusion of a volumetric cloud relies on the splats facing the viewer. In a desktop viewport, we can use the Camera Info node or Object Info (targeting the active camera) to calculate the rotation.28

However, in VR, the "Camera" is dynamic. The HMD pose drives the view.

* **Challenge:** The standard Camera Info node might refer to the scene camera object, which may not update perfectly in sync with the HMD if the VR session is using a specialized offset or if the dependency graph update lags.  
* **Solution:** We must ensure the object used for alignment in GN tracks the VR headset.  
  * Use the **VR Scene Inspection** addon's "Landmarks" or "Custom Camera" feature to link a Scene Camera to the VR view.30  
  * In Geometry Nodes, use the Align Euler to Vector node. The vector is Position (Camera) \- Position (Splat). This forces the instances to rotate towards the HMD every frame.28

### **3.3 Handling Transparency and Sorting**

The primary graphical challenge with this method is transparency sorting. 3DGS relies on strict back-to-front sorting for alpha blending.

* **Eevee Limitations:** Eevee's "Alpha Blend" mode sorts per object, or per triangle within an object, but this sorting can be expensive and sometimes inaccurate for complex intersecting geometry like splats.31  
* **Alpha Hashed:** The recommended setting for VR is **Alpha Hashed**. This uses stochastic transparency (dithering) rather than blending. It effectively bypasses the need for strict sorting, as depth testing works on a per-fragment basis.  
  * *Pros:* Extremely fast, no sorting artifacts, works perfectly with arbitrary head rotation.  
  * *Cons:* Introduces noise (grain). However, the high pixel density of the Quest 3 and Temporal Anti-Aliasing (TAA) in the viewport can smooth this out significantly.  
* **VRSplat Optimization:** Recent research ("VRSplat") suggests that using fewer, higher-quality splats (Mini-Splatting) or foveated rendering can allow for sorting-based methods to remain performant. In a Blender context, "Alpha Hashed" is the closest approximation to this "order-independent" rendering goal without custom C++ rasterizers.5

### **3.4 Implementation Details (Python & Nodes)**

The Python script acts as a pipeline orchestrator.

**Table 1: Data Mapping from 3DGS PLY to Blender Attributes**

| 3DGS Parameter | Blender Attribute Type | Usage in Shader/GN |
| :---- | :---- | :---- |
| Position ($x, y, z$) | Mesh.vertices.co | Base position for Instancing |
| Scale ($s\_x, s\_y, s\_z$) | FLOAT\_VECTOR | Controls Billboard Size |
| Rotation ($r\_w, r\_x, r\_y, r\_z$) | FLOAT\_QUATERNION | Initial orientation offset |
| Opacity ($o$) | FLOAT | Alpha channel (mapped via Sigmoid) |
| Color ($f\_{dc}$) | FLOAT\_COLOR | Base Color (Spherical Harmonics level 0\) |

**Geometry Node Graph Logic:**

1. **Input:** Point Cloud (Vertices).  
2. **Instance on Points:** Instance \= Quad (Grid).  
3. **Scale Instances:** driven by Scale attribute.  
4. **Set Rotation:** Use Align Euler to Vector. Vector \= Object Info (Camera) \-\> Location \- Position. Pivot \= Z axis.  
5. **Output:** Realized Geometry (optional, if instancing is supported by the renderer directly).

*Citation:* The feasibility of this proxy approach is confirmed by the KIRI Engine addon, which uses a "proxy object" to drive rendering, allowing users to toggle between a lightweight edit mode and a render-ready mesh representation.11

## ---

**4\. Solution II: Python-Driven Offscreen Stereo Rendering**

If the application strictly requires custom GLSL shaders (e.g., for specialized accumulation logic that Eevee cannot replicate), the gpu module must be used. To circumvent the draw\_handler invisibility in VR, we must decouple rendering from the viewport and "project" it into the scene.

### **4.1 The Offscreen Buffer Strategy**

Instead of drawing to the screen, we draw to a gpu.types.GPUOffScreen buffer. This is a Framebuffer Object (FBO) that exists purely in GPU memory.15

1. **Initialization:** Create a GPUOffScreen object sized to the Quest 3's per-eye resolution (approx 2064x2208, though 2048x2048 is a safe power-of-two target).34  
2. **The Render Loop:** Hook into bpy.app.handlers.frame\_change\_pre or a modal operator timer. In each cycle:  
   * Bind the Offscreen buffer.  
   * Clear the buffer.  
   * Execute the custom GLSL batch draw.  
   * Unbind.

### **4.2 The "Diegetic Screen" Injection**

Since we cannot write to the OpenXR swapchain, we create a plane in the 3D scene (a "Screen") and map the offscreen texture to it.

* **The Texture Link:** This is the most technically difficult step due to Python/C++ boundaries.  
  * *Naive Approach:* Read pixels from Offscreen to CPU (buffer.read()), write to bpy.data.images.pixels. **Verdict:** Too slow for VR (latency \> 100ms).12  
  * *Optimized Approach:* Use gpu.texture.from\_image to wrap a Blender image, or use the GPUOffScreen.color\_texture directly if using a custom RenderEngine.  
  * *The "Share Handle" Trick:* Some advanced addons use Ctypes to pass the OpenGL texture ID from the GPUOffScreen object directly to a shader node tree that accepts an external texture, though Blender's shader nodes don't natively expose "Texture ID" inputs.  
  * **Viable Workaround:** Use a **Custom Render Engine** (bpy.types.RenderEngine). A custom engine can define a render() method that uses gpu module calls to draw directly into the result pass provided by Blender. When Blender renders the VR view, it calls this engine. This allows the GLSL output to be part of the native render pipeline.36

### **4.3 Handling Stereoscopy in Custom Shaders**

A custom render setup must handle stereo views manually.

1. **Detecting the View:** When RenderEngine.render(scene) is called, check if it's a stereo render.  
2. **Retrieving Matrices:**  
   * The bpy.types.XrSessionState provides the center head pose.37  
   * To get Left/Right eye matrices, one must apply the IPD offset.  
   * **Projection Matrix Calculation:** VR projection matrices are often asymmetric (off-axis). Blender's camera.calc\_matrix\_camera can compute this if the camera data is set to "Stereo" and "Off-Axis", but synchronizing this with the exact OpenXR FOV values requires querying the XrSessionSettings or approximating based on the HMD's reported FOV.19

**Table 2: Manual Stereo Matrix Calculation Logic**

| Matrix Component | Calculation Strategy |
| :---- | :---- |
| **View Matrix (Left)** | $V\_L \= (T\_{IPD/2} \\cdot R\_{Head} \\cdot T\_{Head})^{-1}$ where $T\_{IPD/2}$ shifts left by half IPD. |
| **View Matrix (Right)** | $V\_R \= (T\_{-IPD/2} \\cdot R\_{Head} \\cdot T\_{Head})^{-1}$ where $T\_{-IPD/2}$ shifts right by half IPD. |
| **Projection (Sym)** | Standard gluPerspective using vertical FOV. |
| **Projection (Asym)** | Requires raw tangent values from OpenXR (hard to get in pure Python). Fallback: Use symmetric projection with overscan to compensate.22 |

### **4.4 The "Screen-in-World" Implementation**

A pragmatic alternative to a custom engine is to place a physical plane in the scene, parented to the camera, filling the field of view.

* **Material:** Emission shader using the texture generated by the Offscreen buffer.  
* **Stereo Separation:** The material must be stereo-aware.  
  * Use the **Camera Data** node in the Shader Editor? No, Eevee supports a Camera Data node but it doesn't output "Eye Index".40  
  * **Solution:** Use the **"Multiview"** setup in the texture node. Load the Left render into one Image sequence and the Right into another. Blender's stereoscopic pipeline automatically serves the correct image to the correct eye if the files are named name\_L.png and name\_R.png.42  
  * **Workflow:** The Python script renders the Offscreen buffer to disk (or memory file) as temp\_L.png and temp\_R.png (or side-by-side), reloads the image datablock, and the plane updates. *Note: Disk I/O is a latency killer.*

**Conclusion on Solution II:** While powerful, the data transfer bottleneck (GPU-\>CPU-\>GPU) makes this difficult for 60+ FPS VR unless using C++ extensions to share texture pointers directly.

## ---

**5\. Solution III: Native Extension and Future Pathways**

For professional-grade 3DGS tools, developers often hit the ceiling of the Python API and move to C++.

### **5.1 C++ Integration via Ghost-XR**

The "correct" engineering solution, implemented by projects like **VRSplat** and **VR-GS**, involves modifying Blender's source code or writing a binary addon that interfaces with the OpenXR context directly.13

* **Mechanism:** These systems implement a custom rasterizer (usually CUDA-based) that hooks into the xrEndFrame sequence. They sort the Gaussians and rasterize them directly to the swapchain images provided by the OpenXR runtime, bypassing Blender's renderer entirely.  
* **Performance:** This achieves the native 72/90 FPS required for VR, with zero copy overhead.13

### **5.2 Hybrid Approaches (Python \+ DLL)**

It is possible to write a Python addon that loads a C++ DLL.

1. Python handles UI and data loading.  
2. Python passes the window handle or OpenGL context to the DLL.  
3. The DLL intercepts the render loop or creates a shared context to draw the splats.  
4. This is how the **Freebird XR** and **MARUI** plugins historically added VR features to Blender before native support existed.44

## ---

**6\. Implementation Roadmap: The Geometry Node "Billboard" Strategy**

Given the constraints of a standard Blender Addon (pure Python installation), **Solution I (Geometry Nodes Proxy)** is the recommended path. It balances performance, stability, and ease of distribution.

### **6.1 Step 1: Data Import and Structure**

Create a custom operator OPS\_ImportGaussianSplat.

Python

import bpy  
import numpy as np

def load\_ply\_to\_mesh(filepath):  
    \# Use plyfile or similar library to read vertex data  
    \# Create a mesh data block  
    mesh \= bpy.data.meshes.new(name="GaussianSplat")  
    \# Assign vertices (Mean Positions)  
    \# create Named Attributes for:  
    \# "rot" (Quaternion), "scale" (Vector), "opacity" (Float), "color" (Color)  
    \# Bulk set attributes using foreach\_set for performance  
    return mesh

*Optimization:* Do not create faces or edges. Only vertices. This keeps the memory footprint lower.32

### **6.2 Step 2: The Geometry Node Tree**

Construct a node tree programmatically:

1. **Instance on Points:** Instance a primitive Grid (Resolution 1x1).  
2. **Align Euler to Vector:**  
   * Vector: Position (Target) \- Position (Self).  
   * Target Position comes from an Object Info node referencing the VR Camera.  
3. **Store Named Attribute:** Pass the color and opacity from the source points to the instances.  
4. **Realize Instances:** (Optional) Eevee Next (Blender 4.2+) handles instancing efficiently, but realizing may be needed for certain shader effects.

### **6.3 Step 3: Material and Rendering**

Create a shader that simulates the Gaussian falloff.

* **Texture Coordinate:** Generated (center of quad is 0.5, 0.5).  
* **Math:** Calculate distance from center: $d \= \\sqrt{(x-0.5)^2 \+ (y-0.5)^2}$.  
* **Gaussian Falloff:** $\\alpha' \= \\alpha \\cdot \\exp(-d^2 / \\sigma)$.  
* **Mix Shader:** Mix Transparent BSDF and Emission Shader using $\\alpha'$.  
* **Blend Mode:** Set to **Alpha Hashed**.

### **6.4 Step 4: VR Session Hook**

To ensure the billboards face the user in VR:

1. **Session Start:** When the user clicks "Start VR", the addon must identify the VR View object.  
2. **Update Loop:**  
   Python  
   def update\_camera\_tracker(scene):  
       if bpy.context.window\_manager.xr\_session\_state.is\_running(bpy.context):  
           \# Get VR headset position  
           vr\_pos \= bpy.context.window\_manager.xr\_session\_state.viewer\_pose\_location  
           \# Update the "Target" object used by Geometry Nodes  
           bpy.data.objects.location \= vr\_pos

   bpy.app.handlers.depsgraph\_update\_post.append(update\_camera\_tracker)

   *Insight:* Updating an Empty's location is cheap. The Geometry Nodes modifier will re-evaluate the rotation of the billboards on the GPU (or CPU depending on evaluation mode) based on this new target position.46

## ---

**7\. Optimization and Future Proofing**

### **7.1 Performance Tuning for Quest 3**

The Quest 3 (via Link) has significant GPU power, but pushing millions of transparent quads can bottleneck the rasterizer (overdraw).

* **Point Cloud Culling:** In the Geometry Nodes tree, delete points that are outside the camera frustum or beyond a certain distance. This reduces the number of instances Eevee has to process.34  
* **Resolution Scaling:** In the VR Session settings, reducing the render resolution can significantly improve FPS if fill-rate limited.

### **7.2 The "VRSplat" Horizon**

Research into "VRSplat" highlights that standard 3DGS is inherently problematic for VR due to sorting latency. The paper proposes "Mini-Splatting" (reducing splat count) and foveated rendering.

* **Actionable Insight:** Implement a "Decimate" feature in your addon. Before generating the mesh, filter the PLY data to remove low-opacity or tiny splats that contribute little to visual fidelity but cost high render time.5

## ---

**8\. Conclusion**

The "missing shader" problem is a structural consequence of Blender's segregated VR rendering pipeline. The Python gpu module draws to the window, while OpenXR draws to the headset. Bridging this gap requires moving the rendering logic from "overlay" (draw handlers) to "scene" (geometry).

By adopting the **Geometry Nodes Proxy** workflow, utilizing **Alpha Hashed** transparency for order-independent rendering, and driving billboard orientation via **VR Session State** hooks, developers can achieve a functional, interactive 3D Gaussian Splatting experience in VR. This approach leverages Blender's native strengths, avoids fragile low-level API hacks, and ensures compatibility with future Eevee improvements.

### **References**

1

#### **참고 자료**

1. 3D Gaussian Splatting: A Technical Guide to Real-Time Neural Rendering \- KIRI Engine, 12월 7, 2025에 액세스, [https://www.kiriengine.app/blog/3d-gaussian-splatting-a-technical-guide-to-real-time-neural-rendering](https://www.kiriengine.app/blog/3d-gaussian-splatting-a-technical-guide-to-real-time-neural-rendering)  
2. 3D Gaussian Splatting \- Paper Explained, Training NeRFStudio \- LearnOpenCV, 12월 7, 2025에 액세스, [https://learnopencv.com/3d-gaussian-splatting/](https://learnopencv.com/3d-gaussian-splatting/)  
3. 3D Gaussian Splatting: A new frontier in rendering \- The Chaos Blog, 12월 7, 2025에 액세스, [https://blog.chaos.com/3d-gaussian-splatting-new-frontier-in-rendering](https://blog.chaos.com/3d-gaussian-splatting-new-frontier-in-rendering)  
4. Sort-free Gaussian Splatting via Weighted Sum Rendering \- arXiv, 12월 7, 2025에 액세스, [https://arxiv.org/html/2410.18931v1](https://arxiv.org/html/2410.18931v1)  
5. VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality \- arXiv, 12월 7, 2025에 액세스, [https://arxiv.org/html/2505.10144v1](https://arxiv.org/html/2505.10144v1)  
6. Correct Stereo Panoramic Rendering for VR Headsets \- Blender Artists Community, 12월 7, 2025에 액세스, [https://blenderartists.org/t/correct-stereo-panoramic-rendering-for-vr-headsets/637106](https://blenderartists.org/t/correct-stereo-panoramic-rendering-for-vr-headsets/637106)  
7. Virtual Reality \- OpenXR \- Blender Developer, 12월 7, 2025에 액세스, [https://developer.blender.org/docs/features/gpu/viewports/xr/](https://developer.blender.org/docs/features/gpu/viewports/xr/)  
8. Composition Layers Support | OpenXR Plugin | 1.13.0 \- Unity \- Manual, 12월 7, 2025에 액세스, [https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.13/manual/features/compositionlayers.html](https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.13/manual/features/compositionlayers.html)  
9. Blender VR Scene Inspection addon cannot connect to OpenXR : r/virtualreality\_linux, 12월 7, 2025에 액세스, [https://www.reddit.com/r/virtualreality\_linux/comments/1op9q7o/blender\_vr\_scene\_inspection\_addon\_cannot\_connect/](https://www.reddit.com/r/virtualreality_linux/comments/1op9q7o/blender_vr_scene_inspection_addon_cannot_connect/)  
10. Context override for drawing to screen via the gpu-module \- Blender Artists Community, 12월 7, 2025에 액세스, [https://blenderartists.org/t/context-override-for-drawing-to-screen-via-the-gpu-module/1489250](https://blenderartists.org/t/context-override-for-drawing-to-screen-via-the-gpu-module/1489250)  
11. KIRI Engine 3DGS Render v4.0 for Blender — A Faster Edit→Render Pipeline for Gaussian Splats, 12월 7, 2025에 액세스, [https://www.kiriengine.app/blog/kiri-engine-3DGS%20Render-Blender-v4.0](https://www.kiriengine.app/blog/kiri-engine-3DGS%20Render-Blender-v4.0)  
12. Direct GPU texture to Blender Image data block without CPU readback? \- Python Support, 12월 7, 2025에 액세스, [https://blenderartists.org/t/direct-gpu-texture-to-blender-image-data-block-without-cpu-readback/1614686](https://blenderartists.org/t/direct-gpu-texture-to-blender-image-data-block-without-cpu-readback/1614686)  
13. VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality \- ResearchGate, 12월 7, 2025에 액세스, [https://www.researchgate.net/publication/392802810\_VRSplat\_Fast\_and\_Robust\_Gaussian\_Splatting\_for\_Virtual\_Reality](https://www.researchgate.net/publication/392802810_VRSplat_Fast_and_Robust_Gaussian_Splatting_for_Virtual_Reality)  
14. OSL camera \- the Octane Documentation Portal, 12월 7, 2025에 액세스, [https://docs.otoy.com/blender/OSLcamera.html](https://docs.otoy.com/blender/OSLcamera.html)  
15. GPU Module (gpu) \- Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/gpu.html](https://docs.blender.org/api/current/gpu.html)  
16. Blender/doc/python\_api/examples/gpu.offscreen.1.py at master \- GitHub, 12월 7, 2025에 액세스, [https://github.com/Arlen22/Blender/blob/master/doc/python\_api/examples/gpu.offscreen.1.py](https://github.com/Arlen22/Blender/blob/master/doc/python_api/examples/gpu.offscreen.1.py)  
17. How to design my render loop ? : r/opengl \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/opengl/comments/1mqqqlh/how\_to\_design\_my\_render\_loop/](https://www.reddit.com/r/opengl/comments/1mqqqlh/how_to_design_my_render_loop/)  
18. python \- Off screen window \- Blender Stack Exchange, 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/78687/off-screen-window](https://blender.stackexchange.com/questions/78687/off-screen-window)  
19. CameraStereoData(bpy\_struct) \- Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/bpy.types.CameraStereoData.html](https://docs.blender.org/api/current/bpy.types.CameraStereoData.html)  
20. Best practice for rendering stereo images in VR UI? : r/vrdev \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/vrdev/comments/1m7zw71/best\_practice\_for\_rendering\_stereo\_images\_in\_vr\_ui/](https://www.reddit.com/r/vrdev/comments/1m7zw71/best_practice_for_rendering_stereo_images_in_vr_ui/)  
21. python \- Get view and perspective matrices of the current 3D viewport (not camera), 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/284222/get-view-and-perspective-matrices-of-the-current-3d-viewport-not-camera](https://blender.stackexchange.com/questions/284222/get-view-and-perspective-matrices-of-the-current-3d-viewport-not-camera)  
22. Perspective projection \- help me duplicate blender's default scene \- Stack Overflow, 12월 7, 2025에 액세스, [https://stackoverflow.com/questions/28143783/perspective-projection-help-me-duplicate-blenders-default-scene](https://stackoverflow.com/questions/28143783/perspective-projection-help-me-duplicate-blenders-default-scene)  
23. OpenXR not detected? Can't get Blender, Godot, and Alice VR to work \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/virtualreality\_linux/comments/1cas7xe/openxr\_not\_detected\_cant\_get\_blender\_godot\_and/](https://www.reddit.com/r/virtualreality_linux/comments/1cas7xe/openxr_not_detected_cant_get_blender_godot_and/)  
24. Single-Pass Stereo rendering \- Unity \- Manual, 12월 7, 2025에 액세스, [https://docs.unity3d.com/550/Documentation/Manual/SinglePassStereoRendering.html](https://docs.unity3d.com/550/Documentation/Manual/SinglePassStereoRendering.html)  
25. Geometry node extremely slow playback speed with large collection of instances (+20000), 12월 7, 2025에 액세스, [https://blenderartists.org/t/geometry-node-extremely-slow-playback-speed-with-large-collection-of-instances-20000/1545599](https://blenderartists.org/t/geometry-node-extremely-slow-playback-speed-with-large-collection-of-instances-20000/1545599)  
26. \[2402.00525\] StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering : r/GaussianSplatting \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/GaussianSplatting/comments/1ah0aaz/240200525\_stopthepop\_sorted\_gaussian\_splatting/](https://www.reddit.com/r/GaussianSplatting/comments/1ah0aaz/240200525_stopthepop_sorted_gaussian_splatting/)  
27. BlenderNeRF \- Zenodo, 12월 7, 2025에 액세스, [https://zenodo.org/records/13250725](https://zenodo.org/records/13250725)  
28. Geometry Nodes \- Billboards (similar to track to constraint) \- Blender Stack Exchange, 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/243305/geometry-nodes-billboards-similar-to-track-to-constraint](https://blender.stackexchange.com/questions/243305/geometry-nodes-billboards-similar-to-track-to-constraint)  
29. Camera Info Node \- Blender 5.0 Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/latest/modeling/geometry\_nodes/input/scene/camera\_info.html](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/input/scene/camera_info.html)  
30. VR Scene Inspection \- Blender 5.0 Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/latest/addons/3d\_view/vr\_scene\_inspection.html](https://docs.blender.org/manual/en/latest/addons/3d_view/vr_scene_inspection.html)  
31. Materials \- Blender 5.0 Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/latest/render/eevee/material\_settings.html](https://docs.blender.org/manual/en/latest/render/eevee/material_settings.html)  
32. 3DGS Render Blender Addon by KIRI Engine \- GitHub, 12월 7, 2025에 액세스, [https://github.com/Kiri-Innovation/3dgs-render-blender-addon](https://github.com/Kiri-Innovation/3dgs-render-blender-addon)  
33. gpu.types. \- Blender Python API \- Blender Documentation, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/gpu.types.html](https://docs.blender.org/api/current/gpu.types.html)  
34. VR-Splatting: Foveated Radiance Field Rendering via 3D Gaussian Splatting and Neural Points \- Linus Franke, 12월 7, 2025에 액세스, [https://lfranke.github.io/vr\_splatting/](https://lfranke.github.io/vr_splatting/)  
35. offscreen.draw\_view3d is washed out \#74299 \- Blender Developer, 12월 7, 2025에 액세스, [https://developer.blender.org/T74299](https://developer.blender.org/T74299)  
36. RenderEngine(bpy\_struct) \- Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/bpy.types.RenderEngine.html](https://docs.blender.org/api/current/bpy.types.RenderEngine.html)  
37. XrSessionState(bpy\_struct) \- Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/bpy.types.XrSessionState.html](https://docs.blender.org/api/current/bpy.types.XrSessionState.html)  
38. Camera Intrinsic Matrix with Example in Python | by Neeraj Krishna | TDS Archive | Medium, 12월 7, 2025에 액세스, [https://medium.com/data-science/camera-intrinsic-matrix-with-example-in-python-d79bf2478c12](https://medium.com/data-science/camera-intrinsic-matrix-with-example-in-python-d79bf2478c12)  
39. Can I use my own projection matrix to render left & right eye scene? :: SteamVR Developer Hardware General Discussions \- Steam Community, 12월 7, 2025에 액세스, [https://steamcommunity.com/app/358720/discussions/0/359543542248432114/](https://steamcommunity.com/app/358720/discussions/0/359543542248432114/)  
40. Camera Data Node \- Blender 5.0 Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/latest/render/shader\_nodes/input/camera\_data.html](https://docs.blender.org/manual/en/latest/render/shader_nodes/input/camera_data.html)  
41. Eevee \- Material Shader \- "Ray Length" or Z-index equivalent? \- Blender Stack Exchange, 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/153560/eevee-material-shader-ray-length-or-z-index-equivalent](https://blender.stackexchange.com/questions/153560/eevee-material-shader-ray-length-or-z-index-equivalent)  
42. Usage — Blender Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/2.90/render/output/stereoscopy/usage.html](https://docs.blender.org/manual/en/2.90/render/output/stereoscopy/usage.html)  
43. Activity · YingJiang96/VR-GS \- GitHub, 12월 7, 2025에 액세스, [https://github.com/YingJiang96/VR-GS/activity](https://github.com/YingJiang96/VR-GS/activity)  
44. v2.0.2 update for Freebird (VR plugin for Blender): Pose and edit bones using VR in Blender. It's pretty easy to grab and move the bones to create natural-looking poses using VR : r/oculus \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/oculus/comments/1aozfb6/v202\_update\_for\_freebird\_vr\_plugin\_for\_blender/](https://www.reddit.com/r/oculus/comments/1aozfb6/v202_update_for_freebird_vr_plugin_for_blender/)  
45. VR/AR User Interface \- Blender Devtalk, 12월 7, 2025에 액세스, [https://devtalk.blender.org/t/vr-ar-user-interface/2053](https://devtalk.blender.org/t/vr-ar-user-interface/2053)  
46. Application Handlers (bpy.app.handlers) — Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/3.3/bpy.app.handlers.html](https://docs.blender.org/api/3.3/bpy.app.handlers.html)  
47. Geometry nodes evaluating every frame (even when values do not change) \#123598, 12월 7, 2025에 액세스, [https://projects.blender.org/blender/blender/issues/123598](https://projects.blender.org/blender/blender/issues/123598)  
48. VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality \- Human Sensing Laboratory, 12월 7, 2025에 액세스, [http://www.humansensing.cs.cmu.edu/sites/default/files/VR\_GS.pdf](http://www.humansensing.cs.cmu.edu/sites/default/files/VR_GS.pdf)  
49. Blender \+ Virtual Desktop passthrough on quest 3 is something else.. \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/blender/comments/17bm6ay/blender\_virtual\_desktop\_passthrough\_on\_quest\_3\_is/](https://www.reddit.com/r/blender/comments/17bm6ay/blender_virtual_desktop_passthrough_on_quest_3_is/)  
50. Sort Elements Node \- Blender 5.0 Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/latest/modeling/geometry\_nodes/geometry/operations/sort\_elements.html](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/geometry/operations/sort_elements.html)  
51. Virtual Reality \- Blender Developer Documentation, 12월 7, 2025에 액세스, [https://developer.blender.org/docs/release\_notes/2.83/virtual\_reality/](https://developer.blender.org/docs/release_notes/2.83/virtual_reality/)  
52. Usage \- Blender 5.0 Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/en/latest/render/output/properties/stereoscopy/usage.html](https://docs.blender.org/manual/en/latest/render/output/properties/stereoscopy/usage.html)  
53. Shader nodes help: Make a texture follow an object in camera space : r/blenderhelp \- Reddit, 12월 7, 2025에 액세스, [https://www.reddit.com/r/blenderhelp/comments/1j7pnst/shader\_nodes\_help\_make\_a\_texture\_follow\_an\_object/](https://www.reddit.com/r/blenderhelp/comments/1j7pnst/shader_nodes_help_make_a_texture_follow_an_object/)  
54. Application Handlers (bpy.app.handlers) \- Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/bpy.app.handlers.html](https://docs.blender.org/api/current/bpy.app.handlers.html)  
55. frame\_change\_pre handlers : accessing a custom variable in render, 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/322315/frame-change-pre-handlers-accessing-a-custom-variable-in-render](https://blender.stackexchange.com/questions/322315/frame-change-pre-handlers-accessing-a-custom-variable-in-render)  
56. What is the performance cost of adding a SpaceView3D draw callback?, 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/252055/what-is-the-performance-cost-of-adding-a-spaceview3d-draw-callback](https://blender.stackexchange.com/questions/252055/what-is-the-performance-cost-of-adding-a-spaceview3d-draw-callback)  
57. OpenXR Stereoscopic matrix not returning the correct projection matrix · Issue \#101142 · godotengine/godot \- GitHub, 12월 7, 2025에 액세스, [https://github.com/godotengine/godot/issues/101142](https://github.com/godotengine/godot/issues/101142)  
58. Usage — Blender Manual, 12월 7, 2025에 액세스, [https://docs.blender.org/manual/de/2.81/render/output/multiview/usage.html](https://docs.blender.org/manual/de/2.81/render/output/multiview/usage.html)  
59. Math Types & Utilities (mathutils) \- Blender Python API, 12월 7, 2025에 액세스, [https://docs.blender.org/api/current/mathutils.html](https://docs.blender.org/api/current/mathutils.html)