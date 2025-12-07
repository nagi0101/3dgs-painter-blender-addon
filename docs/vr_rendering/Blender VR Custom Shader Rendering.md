# **Architectural Implementation Strategy for High-Performance Custom GLSL Shading in Blender OpenXR: The 3D Gaussian Splatting Use Case**

## **1\. Executive Summary**

This report serves as a definitive architectural guide for integrating custom high-performance rendering pipelines—specifically 3D Gaussian Splatting (3DGS)—into Blender’s Virtual Reality (VR) ecosystem. The core technical challenge identified is the structural inability of Blender’s standard Python API hooks (draw\_handler\_add) to execute within the OpenXR headset’s render loop. While these handlers function correctly in the desktop SpaceView3D region, the VR session, managed by the underlying GHOST (Generic Handy Operating System Toolkit) library and the wm\_xr\_draw.c execution flow, utilizes a segregated offscreen rendering pipeline that explicitly excludes window-level overlays to maintain performance and visual stability.

Our analysis, grounded in an exhaustive review of Blender 5.0 source code, OpenXR specifications, and existing VR integration projects, confirms that a purely Python-based "overlay" approach using standard draw handlers is architecturally inviable for VR. The VR render loop bypasses the standard window region drawing context entirely, rendering directly from the scene graph to an internal offscreen buffer that is subsequently blitted to the OpenXR swapchain.

To overcome this fundamental limitation and achieve the user’s goal of rendering custom GLSL shaders for 3DGS directly to a Quest 3 headset at 72+ FPS, this report delineates a tiered solution strategy. While a custom RenderEngine implementation provides a partial pathway, it forces a binary choice between the custom renderer and Blender’s native engines (Eevee/Cycles). The recommended solution, therefore, is the development of a custom **OpenXR API Layer**. This architectural pattern allows for the interception of the xrEndFrame submission call at the driver level, enabling the injection of a secondary composition layer containing the 3DGS content. This approach ensures stereo-correct rendering, decoupling the heavy Gaussian sorting workload from Blender’s main thread, and provides the low-level graphics context access required for high-performance rasterization, all without necessitating a fragile fork of the Blender codebase.

## ---

**2\. Introduction: The Intersection of Volumetric Rendering and OpenXR**

The visualization of neural radiance fields and point-based volumetric data, particularly through the technique of 3D Gaussian Splatting (3DGS), represents a paradigm shift in real-time computer graphics. Unlike traditional triangle-based rasterization, which relies on well-established fixed-function hardware pipelines, 3DGS necessitates a specialized rendering approach involving the continuous sorting of millions of semi-transparent anisotropic Gaussian primitives. This requirement for custom rasterization logic—often implemented via Compute Shaders or CUDA kernels—clashes with the rigid, pre-defined rendering pipelines of established content creation tools like Blender.

### **2.1 The Context of the Challenge**

Blender has evolved significantly in its support for Virtual Reality, moving from experimental branches to a native implementation underpinned by the OpenXR standard. This transition, solidified in Blender 3.0 and refined in 5.0, provides a stable, vendor-agnostic foundation for VR scene inspection. However, this stability comes at the cost of flexibility. The integration is designed primarily for *inspection*—viewing the scene as rendered by Eevee or Cycles—rather than *extension*.

The specific technical problem addressed in this report is the "Viewport Disconnect." Developers accustomed to extending Blender’s viewport via the Python API find that their tools—gizmos, custom overlays, and immediate-mode drawing commands—disappear when the user dons a VR headset. This is not a bug, but a deliberate architectural decision within Blender to isolate the VR rendering workload from the heavy, often unoptimized drawing routines of the desktop user interface.

### **2.2 Goals and Technical Constraints**

The objective is to implement a pipeline that allows for the rendering of custom GLSL shaders for 3D Gaussian Splatting directly into the VR headset view. The constraints are rigorous:

* **Platform:** Blender 5.0 on Windows (implied by Oculus Link usage), targeting Meta Quest 3 via OpenXR.  
* **Performance:** The system must maintain a steady 72 Hz (or 90 Hz) frame rate. 3DGS rendering is computationally expensive, primarily due to the sorting phase. Missed frames in VR result in reprojection artifacts and motion sickness, making performance optimization paramount.  
* **Stereoscopy:** The solution must handle stereo rendering correctly, utilizing unique view and projection matrices for the left and right eyes to ensure correct depth perception.  
* **Integration:** The solution must avoid forking Blender. Modifying and recompiling the Blender source code creates a maintenance nightmare, decoupling the tool from official updates. Therefore, the solution must rely on the Python API, C++ extensions (addons), or external libraries.

## ---

**3\. Architectural Autopsy: The Blender VR Render Pipeline**

To engineer a bypass, one must first deeply understand the system being bypassed. We have analyzed the call stack and data flow of Blender’s VR implementation to identify exactly where the "break" in the draw chain occurs.

### **3.1 The GHOST-XR Bridge: Anatomy of wm\_xr**

Blender’s windowing and system event handling is managed by the **GHOST (Generic Handy Operating System Toolkit)** library. GHOST abstracts the differences between Windows, macOS, and Linux, providing a consistent interface for window creation and OpenGL context management.

For VR, Blender introduces the GHOST\_XrContext and GHOST\_IXrGraphicsBinding interfaces.1 These C++ classes manage the lifecycle of the OpenXR instance and session.

* **Session Initialization:** When a user clicks "Start VR Session," wm\_xr.c requests a session from GHOST. GHOST initializes the OpenXR loader, negotiates extensions, and creates an XrSession.  
* **Graphics Binding:** This is a critical junction. OpenXR requires the application to pass its graphics context info to the runtime. GHOST\_XrGraphicsBindingOpenGL 3 handles this for OpenGL. On Windows, where many OpenXR runtimes (like WMR) operate natively in DirectX 12, this binding often manages an interop layer (e.g., WGL\_NV\_DX\_interop) to bridge Blender’s OpenGL context with the runtime’s DirectX requirements.

### **3.2 The Offscreen Render Loop: wm\_xr\_draw.c**

The most critical finding regarding the user's problem lies in source/blender/windowmanager/intern/wm\_xr\_draw.c. This file dictates how Blender draws to the headset.

Unlike the desktop viewport, which draws to a window surface, the VR session draws to an **offscreen framebuffer**.

1. **Swapchain Acquisition:** The system calls xrAcquireSwapchainImage to get the next available texture from the VR runtime.  
2. **Framebuffer Binding:** Blender binds an internal GPUOffScreen buffer.  
3. **View Iteration:** The code iterates through the views (Left Eye, Right Eye). For each view, it sets up the camera matrices based on the XrView pose returned by xrLocateViews.  
4. **The Draw Call:** The function DRW\_draw\_render\_loop\_ex is invoked to render the scene.

The "Why" of the Missing Draw Handlers:  
The call to DRW\_draw\_render\_loop\_ex inside wm\_xr\_draw.c is configured with a specific set of flags. These flags explicitly instruct the draw engine to skip the overlay pass.

* The logic is: "We are rendering the *scene* for inspection. We do not want the 3D cursor, the grid, the selection outlines, or the Python UI scripts cluttering the immersive view."  
* Standard Python draw handlers (bpy.types.SpaceView3D.draw\_handler\_add) are logically attached to the WINDOW region of the SpaceView3D. The VR render loop does not render a "Window Region"; it renders the "Scene Data." Therefore, the callback list associated with the window region is never iterated.

### **3.3 The Texture Submission Blit**

Once the scene is rendered to Blender's internal framebuffer, GHOST\_XrGraphicsBinding::submitToSwapchainImage() is called.3

* This function performs a **blit** (bit-block transfer): it copies the pixels from Blender's internal FBO to the OpenXR swapchain image.  
* Crucially, this blit happens immediately after the internal engine finishes. There is no exposed "hook" or callback slot between the end of the scene render and the submission to OpenXR. The pipeline is closed.

## ---

**4\. The Rendering Challenge: 3D Gaussian Splatting in VR**

Integrating 3D Gaussian Splatting (3DGS) adds a layer of complexity beyond simple line drawing. Understanding 3DGS requirements is essential for selecting the right architecture.

### **4.1 The Sorting Bottleneck**

3DGS represents a scene as a collection of millions of anisotropic 3D Gaussians. To render these correctly, they must be drawn in a specific order.

* **Alpha Blending Dependency:** The color of a pixel is determined by accumulating the contribution of overlapping Gaussians. The simplified blending equation requires strict **back-to-front sorting** relative to the camera.  
* **Per-Frame Sort:** This sorting must happen *every frame*. If the camera moves (which it does constantly in VR), the depth order changes.  
* **Stereo Duplication:** In VR, you have two camera positions. The sorting order for the left eye is different from the right eye. Therefore, the sorting workload is effectively doubled (or requires advanced compute shader techniques to handle multiview).

### **4.2 Why Python Failures are Inevitable**

Trying to implement the sort in Python is futile. Even with NumPy, sorting 1-5 million particles and uploading them to the GPU every 13 milliseconds (to hit 72 FPS) is impossible due to Python's interpreter overhead and the PCI-e bus transfer latency.

* **Requirement:** The sorting and draw command generation must happen on the GPU (via Compute Shaders) or in highly optimized C++/CUDA code.  
* **Implication for Blender:** We cannot just iterate over particles in a Python script and issue gpu.batch.draw(). We need a mechanism that allows a compiled renderer to take over the pipeline.

## ---

**5\. Solution Pathways**

We present three architectural strategies. The first two act as educational failures or partial solutions, leading to the third, which is the recommended professional approach.

### **5.1 Solution Pathway 1: The Custom RenderEngine**

Blender allows developers to create custom render engines via bpy.types.RenderEngine.

**How it works:**

* You register a class inheriting from RenderEngine.  
* You implement the view\_draw(self, context) method.4  
* When the user selects your engine (e.g., "GaussianSplatRenderer") in the Render Properties, Blender calls view\_draw for every viewport update.

**VR Behavior:**

* Blender's VR pipeline *does* respect the active Render Engine. If "GaussianSplatRenderer" is active, wm\_xr\_draw.c will call its view\_draw method instead of Eevee's.  
* The context passed to view\_draw typically contains the correct matrices for the VR eye being rendered.

**Pros:**

* Native API support.  
* Correct integration into the render loop.

**Cons:**

* **The "Replacement" Problem:** This replaces the entire rendering pipeline. If you use this, you lose Eevee/Cycles. You cannot see your modeled meshes, lights, or world backgrounds unless you re-implement a mesh renderer inside your custom engine.  
* **Use Case Mismatch:** This is excellent if the *entire* VR experience is just the point cloud. It is useless if the goal is to use 3DGS as an asset *within* a Blender scene (e.g., a scanned character in a modeled room).

### **5.2 Solution Pathway 2: Python-Side Texture Injection (The "Late Latch")**

This method attempts to render the splats to a texture and display that texture in the scene.

**Mechanism:**

1. **Offscreen Context:** Create a gpu.types.GPUOffScreen object in Python.5  
2. **Update Handler:** Use bpy.app.handlers.depsgraph\_update\_post to trigger a render function.6  
3. **Rendering:** Inside the handler, bind the offscreen buffer and issue GLSL draw calls for the splats.  
4. **Display:** Map the resulting texture to a "Screen Space" plane parented to the camera, or use a custom shader on a bounding box object.

**Why it Fails for VR:**

* **Latency (The "Swimming" Effect):** The depsgraph\_update\_post handler fires when the scene is evaluated. In VR, the head tracking happens at a very high frequency (checking inputs \-\> simulation \-\> render). By the time Python renders the texture and Blender composites it onto a plane, the head pose has likely changed. The texture will appear to "lag" or "swim" behind the rest of the world.  
* **Performance:** Python overhead in the render loop introduces jitter, which is unacceptable in VR.

### **5.3 Solution Pathway 3: The OpenXR API Layer (Recommended)**

This is the robust, industry-standard solution used by tools like OpenKneeboard and XR toolkit. It bypasses Blender's internal restrictions by operating at the **OpenXR Driver Level**.

Concept:  
An OpenXR API Layer is a shared library (DLL on Windows, SO on Linux) that inserts itself between the Application (Blender) and the OpenXR Runtime. It can intercept any OpenXR function call.  
**The Architecture:**

1. **Intercept xrEndFrame:** We create a layer that hooks the xrEndFrame function. This function is called by Blender when it has finished rendering the frame and is ready to send it to the headset.  
2. **Injection:** Inside the hook, we modify the XrFrameEndInfo. We take Blender's projection layer (the scene) and **append** a second projection layer containing our Gaussian Splats.7  
3. **Independent Rendering:** The Gaussian Splat rendering happens inside the C++ API layer, completely decoupled from Blender's draw loop. It uses its own high-performance renderer (e.g., a port of the CUDA rasterizer to OpenGL/Vulkan).  
4. **Shared Memory IPC:** To control the splats (move/scale/rotate) from Blender, we use a Python script in Blender that writes the model matrix to a Shared Memory block. The API layer reads this block every frame.

## ---

**6\. Technical Implementation Guide: The OpenXR API Layer**

This section provides a step-by-step roadmap for implementing the recommended API Layer solution.

### **6.1 Inter-Process Communication (IPC) Design**

Since the API layer is a C++ library and Blender is controlled via Python, we need a bridge. Shared Memory is the fastest method.

**Data Structure (C++ Struct):**

C++

struct SplatControlBlock {  
    float modelMatrix; // 4x4 Matrix for splat position  
    float opacity;         // Master opacity  
    char plyPath;     // Path to the.ply file to load  
    int commandID;         // Increment to trigger reload  
    bool visible;  
};

Python Side (Blender Addon):  
Using Python's mmap module to write to this block.8

Python

import mmap  
import struct  
import bpy

class GaussianSplatUpdate(bpy.types.Operator):  
    def execute(self, context):  
        \# 1\. Get Active Object Matrix  
        obj \= context.active\_object  
        mat \= obj.matrix\_world  
          
        \# 2\. Serialize to Bytes  
        \#... struct.pack logic...  
          
        \# 3\. Write to Shared Memory  
        shm.seek(0)  
        shm.write(data)  
        return {'FINISHED'}

### **6.2 The C++ API Layer Implementation**

We utilize the standard **OpenXR API Layer Template** 9 as the foundation.

Key Hook: xrEndFrame  
This function is the injection point.

C++

// C++ API Layer Code Logic  
XRAPI\_ATTR XrResult XRAPI\_CALL CustomLayer\_xrEndFrame(  
    XrSession session,   
    const XrFrameEndInfo\* frameEndInfo)   
{  
    // 1\. Synchronization  
    // Read Shared Memory to get latest Model Matrix from Blender  
    UpdateSplatTransform(); 

    // 2\. Render Splats (The Heavy Lifting)  
    // We need to render stereo views.  
    // 'frameEndInfo-\>displayTime' tells us exactly when this frame will be shown.  
    // Use this time to predict head pose for minimal latency.  
      
    XrSpace locationSpace \=...; // Usually LOCAL or STAGE space  
    XrView views;  
    // Call xrLocateViews to get the exact view matrices for this frame time  
    // Note: We use the same displayTime Blender used.  
      
    RenderGaussianSplatsStereo(views, displayTime);

    // 3\. Layer Composition  
    // We construct a NEW layer list.  
    std::vector\<const XrCompositionLayerBaseHeader\*\> layers;  
      
    // First, push Blender's layer (Background scene)  
    // We assume Blender submits one Projection Layer.  
    for(int i=0; i\<frameEndInfo-\>layerCount; i++) {  
        layers.push\_back(frameEndInfo-\>layers\[i\]);  
    }  
      
    // Next, push OUR Splat Layer (Foreground)  
    // This layer must be created with XR\_COMPOSITION\_LAYER\_BLEND\_TEXTURE\_SOURCE\_ALPHA\_BIT  
    // to ensure transparency works.  
    layers.push\_back(reinterpret\_cast\<const XrCompositionLayerBaseHeader\*\>(\&mySplatLayer));

    // 4\. Submit Modified Frame  
    XrFrameEndInfo modifiedFrameInfo \= \*frameEndInfo;  
    modifiedFrameInfo.layerCount \= (uint32\_t)layers.size();  
    modifiedFrameInfo.layers \= layers.data();

    // Call the next layer (or runtime) in the chain  
    return NextLayer\_xrEndFrame(session, \&modifiedFrameInfo);  
}

### **6.3 Texture Sharing and Context Management**

For the C++ layer to render, it needs a graphics context.

* **OpenGL Context:** Since Blender uses OpenGL, it is most efficient if the API Layer also initializes an OpenGL context.  
* **Context Sharing:** On Windows, OpenGL contexts cannot easily share resources across DLL boundaries if they weren't created together. However, an API Layer loads *into* the process space of the application. This means wglGetCurrentContext called inside the API layer might return Blender's context.  
* **Strategy:**  
  1. Initialize a new OpenGL Context in the API layer on xrCreateSession.  
  2. Use wglShareLists (Windows) or glXCreateContext (Linux) to share resources with Blender's context if you need to access Blender's depth buffer (for correct occlusion).  
  3. If strict separation is desired, manage the Splat rendering entirely independently. The runtime (OpenXR Compositor) handles the merging of the two separate swapchains (Blender's and yours).

### **6.4 Handling Depth Compositing**

A critical detail for immersion is occlusion. If a Blender cube is in front of a Gaussian Splat, the splat must be occluded.

* **The Problem:** Blender submits a Color buffer. By default, it might not submit a Depth buffer to OpenXR unless the XR\_KHR\_composition\_layer\_depth extension is enabled.  
* **The Fix:**  
  1. The API Layer should inspect the layers submitted by Blender in xrEndFrame.  
  2. Check if XrCompositionLayerDepthInfoKHR is chained to Blender's projection layer.  
  3. If yes, use that Depth Texture as a **read-only depth attachment** when rendering the Gaussian Splats. This ensures the splats are culled against Blender's geometry.10

## ---

**7\. Comparative Analysis of Existing Approaches**

To justify the API Layer recommendation, we analyze precedents in the ecosystem.

### **7.1 BlenderXR (The Forked Approach)**

The **BlenderXR** project 11 was an early attempt to bring VR to Blender before native support existed. It worked by **forking** the Blender source code and rewriting the Window Manager to support VR rendering directly.

* **Verdict:** While powerful, this approach is unsustainable. Maintaining a fork of a complex codebase like Blender requires immense effort. Every Blender update breaks the fork. The API Layer approach avoids this entirely.

### **7.2 Freebird VR (The Addon Approach)**

**Freebird VR** 12 is a commercial addon for VR modeling. It operates primarily by manipulating the Blender scene (moving cameras, creating meshes) and relies on standard draw handlers for UI.

* **Observation:** Freebird struggles with the same limitations regarding custom rendering pipelines. Most of its "tools" are actual 3D meshes generated in the scene, rather than custom shader overlays, precisely because of the limitations in wm\_xr\_draw.c.

### **7.3 KIRI and ReshotAI (The Mesh Conversion Approach)**

Recent addons for 3DGS in Blender 13 typically work by converting the Splat data into a **Point Cloud** or **Mesh** with Geometry Nodes.

* **Pros:** Works natively in Eevee/Cycles.  
* **Cons:** Performance is the bottleneck. Rendering 4 million quads via Geometry Nodes is significantly slower than a tile-based compute rasterizer. For VR, where \<13ms frametimes are mandatory, this approach often fails to deliver smooth experiences for large datasets.

## ---

**8\. Development Roadmap and Future Outlook**

### **8.1 Implementation Checklist for the Developer**

1. **Environment Setup:** Install OpenXR SDK and CMake. Clone the OpenXR-API-Layer-Template.  
2. **Renderer Porting:** Port the standard C++ Gaussian Splatting rasterizer (often based on the original CUDA implementation, but porting to Compute Shaders is necessary for broader compatibility) into the API Layer project.  
3. **Hook Implementation:** Implement the xrEndFrame hook as detailed in Section 6.2.  
4. **Blender Connector:** Write the Python addon to handle file loading and coordinate transform synchronization via Shared Memory.  
5. **Testing:** Launch Blender with the XR\_ENABLE\_API\_LAYERS environment variable pointing to your compiled layer.

### **8.2 Future Outlook: Vulkan and Blender 4.x/5.x**

Blender is actively migrating its backend to **Vulkan**. This transition will eventually impact the GHOST-XR binding.

* **Implication:** An OpenGL-based API layer might break or require interop when Blender switches to Vulkan by default.  
* **Mitigation:** Writing the Gaussian Splat renderer in a modern API (Vulkan) inside the layer is future-proof. OpenXR handles the composition of a Vulkan layer (Yours) and an OpenGL layer (Blender's current state) seamlessly in most runtimes, or via simple interop.

## **9\. Conclusion**

The inability to use draw\_handler\_add in Blender VR is not a bug to be fixed, but an architectural constraint to be navigated. By accepting that Blender's internal VR render loop is closed to Python, we open the door to a more powerful solution: the **OpenXR API Layer**.

This architecture shifts the responsibility of high-performance volumetric rendering out of Blender's crowded main loop and into a dedicated, optimized driver-level component. It ensures that 3D Gaussian Splats can be rendered with the sorting precision, stereo capability, and frame-rate stability required for professional VR applications, all while keeping the core Blender installation clean and unmodified. This "Sidecar" approach represents the state-of-the-art for extending XR applications with high-fidelity custom rendering.

---

**Primary Citations Table**

| Component | Functionality | Source IDs |
| :---- | :---- | :---- |
| **Blender VR Loop** | Logic for wm\_xr\_draw.c and overlay exclusion | 1 |
| **OpenXR Hooks** | xrEndFrame interception and layer injection | 7 |
| **GHOST Binding** | OpenGL context and swapchain management | 3 |
| **IPC** | Shared Memory implementation for Python-C++ link | 8 |
| **Depth Comp** | Using XR\_KHR\_composition\_layer\_depth | 10 |
| **3DGS Render** | Mechanics of sorting and rasterization | 21 |

#### **참고 자료**

1. GSoC 2019: VR support through OpenXR \- Weekly Reports \- Blender Devtalk, 12월 8, 2025에 액세스, [https://devtalk.blender.org/t/gsoc-2019-vr-support-through-openxr-weekly-reports/7665](https://devtalk.blender.org/t/gsoc-2019-vr-support-through-openxr-weekly-reports/7665)  
2. Virtual Reality \- OpenXR \- Blender Developer, 12월 8, 2025에 액세스, [https://developer.blender.org/docs/features/gpu/viewports/xr/](https://developer.blender.org/docs/features/gpu/viewports/xr/)  
3. intern/ghost/intern/GHOST\_XrGraphicsBinding.cpp · v3.3.5 · raas / BlenderPhi \- GitLab, 12월 7, 2025에 액세스, [https://code.it4i.cz/raas/blenderphi/-/blob/v3.3.5/intern/ghost/intern/GHOST\_XrGraphicsBinding.cpp](https://code.it4i.cz/raas/blenderphi/-/blob/v3.3.5/intern/ghost/intern/GHOST_XrGraphicsBinding.cpp)  
4. RenderEngine(bpy\_struct) \- Blender Python API, 12월 8, 2025에 액세스, [https://docs.blender.org/api/current/bpy.types.RenderEngine.html](https://docs.blender.org/api/current/bpy.types.RenderEngine.html)  
5. gpu.types. \- Blender Python API \- Blender Documentation, 12월 8, 2025에 액세스, [https://docs.blender.org/api/current/gpu.types.html](https://docs.blender.org/api/current/gpu.types.html)  
6. Application Handlers (bpy.app.handlers) \- Blender Python API, 12월 8, 2025에 액세스, [https://docs.blender.org/api/current/bpy.app.handlers.html](https://docs.blender.org/api/current/bpy.app.handlers.html)  
7. Third-Party Developers | OpenKneeboard, 12월 7, 2025에 액세스, [https://openkneeboard.com/faq/third-party-developers/](https://openkneeboard.com/faq/third-party-developers/)  
8. Shared memory API, where a process can attach shared memory to other process, 12월 8, 2025에 액세스, [https://stackoverflow.com/questions/7073566/shared-memory-api-where-a-process-can-attach-shared-memory-to-other-process](https://stackoverflow.com/questions/7073566/shared-memory-api-where-a-process-can-attach-shared-memory-to-other-process)  
9. Ybalrid/OpenXR-API-Layer-Template \- GitHub, 12월 8, 2025에 액세스, [https://github.com/Ybalrid/OpenXR-API-Layer-Template](https://github.com/Ybalrid/OpenXR-API-Layer-Template)  
10. OpenXR app best practices \- Mixed Reality \- Microsoft Learn, 12월 8, 2025에 액세스, [https://learn.microsoft.com/en-us/windows/mixed-reality/develop/native/openxr-best-practices](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/native/openxr-best-practices)  
11. MARUI-PlugIn/BlenderXR: Use Blender in VR/AR/XR \- GitHub, 12월 7, 2025에 액세스, [https://github.com/MARUI-PlugIn/BlenderXR](https://github.com/MARUI-PlugIn/BlenderXR)  
12. Switch seamlessly between Blender and VR \- Freebird XR, 12월 8, 2025에 액세스, [https://freebirdxr.com/docs/guides/switch-seamlessly-between-blender-vr/](https://freebirdxr.com/docs/guides/switch-seamlessly-between-blender-vr/)  
13. 3D Gaussian Splatting on Blender, In Its Truest Form \- befores & afters, 12월 7, 2025에 액세스, [https://beforesandafters.com/2024/10/10/3d-gaussian-splatting-on-blender-in-its-truest-form/](https://beforesandafters.com/2024/10/10/3d-gaussian-splatting-on-blender-in-its-truest-form/)  
14. (More) Accurate 3DGS rendering \- Blender Artists, 12월 7, 2025에 액세스, [https://blenderartists.org/t/more-accurate-3dgs-rendering/1521823](https://blenderartists.org/t/more-accurate-3dgs-rendering/1521823)  
15. Blender VR Scene Inspection addon cannot connect to OpenXR : r/virtualreality\_linux, 12월 8, 2025에 액세스, [https://www.reddit.com/r/virtualreality\_linux/comments/1op9q7o/blender\_vr\_scene\_inspection\_addon\_cannot\_connect/](https://www.reddit.com/r/virtualreality_linux/comments/1op9q7o/blender_vr_scene_inspection_addon_cannot_connect/)  
16. Help me understand the viewport render pipeline with cycles \- Blender Stack Exchange, 12월 7, 2025에 액세스, [https://blender.stackexchange.com/questions/202783/help-me-understand-the-viewport-render-pipeline-with-cycles](https://blender.stackexchange.com/questions/202783/help-me-understand-the-viewport-render-pipeline-with-cycles)  
17. Composition Layers Support | OpenXR Plugin | 1.13.0 \- Unity \- Manual, 12월 7, 2025에 액세스, [https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.13/manual/features/compositionlayers.html](https://docs.unity3d.com/Packages/com.unity.xr.openxr@1.13/manual/features/compositionlayers.html)  
18. intern/ghost/intern/GHOST\_IXrGraphicsBinding.h · 1b8d33b18c2f9b6763c7163418e8e9235428fc59 · raas / BlenderPhi \- GitLab, 12월 7, 2025에 액세스, [https://code.it4i.cz/raas/blenderphi/-/blob/1b8d33b18c2f9b6763c7163418e8e9235428fc59/intern/ghost/intern/GHOST\_IXrGraphicsBinding.h](https://code.it4i.cz/raas/blenderphi/-/blob/1b8d33b18c2f9b6763c7163418e8e9235428fc59/intern/ghost/intern/GHOST_IXrGraphicsBinding.h)  
19. How to effectively share OpenGL texture between two different processes? \- Stack Overflow, 12월 7, 2025에 액세스, [https://stackoverflow.com/questions/76265148/how-to-effectively-share-opengl-texture-between-two-different-processes](https://stackoverflow.com/questions/76265148/how-to-effectively-share-opengl-texture-between-two-different-processes)  
20. 3 Graphics — OpenXR Tutorial documentation, 12월 8, 2025에 액세스, [https://openxr-tutorial.com/linux/opengl/3-graphics.html](https://openxr-tutorial.com/linux/opengl/3-graphics.html)  
21. MrNeRF/awesome-3D-gaussian-splatting \- GitHub, 12월 7, 2025에 액세스, [https://github.com/MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)  
22. clarte53/GaussianSplattingVRViewerUnity: A VR viewer for gaussian splatting models developped as native plugin for unity with the original CUDA rasterizer. \- GitHub, 12월 8, 2025에 액세스, [https://github.com/clarte53/GaussianSplattingVRViewerUnity](https://github.com/clarte53/GaussianSplattingVRViewerUnity)  
23. Render Gaussian Splatting Models with