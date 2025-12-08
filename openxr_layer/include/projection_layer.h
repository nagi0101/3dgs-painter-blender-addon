#pragma once

/**
 * Projection Layer for OpenXR Stereo Rendering (3D VR Gaussians)
 * 
 * Uses XrCompositionLayerProjection with stereo swapchain (arraySize=2)
 * to render Gaussians in true 3D space, not on a 2D quad.
 * 
 * Each eye gets its own view/projection matrix from xrLocateViews().
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// OpenGL headers
#include <GL/gl.h>

// OpenXR with OpenGL support
#define XR_USE_GRAPHICS_API_OPENGL
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <vector>
#include <array>

namespace gaussian {

/**
 * Stereo Projection Layer - True 3D VR Rendering
 * 
 * Unlike QuadLayer (2D plane), this renders to both eyes with
 * proper stereo separation for true depth perception.
 */
class ProjectionLayer {
public:
    static constexpr uint32_t EYE_COUNT = 2;  // Left, Right
    
    ProjectionLayer();
    ~ProjectionLayer();
    
    // Non-copyable
    ProjectionLayer(const ProjectionLayer&) = delete;
    ProjectionLayer& operator=(const ProjectionLayer&) = delete;
    
    /**
     * Initialize stereo rendering
     * @param instance OpenXR instance
     * @param session OpenXR session (must be OpenGL-based)
     * @param width Per-eye texture width
     * @param height Per-eye texture height
     */
    bool Initialize(
        XrInstance instance,
        XrSession session,
        uint32_t width = 1024,
        uint32_t height = 1024);
    
    /**
     * Shutdown and release resources
     */
    void Shutdown();
    
    /**
     * Check if initialized
     */
    bool IsInitialized() const { return m_swapchains[0] != XR_NULL_HANDLE; }
    
    /**
     * Begin rendering for a specific eye
     * @param eyeIndex 0=Left, 1=Right
     * @return OpenGL texture ID to render to, or 0 on failure
     */
    GLuint BeginRenderEye(uint32_t eyeIndex);
    
    /**
     * End rendering for current eye
     */
    void EndRenderEye();
    
    /**
     * Locate views for the current frame
     * Must be called once per frame before BeginRenderEye
     * @param displayTime Predicted display time from xrWaitFrame
     */
    bool LocateViews(XrTime displayTime);
    
    /**
     * Get view matrix for specified eye (column-major, OpenGL format)
     */
    const float* GetViewMatrix(uint32_t eyeIndex) const;
    
    /**
     * Get projection matrix for specified eye (column-major, OpenGL format)
     */
    const float* GetProjectionMatrix(uint32_t eyeIndex) const;
    
    /**
     * Get the composition layer header for xrEndFrame
     * Call after rendering both eyes
     */
    const XrCompositionLayerBaseHeader* GetLayer();
    
private:
    bool LoadOpenXRFunctions();
    bool CreateStereoSwapchain(uint32_t width, uint32_t height);
    bool CreateReferenceSpace();
    bool CreateFBOs();
    void ComputeViewMatrix(const XrPosef& pose, float* outMatrix);
    void ComputeProjectionMatrix(const XrFovf& fov, float nearZ, float farZ, float* outMatrix);
    
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSession m_session = XR_NULL_HANDLE;
    std::array<XrSwapchain, EYE_COUNT> m_swapchains = {};  // One per eye
    XrSpace m_localSpace = XR_NULL_HANDLE;
    
    // Per-eye swapchain images (separate for each eye)
    std::array<std::vector<XrSwapchainImageOpenGLKHR>, EYE_COUNT> m_swapchainImages;
    std::array<uint32_t, EYE_COUNT> m_currentImageIndex = {};
    uint32_t m_width = 1024;
    uint32_t m_height = 1024;
    
    // Per-eye FBOs
    std::array<GLuint, EYE_COUNT> m_fbos = {};
    
    // Views from xrLocateViews
    std::array<XrView, EYE_COUNT> m_views = {};
    bool m_viewsValid = false;
    
    // Computed matrices (column-major)
    std::array<float, 16> m_viewMatrices[EYE_COUNT] = {};
    std::array<float, 16> m_projMatrices[EYE_COUNT] = {};
    
    // Projection layer structure
    XrCompositionLayerProjection m_projectionLayer = {};
    std::array<XrCompositionLayerProjectionView, EYE_COUNT> m_projectionViews = {};
    
    uint32_t m_currentEye = 0;
    bool m_renderInProgress = false;
    bool m_functionsLoaded = false;
};

/**
 * Get global ProjectionLayer instance
 */
ProjectionLayer& GetProjectionLayer();

}  // namespace gaussian
