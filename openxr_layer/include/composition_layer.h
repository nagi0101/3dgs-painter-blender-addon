#pragma once

/**
 * Composition Layer for OpenXR Quad Overlay (Phase 3)
 * 
 * Uses OpenGL for compatibility with Blender's OpenGL-based OpenXR session.
 * Manages XrSwapchain and XrCompositionLayerQuad to display
 * rendered Gaussians as an overlay in VR.
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

namespace gaussian {

/**
 * OpenGL Quad Layer - Manages XrCompositionLayerQuad with OpenGL textures
 */
class QuadLayer {
public:
    QuadLayer();
    ~QuadLayer();
    
    // Non-copyable
    QuadLayer(const QuadLayer&) = delete;
    QuadLayer& operator=(const QuadLayer&) = delete;
    
    /**
     * Initialize the quad layer with OpenGL
     * @param instance OpenXR instance
     * @param session OpenXR session (must be OpenGL-based)
     * @param width Texture width
     * @param height Texture height
     */
    bool Initialize(
        XrInstance instance,
        XrSession session,
        uint32_t width = 512,
        uint32_t height = 512);
    
    /**
     * Shutdown and release resources
     */
    void Shutdown();
    
    /**
     * Check if initialized
     */
    bool IsInitialized() const { return m_swapchain != XR_NULL_HANDLE; }
    
    /**
     * Begin rendering - acquire swapchain image
     * @return OpenGL texture ID to render to, or 0 on failure
     */
    GLuint BeginRender();
    
    /**
     * End rendering - release swapchain image
     */
    void EndRender();
    
    /**
     * Get the composition layer header for xrEndFrame
     * @param displayTime Predicted display time
     * @return Pointer to the layer (valid until next BeginRender)
     */
    const XrCompositionLayerBaseHeader* GetLayer(XrTime displayTime);
    
    /**
     * Set quad position (meters from reference space origin)
     */
    void SetPosition(float x, float y, float z);
    
    /**
     * Set quad size (meters)
     */
    void SetSize(float width, float height);

    /**
     * Clear the current texture with a solid color
     * Call this between BeginRender() and EndRender()
     */
    void ClearWithColor(float r, float g, float b, float a);

private:
    bool LoadOpenXRFunctions();
    bool CreateSwapchain(uint32_t width, uint32_t height);
    bool CreateReferenceSpace();
    bool CreateFBO();
    
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSession m_session = XR_NULL_HANDLE;
    XrSwapchain m_swapchain = XR_NULL_HANDLE;
    XrSpace m_localSpace = XR_NULL_HANDLE;
    
    // OpenGL swapchain images
    std::vector<XrSwapchainImageOpenGLKHR> m_swapchainImages;
    uint32_t m_currentImageIndex = 0;
    uint32_t m_width = 512;
    uint32_t m_height = 512;
    
    // OpenGL FBO for rendering
    GLuint m_fbo = 0;
    
    // Quad layer configuration
    XrCompositionLayerQuad m_quadLayer = {};
    XrPosef m_pose = {};
    XrExtent2Df m_size = { 0.5f, 0.5f };  // 0.5m x 0.5m default
    
    bool m_renderInProgress = false;
    bool m_functionsLoaded = false;
};

/**
 * Get global QuadLayer instance
 */
QuadLayer& GetQuadLayer();

}  // namespace gaussian
