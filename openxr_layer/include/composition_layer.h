#pragma once

/**
 * Composition Layer for OpenXR Quad Overlay (Phase 3)
 * 
 * Manages XrSwapchain and XrCompositionLayerQuad to display
 * rendered Gaussians as an overlay in VR.
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// D3D11 headers BEFORE OpenXR
#include <d3d11.h>

// Now define XR_USE_GRAPHICS_API_D3D11 for openxr_platform.h
#define XR_USE_GRAPHICS_API_D3D11
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <vector>
#include <memory>

namespace gaussian {

/**
 * Quad Layer - Manages a single XrCompositionLayerQuad for overlay rendering
 */
class QuadLayer {
public:
    QuadLayer();
    ~QuadLayer();
    
    // Non-copyable
    QuadLayer(const QuadLayer&) = delete;
    QuadLayer& operator=(const QuadLayer&) = delete;
    
    /**
     * Initialize the quad layer
     * @param instance OpenXR instance
     * @param session OpenXR session
     * @param d3d11Device D3D11 device for swapchain
     * @param width Texture width
     * @param height Texture height
     */
    bool Initialize(
        XrInstance instance,
        XrSession session,
        ID3D11Device* d3d11Device,
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
     * @return D3D11 texture to render to, or nullptr on failure
     */
    ID3D11Texture2D* BeginRender();
    
    /**
     * End rendering - release swapchain image
     */
    void EndRender();
    
    /**
     * Get the composition layer header for xrEndFrame
     * @param space Reference space for positioning
     * @param displayTime Predicted display time
     * @return Pointer to the layer (valid until next BeginRender)
     */
    const XrCompositionLayerBaseHeader* GetLayer(
        XrSpace space,
        XrTime displayTime);
    
    /**
     * Set quad position (meters from reference space origin)
     */
    void SetPosition(float x, float y, float z);
    
    /**
     * Set quad size (meters)
     */
    void SetSize(float width, float height);

private:
    bool CreateSwapchain(ID3D11Device* device, uint32_t width, uint32_t height);
    bool CreateReferenceSpace();
    
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSession m_session = XR_NULL_HANDLE;
    XrSwapchain m_swapchain = XR_NULL_HANDLE;
    XrSpace m_localSpace = XR_NULL_HANDLE;
    
    std::vector<XrSwapchainImageD3D11KHR> m_swapchainImages;
    uint32_t m_currentImageIndex = 0;
    uint32_t m_width = 512;
    uint32_t m_height = 512;
    
    // Quad layer configuration
    XrCompositionLayerQuad m_quadLayer = {};
    XrPosef m_pose = {};
    XrExtent2Df m_size = { 1.0f, 1.0f };  // 1m x 1m default
    
    bool m_renderInProgress = false;
};

/**
 * Get global QuadLayer instance
 */
QuadLayer& GetQuadLayer();

}  // namespace gaussian
