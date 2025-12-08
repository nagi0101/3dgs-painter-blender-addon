#pragma once

/**
 * GPU Context for D3D11 rendering (Phase 3)
 * 
 * Manages D3D11 device, context, and resources for
 * rendering Gaussians to OpenXR composition layers.
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>  // ComPtr

#include <openxr/openxr.h>

namespace gaussian {

using Microsoft::WRL::ComPtr;

/**
 * GPU Context - Manages D3D11 resources for VR rendering
 */
class GPUContext {
public:
    GPUContext();
    ~GPUContext();
    
    // Non-copyable
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;
    
    /**
     * Initialize D3D11 device (creates our own device)
     */
    bool Initialize();
    
    /**
     * Initialize using existing device from OpenXR runtime
     * Call this if we can get the app's D3D11 device
     */
    bool InitializeWithDevice(ID3D11Device* device);
    
    /**
     * Shutdown and release resources
     */
    void Shutdown();
    
    /**
     * Check if initialized
     */
    bool IsInitialized() const { return m_device != nullptr; }
    
    /**
     * Get D3D11 device
     */
    ID3D11Device* GetDevice() const { return m_device.Get(); }
    
    /**
     * Get immediate context
     */
    ID3D11DeviceContext* GetContext() const { return m_context.Get(); }
    
    /**
     * Create a render target texture
     * @param width Texture width
     * @param height Texture height
     * @param format DXGI format (default RGBA8)
     * @return Texture2D pointer (caller must Release)
     */
    ID3D11Texture2D* CreateRenderTarget(
        uint32_t width,
        uint32_t height,
        DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM);
    
    /**
     * Clear a render target to a solid color
     */
    void ClearRenderTarget(
        ID3D11RenderTargetView* rtv,
        float r, float g, float b, float a);

private:
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGIAdapter> m_adapter;
    
    D3D_FEATURE_LEVEL m_featureLevel = D3D_FEATURE_LEVEL_11_0;
    bool m_ownsDevice = false;  // true if we created the device
};

/**
 * Get global GPU context instance
 */
GPUContext& GetGPUContext();

}  // namespace gaussian
