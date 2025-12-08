/**
 * GPU Context Implementation (Phase 3)
 */

#include "gpu_context.h"
#include <iostream>

namespace gaussian {

// ============================================
// Logging
// ============================================
static void LogGPU(const char* msg) {
    OutputDebugStringA("[GaussianGPU] ");
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
}

// ============================================
// Global Instance
// ============================================
static GPUContext g_gpuContext;

GPUContext& GetGPUContext() {
    return g_gpuContext;
}

// ============================================
// GPUContext Implementation
// ============================================
GPUContext::GPUContext() = default;

GPUContext::~GPUContext() {
    Shutdown();
}

bool GPUContext::Initialize() {
    if (m_device) {
        return true;  // Already initialized
    }
    
    LogGPU("Creating D3D11 device...");
    
    UINT createFlags = 0;
#ifdef _DEBUG
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Hardware acceleration
        nullptr,                    // No software rasterizer
        createFlags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        m_device.GetAddressOf(),
        &m_featureLevel,
        m_context.GetAddressOf()
    );
    
    if (FAILED(hr)) {
        LogGPU("Failed to create D3D11 device");
        return false;
    }
    
    m_ownsDevice = true;
    
    char msg[128];
    sprintf_s(msg, "D3D11 device created (feature level: 0x%X)", m_featureLevel);
    LogGPU(msg);
    
    return true;
}

bool GPUContext::InitializeWithDevice(ID3D11Device* device) {
    if (!device) {
        return false;
    }
    
    if (m_device) {
        if (m_device.Get() == device) {
            return true;  // Already using this device
        }
        Shutdown();  // Different device, reset
    }
    
    LogGPU("Using existing D3D11 device");
    
    m_device = device;
    device->GetImmediateContext(m_context.GetAddressOf());
    m_featureLevel = device->GetFeatureLevel();
    m_ownsDevice = false;
    
    return true;
}

void GPUContext::Shutdown() {
    if (!m_device) {
        return;
    }
    
    LogGPU("Shutting down GPU context");
    
    m_context.Reset();
    m_adapter.Reset();
    m_device.Reset();
}

ID3D11Texture2D* GPUContext::CreateRenderTarget(
    uint32_t width,
    uint32_t height,
    DXGI_FORMAT format)
{
    if (!m_device) {
        return nullptr;
    }
    
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;  // For OpenXR interop
    
    ID3D11Texture2D* texture = nullptr;
    HRESULT hr = m_device->CreateTexture2D(&desc, nullptr, &texture);
    
    if (FAILED(hr)) {
        LogGPU("Failed to create render target texture");
        return nullptr;
    }
    
    char msg[128];
    sprintf_s(msg, "Created render target: %ux%u", width, height);
    LogGPU(msg);
    
    return texture;
}

void GPUContext::ClearRenderTarget(
    ID3D11RenderTargetView* rtv,
    float r, float g, float b, float a)
{
    if (!m_context || !rtv) {
        return;
    }
    
    float color[4] = { r, g, b, a };
    m_context->ClearRenderTargetView(rtv, color);
}

}  // namespace gaussian
