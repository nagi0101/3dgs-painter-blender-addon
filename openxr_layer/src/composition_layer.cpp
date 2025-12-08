/**
 * Composition Layer Implementation (Phase 3)
 * 
 * Uses xrGetInstanceProcAddr to call OpenXR functions dynamically
 * (API Layers cannot link to loader)
 */

#include "composition_layer.h"
#include "xr_dispatch.h"
#include <cstring>
#include <cstdio>
#include <cstdarg>

namespace gaussian {

// ============================================
// Function Pointers (loaded via xrGetInstanceProcAddr)
// ============================================
static PFN_xrCreateReferenceSpace pfn_xrCreateReferenceSpace = nullptr;
static PFN_xrDestroySpace pfn_xrDestroySpace = nullptr;
static PFN_xrCreateSwapchain pfn_xrCreateSwapchain = nullptr;
static PFN_xrDestroySwapchain pfn_xrDestroySwapchain = nullptr;
static PFN_xrEnumerateSwapchainImages pfn_xrEnumerateSwapchainImages = nullptr;
static PFN_xrAcquireSwapchainImage pfn_xrAcquireSwapchainImage = nullptr;
static PFN_xrWaitSwapchainImage pfn_xrWaitSwapchainImage = nullptr;
static PFN_xrReleaseSwapchainImage pfn_xrReleaseSwapchainImage = nullptr;

// ============================================
// Logging
// ============================================
static void LogLayer(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    OutputDebugStringA("[GaussianQuad] ");
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");
}

// ============================================
// Load Function Pointers
// ============================================
static bool LoadFunctions(XrInstance instance) {
    auto& state = GetLayerState();
    auto getProcAddr = state.next_xrGetInstanceProcAddr;
    
    if (!getProcAddr) {
        LogLayer("No xrGetInstanceProcAddr available");
        return false;
    }
    
    #define LOAD_XR_FUNC(name) \
        if (XR_FAILED(getProcAddr(instance, #name, reinterpret_cast<PFN_xrVoidFunction*>(&pfn_##name)))) { \
            LogLayer("Failed to load " #name); \
            return false; \
        }
    
    LOAD_XR_FUNC(xrCreateReferenceSpace);
    LOAD_XR_FUNC(xrDestroySpace);
    LOAD_XR_FUNC(xrCreateSwapchain);
    LOAD_XR_FUNC(xrDestroySwapchain);
    LOAD_XR_FUNC(xrEnumerateSwapchainImages);
    LOAD_XR_FUNC(xrAcquireSwapchainImage);
    LOAD_XR_FUNC(xrWaitSwapchainImage);
    LOAD_XR_FUNC(xrReleaseSwapchainImage);
    
    #undef LOAD_XR_FUNC
    
    LogLayer("All OpenXR functions loaded");
    return true;
}

// ============================================
// Global Instance
// ============================================
static QuadLayer g_quadLayer;

QuadLayer& GetQuadLayer() {
    return g_quadLayer;
}

// ============================================
// QuadLayer Implementation
// ============================================
QuadLayer::QuadLayer() {
    m_pose.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_pose.position = { 0.0f, 0.0f, -2.0f };
}

QuadLayer::~QuadLayer() {
    Shutdown();
}

bool QuadLayer::Initialize(
    XrInstance instance,
    XrSession session,
    ID3D11Device* d3d11Device,
    uint32_t width,
    uint32_t height)
{
    if (m_swapchain != XR_NULL_HANDLE) {
        return true;
    }
    
    LogLayer("Initializing QuadLayer %ux%u", width, height);
    
    // Load OpenXR functions
    if (!LoadFunctions(instance)) {
        LogLayer("Failed to load OpenXR functions");
        return false;
    }
    
    m_instance = instance;
    m_session = session;
    m_width = width;
    m_height = height;
    
    if (!CreateReferenceSpace()) {
        LogLayer("Failed to create reference space");
        return false;
    }
    
    if (!CreateSwapchain(d3d11Device, width, height)) {
        LogLayer("Failed to create swapchain");
        Shutdown();
        return false;
    }
    
    LogLayer("QuadLayer initialized successfully");
    return true;
}

void QuadLayer::Shutdown() {
    if (m_swapchain != XR_NULL_HANDLE && pfn_xrDestroySwapchain) {
        LogLayer("Destroying swapchain");
        pfn_xrDestroySwapchain(m_swapchain);
        m_swapchain = XR_NULL_HANDLE;
    }
    
    if (m_localSpace != XR_NULL_HANDLE && pfn_xrDestroySpace) {
        pfn_xrDestroySpace(m_localSpace);
        m_localSpace = XR_NULL_HANDLE;
    }
    
    m_swapchainImages.clear();
    m_instance = XR_NULL_HANDLE;
    m_session = XR_NULL_HANDLE;
}

bool QuadLayer::CreateReferenceSpace() {
    if (!pfn_xrCreateReferenceSpace) return false;
    
    XrReferenceSpaceCreateInfo createInfo = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    createInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    createInfo.poseInReferenceSpace.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    createInfo.poseInReferenceSpace.position = { 0.0f, 0.0f, 0.0f };
    
    XrResult result = pfn_xrCreateReferenceSpace(m_session, &createInfo, &m_localSpace);
    if (XR_FAILED(result)) {
        LogLayer("xrCreateReferenceSpace failed: %d", result);
        return false;
    }
    
    return true;
}

bool QuadLayer::CreateSwapchain(ID3D11Device* device, uint32_t width, uint32_t height) {
    if (!pfn_xrCreateSwapchain || !pfn_xrEnumerateSwapchainImages) return false;
    
    XrSwapchainCreateInfo createInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
    createInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | 
                           XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    createInfo.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    createInfo.sampleCount = 1;
    createInfo.width = width;
    createInfo.height = height;
    createInfo.faceCount = 1;
    createInfo.arraySize = 1;
    createInfo.mipCount = 1;
    
    XrResult result = pfn_xrCreateSwapchain(m_session, &createInfo, &m_swapchain);
    if (XR_FAILED(result)) {
        LogLayer("xrCreateSwapchain failed: %d", result);
        return false;
    }
    
    uint32_t imageCount = 0;
    result = pfn_xrEnumerateSwapchainImages(m_swapchain, 0, &imageCount, nullptr);
    if (XR_FAILED(result) || imageCount == 0) {
        LogLayer("xrEnumerateSwapchainImages count failed: %d", result);
        return false;
    }
    
    m_swapchainImages.resize(imageCount);
    for (auto& img : m_swapchainImages) {
        img.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR;
        img.next = nullptr;
    }
    
    result = pfn_xrEnumerateSwapchainImages(
        m_swapchain,
        imageCount,
        &imageCount,
        reinterpret_cast<XrSwapchainImageBaseHeader*>(m_swapchainImages.data())
    );
    
    if (XR_FAILED(result)) {
        LogLayer("xrEnumerateSwapchainImages failed: %d", result);
        return false;
    }
    
    LogLayer("Swapchain created with %u images", imageCount);
    return true;
}

ID3D11Texture2D* QuadLayer::BeginRender() {
    if (m_swapchain == XR_NULL_HANDLE || !pfn_xrAcquireSwapchainImage || !pfn_xrWaitSwapchainImage) {
        return nullptr;
    }
    
    if (m_renderInProgress) {
        return nullptr;
    }
    
    XrSwapchainImageAcquireInfo acquireInfo = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    XrResult result = pfn_xrAcquireSwapchainImage(m_swapchain, &acquireInfo, &m_currentImageIndex);
    if (XR_FAILED(result)) {
        LogLayer("xrAcquireSwapchainImage failed: %d", result);
        return nullptr;
    }
    
    XrSwapchainImageWaitInfo waitInfo = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    waitInfo.timeout = XR_INFINITE_DURATION;
    result = pfn_xrWaitSwapchainImage(m_swapchain, &waitInfo);
    if (XR_FAILED(result)) {
        LogLayer("xrWaitSwapchainImage failed: %d", result);
        return nullptr;
    }
    
    m_renderInProgress = true;
    return m_swapchainImages[m_currentImageIndex].texture;
}

void QuadLayer::EndRender() {
    if (!m_renderInProgress || !pfn_xrReleaseSwapchainImage) {
        return;
    }
    
    XrSwapchainImageReleaseInfo releaseInfo = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    pfn_xrReleaseSwapchainImage(m_swapchain, &releaseInfo);
    
    m_renderInProgress = false;
}

const XrCompositionLayerBaseHeader* QuadLayer::GetLayer(
    XrSpace space,
    XrTime displayTime)
{
    if (m_swapchain == XR_NULL_HANDLE) {
        return nullptr;
    }
    
    m_quadLayer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    m_quadLayer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    m_quadLayer.space = m_localSpace;
    m_quadLayer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    m_quadLayer.pose = m_pose;
    m_quadLayer.size = m_size;
    
    m_quadLayer.subImage.swapchain = m_swapchain;
    m_quadLayer.subImage.imageArrayIndex = 0;
    m_quadLayer.subImage.imageRect.offset = { 0, 0 };
    m_quadLayer.subImage.imageRect.extent = { 
        static_cast<int32_t>(m_width), 
        static_cast<int32_t>(m_height) 
    };
    
    return reinterpret_cast<const XrCompositionLayerBaseHeader*>(&m_quadLayer);
}

void QuadLayer::SetPosition(float x, float y, float z) {
    m_pose.position = { x, y, z };
}

void QuadLayer::SetSize(float width, float height) {
    m_size = { width, height };
}

}  // namespace gaussian
