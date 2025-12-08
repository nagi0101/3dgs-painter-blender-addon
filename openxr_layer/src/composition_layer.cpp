/**
 * Composition Layer Implementation (Phase 3) - OpenGL Version
 * 
 * Uses OpenGL for compatibility with Blender's OpenXR session.
 * Implements XrSwapchain and XrCompositionLayerQuad for VR overlay
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
// OpenGL Extension Function Pointers for FBO
// ============================================
// FBO constants
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5

// Function pointer types
typedef void (APIENTRY *PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint *framebuffers);
typedef void (APIENTRY *PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint *framebuffers);
typedef void (APIENTRY *PFNGLBINDFRAMEBUFFERPROC)(GLenum target, GLuint framebuffer);
typedef void (APIENTRY *PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
typedef GLenum (APIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum target);

// Static function pointers
static PFNGLGENFRAMEBUFFERSPROC pfn_glGenFramebuffers = nullptr;
static PFNGLDELETEFRAMEBUFFERSPROC pfn_glDeleteFramebuffers = nullptr;
static PFNGLBINDFRAMEBUFFERPROC pfn_glBindFramebuffer = nullptr;
static PFNGLFRAMEBUFFERTEXTURE2DPROC pfn_glFramebufferTexture2D = nullptr;
static PFNGLCHECKFRAMEBUFFERSTATUSPROC pfn_glCheckFramebufferStatus = nullptr;
static bool g_glExtensionsLoaded = false;

// Load GL extensions
static bool LoadGLExtensions() {
    if (g_glExtensionsLoaded) return true;
    
    HMODULE hOpenGL = GetModuleHandleA("opengl32.dll");
    if (!hOpenGL) return false;
    
    typedef PROC (WINAPI *PFNWGLGETPROCADDRESSPROC)(LPCSTR);
    PFNWGLGETPROCADDRESSPROC wglGetProcAddress = 
        (PFNWGLGETPROCADDRESSPROC)GetProcAddress(hOpenGL, "wglGetProcAddress");
    if (!wglGetProcAddress) return false;
    
    pfn_glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)wglGetProcAddress("glGenFramebuffers");
    pfn_glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)wglGetProcAddress("glDeleteFramebuffers");
    pfn_glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)wglGetProcAddress("glBindFramebuffer");
    pfn_glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)wglGetProcAddress("glFramebufferTexture2D");
    pfn_glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC)wglGetProcAddress("glCheckFramebufferStatus");
    
    g_glExtensionsLoaded = (pfn_glGenFramebuffers && pfn_glBindFramebuffer);
    return g_glExtensionsLoaded;
}

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
    // Initialize pose to 2 meters in front, at eye level
    m_pose.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };  // Identity quaternion
    m_pose.position = { 0.0f, 0.0f, -2.0f };  // 2m in front
}

QuadLayer::~QuadLayer() {
    Shutdown();
}

bool QuadLayer::LoadOpenXRFunctions() {
    if (m_functionsLoaded) return true;
    
    auto& state = GetLayerState();
    auto getProcAddr = state.next_xrGetInstanceProcAddr;
    
    if (!getProcAddr || m_instance == XR_NULL_HANDLE) {
        LogLayer("Cannot load functions - no getProcAddr or instance");
        return false;
    }
    
    #define LOAD_XR_FUNC(name) \
        if (XR_FAILED(getProcAddr(m_instance, #name, reinterpret_cast<PFN_xrVoidFunction*>(&pfn_##name)))) { \
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
    m_functionsLoaded = true;
    return true;
}

bool QuadLayer::Initialize(
    XrInstance instance,
    XrSession session,
    uint32_t width,
    uint32_t height)
{
    if (m_swapchain != XR_NULL_HANDLE) {
        return true;
    }
    
    LogLayer("Initializing OpenGL QuadLayer %ux%u", width, height);
    
    m_instance = instance;
    m_session = session;
    m_width = width;
    m_height = height;
    
    // Load OpenXR functions
    if (!LoadOpenXRFunctions()) {
        LogLayer("Failed to load OpenXR functions");
        return false;
    }
    
    // Create reference space
    if (!CreateReferenceSpace()) {
        LogLayer("Failed to create reference space");
        return false;
    }
    
    // Create OpenGL swapchain
    if (!CreateSwapchain(width, height)) {
        LogLayer("Failed to create OpenGL swapchain");
        Shutdown();
        return false;
    }
    
    LogLayer("OpenGL QuadLayer initialized successfully");
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
    
    LogLayer("Reference space created");
    return true;
}

bool QuadLayer::CreateSwapchain(uint32_t width, uint32_t height) {
    if (!pfn_xrCreateSwapchain || !pfn_xrEnumerateSwapchainImages) return false;
    
    // Create OpenGL swapchain with RGBA format
    XrSwapchainCreateInfo createInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
    createInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | 
                           XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    createInfo.format = GL_RGBA8;  // OpenGL format
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
    
    // Enumerate swapchain images
    uint32_t imageCount = 0;
    result = pfn_xrEnumerateSwapchainImages(m_swapchain, 0, &imageCount, nullptr);
    if (XR_FAILED(result) || imageCount == 0) {
        LogLayer("xrEnumerateSwapchainImages count failed: %d", result);
        return false;
    }
    
    m_swapchainImages.resize(imageCount);
    for (auto& img : m_swapchainImages) {
        img.type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
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
    
    LogLayer("OpenGL swapchain created with %u images", imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        LogLayer("  Image %u: GL texture %u", i, m_swapchainImages[i].image);
    }
    return true;
}

GLuint QuadLayer::BeginRender() {
    if (m_swapchain == XR_NULL_HANDLE || !pfn_xrAcquireSwapchainImage || !pfn_xrWaitSwapchainImage) {
        return 0;
    }
    
    if (m_renderInProgress) {
        return 0;
    }
    
    // Acquire swapchain image
    XrSwapchainImageAcquireInfo acquireInfo = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    XrResult result = pfn_xrAcquireSwapchainImage(m_swapchain, &acquireInfo, &m_currentImageIndex);
    if (XR_FAILED(result)) {
        LogLayer("xrAcquireSwapchainImage failed: %d", result);
        return 0;
    }
    
    // Wait for image to be ready
    XrSwapchainImageWaitInfo waitInfo = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    waitInfo.timeout = XR_INFINITE_DURATION;
    result = pfn_xrWaitSwapchainImage(m_swapchain, &waitInfo);
    if (XR_FAILED(result)) {
        LogLayer("xrWaitSwapchainImage failed: %d", result);
        return 0;
    }
    
    m_renderInProgress = true;
    
    // Create and bind FBO for rendering
    if (!LoadGLExtensions()) {
        return 0;
    }
    
    if (m_fbo == 0) {
        CreateFBO();
    }
    
    GLuint texture = m_swapchainImages[m_currentImageIndex].image;
    
    // Bind FBO and attach texture
    pfn_glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    pfn_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    
    // Set viewport
    glViewport(0, 0, m_width, m_height);
    
    return texture;
}

void QuadLayer::EndRender() {
    if (!m_renderInProgress || !pfn_xrReleaseSwapchainImage) {
        return;
    }
    
    // Unbind FBO
    if (pfn_glBindFramebuffer) {
        pfn_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    XrSwapchainImageReleaseInfo releaseInfo = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    pfn_xrReleaseSwapchainImage(m_swapchain, &releaseInfo);
    
    m_renderInProgress = false;
}

const XrCompositionLayerBaseHeader* QuadLayer::GetLayer(XrTime displayTime) {
    if (m_swapchain == XR_NULL_HANDLE) {
        return nullptr;
    }
    
    // Configure quad layer
    m_quadLayer = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    m_quadLayer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    m_quadLayer.space = m_localSpace;
    m_quadLayer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    m_quadLayer.pose = m_pose;
    m_quadLayer.size = m_size;
    
    // Swapchain subimage
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

bool QuadLayer::CreateFBO() {
    if (!LoadGLExtensions()) {
        LogLayer("Failed to load GL extensions");
        return false;
    }
    
    if (m_fbo != 0) {
        return true;  // Already created
    }
    
    pfn_glGenFramebuffers(1, &m_fbo);
    LogLayer("Created FBO: %u", m_fbo);
    return m_fbo != 0;
}

void QuadLayer::ClearWithColor(float r, float g, float b, float a) {
    if (!m_renderInProgress) {
        LogLayer("ClearWithColor called outside of render");
        return;
    }
    
    if (!LoadGLExtensions()) {
        return;
    }
    
    GLuint texture = m_swapchainImages[m_currentImageIndex].image;
    
    // Create FBO if needed
    if (m_fbo == 0) {
        CreateFBO();
    }
    
    if (m_fbo == 0) {
        return;
    }
    
    // Bind our FBO
    pfn_glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    
    // Attach texture to FBO
    pfn_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    
    // Check FBO completeness
    GLenum status = pfn_glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LogLayer("FBO not complete: 0x%X", status);
        pfn_glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return;
    }
    
    // Set viewport
    glViewport(0, 0, m_width, m_height);
    
    // Clear with color (FBO is already bound from BeginRender)
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Keep FBO bound for subsequent rendering operations
}

}  // namespace gaussian
