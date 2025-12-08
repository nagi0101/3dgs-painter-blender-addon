/**
 * Projection Layer Implementation - Stereo 3D VR Rendering
 * 
 * Implements XrCompositionLayerProjection with stereo swapchain
 * for true 3D Gaussian rendering in VR.
 */

#include "projection_layer.h"
#include "xr_dispatch.h"  // For GetLayerState()

#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <cstring>

// OpenGL constants
#define GL_RGBA8 0x8058
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_TEXTURE_2D_ARRAY 0x8C1A

// Function pointer types
typedef void (APIENTRY *PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint *framebuffers);
typedef void (APIENTRY *PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint *framebuffers);
typedef void (APIENTRY *PFNGLBINDFRAMEBUFFERSPROC)(GLenum target, GLuint framebuffer);
typedef void (APIENTRY *PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
typedef GLenum (APIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum target);

// Static function pointers
static PFNGLGENFRAMEBUFFERSPROC pfn_glGenFramebuffers = nullptr;
static PFNGLDELETEFRAMEBUFFERSPROC pfn_glDeleteFramebuffers = nullptr;
static PFNGLBINDFRAMEBUFFERSPROC pfn_glBindFramebuffer = nullptr;
static PFNGLFRAMEBUFFERTEXTURE2DPROC pfn_glFramebufferTexture2D = nullptr;
static PFNGLCHECKFRAMEBUFFERSTATUSPROC pfn_glCheckFramebufferStatus = nullptr;
static bool g_glExtensionsLoaded = false;

namespace gaussian {

// OpenXR function pointers
static PFN_xrEnumerateSwapchainFormats pfn_xrEnumerateSwapchainFormats = nullptr;
static PFN_xrCreateSwapchain pfn_xrCreateSwapchain = nullptr;
static PFN_xrDestroySwapchain pfn_xrDestroySwapchain = nullptr;
static PFN_xrEnumerateSwapchainImages pfn_xrEnumerateSwapchainImages = nullptr;
static PFN_xrAcquireSwapchainImage pfn_xrAcquireSwapchainImage = nullptr;
static PFN_xrWaitSwapchainImage pfn_xrWaitSwapchainImage = nullptr;
static PFN_xrReleaseSwapchainImage pfn_xrReleaseSwapchainImage = nullptr;
static PFN_xrCreateReferenceSpace pfn_xrCreateReferenceSpace = nullptr;
static PFN_xrDestroySpace pfn_xrDestroySpace = nullptr;
static PFN_xrLocateViews pfn_xrLocateViews = nullptr;

// Load GL extensions
static bool LoadGLExtensions() {
    if (g_glExtensionsLoaded) return true;
    
    pfn_glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)wglGetProcAddress("glGenFramebuffers");
    pfn_glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)wglGetProcAddress("glDeleteFramebuffers");
    pfn_glBindFramebuffer = (PFNGLBINDFRAMEBUFFERSPROC)wglGetProcAddress("glBindFramebuffer");
    pfn_glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)wglGetProcAddress("glFramebufferTexture2D");
    pfn_glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC)wglGetProcAddress("glCheckFramebufferStatus");
    
    g_glExtensionsLoaded = (pfn_glGenFramebuffers && pfn_glBindFramebuffer && pfn_glFramebufferTexture2D);
    return g_glExtensionsLoaded;
}

// Logging
static void LogProj(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    printf("[ProjectionLayer] %s\n", buffer);
    fflush(stdout);
}

// ============================================
// Global Instance
// ============================================
static ProjectionLayer g_projectionLayer;

ProjectionLayer& GetProjectionLayer() {
    return g_projectionLayer;
}

// ============================================
// ProjectionLayer Implementation
// ============================================
ProjectionLayer::ProjectionLayer() {
    for (int i = 0; i < EYE_COUNT; i++) {
        m_views[i] = { XR_TYPE_VIEW };
    }
}

ProjectionLayer::~ProjectionLayer() {
    Shutdown();
}

bool ProjectionLayer::LoadOpenXRFunctions() {
    if (m_functionsLoaded) return true;
    
    // Use the passed function pointer from layer dispatch (like composition_layer.cpp)
    auto& state = GetLayerState();
    auto getProcAddr = state.next_xrGetInstanceProcAddr;
    
    if (!getProcAddr || m_instance == XR_NULL_HANDLE) {
        LogProj("Cannot load functions - no getProcAddr or instance");
        return false;
    }
    
    #define LOAD_XR_FUNC(name) \
        if (XR_FAILED(getProcAddr(m_instance, #name, reinterpret_cast<PFN_xrVoidFunction*>(&pfn_##name)))) { \
            LogProj("Failed to load " #name); \
            return false; \
        }
    
    LOAD_XR_FUNC(xrEnumerateSwapchainFormats);
    LOAD_XR_FUNC(xrCreateSwapchain);
    LOAD_XR_FUNC(xrDestroySwapchain);
    LOAD_XR_FUNC(xrEnumerateSwapchainImages);
    LOAD_XR_FUNC(xrAcquireSwapchainImage);
    LOAD_XR_FUNC(xrWaitSwapchainImage);
    LOAD_XR_FUNC(xrReleaseSwapchainImage);
    LOAD_XR_FUNC(xrCreateReferenceSpace);
    LOAD_XR_FUNC(xrDestroySpace);
    LOAD_XR_FUNC(xrLocateViews);
    
    #undef LOAD_XR_FUNC
    
    m_functionsLoaded = true;
    return true;
}

bool ProjectionLayer::Initialize(
    XrInstance instance,
    XrSession session,
    uint32_t width,
    uint32_t height)
{
    m_instance = instance;
    m_session = session;
    m_width = width;
    m_height = height;
    
    LogProj("Initializing stereo projection layer (%dx%d per eye)", width, height);
    
    if (!LoadOpenXRFunctions()) {
        LogProj("Failed to load OpenXR functions");
        return false;
    }
    
    if (!LoadGLExtensions()) {
        LogProj("Failed to load GL extensions");
        return false;
    }
    
    if (!CreateReferenceSpace()) {
        LogProj("Failed to create reference space");
        return false;
    }
    
    if (!CreateStereoSwapchain(width, height)) {
        LogProj("Failed to create stereo swapchain");
        return false;
    }
    
    if (!CreateFBOs()) {
        LogProj("Failed to create FBOs");
        return false;
    }
    
    LogProj("Stereo projection layer initialized successfully");
    return true;
}

void ProjectionLayer::Shutdown() {
    for (int i = 0; i < EYE_COUNT; i++) {
        if (m_swapchains[i] != XR_NULL_HANDLE && pfn_xrDestroySwapchain) {
            pfn_xrDestroySwapchain(m_swapchains[i]);
            m_swapchains[i] = XR_NULL_HANDLE;
        }
    }
    
    if (m_localSpace != XR_NULL_HANDLE && pfn_xrDestroySpace) {
        pfn_xrDestroySpace(m_localSpace);
        m_localSpace = XR_NULL_HANDLE;
    }
    
    if (pfn_glDeleteFramebuffers) {
        for (int i = 0; i < EYE_COUNT; i++) {
            if (m_fbos[i] != 0) {
                pfn_glDeleteFramebuffers(1, &m_fbos[i]);
                m_fbos[i] = 0;
            }
        }
    }
    
    for (int i = 0; i < EYE_COUNT; i++) {
        m_swapchainImages[i].clear();
    }
}

bool ProjectionLayer::CreateReferenceSpace() {
    XrReferenceSpaceCreateInfo createInfo = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    createInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    createInfo.poseInReferenceSpace.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    createInfo.poseInReferenceSpace.position = { 0.0f, 0.0f, 0.0f };
    
    XrResult result = pfn_xrCreateReferenceSpace(m_session, &createInfo, &m_localSpace);
    if (result != XR_SUCCESS) {
        LogProj("xrCreateReferenceSpace failed: %d", result);
        return false;
    }
    
    return true;
}

bool ProjectionLayer::CreateStereoSwapchain(uint32_t width, uint32_t height) {
    // Create SEPARATE swapchain for each eye (not texture array)
    // This is more compatible with OpenGL FBO binding
    for (uint32_t eye = 0; eye < EYE_COUNT; eye++) {
        XrSwapchainCreateInfo createInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        createInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        createInfo.format = GL_RGBA8;
        createInfo.sampleCount = 1;
        createInfo.width = width;
        createInfo.height = height;
        createInfo.faceCount = 1;
        createInfo.arraySize = 1;  // Single texture per eye
        createInfo.mipCount = 1;
        
        XrResult result = pfn_xrCreateSwapchain(m_session, &createInfo, &m_swapchains[eye]);
        if (result != XR_SUCCESS) {
            LogProj("xrCreateSwapchain failed for eye %d: %d", eye, result);
            return false;
        }
        
        // Enumerate swapchain images for this eye
        uint32_t imageCount = 0;
        pfn_xrEnumerateSwapchainImages(m_swapchains[eye], 0, &imageCount, nullptr);
        
        m_swapchainImages[eye].resize(imageCount);
        for (auto& img : m_swapchainImages[eye]) {
            img.type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
            img.next = nullptr;
        }
        
        result = pfn_xrEnumerateSwapchainImages(
            m_swapchains[eye],
            imageCount,
            &imageCount,
            (XrSwapchainImageBaseHeader*)m_swapchainImages[eye].data());
        
        if (result != XR_SUCCESS) {
            LogProj("xrEnumerateSwapchainImages failed for eye %d: %d", eye, result);
            return false;
        }
        
        LogProj("Eye %d: %d swapchain images", eye, imageCount);
    }
    
    LogProj("Created separate swapchains for each eye");
    return true;
}

bool ProjectionLayer::CreateFBOs() {
    for (int eye = 0; eye < EYE_COUNT; eye++) {
        pfn_glGenFramebuffers(1, &m_fbos[eye]);
        if (m_fbos[eye] == 0) {
            LogProj("Failed to create FBO for eye %d", eye);
            return false;
        }
    }
    LogProj("Created %d FBOs for stereo rendering", EYE_COUNT);
    return true;
}

bool ProjectionLayer::LocateViews(XrTime displayTime) {
    XrViewLocateInfo locateInfo = { XR_TYPE_VIEW_LOCATE_INFO };
    locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    locateInfo.displayTime = displayTime;
    locateInfo.space = m_localSpace;
    
    XrViewState viewState = { XR_TYPE_VIEW_STATE };
    uint32_t viewCount = 0;
    
    XrResult result = pfn_xrLocateViews(
        m_session,
        &locateInfo,
        &viewState,
        EYE_COUNT,
        &viewCount,
        m_views.data());
    
    if (result != XR_SUCCESS || viewCount != EYE_COUNT) {
        LogProj("xrLocateViews failed: %d (count=%d)", result, viewCount);
        m_viewsValid = false;
        return false;
    }
    
    // Compute view and projection matrices for each eye
    for (uint32_t eye = 0; eye < EYE_COUNT; eye++) {
        ComputeViewMatrix(m_views[eye].pose, m_viewMatrices[eye].data());
        ComputeProjectionMatrix(m_views[eye].fov, 0.1f, 100.0f, m_projMatrices[eye].data());
    }
    
    m_viewsValid = true;
    return true;
}

void ProjectionLayer::ComputeViewMatrix(const XrPosef& pose, float* outMatrix) {
    // Convert XrPosef to view matrix (inverse of pose)
    const XrQuaternionf& q = pose.orientation;
    const XrVector3f& p = pose.position;
    
    // Quaternion to rotation matrix R
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    
    float r00 = 1.0f - 2.0f * (yy + zz);
    float r01 = 2.0f * (xy - wz);
    float r02 = 2.0f * (xz + wy);
    float r10 = 2.0f * (xy + wz);
    float r11 = 1.0f - 2.0f * (xx + zz);
    float r12 = 2.0f * (yz - wx);
    float r20 = 2.0f * (xz - wy);
    float r21 = 2.0f * (yz + wx);
    float r22 = 1.0f - 2.0f * (xx + yy);
    
    // View matrix = inverse(pose) = [R^T | -R^T * p]
    // IMPORTANT: Must use R^T (transpose), not R!
    // Column-major order for OpenGL:
    // Column 0: R^T row 0 = R col 0 = (r00, r01, r02)
    // Column 1: R^T row 1 = R col 1 = (r10, r11, r12)
    // Column 2: R^T row 2 = R col 2 = (r20, r21, r22)
    outMatrix[0] = r00;  outMatrix[4] = r10;  outMatrix[8]  = r20;
    outMatrix[1] = r01;  outMatrix[5] = r11;  outMatrix[9]  = r21;
    outMatrix[2] = r02;  outMatrix[6] = r12;  outMatrix[10] = r22;
    outMatrix[3] = 0.0f; outMatrix[7] = 0.0f; outMatrix[11] = 0.0f;
    
    // Translation: -R^T * p
    outMatrix[12] = -(r00*p.x + r10*p.y + r20*p.z);
    outMatrix[13] = -(r01*p.x + r11*p.y + r21*p.z);
    outMatrix[14] = -(r02*p.x + r12*p.y + r22*p.z);
    outMatrix[15] = 1.0f;
}

void ProjectionLayer::ComputeProjectionMatrix(const XrFovf& fov, float nearZ, float farZ, float* outMatrix) {
    // OpenXR FOV to symmetric/asymmetric projection matrix
    float tanLeft = tanf(fov.angleLeft);
    float tanRight = tanf(fov.angleRight);
    float tanUp = tanf(fov.angleUp);
    float tanDown = tanf(fov.angleDown);
    
    float tanWidth = tanRight - tanLeft;
    float tanHeight = tanUp - tanDown;
    
    // Column-major order for OpenGL
    memset(outMatrix, 0, 16 * sizeof(float));
    
    outMatrix[0] = 2.0f / tanWidth;
    outMatrix[5] = 2.0f / tanHeight;
    outMatrix[8] = (tanRight + tanLeft) / tanWidth;
    outMatrix[9] = (tanUp + tanDown) / tanHeight;
    outMatrix[10] = -(farZ + nearZ) / (farZ - nearZ);
    outMatrix[11] = -1.0f;
    outMatrix[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
}

GLuint ProjectionLayer::BeginRenderEye(uint32_t eyeIndex) {
    if (!IsInitialized() || eyeIndex >= EYE_COUNT || m_renderInProgress) {
        return 0;
    }
    
    // Acquire swapchain image for THIS eye
    XrSwapchainImageAcquireInfo acquireInfo = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    XrResult result = pfn_xrAcquireSwapchainImage(m_swapchains[eyeIndex], &acquireInfo, &m_currentImageIndex[eyeIndex]);
    if (result != XR_SUCCESS) {
        LogProj("xrAcquireSwapchainImage failed for eye %d: %d", eyeIndex, result);
        return 0;
    }
    
    XrSwapchainImageWaitInfo waitInfo = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    waitInfo.timeout = XR_INFINITE_DURATION;
    result = pfn_xrWaitSwapchainImage(m_swapchains[eyeIndex], &waitInfo);
    if (result != XR_SUCCESS) {
        LogProj("xrWaitSwapchainImage failed for eye %d: %d", eyeIndex, result);
        return 0;
    }
    
    m_currentEye = eyeIndex;
    m_renderInProgress = true;
    
    // Get texture from swapchain (regular 2D texture, not array)
    GLuint texture = m_swapchainImages[eyeIndex][m_currentImageIndex[eyeIndex]].image;
    
    pfn_glBindFramebuffer(GL_FRAMEBUFFER, m_fbos[eyeIndex]);
    // Use glFramebufferTexture2D instead of glFramebufferTextureLayer
    pfn_glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    
    GLenum status = pfn_glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LogProj("FBO incomplete for eye %d: 0x%x", eyeIndex, status);
    }
    
    glViewport(0, 0, m_width, m_height);
    
    return texture;
}

void ProjectionLayer::EndRenderEye() {
    if (!m_renderInProgress) return;
    
    pfn_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Release swapchain image for THIS eye
    XrSwapchainImageReleaseInfo releaseInfo = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    pfn_xrReleaseSwapchainImage(m_swapchains[m_currentEye], &releaseInfo);
    
    m_renderInProgress = false;
}

const float* ProjectionLayer::GetViewMatrix(uint32_t eyeIndex) const {
    if (eyeIndex >= EYE_COUNT) return nullptr;
    return m_viewMatrices[eyeIndex].data();
}

const float* ProjectionLayer::GetProjectionMatrix(uint32_t eyeIndex) const {
    if (eyeIndex >= EYE_COUNT) return nullptr;
    return m_projMatrices[eyeIndex].data();
}

const XrCompositionLayerBaseHeader* ProjectionLayer::GetLayer() {
    if (!IsInitialized() || !m_viewsValid) {
        return nullptr;
    }
    
    // Set up projection views for each eye
    for (uint32_t eye = 0; eye < EYE_COUNT; eye++) {
        m_projectionViews[eye].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
        m_projectionViews[eye].next = nullptr;
        m_projectionViews[eye].pose = m_views[eye].pose;
        m_projectionViews[eye].fov = m_views[eye].fov;
        m_projectionViews[eye].subImage.swapchain = m_swapchains[eye];  // Separate swapchain per eye
        m_projectionViews[eye].subImage.imageArrayIndex = 0;  // Not array, just 0
        m_projectionViews[eye].subImage.imageRect.offset = { 0, 0 };
        m_projectionViews[eye].subImage.imageRect.extent = { (int32_t)m_width, (int32_t)m_height };
    }
    
    // Set up projection layer
    m_projectionLayer.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION;
    m_projectionLayer.next = nullptr;
    m_projectionLayer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    m_projectionLayer.space = m_localSpace;
    m_projectionLayer.viewCount = EYE_COUNT;
    m_projectionLayer.views = m_projectionViews.data();
    
    return reinterpret_cast<const XrCompositionLayerBaseHeader*>(&m_projectionLayer);
}

}  // namespace gaussian
