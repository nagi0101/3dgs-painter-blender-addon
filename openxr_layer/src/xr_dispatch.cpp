/**
 * OpenXR Function Dispatch (PHASE 1)
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include "xr_dispatch.h"
#include "shared_memory.h"
#include "gaussian_data.h"
#include "projection_layer.h"  // NEW: Stereo 3D rendering
#include "composition_layer.h"  // Keep for fallback
#include "gpu_context.h"
#include "gaussian_renderer.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdarg>

namespace gaussian {

// ============================================
// Global State
// ============================================
static LayerState g_layerState;
static SharedMemoryReader g_sharedMemory;

LayerState& GetLayerState() {
    return g_layerState;
}

// ============================================
// Logging
// ============================================
static std::ofstream g_dispatchLog;

static void LogXr(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    // Write to log file
    if (!g_dispatchLog.is_open()) {
        char path[MAX_PATH];
        GetTempPathA(MAX_PATH, path);
        std::string logPath = std::string(path) + "gaussian_layer.log";
        g_dispatchLog.open(logPath, std::ios::app);
    }
    if (g_dispatchLog.is_open()) {
        g_dispatchLog << "[GaussianXR] " << buffer << std::endl;
        g_dispatchLog.flush();
    }
    
    OutputDebugStringA("[GaussianXR] ");
    OutputDebugStringA(buffer);
    OutputDebugStringA("\n");
}

// ============================================
// Initialize Dispatch
// ============================================
bool InitializeDispatch(XrInstance instance, PFN_xrGetInstanceProcAddrFP getProcAddr) {
    LogXr("Initializing dispatch");
    
    g_layerState.next_xrGetInstanceProcAddr = getProcAddr;
    
    XrResult result;
    
    result = getProcAddr(instance, "xrEndFrame", 
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrEndFrame));
    if (XR_FAILED(result)) {
        LogXr("Failed to get xrEndFrame");
        return false;
    }
    
    result = getProcAddr(instance, "xrBeginFrame",
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrBeginFrame));
    if (XR_FAILED(result)) {
        LogXr("Failed to get xrBeginFrame");
        return false;
    }
    
    result = getProcAddr(instance, "xrWaitFrame",
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrWaitFrame));
    if (XR_FAILED(result)) {
        LogXr("Failed to get xrWaitFrame");
        return false;
    }
    
    result = getProcAddr(instance, "xrCreateSession",
        reinterpret_cast<PFN_xrVoidFunction*>(&g_layerState.next_xrCreateSession));
    if (XR_FAILED(result)) {
        LogXr("Failed to get xrCreateSession");
        return false;
    }
    
    if (g_sharedMemory.Open()) {
        LogXr("Shared memory opened");
    } else {
        LogXr("Shared memory not available");
    }
    
    LogXr("Dispatch initialized");
    return true;
}

void ShutdownDispatch() {
    LogXr("Shutting down");
    g_sharedMemory.Close();
    g_layerState = LayerState{};
}

// ============================================
// xrWaitFrame Hook
// ============================================
XrResult XRAPI_CALL gaussian_xrWaitFrame(
    XrSession session,
    const XrFrameWaitInfo* frameWaitInfo,
    XrFrameState* frameState)
{
    XrResult result = g_layerState.next_xrWaitFrame(session, frameWaitInfo, frameState);
    
    if (XR_SUCCEEDED(result)) {
        g_layerState.predicted_display_time = frameState->predictedDisplayTime;
    }
    
    return result;
}

// ============================================
// xrBeginFrame Hook
// ============================================
XrResult XRAPI_CALL gaussian_xrBeginFrame(
    XrSession session,
    const XrFrameBeginInfo* frameBeginInfo)
{
    g_layerState.session = session;
    g_layerState.frame_active = true;
    
    // Initialize ProjectionLayer if not done yet (stereo 3D rendering)
    if (!g_layerState.quadLayerInitialized && g_layerState.instance != XR_NULL_HANDLE) {
        LogXr("Initializing ProjectionLayer (stereo 3D)...");
        if (GetProjectionLayer().Initialize(g_layerState.instance, session, 1024, 1024)) {
            g_layerState.quadLayerInitialized = true;
            LogXr("ProjectionLayer initialized successfully");
        } else {
            LogXr("Failed to initialize ProjectionLayer");
        }
    }
    
    return g_layerState.next_xrBeginFrame(session, frameBeginInfo);
}

// ============================================
// xrEndFrame Hook - MAIN INJECTION POINT
// ============================================
XrResult XRAPI_CALL gaussian_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo)
{
    g_layerState.frame_count++;
    
    // Log every 60 frames
    if (g_layerState.frame_count % 60 == 0) {
        LogXr("Frame %llu, layers: %d", 
            g_layerState.frame_count, 
            frameEndInfo->layerCount);
    }
    
    // Check shared memory
    if (!g_sharedMemory.IsOpen()) {
        g_sharedMemory.Open();
    }
    
    if (g_sharedMemory.IsOpen()) {
        // Direct buffer inspection for debugging (bypass frame_id check)
        const auto* buffer = g_sharedMemory.GetBuffer();
        if (buffer && g_layerState.frame_count % 120 == 0) {
            LogXr("SHM Debug: magic=0x%08X, frame=%u, count=%u", 
                buffer->header.magic,
                buffer->header.frame_id,
                buffer->header.gaussian_count);
        }
        
        // Normal read with frame_id check
        auto header = g_sharedMemory.ReadHeader();
        if (header.has_value() && header->gaussian_count > 0) {
            LogXr("NEW Gaussians received: %d (frame_id=%u)", 
                header->gaussian_count, header->frame_id);
            g_layerState.gaussian_render_count = header->gaussian_count;
        }
    }
    
    // ==========================================
    // STEREO 3D RENDERING with ProjectionLayer
    // ==========================================
    if (g_layerState.quadLayerInitialized && GetProjectionLayer().IsInitialized()) {
        // Initialize GaussianRenderer if needed
        static bool rendererInitialized = false;
        if (!rendererInitialized) {
            if (GetGaussianRenderer().Initialize()) {
                rendererInitialized = true;
                LogXr("GaussianRenderer initialized for stereo");
            } else {
                LogXr("GaussianRenderer initialization FAILED");
            }
        }
        
        // Locate views for this frame (get per-eye poses)
        bool viewsLocated = GetProjectionLayer().LocateViews(frameEndInfo->displayTime);
        
        // Render to each eye
        for (uint32_t eye = 0; eye < 2; eye++) {
            GLuint texture = GetProjectionLayer().BeginRenderEye(eye);
            if (texture != 0) {
                // Clear with transparent background
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                
                // Render Gaussians with per-eye view/projection
                if (rendererInitialized && g_sharedMemory.IsOpen() && g_layerState.gaussian_render_count > 0) {
                    const auto* buffer = g_sharedMemory.GetBuffer();
                    if (buffer && buffer->header.gaussian_count > 0) {
                        // Get per-eye matrices from ProjectionLayer
                        const float* viewMatrix = GetProjectionLayer().GetViewMatrix(eye);
                        const float* projMatrix = GetProjectionLayer().GetProjectionMatrix(eye);
                        
                        if (viewMatrix && projMatrix) {
                            GetGaussianRenderer().RenderFromPrimitivesWithMatrices(
                                buffer->gaussians,
                                buffer->header.gaussian_count,
                                viewMatrix,
                                projMatrix,
                                1024, 1024  // viewport size
                            );
                        }
                    }
                }
                
                GetProjectionLayer().EndRenderEye();
            }
        }
        
        // Get projection layer (replaces quad layer - this is TRUE 3D!)
        auto* projLayerHeader = GetProjectionLayer().GetLayer();
        
        if (projLayerHeader) {
            // Create new layers array with our layer REPLACING Blender's
            std::vector<const XrCompositionLayerBaseHeader*> allLayers;
            allLayers.reserve(frameEndInfo->layerCount + 1);
            
            // Copy original layers (Blender's)
            for (uint32_t i = 0; i < frameEndInfo->layerCount; i++) {
                allLayers.push_back(frameEndInfo->layers[i]);
            }
            
            // Add our projection layer
            allLayers.push_back(projLayerHeader);
            
            // Create modified frameEndInfo
            XrFrameEndInfo modifiedEndInfo = *frameEndInfo;
            modifiedEndInfo.layerCount = static_cast<uint32_t>(allLayers.size());
            modifiedEndInfo.layers = allLayers.data();
            
            if (g_layerState.frame_count % 60 == 0) {
                LogXr("Injecting stereo projection layer (total: %u)", modifiedEndInfo.layerCount);
            }
            
            g_layerState.frame_active = false;
            return g_layerState.next_xrEndFrame(session, &modifiedEndInfo);
        }
    }
    
    // Fallback: Just pass through without modification
    g_layerState.frame_active = false;
    return g_layerState.next_xrEndFrame(session, frameEndInfo);
}

// ============================================
// xrCreateSession Hook - Capture D3D11 Device
// ============================================
XrResult XRAPI_CALL gaussian_xrCreateSession(
    XrInstance instance,
    const XrSessionCreateInfo* createInfo,
    XrSession* session)
{
    LogXr("gaussian_xrCreateSession called");
    
    // Search for graphics binding in the next chain
    const XrBaseInStructure* nextStruct = 
        reinterpret_cast<const XrBaseInStructure*>(createInfo->next);
    
    bool isOpenGL = false;
    while (nextStruct != nullptr) {
        LogXr("Found binding type: %d", nextStruct->type);
        
        // OpenGL binding - Blender uses this (type = 1000023000)
        if (nextStruct->type == XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR) {
            LogXr("App uses OpenGL - we will use OpenGL composition layers");
            isOpenGL = true;
        }
        nextStruct = nextStruct->next;
    }
    
    if (!isOpenGL) {
        LogXr("WARNING: App does not use OpenGL Win32 - quad layer may not work");
    }
    
    // Call next in chain
    XrResult result = g_layerState.next_xrCreateSession(instance, createInfo, session);
    
    if (XR_SUCCEEDED(result)) {
        g_layerState.session = *session;
        g_layerState.instance = instance;
        LogXr("Session created: %p", (void*)(*session));
    }
    
    return result;
}

}  // namespace gaussian

