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
#include "composition_layer.h"
#include "gpu_context.h"

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
    
    // Phase 1: Just pass through
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
    
    // Search for D3D11 graphics binding in the next chain
    const XrBaseInStructure* nextStruct = 
        reinterpret_cast<const XrBaseInStructure*>(createInfo->next);
    
    while (nextStruct != nullptr) {
        if (nextStruct->type == XR_TYPE_GRAPHICS_BINDING_D3D11_KHR) {
            // Found D3D11 binding - extract device
            const XrGraphicsBindingD3D11KHR* d3d11Binding = 
                reinterpret_cast<const XrGraphicsBindingD3D11KHR*>(nextStruct);
            
            g_layerState.d3d11Device = d3d11Binding->device;
            LogXr("Captured D3D11 device: %p", g_layerState.d3d11Device);
            break;
        }
        nextStruct = nextStruct->next;
    }
    
    if (!g_layerState.d3d11Device) {
        LogXr("WARNING: No D3D11 device found in session create info");
    }
    
    // Call next in chain
    XrResult result = g_layerState.next_xrCreateSession(instance, createInfo, session);
    
    if (XR_SUCCEEDED(result)) {
        g_layerState.session = *session;
        LogXr("Session created: %p", (void*)(*session));
    }
    
    return result;
}

}  // namespace gaussian

