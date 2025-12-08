#pragma once

// PHASE 1: Pure OpenXR without platform-specific headers
// Manually define types to avoid openxr_platform.h

#include <openxr/openxr.h>

#include <cstdint>

// Forward declare ID3D11Device for storage (no D3D11 header needed here)
struct ID3D11Device;

namespace gaussian {

// ============================================
// Forward declare types from loader_negotiation.h
// (to avoid openxr_platform.h being pulled in)
// ============================================
typedef XrResult(XRAPI_PTR* PFN_xrGetInstanceProcAddrFP)(XrInstance instance, const char* name, PFN_xrVoidFunction* function);

// ============================================
// Function Pointer Types
// ============================================
using PFN_xrEndFrame = XrResult(XRAPI_PTR*)(XrSession, const XrFrameEndInfo*);
using PFN_xrBeginFrame = XrResult(XRAPI_PTR*)(XrSession, const XrFrameBeginInfo*);
using PFN_xrWaitFrame = XrResult(XRAPI_PTR*)(XrSession, const XrFrameWaitInfo*, XrFrameState*);
using PFN_xrCreateSession = XrResult(XRAPI_PTR*)(XrInstance, const XrSessionCreateInfo*, XrSession*);

// ============================================
// Layer State
// ============================================
struct LayerState {
    XrInstance instance = XR_NULL_HANDLE;
    XrSession session = XR_NULL_HANDLE;
    
    // Captured D3D11 device from xrCreateSession
    ID3D11Device* d3d11Device = nullptr;
    
    // Original function pointers (to call next in chain)
    PFN_xrEndFrame next_xrEndFrame = nullptr;
    PFN_xrBeginFrame next_xrBeginFrame = nullptr;
    PFN_xrWaitFrame next_xrWaitFrame = nullptr;
    PFN_xrCreateSession next_xrCreateSession = nullptr;
    PFN_xrGetInstanceProcAddrFP next_xrGetInstanceProcAddr = nullptr;
    
    // Frame timing
    XrTime predicted_display_time = 0;
    bool frame_active = false;
    
    // Quad layer initialized
    bool quadLayerInitialized = false;
    
    // Statistics
    uint64_t frame_count = 0;
    uint64_t gaussian_render_count = 0;
};

// ============================================
// Global State Access
// ============================================
LayerState& GetLayerState();

// ============================================
// Hooked Functions
// ============================================
XrResult XRAPI_CALL gaussian_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo);

XrResult XRAPI_CALL gaussian_xrBeginFrame(
    XrSession session,
    const XrFrameBeginInfo* frameBeginInfo);

XrResult XRAPI_CALL gaussian_xrWaitFrame(
    XrSession session,
    const XrFrameWaitInfo* frameWaitInfo,
    XrFrameState* frameState);

XrResult XRAPI_CALL gaussian_xrCreateSession(
    XrInstance instance,
    const XrSessionCreateInfo* createInfo,
    XrSession* session);

// ============================================
// Initialization
// ============================================
bool InitializeDispatch(XrInstance instance, PFN_xrGetInstanceProcAddrFP getProcAddr);
void ShutdownDispatch();

}  // namespace gaussian
