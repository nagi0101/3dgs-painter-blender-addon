/**
 * OpenXR API Layer - DLL Entry Point (PHASE 1)
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include "xr_dispatch.h"
#include "shared_memory.h"

#include <iostream>
#include <fstream>
#include <cstdarg>
#include <cstring>

namespace {
    std::ofstream g_logFile;
    void Log(const char* format, ...) {
        if (!g_logFile.is_open()) {
            char path[MAX_PATH];
            GetTempPathA(MAX_PATH, path);
            std::string logPath = std::string(path) + "gaussian_layer.log";
            g_logFile.open(logPath, std::ios::app);
        }
        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        g_logFile << "[GaussianLayer] " << buffer << std::endl;
        g_logFile.flush();
        OutputDebugStringA("[GaussianLayer] ");
        OutputDebugStringA(buffer);
        OutputDebugStringA("\n");
    }
}

XrResult XRAPI_CALL layer_xrGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function);
XrResult XRAPI_CALL layer_xrCreateApiLayerInstance(const XrInstanceCreateInfo* createInfo, const XrApiLayerCreateInfo* layerInfo, XrInstance* instance);

extern "C" __declspec(dllexport) XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(
    const XrNegotiateLoaderInfo* loaderInfo, const char* layerName, XrNegotiateApiLayerRequest* layerRequest) {
    Log("xrNegotiateLoaderApiLayerInterface called, layer: %s", layerName);
    if (!loaderInfo || !layerRequest) return XR_ERROR_INITIALIZATION_FAILED;
    if (loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO ||
        loaderInfo->structVersion != XR_LOADER_INFO_STRUCT_VERSION ||
        loaderInfo->structSize != sizeof(XrNegotiateLoaderInfo)) return XR_ERROR_INITIALIZATION_FAILED;
    if (layerRequest->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST ||
        layerRequest->structVersion != XR_API_LAYER_INFO_STRUCT_VERSION ||
        layerRequest->structSize != sizeof(XrNegotiateApiLayerRequest)) return XR_ERROR_INITIALIZATION_FAILED;
    layerRequest->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
    layerRequest->layerApiVersion = XR_CURRENT_API_VERSION;
    layerRequest->getInstanceProcAddr = layer_xrGetInstanceProcAddr;
    layerRequest->createApiLayerInstance = layer_xrCreateApiLayerInstance;
    Log("Layer negotiation successful");
    return XR_SUCCESS;
}

XrResult XRAPI_CALL layer_xrCreateApiLayerInstance(const XrInstanceCreateInfo* createInfo, const XrApiLayerCreateInfo* layerInfo, XrInstance* instance) {
    Log("layer_xrCreateApiLayerInstance: App=%s", createInfo->applicationInfo.applicationName);
    PFN_xrCreateApiLayerInstance nextCreate = layerInfo->nextInfo->nextCreateApiLayerInstance;
    XrApiLayerCreateInfo modifiedLayerInfo = *layerInfo;
    modifiedLayerInfo.nextInfo = layerInfo->nextInfo->next;
    XrResult result = nextCreate(createInfo, &modifiedLayerInfo, instance);
    if (XR_FAILED(result)) { Log("Instance creation failed: %d", result); return result; }
    Log("Instance created");
    gaussian::GetLayerState().instance = *instance;
    gaussian::InitializeDispatch(*instance, layerInfo->nextInfo->nextGetInstanceProcAddr);
    return XR_SUCCESS;
}

XrResult XRAPI_CALL layer_xrGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function) {
    if (strcmp(name, "xrEndFrame") == 0) { *function = (PFN_xrVoidFunction)gaussian::gaussian_xrEndFrame; return XR_SUCCESS; }
    if (strcmp(name, "xrBeginFrame") == 0) { *function = (PFN_xrVoidFunction)gaussian::gaussian_xrBeginFrame; return XR_SUCCESS; }
    if (strcmp(name, "xrWaitFrame") == 0) { *function = (PFN_xrVoidFunction)gaussian::gaussian_xrWaitFrame; return XR_SUCCESS; }
    if (strcmp(name, "xrCreateSession") == 0) { *function = (PFN_xrVoidFunction)gaussian::gaussian_xrCreateSession; return XR_SUCCESS; }
    auto& state = gaussian::GetLayerState();
    if (state.next_xrGetInstanceProcAddr) return state.next_xrGetInstanceProcAddr(instance, name, function);
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
    switch (reason) {
        case DLL_PROCESS_ATTACH: Log("Gaussian Layer loaded"); DisableThreadLibraryCalls(hModule); break;
        case DLL_PROCESS_DETACH: Log("Gaussian Layer unloaded"); gaussian::ShutdownDispatch(); if (g_logFile.is_open()) g_logFile.close(); break;
    }
    return TRUE;
}
