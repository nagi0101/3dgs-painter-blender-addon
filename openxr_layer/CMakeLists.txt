cmake_minimum_required(VERSION 3.20)
project(GaussianOpenXRLayer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# OpenXR SDK - HEADERS ONLY
# ============================================
# API Layers don't link to the loader - the loader loads us!
# We only need the headers, not the full SDK build.
set(OPENXR_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenXR-SDK)

# ============================================
# Windows SDK
# ============================================
# Note: We do NOT define XR_USE_PLATFORM_WIN32 or any XR_USE_GRAPHICS_API_*
# because API Layers should be graphics/platform-agnostic.
# The loader negotiation types are available without these defines.

# ============================================
# Source Files
# ============================================
set(SOURCES
    src/main.cpp
    src/xr_dispatch.cpp
    src/composition_layer.cpp
    src/shared_memory.cpp
    src/gaussian_renderer.cpp
    src/gpu_context.cpp
)

set(HEADERS
    include/gaussian_data.h
    include/xr_dispatch.h
    include/shared_memory.h
)

# ============================================
# Create DLL
# ============================================
add_library(gaussian_layer SHARED ${SOURCES} ${HEADERS})

target_include_directories(gaussian_layer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENXR_SDK_DIR}/include
)

# ============================================
# D3D11 for GPU rendering (Phase 3)
# ============================================
target_link_libraries(gaussian_layer PRIVATE
    d3d11
    dxgi
    d3dcompiler
)

# ============================================
# Export Functions
# ============================================
set_target_properties(gaussian_layer PROPERTIES
    PREFIX ""
    OUTPUT_NAME "gaussian_layer"
)

# ============================================
# Copy Manifest
# ============================================
add_custom_command(TARGET gaussian_layer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/manifest/gaussian_layer.json
        $<TARGET_FILE_DIR:gaussian_layer>/gaussian_layer.json
)

# ============================================
# Install
# ============================================
install(TARGETS gaussian_layer RUNTIME DESTINATION bin)
install(FILES manifest/gaussian_layer.json DESTINATION bin)
