#pragma once

#include <cstdint>
#include <array>

namespace gaussian {

// ============================================
// Shared Memory Constants
// ============================================
constexpr const wchar_t* SHARED_MEMORY_NAME = L"Local\\3DGS_Gaussian_Data";
constexpr size_t MAX_GAUSSIANS = 100000;

// ============================================
// Single Gaussian Data (56 bytes)
// ============================================
struct alignas(4) GaussianPrimitive {
    // Position (12 bytes)
    float position[3];
    
    // Color RGBA (16 bytes)
    float color[4];
    
    // Scale (12 bytes)
    float scale[3];
    
    // Rotation quaternion (16 bytes)
    float rotation[4];
};

static_assert(sizeof(GaussianPrimitive) == 56, "GaussianPrimitive size mismatch");

// ============================================
// Shared Memory Header (180 bytes)
// ============================================
struct SharedMemoryHeader {
    uint32_t magic;              // 0x3DGS
    uint32_t version;            // 1
    uint32_t frame_id;           // Incremental frame counter
    uint32_t gaussian_count;     // Number of valid gaussians
    uint32_t flags;              // Bit 0: indices_valid
    float view_matrix[16];       // Optional: Blender's current view
    float proj_matrix[16];       // Optional: Blender's current projection
    float camera_rotation[4];    // Blender camera rotation quaternion (w,x,y,z)
    float camera_position[3];    // Blender camera world position
    float _padding;              // Alignment padding
};

// ============================================
// Memory Layout Offsets
// ============================================
constexpr size_t HEADER_SIZE = sizeof(SharedMemoryHeader);  // 180 bytes
constexpr size_t INDEX_ARRAY_SIZE = MAX_GAUSSIANS * sizeof(uint32_t);  // 400,000 bytes
constexpr size_t INDEX_ARRAY_OFFSET = HEADER_SIZE;  // Indices start after header
constexpr size_t GAUSSIAN_DATA_OFFSET = HEADER_SIZE + INDEX_ARRAY_SIZE;  // Data after indices
constexpr size_t SHARED_MEMORY_SIZE = HEADER_SIZE + INDEX_ARRAY_SIZE + (MAX_GAUSSIANS * sizeof(GaussianPrimitive));

// Flag constants
constexpr uint32_t SHMEM_FLAG_INDICES_VALID = 0x00000001;

// ============================================
// Complete Shared Memory Layout
// ============================================
struct SharedMemoryBuffer {
    SharedMemoryHeader header;
    uint32_t indices[MAX_GAUSSIANS];           // Sorted indices for back-to-front rendering
    GaussianPrimitive gaussians[MAX_GAUSSIANS];
};

static_assert(sizeof(SharedMemoryBuffer) == SHARED_MEMORY_SIZE, "Buffer size mismatch");

// Magic number for validation
constexpr uint32_t MAGIC_NUMBER = 0x33444753;  // "3DGS" in little-endian

}  // namespace gaussian

