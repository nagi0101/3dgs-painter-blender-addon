#pragma once

/**
 * Gaussian Renderer for VR (Phase B - Full Splatting)
 * 
 * Renders Gaussians with proper elliptical splatting using:
 * - Billboard quads with instanced rendering
 * - 3D to 2D covariance projection
 * - Proper Gaussian evaluation in fragment shader
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <GL/gl.h>

#include "gaussian_data.h"
#include <vector>

namespace gaussian {

/**
 * Full Gaussian Splatting renderer
 */
class GaussianRenderer {
public:
    GaussianRenderer();
    ~GaussianRenderer();
    
    bool Initialize();
    void Shutdown();
    bool IsInitialized() const { return m_initialized; }
    
    /**
     * Render Gaussians with full splatting
     * @param gaussians Array of GaussianPrimitive data
     * @param count Number of Gaussians
     * @param header Contains view/proj matrices and viewport info
     */
    void RenderFromPrimitives(
        const GaussianPrimitive* gaussians,
        uint32_t count,
        const SharedMemoryHeader* header = nullptr);
    
    /**
     * Render with explicit view/projection matrices (for stereo per-eye rendering)
     * @param gaussians Array of GaussianPrimitive data
     * @param count Number of Gaussians
     * @param viewMatrix 4x4 column-major view matrix
     * @param projMatrix 4x4 column-major projection matrix
     * @param cameraPosition Blender camera world position for VR origin offset
     * @param cameraRotation Blender camera rotation quaternion (w,x,y,z) for coordinate alignment
     * @param sortedIndices Array of indices for back-to-front rendering order
     * @param indicesValid True if sortedIndices should be used
     * @param viewportWidth Render target width
     * @param viewportHeight Render target height
     */
    void RenderFromPrimitivesWithMatrices(
        const GaussianPrimitive* gaussians,
        uint32_t count,
        const float* viewMatrix,
        const float* projMatrix,
        const float* cameraPosition,
        const float* cameraRotation,
        const uint32_t* sortedIndices,
        bool indicesValid,
        uint32_t viewportWidth,
        uint32_t viewportHeight);

private:
    bool CreateShader();
    bool CreateBuffers();
    
    bool m_initialized = false;
    
    // Shader program
    GLuint m_shaderProgram = 0;
    GLuint m_vertexShader = 0;
    GLuint m_fragmentShader = 0;
    
    // Vertex buffers
    GLuint m_vao = 0;
    GLuint m_quadBuffer = 0;      // Billboard quad vertices
    GLuint m_instanceBuffer = 0;  // Per-instance gaussian data
    
    // Uniform locations
    GLint m_viewProjMatrixLoc = -1;
    GLint m_viewMatrixLoc = -1;
    GLint m_camPosAndFocalXLoc = -1;
    GLint m_viewportAndFocalYLoc = -1;
};

/**
 * Get global renderer instance
 */
GaussianRenderer& GetGaussianRenderer();

}  // namespace gaussian
