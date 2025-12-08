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
    GLint m_viewMatrixLoc = -1;
    GLint m_projMatrixLoc = -1;
    GLint m_viewportLoc = -1;
    GLint m_focalLoc = -1;
};

/**
 * Get global renderer instance
 */
GaussianRenderer& GetGaussianRenderer();

}  // namespace gaussian

