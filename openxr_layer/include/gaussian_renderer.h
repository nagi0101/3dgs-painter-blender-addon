#pragma once

/**
 * Gaussian Renderer for VR (Phase 3.5)
 * 
 * Renders Gaussians from shared memory to OpenGL texture.
 * Starts with simple point rendering, will evolve to full splatting.
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
 * Simple point-based Gaussian renderer
 */
class GaussianRenderer {
public:
    GaussianRenderer();
    ~GaussianRenderer();
    
    /**
     * Initialize OpenGL resources
     */
    bool Initialize();
    
    /**
     * Shutdown and cleanup
     */
    void Shutdown();
    
    /**
     * Check if initialized
     */
    bool IsInitialized() const { return m_initialized; }
    
    /**
     * Render Gaussians to the currently bound FBO
     * @param positions Array of vec3 positions
     * @param colors Array of vec4 colors (RGBA)
     * @param count Number of Gaussians
     * @param viewMatrix View matrix (4x4 column-major)
     * @param projMatrix Projection matrix (4x4 column-major)
     */
    void Render(
        const float* positions,
        const float* colors,
        uint32_t count,
        const float* viewMatrix,
        const float* projMatrix);
    
    /**
     * Render from GaussianPrimitive array with optional view/proj matrices from header
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
    GLuint m_positionBuffer = 0;
    GLuint m_colorBuffer = 0;
    
    // Uniform locations
    GLint m_viewMatrixLoc = -1;
    GLint m_projMatrixLoc = -1;
    GLint m_useMatricesLoc = -1;
};

/**
 * Get global renderer instance
 */
GaussianRenderer& GetGaussianRenderer();

}  // namespace gaussian
