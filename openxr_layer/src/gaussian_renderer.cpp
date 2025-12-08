/**
 * Gaussian Renderer Implementation (Phase 3.5)
 * 
 * Simple point-based rendering as first step toward full Gaussian splatting.
 */

#include "gaussian_renderer.h"
#include <cstdio>
#include <cstring>

namespace gaussian {

// ============================================
// Missing GL Type Definitions
// ============================================
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef int GLboolean;
typedef float GLfloat;

// ============================================
// OpenGL Extension Function Pointers
// ============================================
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint program);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORMMATRIX4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint v0);

typedef void (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
typedef void (APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint *arrays);
typedef void (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);

// OpenGL Constants
#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_ARRAY_BUFFER 0x8892
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_PROGRAM_POINT_SIZE 0x8642

// Function pointers
static PFNGLCREATESHADERPROC pfn_glCreateShader = nullptr;
static PFNGLSHADERSOURCEPROC pfn_glShaderSource = nullptr;
static PFNGLCOMPILESHADERPROC pfn_glCompileShader = nullptr;
static PFNGLGETSHADERIVPROC pfn_glGetShaderiv = nullptr;
static PFNGLGETSHADERINFOLOGPROC pfn_glGetShaderInfoLog = nullptr;
static PFNGLCREATEPROGRAMPROC pfn_glCreateProgram = nullptr;
static PFNGLATTACHSHADERPROC pfn_glAttachShader = nullptr;
static PFNGLLINKPROGRAMPROC pfn_glLinkProgram = nullptr;
static PFNGLGETPROGRAMIVPROC pfn_glGetProgramiv = nullptr;
static PFNGLUSEPROGRAMPROC pfn_glUseProgram = nullptr;
static PFNGLDELETESHADERPROC pfn_glDeleteShader = nullptr;
static PFNGLDELETEPROGRAMPROC pfn_glDeleteProgram = nullptr;
static PFNGLGETUNIFORMLOCATIONPROC pfn_glGetUniformLocation = nullptr;
static PFNGLUNIFORMMATRIX4FVPROC pfn_glUniformMatrix4fv = nullptr;
static PFNGLUNIFORM1IPROC pfn_glUniform1i = nullptr;
static PFNGLGENVERTEXARRAYSPROC pfn_glGenVertexArrays = nullptr;
static PFNGLDELETEVERTEXARRAYSPROC pfn_glDeleteVertexArrays = nullptr;
static PFNGLBINDVERTEXARRAYPROC pfn_glBindVertexArray = nullptr;
static PFNGLGENBUFFERSPROC pfn_glGenBuffers = nullptr;
static PFNGLDELETEBUFFERSPROC pfn_glDeleteBuffers = nullptr;
static PFNGLBINDBUFFERPROC pfn_glBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC pfn_glBufferData = nullptr;
static PFNGLVERTEXATTRIBPOINTERPROC pfn_glVertexAttribPointer = nullptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC pfn_glEnableVertexAttribArray = nullptr;

static bool g_rendererExtLoaded = false;

static void LogRenderer(const char* msg) {
    OutputDebugStringA("[GaussianRender] ");
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
}

static bool LoadRendererExtensions() {
    if (g_rendererExtLoaded) return true;
    
    HMODULE hGL = GetModuleHandleA("opengl32.dll");
    if (!hGL) return false;
    
    typedef PROC (WINAPI *PFNWGLGETPROCADDRESSPROC)(LPCSTR);
    auto wglGetProcAddress = (PFNWGLGETPROCADDRESSPROC)GetProcAddress(hGL, "wglGetProcAddress");
    if (!wglGetProcAddress) return false;
    
    #define LOAD_GL(name) pfn_##name = (decltype(pfn_##name))wglGetProcAddress(#name);
    
    LOAD_GL(glCreateShader);
    LOAD_GL(glShaderSource);
    LOAD_GL(glCompileShader);
    LOAD_GL(glGetShaderiv);
    LOAD_GL(glGetShaderInfoLog);
    LOAD_GL(glCreateProgram);
    LOAD_GL(glAttachShader);
    LOAD_GL(glLinkProgram);
    LOAD_GL(glGetProgramiv);
    LOAD_GL(glUseProgram);
    LOAD_GL(glDeleteShader);
    LOAD_GL(glDeleteProgram);
    LOAD_GL(glGetUniformLocation);
    LOAD_GL(glUniformMatrix4fv);
    LOAD_GL(glUniform1i);
    LOAD_GL(glGenVertexArrays);
    LOAD_GL(glDeleteVertexArrays);
    LOAD_GL(glBindVertexArray);
    LOAD_GL(glGenBuffers);
    LOAD_GL(glDeleteBuffers);
    LOAD_GL(glBindBuffer);
    LOAD_GL(glBufferData);
    LOAD_GL(glVertexAttribPointer);
    LOAD_GL(glEnableVertexAttribArray);
    
    #undef LOAD_GL
    
    g_rendererExtLoaded = (pfn_glCreateShader && pfn_glCreateProgram);
    return g_rendererExtLoaded;
}

// ============================================
// GLSL Shaders - Simple point rendering
// ============================================
static const char* VERTEX_SHADER = R"(
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;

uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;
uniform bool uUseMatrices;

out vec4 vColor;

void main() {
    if (uUseMatrices) {
        // Apply view/projection for 3D rendering
        vec4 viewPos = uViewMatrix * vec4(aPosition, 1.0);
        vec4 clipPos = uProjMatrix * viewPos;
        gl_Position = clipPos;
        // Scale point size based on distance
        float dist = length(viewPos.xyz);
        gl_PointSize = max(3.0, 30.0 / dist);
    } else {
        // Fallback: simple 2D projection
        gl_Position = vec4(aPosition.xy, 0.0, 1.0);
        gl_PointSize = 10.0;
    }
    vColor = aColor;
}
)";

static const char* FRAGMENT_SHADER = R"(
#version 330 core
in vec4 vColor;
out vec4 fragColor;

void main() {
    // Soft circular point
    vec2 coord = gl_PointCoord - 0.5;
    float dist = length(coord);
    float alpha = smoothstep(0.5, 0.3, dist);
    fragColor = vec4(vColor.rgb, vColor.a * alpha);
}
)";

// ============================================
// Global Instance
// ============================================
static GaussianRenderer g_renderer;

GaussianRenderer& GetGaussianRenderer() {
    return g_renderer;
}

// ============================================
// Implementation
// ============================================
GaussianRenderer::GaussianRenderer() = default;

GaussianRenderer::~GaussianRenderer() {
    Shutdown();
}

bool GaussianRenderer::Initialize() {
    if (m_initialized) return true;
    
    LogRenderer("Initializing...");
    
    if (!LoadRendererExtensions()) {
        LogRenderer("Failed to load GL extensions");
        return false;
    }
    
    if (!CreateShader()) {
        LogRenderer("Failed to create shader");
        return false;
    }
    
    if (!CreateBuffers()) {
        LogRenderer("Failed to create buffers");
        return false;
    }
    
    m_initialized = true;
    LogRenderer("Initialized successfully");
    return true;
}

void GaussianRenderer::Shutdown() {
    if (!m_initialized) return;
    
    if (pfn_glDeleteProgram && m_shaderProgram) {
        pfn_glDeleteProgram(m_shaderProgram);
    }
    if (pfn_glDeleteVertexArrays && m_vao) {
        pfn_glDeleteVertexArrays(1, &m_vao);
    }
    if (pfn_glDeleteBuffers) {
        if (m_positionBuffer) pfn_glDeleteBuffers(1, &m_positionBuffer);
        if (m_colorBuffer) pfn_glDeleteBuffers(1, &m_colorBuffer);
    }
    
    m_initialized = false;
}

bool GaussianRenderer::CreateShader() {
    // Create vertex shader
    m_vertexShader = pfn_glCreateShader(GL_VERTEX_SHADER);
    pfn_glShaderSource(m_vertexShader, 1, &VERTEX_SHADER, nullptr);
    pfn_glCompileShader(m_vertexShader);
    
    GLint success;
    pfn_glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        pfn_glGetShaderInfoLog(m_vertexShader, 512, nullptr, log);
        LogRenderer("Vertex shader error:");
        LogRenderer(log);
        return false;
    }
    
    // Create fragment shader
    m_fragmentShader = pfn_glCreateShader(GL_FRAGMENT_SHADER);
    pfn_glShaderSource(m_fragmentShader, 1, &FRAGMENT_SHADER, nullptr);
    pfn_glCompileShader(m_fragmentShader);
    
    pfn_glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        pfn_glGetShaderInfoLog(m_fragmentShader, 512, nullptr, log);
        LogRenderer("Fragment shader error:");
        LogRenderer(log);
        return false;
    }
    
    // Create program
    m_shaderProgram = pfn_glCreateProgram();
    pfn_glAttachShader(m_shaderProgram, m_vertexShader);
    pfn_glAttachShader(m_shaderProgram, m_fragmentShader);
    pfn_glLinkProgram(m_shaderProgram);
    
    pfn_glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        LogRenderer("Shader link error");
        return false;
    }
    
    // Get uniform locations
    m_viewMatrixLoc = pfn_glGetUniformLocation(m_shaderProgram, "uViewMatrix");
    m_projMatrixLoc = pfn_glGetUniformLocation(m_shaderProgram, "uProjMatrix");
    m_useMatricesLoc = pfn_glGetUniformLocation(m_shaderProgram, "uUseMatrices");
    
    // Delete shaders (now linked)
    pfn_glDeleteShader(m_vertexShader);
    pfn_glDeleteShader(m_fragmentShader);
    
    LogRenderer("Shader created");
    return true;
}

bool GaussianRenderer::CreateBuffers() {
    pfn_glGenVertexArrays(1, &m_vao);
    pfn_glGenBuffers(1, &m_positionBuffer);
    pfn_glGenBuffers(1, &m_colorBuffer);
    
    LogRenderer("Buffers created");
    return true;
}

void GaussianRenderer::Render(
    const float* positions,
    const float* colors,
    uint32_t count,
    const float* viewMatrix,
    const float* projMatrix)
{
    if (!m_initialized || count == 0) return;
    
    // Bind VAO
    pfn_glBindVertexArray(m_vao);
    
    // Upload positions
    pfn_glBindBuffer(GL_ARRAY_BUFFER, m_positionBuffer);
    pfn_glBufferData(GL_ARRAY_BUFFER, count * 3 * sizeof(float), positions, GL_DYNAMIC_DRAW);
    pfn_glVertexAttribPointer(0, 3, GL_FLOAT, 0, 0, nullptr);
    pfn_glEnableVertexAttribArray(0);
    
    // Upload colors
    pfn_glBindBuffer(GL_ARRAY_BUFFER, m_colorBuffer);
    pfn_glBufferData(GL_ARRAY_BUFFER, count * 4 * sizeof(float), colors, GL_DYNAMIC_DRAW);
    pfn_glVertexAttribPointer(1, 4, GL_FLOAT, 0, 0, nullptr);
    pfn_glEnableVertexAttribArray(1);
    
    // Use shader
    pfn_glUseProgram(m_shaderProgram);
    
    // Upload matrices if available
    bool useMatrices = (viewMatrix != nullptr && projMatrix != nullptr);
    if (pfn_glUniform1i && m_useMatricesLoc >= 0) {
        pfn_glUniform1i(m_useMatricesLoc, useMatrices ? 1 : 0);
    }
    if (useMatrices && pfn_glUniformMatrix4fv) {
        if (m_viewMatrixLoc >= 0) {
            pfn_glUniformMatrix4fv(m_viewMatrixLoc, 1, 0, viewMatrix);
        }
        if (m_projMatrixLoc >= 0) {
            pfn_glUniformMatrix4fv(m_projMatrixLoc, 1, 0, projMatrix);
        }
    }
    
    // Enable blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Enable point size
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    // Draw points
    glDrawArrays(GL_POINTS, 0, count);
    
    // Cleanup
    pfn_glBindVertexArray(0);
    pfn_glUseProgram(0);
}

void GaussianRenderer::RenderFromPrimitives(
    const GaussianPrimitive* gaussians,
    uint32_t count,
    const SharedMemoryHeader* header)
{
    if (!gaussians || count == 0) return;
    
    // Check if we have valid view/proj matrices
    bool hasMatrices = false;
    const float* viewMatrix = nullptr;
    const float* projMatrix = nullptr;
    
    if (header) {
        // Check if matrices are non-zero (at least one element should be non-zero for valid matrix)
        bool viewNonZero = false;
        bool projNonZero = false;
        for (int i = 0; i < 16; i++) {
            if (header->view_matrix[i] != 0.0f) viewNonZero = true;
            if (header->proj_matrix[i] != 0.0f) projNonZero = true;
        }
        hasMatrices = viewNonZero && projNonZero;
        if (hasMatrices) {
            viewMatrix = header->view_matrix;
            projMatrix = header->proj_matrix;
        }
    }
    
    std::vector<float> positions(count * 3);
    std::vector<float> colors(count * 4);
    
    if (hasMatrices) {
        // Use raw world positions - shader will apply view/proj matrices
        for (uint32_t i = 0; i < count; i++) {
            positions[i * 3 + 0] = gaussians[i].position[0];
            positions[i * 3 + 1] = gaussians[i].position[1];
            positions[i * 3 + 2] = gaussians[i].position[2];
            
            colors[i * 4 + 0] = gaussians[i].color[0];
            colors[i * 4 + 1] = gaussians[i].color[1];
            colors[i * 4 + 2] = gaussians[i].color[2];
            colors[i * 4 + 3] = gaussians[i].color[3];
        }
        
        static int debugCounter = 0;
        if (debugCounter++ % 60 == 0) {
            LogRenderer("Using 3D projection with view/proj matrices");
        }
    } else {
        // Fallback: bounding box normalization for 2D display
        float minX = gaussians[0].position[0], maxX = minX;
        float minY = gaussians[0].position[1], maxY = minY;
        
        for (uint32_t i = 1; i < count; i++) {
            float x = gaussians[i].position[0];
            float y = gaussians[i].position[1];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        
        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;
        float range = (std::max)(maxX - minX, maxY - minY);
        if (range < 0.001f) range = 1.0f;
        float scale = 1.6f / range;
        
        for (uint32_t i = 0; i < count; i++) {
            positions[i * 3 + 0] = (gaussians[i].position[0] - centerX) * scale;
            positions[i * 3 + 1] = (gaussians[i].position[1] - centerY) * scale;
            positions[i * 3 + 2] = 0.0f;
            
            colors[i * 4 + 0] = gaussians[i].color[0];
            colors[i * 4 + 1] = gaussians[i].color[1];
            colors[i * 4 + 2] = gaussians[i].color[2];
            colors[i * 4 + 3] = gaussians[i].color[3];
        }
        
        static int debugCounter = 0;
        if (debugCounter++ % 60 == 0) {
            LogRenderer("Using 2D bounding box normalization (no matrices)");
        }
    }
    
    Render(positions.data(), colors.data(), count, viewMatrix, projMatrix);
}

}  // namespace gaussian
