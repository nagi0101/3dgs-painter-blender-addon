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
#define GL_STATIC_DRAW 0x88E4
#define GL_PROGRAM_POINT_SIZE 0x8642
#define GL_TRIANGLES 0x0004

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
    if (msg) {
        printf("[GaussianRender] %s\n", msg);
        OutputDebugStringA("[GaussianRender] ");
        OutputDebugStringA(msg);
        OutputDebugStringA("\n");
    }
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
// GLSL Shaders - Full Gaussian Splatting
// ============================================
static const char* VERTEX_SHADER = R"(
#version 330 core

// Per-vertex: billboard quad corner (0-3)
layout(location = 0) in vec2 aQuadPos;  // [-1,1] x [-1,1]

// Per-instance: gaussian data
layout(location = 1) in vec3 aPosition;    // World position
layout(location = 2) in vec4 aRotation;    // Quaternion (w,x,y,z)
layout(location = 3) in vec3 aScale;       // Scale (x,y,z)
layout(location = 4) in vec4 aColor;       // RGBA

uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;
uniform vec2 uViewport;      // Viewport size in pixels
uniform vec2 uFocal;         // Focal lengths (fx, fy)

out vec4 vColor;
out vec3 vConic;     // Inverse 2D covariance (a, b, c)
out vec2 vCoordXY;   // Offset from center in pixels

// Quaternion to rotation matrix (column-major)
mat3 quatToMat(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    
    return mat3(
        1.0 - 2.0*(yy + zz), 2.0*(xy + wz), 2.0*(xz - wy),
        2.0*(xy - wz), 1.0 - 2.0*(xx + zz), 2.0*(yz + wx),
        2.0*(xz + wy), 2.0*(yz - wx), 1.0 - 2.0*(xx + yy)
    );
}

// Compute 3D covariance from scale and rotation
mat3 computeCov3D(vec3 scale, vec4 rot) {
    mat3 R = quatToMat(rot);
    mat3 S = mat3(scale.x, 0.0, 0.0,
                  0.0, scale.y, 0.0,
                  0.0, 0.0, scale.z);
    mat3 RS = R * S;
    return RS * transpose(RS);
}

// Project 3D covariance to 2D using Jacobian
vec3 computeCov2D(vec3 posView, mat3 cov3D, vec2 focal) {
    float z = posView.z;
    float z2 = z * z;
    
    // Jacobian of perspective projection (column-major)
    mat3 J = mat3(
        focal.x / z, 0.0, 0.0,
        0.0, focal.y / z, 0.0,
        -focal.x * posView.x / z2, -focal.y * posView.y / z2, 0.0
    );
    
    mat3 cov2D = J * cov3D * transpose(J);
    
    // Anti-aliasing filter
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;
    
    return vec3(cov2D[0][0], cov2D[0][1], cov2D[1][1]);
}

void main() {
    // Transform to clip space
    vec4 posClip = uProjMatrix * uViewMatrix * vec4(aPosition, 1.0);
    
    // Early frustum culling
    if (posClip.w <= 0.0 || abs(posClip.x/posClip.w) > 1.3 || abs(posClip.y/posClip.w) > 1.3) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }
    
    // Compute view-space position
    vec4 posViewFull = uViewMatrix * vec4(aPosition, 1.0);
    vec3 posView = posViewFull.xyz;
    
    // Compute 3D covariance in world space
    mat3 cov3D_world = computeCov3D(aScale, aRotation);
    
    // Transform to view space
    mat3 V = mat3(uViewMatrix);
    mat3 cov3D_view = V * cov3D_world * transpose(V);
    
    // Project to 2D
    vec3 cov2D = computeCov2D(posView, cov3D_view, uFocal);
    
    // Compute inverse covariance (conic)
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0) {
        gl_Position = vec4(-100.0, -100.0, -100.0, 1.0);
        return;
    }
    float detInv = 1.0 / det;
    vec3 conic = vec3(cov2D.z * detInv, -cov2D.y * detInv, cov2D.x * detInv);
    
    // Compute quad extent (3-sigma)
    float maxRadius = 3.0 * sqrt(max(cov2D.x, cov2D.z));
    maxRadius = clamp(maxRadius, 1.0, 500.0);
    
    // Billboard quad in NDC
    vec2 quadExtentNDC = vec2(maxRadius) / uViewport * 2.0;
    posClip.xy += aQuadPos * quadExtentNDC * posClip.w;
    
    gl_Position = posClip;
    
    vColor = aColor;
    vConic = conic;
    vCoordXY = aQuadPos * maxRadius;
}
)";

static const char* FRAGMENT_SHADER = R"(
#version 330 core

in vec4 vColor;
in vec3 vConic;
in vec2 vCoordXY;

out vec4 fragColor;

void main() {
    // Evaluate 2D Gaussian: exp(-0.5 * x^T * conic * x)
    float power = -0.5 * (
        vConic.x * vCoordXY.x * vCoordXY.x +
        vConic.z * vCoordXY.y * vCoordXY.y +
        2.0 * vConic.y * vCoordXY.x * vCoordXY.y
    );
    
    // Discard if outside 3-sigma
    if (power > 0.0) {
        discard;
    }
    
    float alpha = vColor.a * exp(power);
    
    // Discard nearly transparent
    if (alpha < 0.004) {
        discard;
    }
    
    alpha = min(alpha, 0.99);
    
    fragColor = vec4(vColor.rgb * alpha, alpha);
}
)";

// ============================================
// GL Extension for instanced rendering
// ============================================
typedef void (APIENTRY *PFNGLVERTEXATTRIBDIVISORPROC)(GLuint index, GLuint divisor);
typedef void (APIENTRY *PFNGLDRAWARRAYSINSTANCEDPROC)(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
typedef void (APIENTRY *PFNGLUNIFORM2FPROC)(GLint location, GLfloat v0, GLfloat v1);

static PFNGLVERTEXATTRIBDIVISORPROC pfn_glVertexAttribDivisor = nullptr;
static PFNGLDRAWARRAYSINSTANCEDPROC pfn_glDrawArraysInstanced = nullptr;
static PFNGLUNIFORM2FPROC pfn_glUniform2f = nullptr;

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
    
    LogRenderer("Initializing (Full Gaussian Splatting)...");
    
    if (!LoadRendererExtensions()) {
        LogRenderer("Failed to load GL extensions");
        return false;
    }
    
    // Load additional extensions for instancing
    HMODULE hGL = GetModuleHandleA("opengl32.dll");
    if (hGL) {
        typedef PROC (WINAPI *PFNWGLGETPROCADDRESSPROC)(LPCSTR);
        auto wglGetProcAddress = (PFNWGLGETPROCADDRESSPROC)GetProcAddress(hGL, "wglGetProcAddress");
        if (wglGetProcAddress) {
            pfn_glVertexAttribDivisor = (PFNGLVERTEXATTRIBDIVISORPROC)wglGetProcAddress("glVertexAttribDivisor");
            pfn_glDrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCEDPROC)wglGetProcAddress("glDrawArraysInstanced");
            pfn_glUniform2f = (PFNGLUNIFORM2FPROC)wglGetProcAddress("glUniform2f");
        }
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
    LogRenderer("Initialized successfully (Full Splatting)");
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
        if (m_quadBuffer) pfn_glDeleteBuffers(1, &m_quadBuffer);
        if (m_instanceBuffer) pfn_glDeleteBuffers(1, &m_instanceBuffer);
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
        char log[1024];
        pfn_glGetShaderInfoLog(m_vertexShader, 1024, nullptr, log);
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
        char log[1024];
        pfn_glGetShaderInfoLog(m_fragmentShader, 1024, nullptr, log);
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
        char log[1024];
        pfn_glGetShaderInfoLog(m_shaderProgram, 1024, nullptr, log);
        LogRenderer("Shader link error:");
        LogRenderer(log);
        return false;
    }
    
    // Get uniform locations
    m_viewMatrixLoc = pfn_glGetUniformLocation(m_shaderProgram, "uViewMatrix");
    m_projMatrixLoc = pfn_glGetUniformLocation(m_shaderProgram, "uProjMatrix");
    m_viewportLoc = pfn_glGetUniformLocation(m_shaderProgram, "uViewport");
    m_focalLoc = pfn_glGetUniformLocation(m_shaderProgram, "uFocal");
    
    // Delete shaders (now linked)
    pfn_glDeleteShader(m_vertexShader);
    pfn_glDeleteShader(m_fragmentShader);
    
    LogRenderer("Shader created (Full Splatting)");
    return true;
}

bool GaussianRenderer::CreateBuffers() {
    pfn_glGenVertexArrays(1, &m_vao);
    pfn_glBindVertexArray(m_vao);
    
    // Billboard quad vertices (6 vertices = 2 triangles)
    // Each vertex is vec2 in [-1, 1] range
    static const float quadVertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f,  1.0f
    };
    
    pfn_glGenBuffers(1, &m_quadBuffer);
    pfn_glBindBuffer(GL_ARRAY_BUFFER, m_quadBuffer);
    pfn_glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    // Location 0: quad position (per-vertex)
    pfn_glVertexAttribPointer(0, 2, GL_FLOAT, 0, 2 * sizeof(float), nullptr);
    pfn_glEnableVertexAttribArray(0);
    
    // Instance buffer (per-gaussian data)
    pfn_glGenBuffers(1, &m_instanceBuffer);
    
    pfn_glBindVertexArray(0);
    
    LogRenderer("Buffers created (instanced quads)");
    return true;
}

void GaussianRenderer::RenderFromPrimitives(
    const GaussianPrimitive* gaussians,
    uint32_t count,
    const SharedMemoryHeader* header)
{
    if (!m_initialized || !gaussians || count == 0) return;
    
    // Debug logging
    static int debugCounter = 0;
    if (debugCounter++ % 60 == 0) {
        char buf[256];
        sprintf(buf, "RenderFromPrimitives (splatting): count=%u", count);
        LogRenderer(buf);
    }
    
    // Check for valid matrices
    bool hasMatrices = false;
    if (header) {
        for (int i = 0; i < 16; i++) {
            if (header->view_matrix[i] != 0.0f || header->proj_matrix[i] != 0.0f) {
                hasMatrices = true;
                break;
            }
        }
    }
    
    if (!hasMatrices) {
        // Fallback: can't render without matrices in splatting mode
        static int warnCounter = 0;
        if (warnCounter++ % 60 == 0) {
            LogRenderer("Warning: No view/proj matrices, skipping splat render");
        }
        return;
    }
    
    // Prepare instance data: [position(3) + rotation(4) + scale(3) + color(4)] = 14 floats per gaussian
    const int floatsPerInstance = 14;
    std::vector<float> instanceData(count * floatsPerInstance);
    
    for (uint32_t i = 0; i < count; i++) {
        int base = i * floatsPerInstance;
        const auto& g = gaussians[i];
        
        // Position (3)
        instanceData[base + 0] = g.position[0];
        instanceData[base + 1] = g.position[1];
        instanceData[base + 2] = g.position[2];
        
        // Rotation quaternion (4) - w, x, y, z
        instanceData[base + 3] = g.rotation[0];  // w
        instanceData[base + 4] = g.rotation[1];  // x
        instanceData[base + 5] = g.rotation[2];  // y
        instanceData[base + 6] = g.rotation[3];  // z
        
        // Scale (3)
        instanceData[base + 7] = g.scale[0];
        instanceData[base + 8] = g.scale[1];
        instanceData[base + 9] = g.scale[2];
        
        // Color (4)
        instanceData[base + 10] = g.color[0];
        instanceData[base + 11] = g.color[1];
        instanceData[base + 12] = g.color[2];
        instanceData[base + 13] = g.color[3];
    }
    
    // Bind VAO
    pfn_glBindVertexArray(m_vao);
    
    // Upload instance data
    pfn_glBindBuffer(GL_ARRAY_BUFFER, m_instanceBuffer);
    pfn_glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(float), instanceData.data(), GL_DYNAMIC_DRAW);
    
    // Setup instance attributes
    const int stride = floatsPerInstance * sizeof(float);
    
    // Location 1: position (3 floats)
    pfn_glVertexAttribPointer(1, 3, GL_FLOAT, 0, stride, (void*)(0 * sizeof(float)));
    pfn_glEnableVertexAttribArray(1);
    if (pfn_glVertexAttribDivisor) pfn_glVertexAttribDivisor(1, 1);
    
    // Location 2: rotation (4 floats)
    pfn_glVertexAttribPointer(2, 4, GL_FLOAT, 0, stride, (void*)(3 * sizeof(float)));
    pfn_glEnableVertexAttribArray(2);
    if (pfn_glVertexAttribDivisor) pfn_glVertexAttribDivisor(2, 1);
    
    // Location 3: scale (3 floats)
    pfn_glVertexAttribPointer(3, 3, GL_FLOAT, 0, stride, (void*)(7 * sizeof(float)));
    pfn_glEnableVertexAttribArray(3);
    if (pfn_glVertexAttribDivisor) pfn_glVertexAttribDivisor(3, 1);
    
    // Location 4: color (4 floats)
    pfn_glVertexAttribPointer(4, 4, GL_FLOAT, 0, stride, (void*)(10 * sizeof(float)));
    pfn_glEnableVertexAttribArray(4);
    if (pfn_glVertexAttribDivisor) pfn_glVertexAttribDivisor(4, 1);
    
    // Use shader
    pfn_glUseProgram(m_shaderProgram);
    
    // Upload uniforms
    if (pfn_glUniformMatrix4fv) {
        if (m_viewMatrixLoc >= 0) {
            pfn_glUniformMatrix4fv(m_viewMatrixLoc, 1, 0, header->view_matrix);
        }
        if (m_projMatrixLoc >= 0) {
            pfn_glUniformMatrix4fv(m_projMatrixLoc, 1, 0, header->proj_matrix);
        }
    }
    
    // Viewport and focal length from projection matrix
    // Standard VR viewport is 1024x1024 (our quad texture size)
    float viewportW = 1024.0f;
    float viewportH = 1024.0f;
    
    // Extract focal lengths from projection matrix
    // P[0][0] = 2*fx/width, P[1][1] = 2*fy/height
    float focalX = header->proj_matrix[0] * viewportW * 0.5f;
    float focalY = header->proj_matrix[5] * viewportH * 0.5f;
    
    if (pfn_glUniform2f) {
        if (m_viewportLoc >= 0) {
            pfn_glUniform2f(m_viewportLoc, viewportW, viewportH);
        }
        if (m_focalLoc >= 0) {
            pfn_glUniform2f(m_focalLoc, focalX, focalY);
        }
    }
    
    // Enable blending for proper alpha compositing
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  // Premultiplied alpha
    
    // Disable depth test for painterly overlapping
    glDisable(GL_DEPTH_TEST);
    
    // Draw instanced quads
    if (pfn_glDrawArraysInstanced) {
        pfn_glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count);
    }
    
    // Cleanup
    pfn_glBindVertexArray(0);
    pfn_glUseProgram(0);
}

}  // namespace gaussian

