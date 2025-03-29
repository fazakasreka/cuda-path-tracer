#include "math.cuh"

__host__ __device__ mat4 mat4::inverse() const {
    // Get the rotation/scale part (upper 3x3)
    vec3 a = vec3(rows[0][0], rows[0][1], rows[0][2]);
    vec3 b = vec3(rows[1][0], rows[1][1], rows[1][2]);
    vec3 c = vec3(rows[2][0], rows[2][1], rows[2][2]);
    
    // Get the translation part
    vec3 t = vec3(rows[3][0], rows[3][1], rows[3][2]);
    
    // Calculate inverse of rotation/scale using transpose and determinant
    float det = dot(cross(a, b), c);
    if (abs(det) < epsilon) {
        // Return identity if matrix is singular
        return mat4(1.0f, 0.0f, 0.0f, 0.0f, 
                    0.0f, 1.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f, 0.0f, 
                    0.0f, 0.0f, 0.0f, 1.0f
        );
    }
    
    float invDet = 1.0f / det;
    vec3 r0 = cross(b, c) * invDet;
    vec3 r1 = cross(c, a) * invDet;
    vec3 r2 = cross(a, b) * invDet;
    
    // Calculate -R^T * t for the translation part
    float tx = -dot(r0, t);
    float ty = -dot(r1, t);
    float tz = -dot(r2, t);
    
    return mat4(
        r0.x, r0.y, r0.z, 0.0f,
        r1.x, r1.y, r1.z, 0.0f,
        r2.x, r2.y, r2.z, 0.0f,
        tx,   ty,   tz,   1.0f
    );
}