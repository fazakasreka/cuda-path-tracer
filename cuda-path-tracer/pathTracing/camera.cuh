#pragma once

#include "math/math.cuh"
#include "pathTracing/ray.cuh"

class Camera {
	vec3 eye, lookat, right, up;
public:
	inline void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = w.length();
		right = cross(vup, w).normalize() * f * tanf(fov / 2.0f);	// orthogonalization
		up = cross(w, right).normalize() * f * tanf(fov / 2.0f);
	}
	__host__ __device__ inline Ray getRay(float X, float Y) {	// integer parts of X, Y define the pixel, fractional parts the point inside pixel
		vec3 dir = lookat + right * (2.0f * X / screenWidth - 1.0f) + up * (2.0f * Y / screenHeight - 1.0f) - eye;
		return Ray(eye, dir);
	}
};