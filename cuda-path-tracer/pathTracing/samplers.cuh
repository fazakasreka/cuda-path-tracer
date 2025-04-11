#pragma once

#include "math/math.cuh"
#include "constants/constants.cuh"
#include "random.cuh"

// sample direction with cosine distribution, returns the pdf
__device__ inline float SampleDiffuse(const vec3& N, const vec3& inDir, vec3& outDir, curandState_t* state) {
	vec3 T = cross(N, vec3(1.0f, 0.0f, 0.0f));	// Find a Cartesian frame T, B, N where T, B are in the plane
	if (T.length() < epsilon) T = cross(N, vec3(0.0f, 0.0f, 1.0f));
	T = T.normalize();
	vec3 B = cross(N, T);

	float x, y, z;
	do {
		x = 2.0f * Rand::random(state) - 1.0f;    // random point in [-1,-1;1,1]
		y = 2.0f * Rand::random(state) - 1.0f;
	} while (x * x + y * y > 1.0f);  // reject if not in circle
	z = sqrtf(1.0f - x * x - y * y);  // project to hemisphere

	outDir = N * z + T * x + B * y;
	return z / PI;	// pdf
}

// sample direction with cosine distribution, returns the pdf
__device__ inline float SamplePhong(const vec3& N, const vec3& inDir, vec3& outDir, curandState_t* state) {
	vec3 R = inDir - N * dot(N, inDir) * 2.0f;
	vec3 T = cross(R, N);
	if (T.length() < epsilon) T = cross(R, vec3(0.0f, 0.0f, 1.0f));
	T = T.normalize();
	vec3 B = cross(R, T);

	float u = Rand::random(state);
	float v = Rand::random(state);

	float n = 4.0f;
	float alpha = PI * 2.0f * u;
	float beta = powf(acosf(1.0f-v), 1.0f/(n+1.0f));


	outDir = (T * cosf(alpha) + B * sinf(alpha)) * sinf(beta) + R * cosf(beta);
	return (1.0f / (2.0f * PI)) * (n+1.0f) * powf(cosf(beta), n) * sinf(beta);    // pdf
}

// sample direction of a Dirac delta distribution, returns the pdf
__device__ inline float SampleMirror(const vec3& N, const vec3& inDir, vec3& outDir) {
	outDir = inDir - N * dot(N, inDir) * 2.0f;
	return 1.0f;    // pdf
}