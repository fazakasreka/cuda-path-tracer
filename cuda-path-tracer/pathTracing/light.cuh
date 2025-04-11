#pragma once

#include "math/math.cuh"
#include "pathTracing/random.cuh"
#include "constants/constants.cuh"

struct Light {
	vec3 location;
	vec3 power;

	inline Light() {
	
	}

	inline Light(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	__host__ __device__ inline float distanceOf(vec3 point) {
		return (location - point).length();
	}
	__host__ __device__ inline vec3 directionOf(vec3 point) {
		return (location - point).normalize();
	}
	__host__ __device__ inline vec3 radianceAt(vec3 point) {
		float distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2 / 4.0f / PI;
	}
	__device__  inline void randomSampleRay(vec3& outDir, curandState_t* state) {
		vec3 X = vec3(0.0f, 0.0f, 1.0f);
		vec3 Y = vec3(0.0f, 1.0f, 0.0f);
		vec3 Z = vec3(1.0f, 0.0f, 0.0f);

		float alpha = Rand::random(state) * 2.0f * PI;
		float beta = Rand::random(state) * 2.0f * PI;

		outDir = ((X * cosf(alpha) + Z * sinf(alpha)) * sinf(beta) + Y * cosf(beta)).normalize();
	}
};