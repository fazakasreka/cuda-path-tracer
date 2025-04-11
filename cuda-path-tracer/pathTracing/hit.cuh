#pragma once

#include <cuda_runtime.h>

#include "math/math.cuh"
#include "material.cuh"


struct Hit {
	float t;		// ray parameter
	vec3 position;	// position of the intersection
	vec3 normal;	// normal of the intersected surface
	Material* material;	// material of the intersected surface
	__host__ __device__ inline Hit() { t = -1; }
};