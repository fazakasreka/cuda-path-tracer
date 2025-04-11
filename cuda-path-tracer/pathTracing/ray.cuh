#pragma once

#include <cuda_runtime.h>

#include "math/math.cuh"


struct Ray {
	vec3 start, dir;
	__host__ __device__ inline Ray(vec3 _start, vec3 _dir) { start = _start; dir = _dir.normalize(); }
};

