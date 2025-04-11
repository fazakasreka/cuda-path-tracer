#pragma once

#include <curand_kernel.h>
#include <cuda_runtime.h>

struct Rand {
	__device__ inline static float random(curandState_t* state) {
		return curand_uniform(state);
	}
};