#pragma once

#include <cuda_runtime.h>

#include "pathTracing/intersectable.cuh"
#include "pathTracing/ray.cuh"
#include "pathTracing/hit.cuh"

struct Plane : public Intersectable {
	vec3 point, normal;

	Plane() : Intersectable() {}

	Plane(const vec3& _point, const vec3& _normal, Material* mat) : Intersectable(mat) {
		point = _point;
		normal = _normal.normalize();
	}
    __host__ __device__ inline Hit intersect(const Ray& ray) {
        Hit hit;
        float NdotV = dot(normal, ray.dir);
        if (fabsf(NdotV) < epsilon) return hit;
        float t = dot(normal, point - ray.start) / NdotV;
        if (t < epsilon) return hit;
        hit.t = t;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        if (dot(hit.normal, ray.dir) > 0.0f) hit.normal = hit.normal * (-1.0f); // flip the normal, we are inside the sphere
        hit.material = material;
        return hit;
    }
};