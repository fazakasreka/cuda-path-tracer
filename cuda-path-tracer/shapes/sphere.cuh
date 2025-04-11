#pragma once

#include <cuda_runtime.h>

#include "pathTracing/intersectable.cuh"
#include "pathTracing/ray.cuh"
#include "pathTracing/hit.cuh"

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere() : Intersectable() {}

	Sphere(const vec3& _center, float _radius, Material* mat1) : Intersectable(mat1) {
		center = _center;
		radius = _radius;
	}
	__host__ __device__ inline Hit Sphere::intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0.0f) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0.0f) return hit;
        hit.t = (t2 > 0.0f) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) / radius;
        hit.material = material;
        if (dot(hit.normal, ray.dir) > 0.0f) hit.normal = hit.normal * (-1.0f); // flip the normal, we are inside the sphere
        return hit;
    }
};
