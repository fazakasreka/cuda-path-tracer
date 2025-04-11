#pragma once

#include "math/math.cuh"

struct Material {
	vec3 diffuseAlbedo;	// probability of diffuse reflection
	vec3 mirrorAlbedo;	// probability of mirror like reflection

	inline Material(vec3 _diffuseAlbedo, vec3 _mirrorAlbedo) {
		diffuseAlbedo = _diffuseAlbedo;
		mirrorAlbedo = _mirrorAlbedo;
	}
};