#pragma once

#include "material.cuh"
#include "copyToDevice/copyToDevice.h"

class Intersectable {
protected:
	Material* material;
public:
	inline Intersectable() {}
	inline Intersectable(Material* mat) { 
		copyToDevice(material, *mat, "material");
	}
};