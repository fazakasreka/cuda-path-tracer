#pragma once

#include "HDRLoader/HDRLoader.h"
#include "copyToDevice/copyToDevice.h"
#include "math/math.cuh"
#include <string>


class EnvMap {
	HDRLoaderResult px;
	HDRLoaderResult py;
	HDRLoaderResult pz;
	HDRLoaderResult nx;
	HDRLoaderResult ny;
	HDRLoaderResult nz;


	HDRLoaderResult* device_px;
	HDRLoaderResult* device_py;
	HDRLoaderResult* device_pz;
	HDRLoaderResult* device_nx;
	HDRLoaderResult* device_ny;
	HDRLoaderResult* device_nz;

	inline void copyHDRToDevice(HDRLoaderResult &cpu_result, HDRLoaderResult* &device_result) {
		float* device_cols;
		copyToDevice(device_cols, cpu_result.cols, cpu_result.size(), "HDR.cols");
		delete cpu_result.cols;
		cpu_result.cols = device_cols;

		copyToDevice(device_result, cpu_result, "HDR");
	}
public:
	inline EnvMap(std::string folderName) {
		HDRLoader::load((folderName + "px.hdr").c_str(), px);
		HDRLoader::load((folderName + "py.hdr").c_str(), py);
		HDRLoader::load((folderName + "pz.hdr").c_str(), pz);
		HDRLoader::load((folderName + "nx.hdr").c_str(), nx);
		HDRLoader::load((folderName + "ny.hdr").c_str(), ny);
		HDRLoader::load((folderName + "nz.hdr").c_str(), nz);

		//Upload to GPU
		copyHDRToDevice(px, device_px);
		copyHDRToDevice(py, device_py);
		copyHDRToDevice(pz, device_pz);
		copyHDRToDevice(nx, device_nx);
		copyHDRToDevice(ny, device_ny);
		copyHDRToDevice(nz, device_nz);
	}
	__host__ __device__ inline vec3 getPixelColor(vec3 ray) {
		HDRLoaderResult* hit = device_px;
		if ((abs(ray.x) >= abs(ray.y)) && (abs(ray.x) >= abs(ray.z))) {
			if (ray.x > 0.0f) hit = device_px;
			else hit = device_nx;

			int pos = ((int)((1.0f - ((ray.y / abs(ray.x) + 1.0f) / 2.0f)) * hit->height)) * 3 * hit->width
				+ ((int)((1.0f - ((ray.z / ray.x + 1.0f) / 2.0f)) * hit->width)) * 3;
			if (pos > hit->height * hit->width * 3 - 3) pos = hit->height * hit->width * 3 - 3;
			return vec3(
				hit->cols[pos],
				hit->cols[pos + 1],
				hit->cols[pos + 2]
			);
		}
		else if ((abs(ray.y) >= abs(ray.x)) && (abs(ray.y) >= abs(ray.z))) {
			if (ray.y > 0.0f) hit = device_py;
			else hit = device_ny;
			int pos = ((int)((ray.z / ray.y + 1.0f) / 2.0f * hit->height)) * 3 * hit->width
				+ ((int)((ray.x / abs(ray.y) + 1.0f) / 2.0f * hit->width)) * 3;
			if (pos > hit->height * hit->width * 3 - 3) pos = hit->height * hit->width * 3 - 3;
			return vec3(
				hit->cols[pos],
				hit->cols[pos + 1],
				hit->cols[pos + 2]
			);
		}
		//if ((abs(ray.z) >= abs(ray.x)) && (abs(ray.z) >= abs(ray.y)))
		if (ray.z > 0.0f) hit = device_pz;
		else hit = device_nz;
		int pos = ((int)((1.0f - ((ray.y / abs(ray.z) + 1.0f) / 2.0f)) * hit->height)) * 3 * hit->width
			+ ((int)((ray.x / ray.z + 1.0f) / 2.0f * hit->width)) * 3;
		if (pos > hit->height * hit->width * 3 - 3) pos = hit->height * hit->width * 3 - 3;
		return vec3(
			hit->cols[pos],
			hit->cols[pos + 1],
			hit->cols[pos + 2]
		);
		
	}
};
