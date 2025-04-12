//=============================================================================================
// Path tracing program
//=============================================================================================
#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <queue> 

#include "copyToDevice/copyToDevice.h"
#include "HDREnvMap/envMap.cuh"
#include "math/math.cuh"
#include "constants/constants.cuh"

#include "pathTracing/camera.cuh"
#include "pathTracing/hit.cuh"
#include "pathTracing/intersectable.cuh"
#include "pathTracing/light.cuh"
#include "pathTracing/material.cuh"
#include "pathTracing/random.cuh"
#include "pathTracing/ray.cuh"
#include "pathTracing/samplers.cuh"

#include "shapes/mesh.cuh"
#include "shapes/sphere.cuh"
#include "shapes/plane.cuh"

class Scene {
	Sphere* device_spheres;
	int* device_sphere_size;

	MeshObject* device_meshes;
	int* device_meshes_size;

	Light* device_lights;
	int* device_lights_size;

	Plane* device_planes;
	int* device_plane_size;

public:
	Camera camera;
	EnvMap envMap = EnvMap("resources/hdr-env-cube/");
	void build() {
		vec3 eye = vec3(0.0f, 0.0f, 3.0f);
		vec3 vup = vec3(0.0f, 1.0f, 0.0f);
		vec3 lookat = vec3(0.0f, 0.0f, 0.0f);
		float fov = 100.0f * PI / 180.0f;
		camera.set(eye, lookat, vup, fov);

		//LIGHTS
		int light_size = 1;
		copyToDevice(device_lights_size, light_size, "device_lights_size");

		Light* lights = new Light[light_size]; 
		lights[0] = Light(vec3(0.0f, -4.0f, -4.5f), vec3(1000.0f, 1000.0f, 1000.0f));
		//lights[1] = Light(vec3(0, 10, -3), vec3(2000, 2000, 2000));
		//lights[1] = Light(vec3(0, 6, 4), vec3(2000, 2000, 2000));
		//lights[2] = Light(vec3(0, 2, -2), vec3(2000, 2000, 2000));
		//lights[3] = Light(vec3(0, 6, 2), vec3(2000, 2000, 2000));

		copyToDevice(device_lights, lights, light_size, "device_lights");

		delete lights;

		//SPHERE
		int spheres_size = 0;
		copyToDevice(device_sphere_size, spheres_size, "device_sphere_size");

		Sphere* spheres = new Sphere[spheres_size];
		//spheres[0] = Sphere(vec3(0, 0, 0), 10, new Material(vec3(0.2,0.9,0.2), vec3(0,0,0)));
		//spheres[1] = Sphere(vec3(-1.5, 0, 0), 0.6, new Material(vec3(0, 0, 0), vec3(1, 1, 1)));

		copyToDevice(device_spheres, spheres, spheres_size, "device_spheres");

		delete spheres;

		//PLANES
		int planes_size = 6;
		copyToDevice(device_plane_size, planes_size, "device_plane_size");

		Plane* planes = new Plane[planes_size];
		planes[0] = Plane(vec3(0.0f, -5.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), new Material(vec3(0.9f, 0.9f, 0.9f), vec3(0.0f, 0.0f, 0.0f)));
		planes[1] = Plane(vec3(0.0f, 0.0f, 5.0f), vec3(0.0f, 0.0f, 1.0f), new Material(vec3(0.9f, 0.9f, 0.9f), vec3(0.0f, 0.0f, 0.0f)));
		planes[2] = Plane(vec3(0.0f, 0.0f, -5.0f), vec3(0.0f, 0.0f, 1.0f), new Material(vec3(0.9f, 0.9f, 0.9f), vec3(0.0f, 0.0f, 0.0f)));
		planes[3] = Plane(vec3(5.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), new Material(vec3(0.9f, 0.9f, 0.9f), vec3(0.0f, 0.0f, 0.0f)));
		planes[4] = Plane(vec3(-5.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), new Material(vec3(0.9f, 0.9f, 0.9f), vec3(0.0f, 0.0f, 0.0f)));
		planes[5] = Plane(vec3(0.0f, 5.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), new Material(vec3(0.9f, 0.9f, 0.9f), vec3(0.0f, 0.0f, 0.0f)));

		copyToDevice(device_planes, planes, planes_size, "device_planes");

		delete planes;

		//MESHES
		int mesh_size = 3;
		copyToDevice(device_meshes_size, mesh_size, "device_meshes_size");

		MeshObject* meshes = new MeshObject[mesh_size];

		//meshes[0] = MeshObject(
		//	"resources/objects/bunny.obj",
		//	vec3(0.3, 0, 0), //position
		//	vec3(-3.1415/ 2.0, 0.8, 0.0),  //rotate
		//	vec3(0.5, 0.5, 0.5),  //scale
		//	new Material(vec3(0.0, 0.0, 0.0), vec3(0.9, 0.9, 0.9))
		//);

		meshes[0] = MeshObject(
			"resources/objects/cube.obj",
			vec3(2.0f, -4.0f, -4.0f), //position
			vec3(0.0f, 0.0f, 0.0f),  //rotate
			vec3(1.0f, 1.0f, 1.0f),  //scale
			new Material(vec3(0.8392f, 0.0f, 0.4392f), vec3(0.0f, 0.0f, 0.0f))
		);

		meshes[1] = MeshObject(
			"resources/objects/cube.obj",
			vec3(0.0f, -4.0f, -2.5f), //position
			vec3(0.0f, 0.0f, 0.0f),  //rotate
			vec3(1.0f, 1.0f, 1.0f),  //scale
			new Material(vec3(0.6078f, 0.3098f, 0.5882f), vec3(0.0f, 0.0f, 0.0f))
		);
		meshes[2] = MeshObject(
			"resources/objects/cube.obj",
			vec3(-2.0f, -4.0f, -4.0f), //position
			vec3(0.0f, PI, 0.0f),  //rotate
			vec3(1.0f, 1.0f, 1.0f),  //scale
			new Material(vec3(0.0f, 0.2196f, 0.6588f), vec3(0.0f, 0.0f, 0.0f))
		);

		copyToDevice(device_meshes, meshes, mesh_size, "device_meshes");

		delete meshes;
	}

	// Find the first intersection of the ray with objects
	__device__ Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (int i = 0; i < *device_meshes_size; i++) {
			Hit hit = device_meshes[i].intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		for (int i = 0; i < *device_sphere_size; i++) {
			Hit hit = device_spheres[i].intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		for (int i = 0; i < *device_plane_size; i++) {
			Hit hit = device_planes[i].intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		return bestHit;
	}

	__device__ float clamp(float x, float min, float max) {
		if (x < min) {
			return min;
		}
		if (x > max) {
			return max;
		}
		return x;
	}

	// Bidirectional Path Tracer
	__device__ vec3 trace(Ray _ray, EnvMap envMap, curandState_t* state) {
		vec3 outRad(0.0f, 0.0f, 0.0f);
		//LIGHT PATH

		//choose a random light source
		int light_idx = (int)(Rand::random(state) * ((float)*device_lights_size));
		vec3 light_out_dir;
		device_lights[light_idx].randomSampleRay(light_out_dir, state);

		//get the first object it hits
		Ray light_ray(device_lights[light_idx].location, light_out_dir);
		Hit hit = firstIntersect(light_ray);
		if (hit.t < 0) return outRad;

		//store the hits and throughputs
		Hit light_hits[maxdepth];
		vec3 light_throughput[maxdepth];
		float light_pdf[maxdepth];

		//initialize the first item
		light_hits[0] = hit;
		light_throughput[0] = device_lights[light_idx].radianceAt(hit.position);
		light_pdf[0] = (1.0f / (float)*device_lights_size) * (1.0f / (4.0f * PI));

		for (int i = 0; i < maxdepth - 1; i++) {
			float diffProb = light_hits[i].material->diffuseAlbedo.average();
			float mirrorProb = light_hits[i].material->mirrorAlbedo.average();
			float rnd = Rand::random(state);

			if (rnd >= diffProb + mirrorProb) break;

			vec3 inDir = (i == 0) ? light_out_dir :
				(light_hits[i].position - light_hits[i - 1].position).normalize();
			vec3 outDir;
			float pdf_brdf;

			if (rnd < diffProb) {
				pdf_brdf = SampleDiffuse(light_hits[i].normal, inDir, outDir, state) * diffProb;
				float cosTheta = dot(light_hits[i].normal, -inDir);
				if (cosTheta < epsilon) break;
				light_throughput[i] = (i == 0) 
					? light_throughput[i] * light_hits[i].material->diffuseAlbedo / PI * cosTheta
					: light_throughput[i - 1] * light_hits[i].material->diffuseAlbedo / PI * cosTheta;
			} else {
				pdf_brdf = SampleMirror(light_hits[i].normal, inDir, outDir) * mirrorProb;
				float cosTheta = dot(light_hits[i].normal, -inDir);
				if (cosTheta < epsilon) break;
				light_throughput[i] = (i == 0) 
					? light_throughput[i] * light_hits[i].material->mirrorAlbedo
					: light_throughput[i - 1] * light_hits[i].material->mirrorAlbedo;
			}

			Ray bounce_ray(light_hits[i].position + light_hits[i].normal * epsilon, outDir);
			Hit next_hit = firstIntersect(bounce_ray);
			if (next_hit.t < 0) break;

			light_hits[i + 1] = next_hit;
			light_pdf[i + 1] = light_pdf[i] * pdf_brdf;
		}

		// CAMERA PATH

		vec3 cam_throughput = vec3(1.0f, 1.0f, 1.0f);
		float cam_pdf = 1.0f;
		Ray cam_ray = _ray;

		for (int i = 0; i < maxdepth; i++) {
			Hit hit = firstIntersect(cam_ray);
			if (hit.t < 0) break;

			//connect current path with the light source
			vec3 light_pos = device_lights[0].location;
			vec3 light_dir = (hit.position - light_pos).normalize();
			float dist = (light_pos - hit.position).length();

			Hit shadow = firstIntersect(Ray(light_pos, light_dir));
			if (shadow.t > 0 && (shadow.position - hit.position).length() < epsilon) {
				vec3 light_radiance = device_lights[0].radianceAt(hit.position);
				float cosTheta = dot(hit.normal, -light_dir);
				if (cosTheta > epsilon) {
					vec3 brdf = hit.material->diffuseAlbedo / PI;
					vec3 contrib = (cam_throughput / cam_pdf) * (light_radiance * brdf * cosTheta);
					outRad += contrib;
				}
			}

			//connect current path with every element of the light path
			for (int j = 0; j < maxdepth; j++) {
				if (light_hits[j].t < 0) break;

				vec3 dir = (light_hits[j].position - hit.position).normalize();
				float dist = (light_hits[j].position - hit.position).length();

				//check if they can be connected
				Hit shadow = firstIntersect(Ray(hit.position + hit.normal * epsilon, dir));
				if (shadow.t < 0 || (shadow.position - light_hits[j].position).length() > epsilon) continue;

				float cosCam = dot(hit.normal, dir);
				float cosLight = dot(light_hits[j].normal, -dir);
				if (cosCam < epsilon || cosLight < epsilon) continue;

				float G = cosCam * cosLight / (dist * dist);
				vec3 brdf = hit.material->diffuseAlbedo / PI;
				vec3 camera_weight = cam_throughput * brdf * cosCam / cam_pdf;
				vec3 light_weight = light_throughput[j] / light_pdf[j];

				outRad += light_weight * G * camera_weight;
			}

			//get next element in camera path
			float diffProb = hit.material->diffuseAlbedo.average();
			float mirrorProb = hit.material->mirrorAlbedo.average();
			float rnd = Rand::random(state);
			vec3 outDir;

			if (rnd < diffProb) {
				float pdf = SampleDiffuse(hit.normal, cam_ray.dir, outDir, state);
				float cosTheta = dot(hit.normal, outDir);
				if (cosTheta < epsilon) break;
				cam_throughput = cam_throughput * hit.material->diffuseAlbedo / PI * cosTheta;
				cam_pdf = cam_pdf * pdf * diffProb;
			} else if (rnd < diffProb + mirrorProb) {
				float pdf = SampleMirror(hit.normal, cam_ray.dir, outDir);
				cam_throughput = cam_throughput * hit.material->mirrorAlbedo;
				cam_pdf = cam_pdf * pdf * mirrorProb;
			} else {
				break;
			}

			cam_ray = Ray(hit.position + hit.normal * epsilon, outDir);
		}

		return outRad;
	}
};

// Render the scene: Trace nSamples rays through each pixel and average radiance values
__global__ void render(vec3* image, Scene scene) {
	//indicies
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("\rProgress: %d%%", (int)(((float)id / (screenHeight * screenWidth)) * 100));
	if (id < screenHeight * screenWidth) {
		int tileIdx = id / tileArea;

		int tile_row = tileIdx / (screenWidth / tileSize);
		int tile_column = tileIdx - (tile_row * (screenWidth / tileSize));

		int offest_on_tile = id - (tileIdx * tileArea);
		int row_offset = offest_on_tile / tileSize;
		int column_offset = offest_on_tile - (row_offset * tileSize);

		int row = tile_row * tileSize + row_offset;
		int column = tile_column * tileSize + column_offset;

		id = row * screenWidth + column;

		//init rand
		curandState_t state;
		curand_init(id, /* the seed controls the sequence of random values that are produced */
			0, /* the sequence number is only important with multiple cores */
			1, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&state);

		//fill image
		image[id] = vec3(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < nSamples; i++) {
			image[id] += scene.trace(
				scene.camera.getRay(column + Rand::random(&state), row + Rand::random(&state)),
				scene.envMap,
				&state
			) / nSamples;
		}
	}
}

// Save image into a Targa format file
void SaveTGAFile(char* fileName, const vec3* image) {
	FILE* tgaFile = fopen(fileName, "wb");
	if (!tgaFile) {
		printf("File %s cannot be opened\n", fileName);
		return;
	}
	// File header
	fputc(0, tgaFile); fputc(0, tgaFile); fputc(2, tgaFile);
	for (int i = 3; i < 12; i++) { fputc(0, tgaFile); }
	fputc(screenWidth % 256, tgaFile); fputc(screenWidth / 256, tgaFile);
	fputc(screenHeight % 256, tgaFile); fputc(screenHeight / 256, tgaFile);
	fputc(24, tgaFile); fputc(32, tgaFile);
	// List of pixel colors
	for (int Y = screenHeight - 1; Y >= 0; Y--) {
		for (int X = 0; X < screenWidth; X++) {
			int R = (int)fmaxf(fminf(image[Y * screenWidth + X].x * 255.5f, 255.5f), 0.0f);
			int G = (int)fmaxf(fminf(image[Y * screenWidth + X].y * 255.5f, 255.5f), 0.0f);
			int B = (int)fmaxf(fminf(image[Y * screenWidth + X].z * 255.5f, 255.5f), 0.0f);
			fputc(B, tgaFile); fputc(G, tgaFile); fputc(R, tgaFile);
		}
	}
	fclose(tgaFile);
}


int main(int argc, char* argv[]) {
	//cuda setDevice
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//Scene
	Scene scene;										
	scene.build();

	//Malloc image on GPU
	vec3* device_image;
	cudaStatus = cudaMalloc((void**)&device_image, screenHeight * screenWidth * sizeof(vec3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	//Render scene to image on GPU
	std::cout <<"Render strated" << std::endl;
	auto begin = std::chrono::high_resolution_clock::now();

	render<<<screenWidth * screenHeight / tileArea + 1, tileArea>>>(device_image, scene);

	cudaStatus = cudaGetLastError();
	//check for errors
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "render launch failed! %s", cudaGetErrorString(cudaStatus));
	}
	//sync
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching render!\n", cudaStatus);
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	std::cout << "\nRender ended in " << elapsed.count() * 1e-9 << std::endl;

	//copy image to CPU
	vec3* image = new vec3[screenHeight * screenWidth];
	cudaStatus = cudaMemcpy(image, device_image, screenHeight * screenWidth * sizeof(vec3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "device_image cudaMemcpy failed!");
	}

	//save image
	char location[30] = "resources/out/image.tga";		
	SaveTGAFile(location, image);		
	
	cudaDeviceReset();

	//delete image
	
	delete image;
	return 1;
}