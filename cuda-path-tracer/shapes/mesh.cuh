#pragma once

#include "math/math.cuh"
#include "pathTracing/ray.cuh"
#include "pathTracing/hit.cuh"
#include "pathTracing/intersectable.cuh"
#include "pathTracing/material.cuh"
#include "copyToDevice/copyToDevice.h"
#include <vector>
#include <queue>
#include <algorithm>

struct Vertex {
	vec3 position, normal;
	inline __host__ __device__ Vertex(vec3 position, vec3 normal) {
		this->position = position;
		this->normal = normal;
	}
	inline __host__ __device__  Vertex() {
		position = vec3(0, 0, 0);
		normal = vec3(0, 0, 0);
	}
};

struct Triangle {
	Vertex a, b, c;

	inline Triangle(Vertex a, Vertex b, Vertex c, Material* mat) {
		this->a = a;
		this->b = b;
		this->c = c;
	}

	inline Triangle() {}

	 __host__ __device__ inline Hit intersect(const Ray& ray) {
		Hit hit;

		// compute plane's normal
		vec3 ab = b.position - a.position;
		vec3 ac = c.position - a.position;
		vec3 N = cross(ab, ac); // N 
		float area2 = N.length();

		// Step 1: finding P

		// check if ray and plane are parallel ?
		float NdotRayDirection = dot(N, ray.dir);
		if (fabsf(NdotRayDirection) < epsilon) // almost 0 
			return hit; // they are parallel so they don't intersect ! 

		float d = -dot(N, a.position);
		hit.t = -(dot(N, ray.start) + d) / NdotRayDirection;

		// check if the triangle is in behind the ray
		if (hit.t < 0.0f) return hit; // the triangle is behind 

		// compute the intersection point using equation 1
		vec3 P = ray.start + ray.dir * hit.t;

		// Step 2: inside-outside test
		vec3 C; // vector perpendicular to triangle's plane 

		// edge 0
		vec3 vpA = P - a.position;
		C = cross(ab, vpA);
		if (dot(N, C) < 0.0f) {
			hit.t = -1.0f;
			return hit;
		} // P is on the right side 

		// edge 1
		vec3 bc = c.position - b.position;
		vec3 vpB = P - b.position;
		C = cross(bc, vpB);
		if (dot(N, C) < 0.0f) {
			hit.t = -1.0f;
			return hit;
		} // P is on the right side

		// edge 2
		vec3 ca = a.position - c.position;
		vec3 vpC = P - c.position;
		C = cross(ca, vpC);
		if (dot(N, C) < 0.0f) {
			hit.t = -1.0f;
			return hit;
		}  // P is on the right side; 

		hit.position = ray.start + ray.dir * hit.t;
		
		vec3 v_a, v_c, v_h;

		float bary[3];
		v_a = a.position - b.position;
		v_c = c.position - b.position;
		v_h = hit.position - b.position;

		float daa = dot(v_a, v_a);
		float dac = dot(v_a, v_c);
		float dcc = dot(v_c, v_c);
		float denom = daa * dcc - dac * dac;

		float dha = dot(v_h, v_a);
		float dhc = dot(v_h, v_c);
		bary[0] = (dcc * dha - dac * dhc) / denom;
		bary[1] = (daa * dhc - dac * dha) / denom;
		bary[2] = 1.0f - bary[0] - bary[1];
		hit.normal = (a.normal * bary[0] + b.normal * bary[2] + c.normal * bary[1]).normalize();
		if (dot(hit.normal, ray.dir) > 0.0f) hit.normal = hit.normal * (-1.0f); // flip the normal
		return hit; // this ray hits the triangle
	}
};

struct Box {
	float minX, maxX;
	float minY, maxY;
	float minZ, maxZ;
	inline Box(float minX, float maxX, float minY, float maxY, float minZ, float maxZ) {
		this->minX = minX;
		this->maxX = maxX;
		this->minY = minY;
		this->maxY = maxY;
		this->minZ = minZ;
		this->maxZ = maxZ;
	}
	inline Box() {}

	void half(Box& boxLeft, Box& boxRight, Axis axis, float boxLeftPercent = 0.5f);

	inline __host__ __device__ bool isVertexInBox(vec3 position) {
		return(minX <= position.x &&
			position.x <= maxX + epsilon &&
			minY <= position.y + epsilon &&
			position.y <= maxY + epsilon &&
			minZ <= position.z + epsilon &&
			position.z <= maxZ + epsilon
			);
	}

	inline __host__ __device__ bool isTriangleInBox(Triangle* triangle) {
		return(isVertexInBox(triangle->a.position) ||
			isVertexInBox(triangle->b.position) ||
			isVertexInBox(triangle->c.position)
			);
	}

	inline __host__ __device__ Hit intersect(const Ray& ray) {
		vec3 potentialPoint[6];
		float t[6];
		t[0] = (minX - ray.start.x) / ray.dir.x;
		t[1] = (maxX - ray.start.x) / ray.dir.x;
		potentialPoint[0] = ray.start + ray.dir * t[0];
		potentialPoint[1] = ray.start + ray.dir * t[1];
		t[2] = (minY - ray.start.y) / ray.dir.y;
		t[3] = (maxY - ray.start.y) / ray.dir.y;
		potentialPoint[2] = ray.start + ray.dir * t[2];
		potentialPoint[3] = ray.start + ray.dir * t[3];
		t[4] = (minZ - ray.start.z) / ray.dir.z;
		t[5] = (maxZ - ray.start.z) / ray.dir.z;
		potentialPoint[4] = ray.start + ray.dir * t[4];
		potentialPoint[5] = ray.start + ray.dir * t[5];

		Hit bestHit;
		for (int i = 0; i < 6; i++) {
			if (t[i] > 0.0f && (bestHit.t < 0.0f || t[i] < bestHit.t)
				&& isVertexInBox(potentialPoint[i])) bestHit.t = t[i];
		}
		return bestHit;
	}

	inline __host__ __device__ float widthOnAxis(Axis axis) {
		if (axis == Axis_X) return maxX - minX;
		if (axis == Axis_Y) return maxY - minY;
		if (axis == Axis_Z) return maxZ - minZ;
		printf("Invalid axis passed to widthOnAxis\n");
		return 0.0f;
	}

	inline __host__ __device__ bool inBoundsOnAxis(Axis axis, float t) {
		if (axis == Axis_X) {
			return (t > minX && t < maxX);
		}
		if (axis == Axis_Y) {
			return (t > minY && t < maxY);
		}
		if (axis == Axis_Z) {
			return (t > minZ && t < maxZ);
		}
		printf("Invalid axis passed to inBoundsOnAxis\n");
		return false;
	}

	inline __host__ __device__ float startingPointOnAxis(Axis axis) {
		if (axis == Axis_X) return minX;
		if (axis == Axis_Y) return minY;
		if (axis == Axis_Z) return minZ;
		printf("Invalid axis passed to startingPointOnAxis\n");
		return 0.0f;
	}
};


struct Node {
	//CPU
	Node* left = nullptr;
	Node* right = nullptr;

	//shared
	Box box;
	Axis axis;

	//GPU
	Triangle* device_triangles;
	int* device_triangles_size;

	int left_child_of_parent_idx = -1;
	int right_child_of_parent_idx = -1;
	int left_index = -1;
	int right_index = -1;


	inline __device__ __host__ bool isLeaf() {
		return left == nullptr;
	}

	void makeNodeTree(std::vector<Triangle*> triangles, int depth=0);

	inline int getSize() {
		if (isLeaf()) return 1;
		return 1 + left->getSize() + right->getSize();
	}
};


struct Mesh {
	Node* device_kdTree;
	int* device_kdTreeSize;

	inline Mesh(std::vector<Triangle*> triangles) {
		Node* kdTree = getKdTree(triangles);
		uploadKdTree(kdTree);
		for (auto triangle : triangles) {
			delete triangle;
		}
	}
	Node* getKdTree(std::vector<Triangle*> triangles);
	void processTriangleToBox(Triangle* triangle, Box& box);
    void processVertexToBox(Vertex vertex, Box& box);
	void uploadKdTree(Node* kdTree);
	void putNodeToIdx(Node* currentNode, Node* array, int idx);

	__device__ inline Hit intersect(const Ray& ray) {
		if (device_kdTree[0].box.intersect(ray).t > 0) {

			//put the topnode into the stack
			int stack[maxKdTreeHeight];
			int stack_idx = 0;
			stack[stack_idx++] = 0;

			while (stack_idx > 0) {
				//get the next node
				Node currentNode = device_kdTree[stack[--stack_idx]];
				//tree traversing
				while (!(currentNode.isLeaf())) {
					Hit rightHit =
						device_kdTree[currentNode.right_index].box.intersect(ray);
					Hit leftHit = 
						device_kdTree[currentNode.left_index].box.intersect(ray);
					
					Hit smallerHit = (rightHit.t < leftHit.t) ?
						rightHit : leftHit;
					Hit biggerHit = (rightHit.t < leftHit.t) ?
						leftHit : rightHit;

					int smallerIdx = (rightHit.t < leftHit.t) ?
						currentNode.right_index : currentNode.left_index;
					int biggerIdx = (rightHit.t < leftHit.t) ?
						currentNode.left_index : currentNode.right_index;

					currentNode = (smallerHit.t > 0) ?
						device_kdTree[smallerIdx] : device_kdTree[biggerIdx];

					//if the ray hits both boxes
					if (smallerHit.t > 0 && biggerHit.t > 0) {
						stack[stack_idx++] = biggerIdx;
					}
				}
				//leaf node
				Hit bestHit;
				for (int i = 0; i < *currentNode.device_triangles_size; i++) {
					Hit hit = currentNode.device_triangles[i].intersect(ray); //  hit.t < 0 if no intersection
					if (hit.t > 0 && 
						(bestHit.t < 0 || hit.t < bestHit.t) 
						&& currentNode.box.isVertexInBox(ray.start + ray.dir * hit.t)
					) {
						bestHit = hit;
					}
				}
				if (bestHit.t > 0) return bestHit;
			}
			return Hit();
		}
		return Hit();
	}
};


Mesh* readObjIntoMesh(std::string location, mat4 SRTmtx);


struct MeshObject : public Intersectable {
	Mesh* device_mesh;
	vec3 positon = vec3(0, 0, 0);
	vec3 scale = vec3(1, 1, 1);
	vec3 rotate = vec3(0, 0, 0);

	inline MeshObject() : Intersectable() {
		
	}

	inline MeshObject(std::string location, vec3 position, vec3 rotate, vec3 scale, Material* mat) : Intersectable(mat) {
		Mesh* mesh = readObjIntoMesh(location, SRTmtx(scale, rotate, position));
		this->positon = position;
		this->scale = scale;
		this->rotate = rotate;
		copyToDevice(device_mesh, *mesh, "device_mesh");
	}

	__device__ inline Hit intersect(const Ray& ray) {
		Hit bestHit = device_mesh->intersect(ray);
		bestHit.material = this->material;
		return bestHit;
	}
};