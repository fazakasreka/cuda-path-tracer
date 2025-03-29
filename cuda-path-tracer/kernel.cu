//=============================================================================================
// Path tracing program
//=============================================================================================
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <list>
#include <chrono>
#include <queue> 


#include "HDRLoader/HDRLoader.h"
#include "copyToDevice/copyToDevice.h"
#include "math/math.cuh"
#include "constants/constants.cuh"


// Pseudo-random number in [0,1)
//__host__ __device__ float random() { return (float)rand() / RAND_MAX; }


struct Rand {
	__device__ static float random(curandState_t* state) {
		return curand_uniform(state);
	}
};


// Material class
struct Material {
	vec3 diffuseAlbedo;	// probability of diffuse reflection
	vec3 mirrorAlbedo;	// probability of mirror like reflection

	Material(vec3 _diffuseAlbedo, vec3 _mirrorAlbedo) {
		diffuseAlbedo = _diffuseAlbedo;
		mirrorAlbedo = _mirrorAlbedo;
	}
};

// sample direction with cosine distribution, returns the pdf
__device__ float SampleDiffuse(const vec3& N, const vec3& inDir, vec3& outDir, curandState_t* state) {
	vec3 T = cross(N, vec3(1.0f, 0.0f, 0.0f));	// Find a Cartesian frame T, B, N where T, B are in the plane
	if (T.length() < epsilon) T = cross(N, vec3(0.0f, 0.0f, 1.0f));
	T = T.normalize();
	vec3 B = cross(N, T);


	//float u = Rand::random(state);
	//float v = Rand::random(state);

	//float alpha = M_PI * 2.0f * u;
	//float beta = asinf(sqrtf(v));
	
	//outDir = (T * cosf(alpha) + B * sinf(alpha)) * sinf(beta) + N * cosf(beta);
	//return 1 / (2.0f * M_PI) * sinf(2.0f * beta);	// pdf

	float x, y, z;
	do {
		x = 2.0f * Rand::random(state) - 1.0f;    // random point in [-1,-1;1,1]
		y = 2.0f * Rand::random(state) - 1.0f;
	} while (x * x + y * y > 1.0f);  // reject if not in circle
	z = sqrtf(1.0f - x * x - y * y);  // project to hemisphere

	outDir = N * z + T * x + B * y;
	return z / M_PI;	// pdf
}

// sample direction with cosine distribution, returns the pdf
__device__ float SamplePhong(const vec3& N, const vec3& inDir, vec3& outDir, curandState_t* state) {
	vec3 R = inDir - N * dot(N, inDir) * 2.0f;
	vec3 T = cross(R, N);
	if (T.length() < epsilon) T = cross(R, vec3(0.0f, 0.0f, 1.0f));
	T = T.normalize();
	vec3 B = cross(R, T);

	float u = Rand::random(state);
	float v = Rand::random(state);

	float n = 4.0f;
	float alpha = PI * 2.0f * u;
	float beta = powf(acosf(1.0f-v), 1.0f/(n+1.0f));


	outDir = (T * cosf(alpha) + B * sinf(alpha)) * sinf(beta) + R * cosf(beta);
	return (1.0f / (2.0f * M_PI)) * (n+1.0f) * powf(cosf(beta), n) * sinf(beta);    // pdf
}

// sample direction of a Dirac delta distribution, returns the pdf
__device__ float SampleMirror(const vec3& N, const vec3& inDir, vec3& outDir) {
	outDir = inDir - N * dot(N, inDir) * 2.0f;
	return 1.0f;    // pdf
}

// Hit of ray tracing
struct Hit {
	float t;		// ray parameter
	vec3 position;	// position of the intersection
	vec3 normal;	// normal of the intersected surface
	Material* material;	// material of the intersected surface
	__host__ __device__ Hit() { t = -1; }
};

// The ray to be traced
struct Ray {
	vec3 start, dir;
	__host__ __device__ Ray(vec3 _start, vec3 _dir) { start = _start; dir = _dir.normalize(); }
};


class Intersectable {
protected:
	Material* material;
public:
	Intersectable() {}
	Intersectable(Material* mat) { 
		copyToDevice(material, *mat, "material");
	}
};

// Sphere
struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere() : Intersectable() {}

	Sphere(const vec3& _center, float _radius, Material* mat1) : Intersectable(mat1) {
		center = _center;
		radius = _radius;
	}
	__host__ __device__ Hit intersect(const Ray& ray) {
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

// Plane
struct Plane : public Intersectable {
	vec3 point, normal;

	Plane() : Intersectable() {}

	Plane(const vec3& _point, const vec3& _normal, Material* mat) : Intersectable(mat) {
		point = _point;
		normal = _normal.normalize();
	}
	__host__ __device__ Hit intersect(const Ray& ray) {
		Hit hit;
		float NdotV = dot(normal, ray.dir);
		if (fabsf(NdotV) < epsilon) return hit;
		float t = dot(normal, point - ray.start) / NdotV;
		if (t < epsilon) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		//if ((hit.position - point).length() > 1.5) return Hit();
		if (dot(hit.normal, ray.dir) > 0.0f) hit.normal = hit.normal * (-1.0f); // flip the normal, we are inside the sphere
		hit.material = material;
		return hit;
	}
};


//Ring
struct Ring : public Intersectable {
	vec3 position, normal;
	float radius, height;

	Ring() : Intersectable() {}

	Ring(const vec3& _point, const vec3& _normal, float _height, float _radius, Material* mat) : Intersectable(mat) {
		position = _point;
		normal = _normal.normalize();
		height = _height;
		radius = _radius;
	}
	__host__ __device__ Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 distance = ray.start - position; //start-center

		float coreDotDistance = dot(normal, distance); //core°distance
		float coreDotDir = dot(normal, ray.dir); //core ° dir

		float c = dot(distance - normal * coreDotDistance, distance - normal * coreDotDistance) - radius * radius;
		float b = 2.0f * dot(distance - normal * coreDotDistance, ray.dir - normal * coreDotDir);
		float a = dot(ray.dir - normal * coreDotDir, ray.dir - normal * coreDotDir);

		float discr = (b * b) - (4.0f * a * c);
		if (discr < 0.0f) return hit; //no real solution

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b - sqrt_discr) / 2.0f / a;
		float t2 = (-b + sqrt_discr) / 2.0f / a;

		float t1Height = coreDotDistance + t1 * coreDotDir;
		float t2Height = coreDotDistance + t2 * coreDotDir;

		if (t2 <= 0.0f) { //neither of the solutions are on the "good" side of the ray, no intersection
			return hit;
		}
		if (t1 > 0.0f) {
			if (t1Height > 0.0f && t1Height < height) {
				hit.t = t1;
				hit.position = ray.start + ray.dir * t1;
				hit.normal = (position + normal * t1Height - hit.position).normalize();
				if (dot(hit.normal, ray.dir) > 0.0f) hit.normal = hit.normal * (-1.0f); // flip the normal
				hit.material = material;
				return hit;
			}
		}
		if (t2Height > 0.0f && t2Height < height) {
			hit.t = t2; 
			hit.position = ray.start + ray.dir * t2;
			hit.normal = (position + normal * t2Height - hit.position).normalize();
			if (dot(hit.normal, ray.dir) > 0.0f) hit.normal = hit.normal * (-1.0f); // flip the normal
			hit.material = material;
			return hit;
		}

		return hit;
	}
};


//Mesh
struct Vertex {
	vec3 position, normal;
	__host__ __device__ Vertex(vec3 position, vec3 normal) {
		this->position = position;
		this->normal = normal;
	}
	__host__ __device__  Vertex() {
		position = vec3(0, 0, 0);
		normal = vec3(0, 0, 0);
	}
};

struct Triangle {
	Vertex a, b, c;

	Triangle(Vertex a, Vertex b, Vertex c, Material* mat) {
		this->a = a;
		this->b = b;
		this->c = c;
	}

	Triangle() {}

	__host__ __device__ Hit intersect(const Ray& ray) {
		Hit hit;

		// compute plane's normal
		vec3 ab = b.position - a.position;
		vec3 ac = c.position - a.position;
		// no need to normalize
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
	Box(float minX, float maxX, float minY, float maxY, float minZ, float maxZ) {
		this->minX = minX;
		this->maxX = maxX;
		this->minY = minY;
		this->maxY = maxY;
		this->minZ = minZ;
		this->maxZ = maxZ;
	}
	Box() {}

	void half(Box& boxLeft, Box& boxRight, Axis axis, float boxLeftPercent = 0.5f) {
		boxLeft.minX = minX;
		boxLeft.maxX = maxX;
		boxLeft.minY = minY;
		boxLeft.maxY = maxY;
		boxLeft.minZ = minZ;
		boxLeft.maxZ = maxZ;
		boxRight.minX = minX;
		boxRight.maxX = maxX;
		boxRight.minY = minY;
		boxRight.maxY = maxY;
		boxRight.minZ = minZ;
		boxRight.maxZ = maxZ;
		if (axis == Axis_X) {
			boxLeft.maxX = minX + (maxX - minX) * boxLeftPercent;
			boxRight.minX = boxLeft.maxX;
		}
		if (axis == Axis_Y) {
			boxLeft.maxY = minY + (maxY - minY) * boxLeftPercent;
			boxRight.minY = boxLeft.maxY;
		}
		if (axis == Axis_Z) {
			boxLeft.maxZ = minZ + (maxZ - minZ) * boxLeftPercent;
			boxRight.minZ = boxLeft.maxZ;
		}
	}

	__host__ __device__ bool isVertexInBox(vec3 position) {
		return(minX <= position.x &&
			position.x <= maxX + epsilon &&
			minY <= position.y + epsilon &&
			position.y <= maxY + epsilon &&
			minZ <= position.z + epsilon &&
			position.z <= maxZ + epsilon
			);
	}

	__host__ __device__ bool isTriangleInBox(Triangle* triangle) {
		return(isVertexInBox(triangle->a.position) ||
			isVertexInBox(triangle->b.position) ||
			isVertexInBox(triangle->c.position)
			);
	}

	__host__ __device__ Hit intersect(const Ray& ray) {
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

	__host__ __device__ float widthOnAxis(Axis axis) {
		if (axis == Axis_X) return maxX - minX;
		if (axis == Axis_Y) return maxY - minY;
		if (axis == Axis_Z) return maxZ - minZ;
	}

	__host__ __device__ bool inBoundsOnAxis(Axis axis, float t) {
		if (axis == Axis_X) {
			return (t > minX && t < maxX);
		}
		if (axis == Axis_Y) {
			return (t > minY && t < maxY);
		}
		if (axis == Axis_Z) {
			return (t > minZ && t < maxZ);
		}
	}

	__host__ __device__ float startingPointOnAxis(Axis axis) {
		if (axis == Axis_X) return minX;
		if (axis == Axis_Y) return minY;
		if (axis == Axis_Z) return minZ;
	}
};


struct Node {
	//CPU
	Node* left = nullptr;
	Node* right = nullptr;

	//shared
	Box box;
	Axis axis;

	//CUDA
	Triangle* device_triangles;
	int* device_triangles_size;

	int left_child_of_parent_idx = -1;
	int right_child_of_parent_idx = -1;
	int left_index = -1;
	int right_index = -1;


	__device__ __host__ bool isLeaf() {
		return left == nullptr;
	}

	void makeNodeTree(std::vector<Triangle*> triangles, int depth=0) {
		if (triangles.size() <= maxTriangleNum || depth > maxKdTreeHeight) {
			int size = triangles.size();
			copyToDevice(device_triangles_size, size, "device_triangles_size");
			copyToDevice(device_triangles, triangles, size, "device_triangles");
			return;
		}
		left = new Node();
		right = new Node();


		//get vertices
		int N_vertices = triangles.size() * 3;
		float* vertices = new float[N_vertices];
		for (int i = 0; i < triangles.size(); i++) {
			vertices[3 * i] = triangles[i]->a.position.axisCoordinate(axis);
			vertices[3 * i + 1] = triangles[i]->b.position.axisCoordinate(axis);
			vertices[3 * i + 2] = triangles[i]->c.position.axisCoordinate(axis);
		}

		//remove vertices out of box bounds
		int N_verces_in_bounds = 0;
		for (int i = 0; i < N_vertices; i++) {
			if (box.inBoundsOnAxis(axis, vertices[i])) N_verces_in_bounds++;
		}

		float* vertices_in_bounds = new float[N_verces_in_bounds];
		int j = 0;
		for (int i = 0; i < N_vertices; i++) {
			if (box.inBoundsOnAxis(axis, vertices[i])) {
				vertices_in_bounds[j++] = vertices[i];
			}
		}
		delete[] vertices;

		//sort verticies
		std::sort(vertices_in_bounds, vertices_in_bounds + N_verces_in_bounds, std::less<float>());


		//log2 look for optimal halfing point
		int idx_split_point = N_verces_in_bounds / 2;
		for (int i = 2; i <= triangleOptimumSearchMaxDepth + 1; i++) {
			//split
			float splitPoint = vertices_in_bounds[idx_split_point];
			box.half(left->box, right->box, axis,
				(splitPoint - box.startingPointOnAxis(axis)) / box.widthOnAxis(axis)
			);
			//count triangles in each
			int N_trinagles_left = 0, N_triangles_right = 0;
			for (auto triangle : triangles) {
				if (left->box.isTriangleInBox(triangle)) {
					N_trinagles_left++;
				}
				if (right->box.isTriangleInBox(triangle)) {
					N_triangles_right++;
				}
			}
			//decide next move
			if ((N_trinagles_left - N_triangles_right) < allowedtriangleDifference
				&& -allowedtriangleDifference < (N_trinagles_left - N_triangles_right)) {
				break;
			}
			else if (N_trinagles_left > N_triangles_right) {
				idx_split_point -= N_verces_in_bounds / (2 * i);
				if (idx_split_point < 0) {
					idx_split_point = 0;
					break;
				}
			}
			else {
				idx_split_point += N_verces_in_bounds / (2 * i);
				if (idx_split_point > N_verces_in_bounds - 1) {
					idx_split_point = N_verces_in_bounds - 1;
					break;
				}
			}
		}


		//half at selected point
		box.half(left->box, right->box, axis
			, (vertices_in_bounds[idx_split_point] - box.startingPointOnAxis(axis)) / box.widthOnAxis(axis)
		);

		delete[] vertices_in_bounds;
		/////////////////////////////////////////


		//add triangles
		std::vector<Triangle*> left_triangles;
		std::vector<Triangle*> right_triangles;
		for (auto triangle : triangles) {
			if (left->box.isTriangleInBox(triangle)) {
				left_triangles.push_back(triangle);
			}
			if (right->box.isTriangleInBox(triangle)) {
				right_triangles.push_back(triangle);
			}
		}
		//change axis for next time
		left->axis = nextAxis(axis);
		right->axis = nextAxis(axis);
		left->makeNodeTree(left_triangles, depth + 1);
		right->makeNodeTree(right_triangles, depth + 1);
	}

	int getSize() {
		if (isLeaf()) return 1;
		return 1 + left->getSize() + right->getSize();
	}
};


struct Mesh {
	Node* device_kdTree;
	int* device_kdTreeSize;

	Mesh(std::vector<Triangle*> triangles) {
		Node* kdTree = getKdTree(triangles);
		uploadKdTree(kdTree);
		for (auto triangle : triangles) {
			delete triangle;
		}
	}
	Node* getKdTree(std::vector<Triangle*> triangles) {
		Node* kdTree = new Node();

		kdTree->box.minX = triangles[0]->a.position.x;
		kdTree->box.maxX = triangles[0]->a.position.x;
		kdTree->box.minY = triangles[0]->a.position.y;
		kdTree->box.maxY = triangles[0]->a.position.y;
		kdTree->box.minZ = triangles[0]->a.position.z;
		kdTree->box.maxZ = triangles[0]->a.position.z;
		for (auto triangle : triangles) {
			processTriangleToBox(triangle, kdTree->box);
		}

		kdTree->axis = Axis_X;
		kdTree->makeNodeTree(triangles);

		return kdTree;
	}
	void processTriangleToBox(Triangle* triangle, Box& box) {
		processVertexToBox(triangle->a, box);
		processVertexToBox(triangle->b, box);
		processVertexToBox(triangle->c, box);
	}
	void processVertexToBox(Vertex vertex, Box& box) {
		if (vertex.position.x < box.minX) box.minX = vertex.position.x;
		if (vertex.position.x > box.maxX) box.maxX = vertex.position.x;
		if (vertex.position.y < box.minY) box.minY = vertex.position.y;
		if (vertex.position.y > box.maxY) box.maxY = vertex.position.y;
		if (vertex.position.z < box.minZ) box.minZ = vertex.position.z;
		if (vertex.position.z > box.maxZ) box.maxZ = vertex.position.z;
	}


	void uploadKdTree(Node* kdTree) {
		//save size
		int size = kdTree->getSize();
		copyToDevice(device_kdTreeSize, size, "device_kdTreeSize");

		Node *host_kdTree = new Node[size];

		std::queue<Node*> queue;
		queue.push(kdTree);
		int idx = 0;
		while (queue.size() > 0) {
			Node* currentNode = queue.front();
			queue.pop();

			while (!(currentNode->isLeaf())) {
				putNodeToIdx(currentNode, host_kdTree, idx);
				idx++;
				
				//next while
				queue.push(currentNode->left);
				currentNode = currentNode->right;
			}
			//leaf node
			putNodeToIdx(currentNode, host_kdTree, idx);
			idx++;
		}
		//upload to CUDA
		copyToDevice(device_kdTree, host_kdTree, size, "device_kdTree");
	}

	void putNodeToIdx(Node* currentNode, Node* array, int idx) {
		//go back to parent and update
		if (currentNode->left_child_of_parent_idx != -1) {
			array[currentNode->left_child_of_parent_idx]
				.left_index = idx;
		}
		else if (currentNode->right_child_of_parent_idx != -1) {
			array[currentNode->right_child_of_parent_idx]
				.right_index = idx;
		}

		if (!currentNode->isLeaf()) {
			//save for children where to update later
			currentNode->left->left_child_of_parent_idx = idx;
			currentNode->right->right_child_of_parent_idx = idx;
		}

		//add to array
		array[idx] = *currentNode;
	}

	__device__ Hit intersect(const Ray& ray) {
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

//OBJ
constexpr auto POSITION = "v";
constexpr auto NORMAL = "vn";
constexpr auto FACE = "f";
constexpr auto POSITION_IDX = 0;
constexpr auto NORMAL_IDX = 2;
constexpr auto MAX_IDX = 2;
constexpr auto DEVIDER = '/';
constexpr auto EMPTY = ' ';
Mesh* readObjIntoMesh(std::string location, mat4 SRTmtx) {
	//read into these
	std::vector<vec3> positions;
	std::vector<vec3> normals;
	std::vector<int> position_idx;
	std::vector<int> normal_idx;

	//open file
	std::stringstream ss;
	std::ifstream in_file(location);
	std::string line = "";
	if (!in_file.is_open()) {
		throw "Error opening file at " + location;
	}
	//helper temps
	std::string prefix = "";
	vec3 temp;
	int i;

	//read all lines
	while (std::getline(in_file, line)) {
		//clear, read new line
		ss.clear();
		ss.str(line);
		//get prefix
		ss >> prefix;

		if (prefix == POSITION)
		{
			ss >> temp.x >> temp.y >> temp.z;
			positions.push_back(temp);
		}
		else if (prefix == NORMAL)
		{
			ss >> temp.x >> temp.y >> temp.z;
			normals.push_back(temp);
		}
		else if (prefix == FACE)
		{
			int counter = 0;
			while (ss >> i) {
				//read
				if (counter == POSITION_IDX) {
					position_idx.push_back(i - 1); //we index from 0
				}
				else if (counter == NORMAL_IDX) {
					normal_idx.push_back(i - 1); //we index from 0
				}
				//handle separators
				for (int i = 0; i < 3; i++) {
					if (ss.peek() == DEVIDER) {
						counter++;
						ss.ignore(1, DEVIDER);
					}
				}
				if (ss.peek() == EMPTY) {
					counter = 0;
					ss.ignore(1, EMPTY);
				}
			}

		}
	}

	std::vector<Triangle*> triangles;
	triangles.resize(position_idx.size() / 3);
	for (size_t i = 0; i < triangles.size(); i++) {
		int vertexAIdx = i * 3;
		int vertexBIdx = i * 3 + 1;
		int vertexCIdx = i * 3 + 2;
		vec3 posA = (vec4(positions[position_idx[vertexAIdx]], 1) * SRTmtx ).xyz();
		vec3 normA = (vec4(normals[normal_idx[vertexAIdx]], 0) * SRTmtx.inverse().transpose()).xyz().normalize();
		vec3 posB = (vec4(positions[position_idx[vertexBIdx]], 1) * SRTmtx).xyz();
		vec3 normB = (vec4(normals[normal_idx[vertexBIdx]], 0) * SRTmtx.inverse().transpose()).xyz().normalize();
		vec3 posC = (vec4(positions[position_idx[vertexCIdx]], 1) * SRTmtx).xyz();
		vec3 normC = (vec4(normals[normal_idx[vertexCIdx]], 0) * SRTmtx.inverse().transpose()).xyz().normalize();

		triangles[i] = new Triangle(
			Vertex(posA, normA),
			Vertex(posB, normB),
			Vertex(posC, normC),
			nullptr
		);
	}
	return new Mesh(triangles);
}


struct MeshObject : public Intersectable {
	Mesh* device_mesh;
	vec3 positon = vec3(0, 0, 0);
	vec3 scale = vec3(1, 1, 1);
	vec3 rotate = vec3(0, 0, 0);

	MeshObject() : Intersectable() {
		
	}

	MeshObject(std::string location, vec3 position, vec3 rotate, vec3 scale, Material* mat) : Intersectable(mat) {
		Mesh* mesh = readObjIntoMesh(location, SRTmtx(scale, rotate, position));
		this->positon = position;
		this->scale = scale;
		this->rotate = rotate;
		copyToDevice(device_mesh, *mesh, "device_mesh");
	}

	__device__ Hit intersect(const Ray& ray) {
		Hit bestHit = device_mesh->intersect(ray);
		bestHit.material = this->material;
		return bestHit;
	}
};

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

	void copyHDRToCuda(HDRLoaderResult &cpu_result, HDRLoaderResult* &device_result) {
		//change cols to CUDA cols
		float* device_cols;
		copyToDevice(device_cols, cpu_result.cols, cpu_result.size(), "HDR.cols");
		delete cpu_result.cols;
		cpu_result.cols = device_cols;

		//copy to CUDA
		copyToDevice(device_result, cpu_result, "HDR");
	}
public:
	EnvMap(std::string folderName) {
		HDRLoader::load((folderName + "px.hdr").c_str(), px);
		HDRLoader::load((folderName + "py.hdr").c_str(), py);
		HDRLoader::load((folderName + "pz.hdr").c_str(), pz);
		HDRLoader::load((folderName + "nx.hdr").c_str(), nx);
		HDRLoader::load((folderName + "ny.hdr").c_str(), ny);
		HDRLoader::load((folderName + "nz.hdr").c_str(), nz);

		//Upload to CUDA
		copyHDRToCuda(px, device_px);
		copyHDRToCuda(py, device_py);
		copyHDRToCuda(pz, device_pz);
		copyHDRToCuda(nx, device_nx);
		copyHDRToCuda(ny, device_ny);
		copyHDRToCuda(nz, device_nz);
	}
	__host__ __device__ vec3 getPixelColor(vec3 ray) {
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
		else if ((abs(ray.z) >= abs(ray.x)) && (abs(ray.z) >= abs(ray.y))) {
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
	}
};

// The virtual camera
class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = w.length();
		right = cross(vup, w).normalize() * f * tanf(fov / 2.0f);	// orthogonalization
		up = cross(w, right).normalize() * f * tanf(fov / 2.0f);
	}
	__host__ __device__ Ray getRay(float X, float Y) {	// integer parts of X, Y define the pixel, fractional parts the point inside pixel
		vec3 dir = lookat + right * (2.0f * X / screenWidth - 1.0f) + up * (2.0f * Y / screenHeight - 1.0f) - eye;
		return Ray(eye, dir);
	}
};

// Point light source
struct Light {
	vec3 location;
	vec3 power;

	Light() {
	
	}

	Light(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	__host__ __device__ float distanceOf(vec3 point) {
		return (location - point).length();
	}
	__host__ __device__ vec3 directionOf(vec3 point) {
		return (location - point).normalize();
	}
	__host__ __device__ vec3 radianceAt(vec3 point) {
		float distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2 / 4.0f / M_PI;
	}
	__device__  void randomSampleRay(vec3& outDir, curandState_t* state) {
		vec3 X = vec3(0.0f, 0.0f, 1.0f);
		vec3 Y = vec3(0.0f, 1.0f, 0.0f);
		vec3 Z = vec3(1.0f, 0.0f, 0.0f);

		float alpha = Rand::random(state) * 2.0f * PI;
		float beta = Rand::random(state) * 2.0f * PI;

		outDir = ((X * cosf(alpha) + Z * sinf(alpha)) * sinf(beta) + Y * cosf(beta)).normalize();

		//float x, y, z;
		//do {
		//	x = 2 * Rand::random(state) - 1;    // random point in [-1,-1;1,1]
		//	y = 2 * Rand::random(state) - 1;
		//} while (x * x + y * y > 1);  // reject if not in circle
		//z = sqrtf(1 - x * x - y * y);  // project to hemisphere

		//outDir = X * z + Y * x + Z * y;
	}
};

// Virtual world
class Scene {
	Sphere* device_spheres;
	int* device_sphere_size;

	MeshObject* device_meshes;
	int* device_meshes_size;

	Light* device_lights;
	int* device_lights_size;

	Plane* device_planes;
	int* device_plane_size;

	Ring* device_rings;
	int* device_ring_size;

public:
	Camera camera;
	EnvMap envMap = EnvMap("resources/hdr-env-cube/");
	void build() {
		vec3 eye = vec3(0.0f, 0.0f, 3.0f);
		vec3 vup = vec3(0.0f, 1.0f, 0.0f);
		vec3 lookat = vec3(0.0f, 0.0f, 0.0f);
		float fov = 100.0f * M_PI / 180.0f;
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

		//RINGS
		int rings_size = 0;
		copyToDevice(device_ring_size, rings_size, "device_plane_size");

		Ring* rings = new Ring[rings_size];
		//rings[0] = Ring(
		//	vec3(0, -0.5, 0), 
		//	vec3(0, 1, 0),
		//	3,
		//	1.5,
		//	new Material(vec3(0.0, 0.0, 0.0), vec3(0.9, 0.9, 0.9))
		//);

		copyToDevice(device_rings, rings, rings_size, "device_rings");

		delete rings;

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
			vec3(0.0f, M_PI, 0.0f),  //rotate
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
		for (int i = 0; i < *device_ring_size; i++) {
			Hit hit = device_rings[i].intersect(ray); //  hit.t < 0 if no intersection
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

	// Trace a ray and return the radiance of the visible surface
	__device__ vec3 trace(Ray _ray, EnvMap envMap, curandState_t* state) {
		vec3 outRad(0.0f, 0.0f, 0.0f);

		//random light ray
		int light_idx = (int)(Rand::random(state) * ((float)*device_lights_size));
		vec3 light_source_out_dir;
		device_lights[light_idx].randomSampleRay(light_source_out_dir, state);

		//first hit
		Hit first_light_hit = firstIntersect(Ray(device_lights[light_idx].location, light_source_out_dir));
		if (first_light_hit.t < 0.0f) {
			return outRad;
		}

		Hit light_paths_hits[maxdepth];
		vec3 light_paths_radiance[maxdepth];
		float light_paths_pdf[maxdepth];
		light_paths_hits[0] = first_light_hit;
		light_paths_radiance[0] = device_lights[light_idx].radianceAt(first_light_hit.position);
		light_paths_pdf[0] = 1.0f / ((float)*device_lights_size) * 1.0f / (4.0f * M_PI);
		
		//make light paths
		for (int i = 0; i < maxdepth; i++) {

			float diffuseSelectProb = light_paths_hits[i].material->diffuseAlbedo.average();
			float mirrorSelectProb = light_paths_hits[i].material->mirrorAlbedo.average();
			float rnd = Rand::random(state);	// Russian roulette to find diffuse, mirror or no reflection

			if (rnd < diffuseSelectProb + mirrorSelectProb) {
				vec3 lightDirIn = (i == 0) ?
					light_source_out_dir :
					(light_paths_hits[i].position - light_paths_hits[i - 1].position).normalize();
				vec3 lightDirOut;
				float pdf_brdf;

				if (rnd < diffuseSelectProb) { // diffuse
					pdf_brdf = SampleDiffuse(
						light_paths_hits[i].normal,
						lightDirIn,
						lightDirOut,
						state
					) * diffuseSelectProb;
					float cosThetaL = dot(light_paths_hits[i].normal, lightDirIn * (-1));
					if (cosThetaL > epsilon) {
						if (i == 0) {
							light_paths_radiance[i] = light_paths_radiance[i] * (light_paths_hits[i].material->diffuseAlbedo) / M_PI * cosThetaL;
						}
						else {
							light_paths_radiance[i] = light_paths_radiance[i - 1] * (light_paths_hits[i].material->diffuseAlbedo) / M_PI * cosThetaL;
						}
					}
					else {
						light_paths_hits[i].t = -1;
						break;
					}
				}

				else { // mirror
					pdf_brdf = SampleMirror(
						light_paths_hits[i].normal,
						lightDirIn,
						lightDirOut
					)* mirrorSelectProb;
					float cosThetaL = dot(light_paths_hits[i].normal, lightDirIn * (-1));
					if (cosThetaL > epsilon) {
						if (i == 0) {
							light_paths_radiance[i] = light_paths_radiance[i] * (light_paths_hits[i].material->mirrorAlbedo);
						}
						else {
							light_paths_radiance[i] = light_paths_radiance[i - 1] * (light_paths_hits[i].material->mirrorAlbedo);
						}
					}
					else {
						light_paths_hits[i].t = -1;
						break;
					}
				}

				//evaluate next hit
				if (i != maxdepth - 1) {
					light_paths_hits[i + 1] = firstIntersect(Ray(light_paths_hits[i].position + light_paths_hits[i].normal * epsilon, lightDirOut));
					if (light_paths_hits[i + 1].t < 0) {
						break;
					}
					else {
						light_paths_pdf[i + 1] = light_paths_pdf[i] * pdf_brdf;
					}
				}
			}
			else {
				break;
			}
		}


		vec3 paths_color[maxdepth * (maxdepth + 1)];
		float paths_probability[maxdepth * (maxdepth + 1)];
		int n_paths = 0;


		vec3 pixel_path_brdf = vec3(1.0f, 1.0f, 1.0f);
		float pixel_path_pdf = 1.0f;
		Ray ray = _ray;

		//make paths
		for (int i = 0; i < maxdepth; i++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) {
				//outRad += envMap.getPixelColor(ray.dir) * 6* pixel_path_weight;
				break;
				return outRad;
			}
			else {

				//for (int i = 0; i < *device_lights_size; i++) {	// Direct light source computation
				// vec3 outDir = device_lights[i].directionOf(hit.position);
				//	Hit shadowHit = firstIntersect(Ray(hit.position + hit.normal * epsilon, outDir));
				//	if (shadowHit.t < epsilon || shadowHit.t > device_lights[i].distanceOf(hit.position)) {	// if not in shadow
				//		float cosThetaL = dot(hit.normal, outDir);
				//		if (cosThetaL >= epsilon) {
				//			outRad += hit.material->diffuseAlbedo / M_PI * cosThetaL
				//				* device_lights[i].radianceAt(hit.position) * pixel_path_brdf /pixel_path_pdf;
				//		}
				//	}
				//}


				vec3 outDir_light = device_lights[light_idx].directionOf(hit.position);
				Hit shadowHit = firstIntersect(Ray(hit.position + hit.normal * epsilon, outDir_light));
				if (shadowHit.t < epsilon || shadowHit.t > device_lights[light_idx].distanceOf(hit.position)) {	// if not in shadow
					float cosThetaL = dot(hit.normal, outDir_light);
					if (cosThetaL >= epsilon) {
						paths_color[n_paths] = device_lights[light_idx].radianceAt(hit.position)
							* hit.material->diffuseAlbedo / M_PI * cosThetaL
							* pixel_path_brdf
							;
						paths_probability[n_paths] = pixel_path_pdf;
						n_paths++;
					}
				}


				for (int j = 0; j<maxdepth; j++) {
					if (light_paths_hits[j].t < 0) break;

					Ray pixel_path_to_light_path = Ray(
						hit.position + hit.normal * epsilon,
						(light_paths_hits[j].position - hit.position).normalize()
					);

					Hit pixel_to_light_first_hit = firstIntersect(pixel_path_to_light_path);
					bool clearView = pixel_to_light_first_hit.t > 0 && (pixel_to_light_first_hit.position - light_paths_hits[i].position).length() < epsilon;

					if (clearView) {
						float r = (light_paths_hits[j].position - hit.position).length();
						float cosThetaInPixel = dot(hit.normal, pixel_path_to_light_path.dir);
						float cosThetaInLight = dot(light_paths_hits[j].normal, pixel_path_to_light_path.dir * (-1.0));
						if (cosThetaInPixel < epsilon) {
							continue;
						}
						vec3 brdf_pixel = (hit.material->diffuseAlbedo) / M_PI * cosThetaInPixel;

						paths_color[n_paths] = light_paths_radiance[j]
							* brdf_pixel
							* cosThetaInPixel
							//* 1.0 / (r * r)
							* pixel_path_brdf
							;
						paths_probability[n_paths] = 1;// pixel_path_pdf* light_paths_pdf[j];
						n_paths++;
					}

				}

				float diffuseSelectProb = hit.material->diffuseAlbedo.average();
				float mirrorSelectProb = hit.material->mirrorAlbedo.average();

				float rnd = Rand::random(state);	// Russian roulette to find diffuse, mirror or no reflection
				vec3 outDir;
				if (rnd < diffuseSelectProb) { // diffuse
					float pdf = SampleDiffuse(hit.normal, ray.dir, outDir, state);
					float cosThetaL = dot(hit.normal, outDir);
					if (cosThetaL >= epsilon) {
						pixel_path_brdf = pixel_path_brdf * (hit.material->diffuseAlbedo) / M_PI * cosThetaL; //brdf
						pixel_path_pdf = pixel_path_pdf * pdf * diffuseSelectProb; //pdf
					}
					else {
						break;
					}
				}
				else if (rnd < diffuseSelectProb + mirrorSelectProb) { // mirror
					float pdf = SampleMirror(hit.normal, ray.dir, outDir);
					pixel_path_brdf = pixel_path_brdf * hit.material->mirrorAlbedo; //brdf
					pixel_path_pdf = pixel_path_pdf * pdf * mirrorSelectProb; //pdf
				}
				else {
					break;
				}

				ray = Ray(hit.position + hit.normal * epsilon, outDir);
			}
		}

		if (n_paths != 0) {
			float sum_probability = 0;
			for (int i = 0; i < n_paths; i++) {
				sum_probability += paths_probability[i];
			}
			//return vec3(sum_probability, sum_probability, sum_probability);
			for (int i = 0; i < n_paths; i++) {
				outRad += paths_color[i] / paths_probability[i];// / n_paths;// sum_probability;
			}
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
	std::cout << "Render ended in " << elapsed.count() * 1e-9 << std::endl;

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