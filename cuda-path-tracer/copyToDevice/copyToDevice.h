#pragma once

#include <vector>
#include <stdio.h> 

//copy single data
template <typename T>
void copyToDevice(T* &device_pointer, T &data, const char* errorMessage) {
	cudaError_t cudaStatus = cudaMalloc((void**)&device_pointer, sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
	}

	cudaStatus = cudaMemcpy(device_pointer, &data, sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s cudaMemcpy failed! %s \n", errorMessage, cudaGetErrorString(cudaStatus));
	}

}

//copy array of data
template <typename T>
void copyToDevice(T* &device_pointer, T* data, int size, const char* errorMessage) {
	cudaError_t cudaStatus = cudaMalloc((void**)&device_pointer, size * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
	}

	cudaStatus = cudaMemcpy(device_pointer, data, size * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s cudaMemcpy failed! %s \n", errorMessage, cudaGetErrorString(cudaStatus));
	}

}

//copy std:vector data
template <typename T>
void copyToDevice(T*& device_pointer, std::vector<T*> data, int size, const char* errorMessage) {
	cudaError_t cudaStatus = cudaMalloc((void**)&device_pointer, size * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! \n");
	}

	T* t_array = new T[size];
	for (int i = 0; i < size; i++){
		t_array[i] = *data[i];
	}

	cudaStatus = cudaMemcpy(device_pointer, t_array, size * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s cudaMemcpy failed! %s \n", errorMessage, cudaGetErrorString(cudaStatus));
	}

	delete t_array;
}