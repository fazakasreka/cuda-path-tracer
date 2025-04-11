#pragma once

#include <cuda_runtime.h>

__constant__ const unsigned int screenWidth = 512, screenHeight = 512;	// resolution of the rendered image
__constant__ const float epsilon = 1e-5;	// limit of considering a number to be zero
__constant__ const int maxdepth = 3;		// max depth of recursion
__constant__ const int nSamples = 50;		// number of path samples per pixel
__constant__ const float PI = 3.14159265358979323846f;

__constant__ const int maxTriangleNum = 50; 
__constant__ const int maxKdTreeHeight= 15;
__constant__ const int allowedtriangleDifference = 20;
__constant__ const int triangleOptimumSearchMaxDepth = 20;
__constant__ const int tileSize = 8;
__constant__ const int tileArea = tileSize * tileSize;