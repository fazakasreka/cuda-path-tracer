/***********************************************************************************
	Created:	17:9:2002
	FileName: 	hdrloader.h
	Author:		Igor Kravtchenko

	Info:		Load HDR image and convert to a set of float32 RGB triplet.
************************************************************************************/

class HDRLoaderResult {
public:
	int width, height;
	// each pixel takes 3 float32, each component can be of any value...
	float* cols;

	inline int size() {
		return width * height * 3;
	}
};

class HDRLoader {
public:
	 __host__ static bool load(const char* fileName, HDRLoaderResult& res);
};

