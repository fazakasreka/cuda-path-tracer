#pragma once

#include <cmath>
#include "constants/constants.cuh"

enum Axis { Axis_X, Axis_Y, Axis_Z };
__host__ __device__ inline Axis nextAxis(Axis axis) {
	if (axis == Axis_X) return Axis_Y;
	if (axis == Axis_Y) return Axis_Z;
	if (axis == Axis_Z) return Axis_X;
}

// 3D vector operations
struct vec3 {
	float x, y, z;
	__host__ __device__ inline vec3(float x0 = 0.0f, float y0 = 0.0f, float z0 = 0.0f) { x = x0; y = y0; z = z0; }
	__host__ __device__ inline vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	__host__ __device__ inline vec3 operator/(float d) const { return vec3(x / d, y / d, z / d); }
	__host__ __device__ inline vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	__host__ __device__ inline void operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; }
	__host__ __device__ inline vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	__host__ __device__ inline vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	__host__ __device__ inline vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__ inline vec3 normalize() const { return (*this) * (1 / (length() + epsilon)); }
	__host__ __device__ inline float length() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ inline float average() { return (x + y + z) / 3.0f; }
	__host__ __device__ inline float axisCoordinate(Axis axis) {
		if (axis == Axis_X) return x;
		if (axis == Axis_Y) return y;
		if (axis == Axis_Z) return z;
	}
};


__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {	// dot product
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {	// cross product
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

//4D vector
struct vec4 {
	//--------------------------
	float x, y, z, w;

	__host__ __device__ inline vec4(float x0 = 0.0f, float y0 = 0.0f, float z0 = 0.0f, float w0 = 0.0f) { x = x0; y = y0; z = z0; w = w0; }
	__host__ __device__ inline vec4(vec3 vec3, float w0 = 0.0f) { x = vec3.x; y = vec3.z; z = vec3.y; w = w0; }
	__host__ __device__ inline float& operator[](int j) { return *(&x + j); }
	__host__ __device__ inline float operator[](int j) const { return *(&x + j); }
	__host__ __device__ inline vec3 xyz() {
		return vec3(this->x, this->y, this->z);
	}

	__host__ __device__ inline vec4 operator*(float a) const { return vec4(x * a, y * a, z * a, w * a); }
	__host__ __device__ inline vec4 operator/(float d) const { return vec4(x / d, y / d, z / d, w / d); }
	__host__ __device__ inline vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
	__host__ __device__ inline vec4 operator-(const vec4& v)  const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
	__host__ __device__ inline vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
	__host__ __device__ inline void operator+=(const vec4 right) { x += right.x; y += right.y; z += right.z; w += right.w; }
};

__host__ __device__ inline float dot(const vec4& v1, const vec4& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
}

__host__ __device__ inline vec4 operator*(float a, const vec4& v) {
	return vec4(v.x * a, v.y * a, v.z * a, v.w * a);
}

//mat4
//---------------------------
struct mat4 { // row-major matrix 4x4
//---------------------------
	vec4 rows[4];
public:
	__host__ __device__ inline mat4() {}
	__host__ __device__ inline mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		rows[0][0] = m00; rows[0][1] = m01; rows[0][2] = m02; rows[0][3] = m03;
		rows[1][0] = m10; rows[1][1] = m11; rows[1][2] = m12; rows[1][3] = m13;
		rows[2][0] = m20; rows[2][1] = m21; rows[2][2] = m22; rows[2][3] = m23;
		rows[3][0] = m30; rows[3][1] = m31; rows[3][2] = m32; rows[3][3] = m33;
	}
	__host__ __device__ inline mat4(vec4 it, vec4 jt, vec4 kt, vec4 ot) {
		rows[0] = it; rows[1] = jt; rows[2] = kt; rows[3] = ot;
	}

	__host__ __device__ inline vec4& operator[](int i) { return rows[i]; }
	__host__ __device__ inline vec4 operator[](int i) const { return rows[i]; }
	__host__ __device__ inline operator float* () const { return (float*)this; }

	mat4 inverse() const;

	inline mat4 transpose() {
		return mat4(
			rows[0][0], rows[1][0], rows[2][0], rows[3][0],
			rows[0][1], rows[1][1], rows[2][1], rows[3][1],
			rows[0][2], rows[1][2], rows[2][2], rows[3][2],
			rows[0][3], rows[1][3], rows[2][3], rows[3][3]
		);

	}

};


__host__ __device__ inline vec4 operator*(const vec4& v, const mat4& mat) {
	return v[0] * mat[0] + v[1] * mat[1] + v[2] * mat[2] + v[3] * mat[3];
}

__host__ __device__ inline mat4 operator*(const mat4& left, const mat4& right) {
	mat4 result;
	for (int i = 0; i < 4; i++) result.rows[i] = left.rows[i] * right;
	return result;
}

__host__ __device__ inline mat4 TranslateMatrix(vec3 t) {
	return mat4(vec4(1.0f, 0.0f, 0.0f, 0.0f),
		vec4(0.0f, 1.0f, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, 1.0f, 0.0f),
		vec4(t.x, t.y, t.z, 1.0f));
}

__host__ __device__ inline mat4 ScaleMatrix(vec3 s) {
	return mat4(
		vec4(s.x, 0.0f, 0.0f, 0.0f),
		vec4(0.0f, s.y, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, s.z, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 1.0f));
}

__host__ __device__ inline mat4 RotationMatrix(float angle, vec3 w) {
	float c = cosf(angle), s = sinf(angle);
	w = w.normalize();
	return mat4(
		vec4(c * (1.0f - w.x * w.x) + w.x * w.x, w.x * w.y * (1.0f - c) + w.z * s, w.x * w.z * (1.0f - c) - w.y * s, 0.0f),
		vec4(w.x * w.y * (1.0f - c) - w.z * s, c * (1.0f - w.y * w.y) + w.y * w.y, w.y * w.z * (1.0f - c) + w.x * s, 0.0f),
		vec4(w.x * w.z * (1.0f - c) + w.y * s, w.y * w.z * (1.0f - c) - w.x * s, c * (1.0f - w.z * w.z) + w.z * w.z, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 1.0f));
}

inline mat4 SRTmtx(vec3 scale, vec3 rotation, vec3 translate) {
	return ScaleMatrix(scale)
		* RotationMatrix(rotation.x, vec3(1.0f, 0.0f, 0.0f))
		* RotationMatrix(rotation.y, vec3(0.0f, 1.0f, 0.0f))
		* RotationMatrix(rotation.z, vec3(0.0f, 0.0f, 1.0f))
		* TranslateMatrix(translate);
}