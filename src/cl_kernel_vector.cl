#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define INVALID -2

typedef struct short2
{
	short x;
	short y;
} short2s;
struct float2
{
	float x;
	float y;
};

struct float3 {
	float x;
	float y;
	float z;
};

struct uchar3 {
	unsigned char x;
	unsigned char y;
	unsigned char z;
};

inline float c_sq(float r)
{
	return r * r;
}

typedef struct Matrix4
{
	float4 data[4];
} Matrix4;

inline float3 mul_Matrix4_float3v(const Matrix4 M, const float3 val)
{
	return (float3)(dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), val) + M.data[0].w,
	                dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), val) + M.data[1].w,
	                dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), val) + M.data[2].w);
}

inline float c_clamp_float(float f, float a, float b)
{
	return fmax(a, fmin(f, b));
}

inline float2 getVolume(const uint v_size_x, const uint v_size_y, const uint v_size_z,
                         __global const short2s *v_data, const uint x, const uint y, const uint z)
{
	const short2s d = v_data[x + y * v_size_x + z * v_size_x * v_size_y];
	return (float2)(d.x * 0.00003051944088f, d.y);
}

inline void setVolume(const uint v_size_x, const uint v_size_y, const uint v_size_z,
                      __global short2s *v_data, const uint x, const uint y, const uint z, float2 d)
{
	v_data[x + y * v_size_x + z * v_size_x * v_size_y].x = d.x * 32766.0f;
	v_data[x + y * v_size_x + z * v_size_x * v_size_y].y = d.y;
}

inline float3 add_float3_float3(float3 a, float3 b)
{
	return (float3)(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 v_rotate(const Matrix4 M, const float3 v)
{
	return (float3) (dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v),
	                 dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v),
	                 dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

inline int c_clamp(int f, int a, int b)
{
	return max(a, min(f, b));
}

inline int2 v_clamp_int2(int2 v, int2 a, int2 b)
{
	return (int2)(c_clamp(v.x, a.x, b.x), c_clamp(v.y, a.y, b.y));
}

inline uchar3 v_gs2rgb(float h)
{
	uchar3 rgb;
	int sextant;
	float fractn;
	float vsf;
	float mid1;
	float mid2;
	h *= 6.0f;
	sextant = (int) h;
	fractn = h - sextant;
	vsf = 0.5f*fractn;
	mid1 = 0.25f + vsf;
	mid2 = 0.75f - vsf;
	if (sextant == 0) {
		rgb = (uchar3)(191, mid1 * 255, 64);
	}	else if (sextant == 1) {
		rgb = (uchar3)(mid2 * 255, 191, 64);
	}	else if (sextant == 2) {
		rgb = (uchar3)(64, 191, mid1 * 255);
	}	else if (sextant == 3) {
		rgb = (uchar3)(64, mid2 * 255, 191);
	}	else if (sextant == 4) {
		rgb = (uchar3)(mid1 * 255, 64, 191);
	}	else if (sextant == 5) {
		rgb = (uchar3)(191, 64, mid2 * 255);
	}	else {
		rgb = (uchar3)(0, 0, 0);
	}
	return rgb;
}

typedef struct TrackData
{
	int result;
	float error;
	float J[6];
} TrackData;

inline float c_vs2(const uint x, const uint y, const uint z,
                   const uint v_size_x, const uint v_size_y,
                   const uint v_size_z, __global const short2s *v_data)
{
	return v_data[x + y * v_size_x + z * v_size_x * v_size_y].x;
}

inline float v_interp(const float3 pos, const uint v_size_x,
                      const uint v_size_y, const uint v_size_z,
                      __global const short2s *v_data, const float3 v_dim)
{
	const float3 scaled_pos = (float3)((pos.x * v_size_x / v_dim.x) - 0.5f,
	                                   (pos.y * v_size_y / v_dim.y) - 0.5f,
	                                   (pos.z * v_size_z / v_dim.z) - 0.5f);
	const int3 base = convert_int3(floor(scaled_pos));
	float3 basef = (float3)(0);
	const float3 factor = (float3)(fract(scaled_pos, (float3*) &basef));
	const int3 lower = max(base, (int3)(0));
	const int3 upper = min(base + (int3)(1), convert_int3((uint3)(v_size_x, v_size_y, v_size_z)) - (int3)(1));

	return (((c_vs2(lower.x, lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) * (1 - factor.x) +
	          c_vs2(upper.x, lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) * factor.x) * (1 - factor.y) +
	         (c_vs2(lower.x, upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) * (1 - factor.x) +
	          c_vs2(upper.x, upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) * factor.x) * factor.y) * (1 - factor.z) +
	        ((c_vs2(lower.x, lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) * (1 - factor.x) +
	          c_vs2(upper.x, lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) * factor.x) * (1 - factor.y) +
	         (c_vs2(lower.x, upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) * (1 - factor.x) +
	          c_vs2(upper.x, upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) * factor.x) * factor.y) * factor.z) * 0.00003051944088f;
}

inline float4 v_raycast(const uint v_size_x, const uint v_size_y, const uint v_size_z,
                        __global const short2s *v_data,
                        const float3 v_dim,	const uint2 pos, const Matrix4 view,
                        const float nearPlane, const float farPlane, const float stepVal,
                        const float largestep)
{
	const float3 origin = (float3)(view.data[0].w, view.data[1].w, view.data[2].w);
	const float3 direction = v_rotate(view, (float3)(pos.x, pos.y, 1.0f));

	const float3 invR = (float3)(1.0f) / direction;
	const float3 tbot = (float3) -1 * invR * origin;
	const float3 ttop = invR * (v_dim - origin);

	const float3 tmin = fmin(ttop, tbot);
	const float3 tmax = fmax(ttop, tbot);

	const float largest_tmin = fmax(fmax(tmin.x, tmin.y),	fmax(tmin.x, tmin.z));
	const float smallest_tmax = fmin(fmin(tmax.x, tmax.y), fmin(tmax.x, tmax.z));

	const float tnear = fmax(largest_tmin, nearPlane);
	const float tfar = fmin(smallest_tmax, farPlane);

	if (tnear < tfar) {
		float t = tnear;
		float stepsize = largestep;
		float f_t = v_interp(origin + direction * t, v_size_x, v_size_y, v_size_z, v_data, v_dim);
		float f_tt = 0;
		if (f_t > 0) {
			while (t < tfar) {
				f_tt = v_interp(origin + direction * t, v_size_x, v_size_y, v_size_z, v_data, v_dim);
				if (f_tt < 0) {
					break;
				}
				if (f_tt < 0.8f) {
					stepsize = stepVal;
				}
				f_t = f_tt;
				t += stepsize;
			}
			if (f_tt < 0) {
				t = t + stepsize * f_tt / (f_t - f_tt);
				return (float4)(origin + direction * t, t);
			}
		}
	}
	return (float4)(0);
}

inline float3 v_grad(float3 pos, const uint v_size_x, const uint v_size_y,
                      const uint v_size_z, __global const short2s *v_data,
                      const float3 v_dim)
{
	const float3 scaled_pos = (float3)((pos.x * v_size_x / v_dim.x) - 0.5f,
	                                   (pos.y * v_size_y / v_dim.y) - 0.5f,
	                                   (pos.z * v_size_z / v_dim.z) - 0.5f);
	const int3 base = (int3)(floor(scaled_pos.x), floor(scaled_pos.y),floor(scaled_pos.z));
	const float3 basef = (float3)(0);
	const float3 factor = fract(scaled_pos, (float3*) &basef);
	const int3 lower_lower = max(base - (int3)(1), (int3)(0));
	const int3 lower_upper = max(base, (int3)(0));
	const int3 upper_lower = min(base + (int3)(1), (convert_int3((uint3)(v_size_x, v_size_y, v_size_z)) - (int3)(1)));
	const int3 upper_upper = min(base + (int3)(2), (convert_int3((uint3)(v_size_x, v_size_y, v_size_z)) - (int3)(1)));
	const int3 lower = lower_upper;
	const int3 upper = upper_lower;

	float3 gradient;

	gradient.x = (((c_vs2(upper_lower.x, lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_lower.x, lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper_upper.x, lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_upper.x, lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * (1 - factor.y) +
	              ((c_vs2(upper_lower.x, upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_lower.x, upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper_upper.x, upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_upper.x, upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * factor.y) * (1 - factor.z) +
	             (((c_vs2(upper_lower.x, lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_lower.x, lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper_upper.x, lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_upper.x, lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * (1 - factor.y) +
	              ((c_vs2(upper_lower.x, upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_lower.x, upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper_upper.x, upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower_upper.x, upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * factor.y) * factor.z;

	gradient.y = (((c_vs2(lower.x, upper_lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, lower_lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, upper_lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, lower_lower.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * (1 - factor.y) +
	              ((c_vs2(lower.x, upper_upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, lower_upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, upper_upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, lower_upper.y, lower.z, v_size_x, v_size_y, v_size_z, v_data))	* factor.x) * factor.y) * (1 - factor.z) +
	             (((c_vs2(lower.x, upper_lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, lower_lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, upper_lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, lower_lower.y, upper.z, v_size_x, v_size_y, v_size_z, v_data))	* factor.x) * (1 - factor.y) +
	              ((c_vs2(lower.x, upper_upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, lower_upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, upper_upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, lower_upper.y, upper.z, v_size_x, v_size_y, v_size_z, v_data))	* factor.x) * factor.y) * factor.z;

	gradient.z = (((c_vs2(lower.x, lower.y, upper_lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, lower.y, lower_lower.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, lower.y, upper_lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, lower.y, lower_lower.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * (1 - factor.y) +
	              ((c_vs2(lower.x, upper.y, upper_lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, upper.y, lower_lower.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, upper.y, upper_lower.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, upper.y, lower_lower.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * factor.y) * (1 - factor.z) +
	             (((c_vs2(lower.x, lower.y, upper_upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, lower.y, lower_upper.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, lower.y, upper_upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, lower.y, lower_upper.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * (1 - factor.y) +
	              ((c_vs2(lower.x, upper.y, upper_upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(lower.x, upper.y, lower_upper.z, v_size_x, v_size_y, v_size_z, v_data)) * (1 - factor.x) +
	               (c_vs2(upper.x, upper.y, upper_upper.z, v_size_x, v_size_y, v_size_z, v_data) -
	                c_vs2(upper.x, upper.y, lower_upper.z, v_size_x, v_size_y, v_size_z, v_data)) * factor.x) * factor.y) * factor.z;

	return gradient * (float3)(v_dim.x / v_size_x, v_dim.y / v_size_y, v_dim.z / v_size_z) * (0.5f * 0.00003051944088f);
}

inline float3 v_clamp_float3(float3 v, float a, float b)
{
	return (float3)(c_clamp_float(v.x, a, b), c_clamp_float(v.y, a, b), c_clamp_float(v.z, a, b));
}

/* Code to be extracted into pencil_kernel_core.cl. */
inline void initVolume_core(const uint x, const uint y, const uint z,
                            const uint v_size_x, const uint v_size_y, const uint v_size_z,
                            __global short2s *v_data,
                            const float dxVal, const float dyVal)
{
	short2s dVal;
	dVal.x = dxVal;
	dVal.y = dyVal;
	v_data[x + y * v_size_x + z * v_size_x * v_size_y] = dVal;
}

TrackData track_core(uint refSize_x, uint refSize_y,
                     const TrackData output, const struct float3 inVertex, const struct float3 inNormal,
                     __global const struct float3 *refVertex, __global const struct float3 *refNormal,
                     const Matrix4 Ttrack,	const Matrix4 view, const float dist_threshold,
                     const float normal_threshold)
{
	TrackData row = output;

	if (inNormal.x == INVALID) {
		row.result = -1;
	} else {
		const float3 projectedVertex = mul_Matrix4_float3v(Ttrack, (float3)(inVertex.x, inVertex.y, inVertex.z));
		const float3 projectedPos = mul_Matrix4_float3v(view, projectedVertex);
		const float2 projPixel = (float2)(projectedPos.x/projectedPos.z + 0.5f, projectedPos.y/projectedPos.z + 0.5f);
		if ( projPixel.x < 0 || projPixel.x > refSize_x - 1 ||
		     projPixel.y < 0 || projPixel.y > refSize_y - 1 ) {
			row.result = -2;
		} else {
			const uint2 refPixel = (uint2)(projPixel.x, projPixel.y);
			const struct float3 refNormalStruct = refNormal[refPixel.x + refPixel.y * refSize_x];
			const float3 referenceNormal = (float3)(refNormalStruct.x, refNormalStruct.y, refNormalStruct.z);
			if (referenceNormal.x == INVALID) {
				row.result = -3;
			} else {
				const struct float3 refVertexStruct = refVertex[refPixel.x + refPixel.y * refSize_x];
				const float3 refVertexVec = (float3)(refVertexStruct.x, refVertexStruct.y, refVertexStruct.z);
				const float3 diff = refVertexVec - projectedVertex;
				const float3 projectedNormal = v_rotate(Ttrack,	(float3)(inNormal.x, inNormal.y, inNormal.z));
				if (length(diff) > dist_threshold) {
					row.result = -4;
				} else {
					if (dot(projectedNormal, referenceNormal) < normal_threshold) {
						row.result = -5;
					} else {
						row.result = 1;
						row.error = dot(referenceNormal, diff);
						row.J[0] = referenceNormal.x;
						row.J[1] = referenceNormal.y;
						row.J[2] = referenceNormal.z;
						const float3 crossVal = cross(projectedVertex, referenceNormal);
						row.J[3] = crossVal.x;
						row.J[4] = crossVal.y;
						row.J[5] = crossVal.z;
					}
				}
			}
		}
	}
	return row;
}

inline void raycast_core(const uint x, const uint y,
                         const uint inputSize_x, const uint inputSize_y,
                         __global struct float3 *vertex, __global struct float3 *normal,
                         const uint integration_size_x,
                         const uint integration_size_y,
                         const uint integration_size_z,
                         __global const short2s *integration_data,
                         const struct float3 integration_dim, const Matrix4 view,
                         const float nearPlane, const float farPlane,
                         const float stepVal, const float largestep)
{
	uint coord = y*inputSize_x + x;
	const float4 hit = v_raycast(integration_size_x, integration_size_y,
	                              integration_size_z, integration_data,
	                              (float3)(integration_dim.x, integration_dim.y, integration_dim.z), (uint2)(x, y), view,
	                              nearPlane, farPlane, stepVal, largestep);
	if (hit.w > 0.0) {
		const float3 test = (float3)(hit.x, hit.y, hit.z);
		const float3 surfNorm = v_grad(test, integration_size_x,
		                               integration_size_y, integration_size_z,
		                               integration_data, (float3)(integration_dim.x, integration_dim.y, integration_dim.z));
		vertex[coord].x = (test.x);
		vertex[coord].y = (test.y);
		vertex[coord].z = (test.z);
		if (length(surfNorm) == 0) {
			normal[coord].x = (float)INVALID;
		} else {
			float3 normalVec = normalize(surfNorm);
			normal[coord].x = normalVec.x;
			normal[coord].y = normalVec.y;
			normal[coord].z = normalVec.z;
		}
	} else {
		vertex[coord].x = 0;
		vertex[coord].y = 0;
		vertex[coord].z = 0;
		normal[coord].x = INVALID;
		normal[coord].y = 0;
		normal[coord].z = 0;
	}
}

inline struct uchar3 renderVolume_core(const uint x, const uint y,
                                 const uint volume_size_x,
                                 const uint volume_size_y,
                                 const uint volume_size_z,
                                 __global const short2s *volume_data,
                                 const struct float3 volume_dim, const Matrix4 view,
                                 const float nearPlane, const float farPlane,
                                 const float stepVal, const float largestep,
                                 const struct float3 light, const struct float3 ambient)
{
	struct uchar3 retVal;
	uchar3 retVec;
	float4 hit = v_raycast(volume_size_x, volume_size_y, volume_size_z,
	                        volume_data, (float3)(volume_dim.x, volume_dim.y, volume_dim.z), (uint2)(x, y), view,
	                        nearPlane, farPlane, stepVal, largestep);
	if (hit.w > 0) {
		const float3 test = (float3)(hit.x, hit.y, hit.z);
		const float3 surfNorm = v_grad(test, volume_size_x, volume_size_y,
		                               volume_size_z, volume_data, (float3)(volume_dim.x, volume_dim.y, volume_dim.z));
		if (length(surfNorm) > 0) {
			const float3 diff = normalize((float3)(light.x, light.y, light.z) - test);
			const float dir = fmax(dot(normalize(surfNorm), diff), 0.0f);
			retVec = convert_uchar3((v_clamp_float3((float3)(dir) + (float3)(ambient.x, ambient.y, ambient.z), 0.0f, 1.0f)) * (float3) 255);
		} else {
			retVec = (uchar3)(0, 0, 0);
		}
	} else {
		retVec = (uchar3)(0, 0, 0);
	}
	retVal.x = retVec.x;
	retVal.y = retVec.y;
	retVal.z = retVec.z;
	return retVal;
}

inline struct uchar3 renderTrack_core(const uint x, const uint y,
                                const uint outSize_x, const uint outSize_y,
                                __global const TrackData *data)
{
	int test = data[y*outSize_x + x].result;
	struct uchar3 retVal;
	uchar3 retVec;
	if (test == 1) {
		retVec = (uchar3)(128, 128, 128);
	} else if (test == -1) {
		retVec = (uchar3)(0, 0, 0);
	} else if (test == -2) {
		retVec = (uchar3)(255, 0, 0);
	} else if (test == -3) {
		retVec = (uchar3)(0, 255, 0);
	} else if (test == -4) {
		retVec = (uchar3)(0, 0, 255);
	} else if (test == -5) {
		retVec = (uchar3)(255, 255, 0);
	} else {
		retVec = (uchar3)(255, 128, 128);
	}
	retVal.x = retVec.x;
	retVal.y = retVec.y;
	retVal.z = retVec.z;
	return retVal;
}

inline struct uchar3 renderDepth_core(const uint x, const uint y,
                                const uint depthSize_x, const uint depthSize_y,
                                __global const float *depth,
                                const float nearPlane, const float farPlane,
                                const float rangeScale)
{
	struct uchar3 retVal;
	uchar3 retVec;
	float depthxy = depth[y*depthSize_x + x];
	if (depthxy < nearPlane) {
		retVec = (uchar3)(255, 255, 255);
	} else {
		if (depthxy > farPlane) {
			retVec = (uchar3)(0, 0, 0);
		} else {
			const float d = (depthxy - nearPlane) * rangeScale;
			retVec = v_gs2rgb(d);
		}
	}
	retVal.x = retVec.x;
	retVal.y = retVec.y;
	retVal.z = retVec.z;
	return retVal;
}

inline struct uchar3 renderNormal_core(const uint x, const uint y,
                                 const uint normalSize_x,
                                 const uint normalSize_y,
                                 __global const struct float3 *normal)
{
	uchar3 retVec;
	struct float3 n = normal[x + y * normalSize_x];
	if (n.x == -2) {
		retVec = (uchar3)(0, 0, 0);
	} else {
		float3 m = normalize((float3)(n.x, n.y, n.z));
		retVec = (uchar3)(m.x * 128 + 128, m.y * 128 + 128, m.z * 128 + 128);
	}
	struct uchar3 retVal;
	retVal.x = retVec.x;
	retVal.y = retVec.y;
	retVal.z = retVec.z;
	return retVal;
}

inline float halfSampleRobustImage_core(const uint x, const uint y,
                                        const uint outSize_x,
                                        const uint outSize_y,
                                        const uint inSize_x,
                                        const uint inSize_y,
                                        __global const float *in,
                                        const float e_d, const int r)
{
	uint2 pixel = (uint2) (x, y);
	uint2 centerPixel = 2 * pixel;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel.x	+ centerPixel.y * 2 * outSize_x];
	for (int i = -r + 1; i <= r; ++i) {
		for (int j = -r + 1; j <= r; ++j) {
			int2 cur = v_clamp_int2((int2)(centerPixel.x + j, centerPixel.y + i),
			                        (int2)(0, 0),
			                        (int2)(2 * outSize_x - 1, 2 * outSize_y - 1));
			float current = in[cur.x + cur.y * 2 * outSize_x];
			if (fabs(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	return t / sum;
}

inline struct float3 vertex2normal_core(const uint x, const uint y,
                                  const uint imageSize_x,
                                  const uint imageSize_y,
                                  __global const struct float3 *in)
{
	const uint2 pleft = (uint2)(max((int)x - 1, 0), y);
	const uint2 pright = (uint2)(min((int)x + 1, (int) imageSize_x - 1), y);
	const uint2 pup = (uint2)(x, max((int)y - 1, 0));
	const uint2 pdown = (uint2)(x, min((int)y + 1, ((int) imageSize_y) - 1));

	const struct float3 left_t = in[pleft.x + imageSize_x * pleft.y];
	const struct float3 right_t = in[pright.x + imageSize_x * pright.y];
	const struct float3 up_t = in[pup.x + imageSize_x * pup.y];
	const struct float3 down_t = in[pdown.x + imageSize_x * pdown.y];

	const float3 left = (float3) (left_t.x, left_t.y, left_t.z);
	const float3 right = (float3) (right_t.x, right_t.y, right_t.z);
	const float3 up = (float3) (up_t.x, up_t.y, up_t.z);
	const float3 down = (float3) (down_t.x, down_t.y, down_t.z);

	struct float3 outVal;
	if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
		outVal.x = INVALID;
	} else {
		const float3 dxv = right - left;
		const float3 dyv = down - up;
		float3 outVec = normalize(cross(dyv, dxv));
		outVal.x = outVec.x;
		outVal.y = outVec.y;
		outVal.z = outVec.z;
	}
	return outVal;
}

inline struct float3 depth2vertex_core(const uint x, const uint y,
                                 const uint imageSize_x, const uint imageSize_y,
                                 __global const float *depth,
                                 const Matrix4 invK)
{
	float3 vertex_vec = (float3)(0.0f);
	float depth_val = depth[x + y * imageSize_x];
	if (depth_val > 0) {
		vertex_vec = depth_val * v_rotate(invK, (float3)(x, y, 1.0f));
	}
	struct float3 vertex_val;
	vertex_val.x = vertex_vec.x;
	vertex_val.y = vertex_vec.y;
	vertex_val.z = vertex_vec.z;

	return vertex_val;
}

inline void integrateKernel_core(const uint vol_size_x, const uint vol_size_y,
                                 const uint vol_size_z, const struct float3 vol_dim,
                                 __global short2s *vol_data,
                                 const uint x, const uint y,
                                 const uint depthSize_x, const uint depthSize_y,
                                 __global const float *depth,
                                 const Matrix4 invTrack, const Matrix4 K,
                                 const float mu, const float maxweight,
                                 const struct float3 delta, const struct float3 cameraDelta)
{
	float3 vol_pos = (float3) ((x + 0.5f) * vol_dim.x / vol_size_x,
	                           (y + 0.5f) * vol_dim.y / vol_size_y,
	                           (0 + 0.5f) * vol_dim.z / vol_size_z);
	float3 pos = mul_Matrix4_float3v(invTrack, vol_pos);
	float3 cameraX = mul_Matrix4_float3v(K, pos);
	float3 deltaVec = (float3)(delta.x, delta.y, delta.z);
	float3 cameraDeltaVec = (float3)(cameraDelta.x, cameraDelta.y, cameraDelta.z);

	for (unsigned int z = 0; z < vol_size_z; ++z) {
		if (pos.z >= 0.0001f) {
			const float pixel_x = cameraX.x / cameraX.z + 0.5f;
			const float pixel_y = cameraX.y / cameraX.z + 0.5f;
			if ( !(pixel_x < 0 || pixel_x > depthSize_x - 1 ||
			       pixel_y < 0 || pixel_y > depthSize_y - 1) ) {
				unsigned int px_x;
				unsigned int px_y;
				px_x = pixel_x;
				px_y = pixel_y;
				float depthpx = depth[px_y*depthSize_x + px_x];
				if (depthpx != 0) {
					const float tempfx = c_sq(pos.x / pos.z);
					const float tempfy = c_sq(pos.y / pos.z);
					const float diff = (depthpx - cameraX.z) * sqrt(1 + tempfx + tempfy);
					if (diff > -mu) {
						const float sdf = fmin(1.0f, diff / mu);
						float2 data = getVolume(vol_size_x, vol_size_y, vol_size_z, vol_data, x, y, z);
						data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.0f, 1.0f);
						data.y = fmin(data.y + 1, maxweight);
						setVolume(vol_size_x, vol_size_y, vol_size_z, vol_data, x, y, z, data);
					}
				}
			}
		}
		pos += deltaVec;
		cameraX += cameraDeltaVec;
	}
}

inline float bilateralFilter_core(int x, int y, int size_x, int size_y, int r,
                                  int gaussianS, float e_d,
                                  __global float *in, __global float *gaussian)
{
	float sum = 0.0f;
	float t = 0.0f;

	const float center = in[y*size_x + x];
	for (int i = -r; i <= r; ++i) {
		for (int j = -r; j <= r; ++j) {
			uint2 curPos = (uint2)(clamp(x + i, 0, size_x - 1), clamp(y + j, 0, size_y - 1));
			const float curPix = in[curPos.y*size_x + curPos.x];
			if (curPix > 0) {
				const float mod = c_sq(curPix - center);
				const float factor = gaussian[i + r] * gaussian[j + r] * exp(-mod / (2 * e_d * e_d));
				t += factor * curPix;
				sum += factor;
			}
		}
	}
	return t / sum;
}

inline void reduce_core(__global float *sums, TrackData row)
{
	if (row.result < 1) {
		sums[29] += row.result == -4 ? 1 : 0;
		sums[30] += row.result == -5 ? 1 : 0;
		sums[31] += row.result > -4 ? 1 : 0;
	} else {
		sums[0] += row.error * row.error;
		sums[1] += row.error * row.J[0];
		sums[2] += row.error * row.J[1];
		sums[3] += row.error * row.J[2];
		sums[4] += row.error * row.J[3];
		sums[5] += row.error * row.J[4];
		sums[6] += row.error * row.J[5];
		sums[7] += row.J[0] * row.J[0];
		sums[8] += row.J[0] * row.J[1];
		sums[9] += row.J[0] * row.J[2];
		sums[10] += row.J[0] * row.J[3];
		sums[11] += row.J[0] * row.J[4];
		sums[12] += row.J[0] * row.J[5];
		sums[13] += row.J[1] * row.J[1];
		sums[14] += row.J[1] * row.J[2];
		sums[15] += row.J[1] * row.J[3];
		sums[16] += row.J[1] * row.J[4];
		sums[17] += row.J[1] * row.J[5];
		sums[18] += row.J[2] * row.J[2];
		sums[19] += row.J[2] * row.J[3];
		sums[20] += row.J[2] * row.J[4];
		sums[21] += row.J[2] * row.J[5];
		sums[22] += row.J[3] * row.J[3];
		sums[23] += row.J[3] * row.J[4];
		sums[24] += row.J[3] * row.J[5];
		sums[25] += row.J[4] * row.J[4];
		sums[26] += row.J[4] * row.J[5];
		sums[27] += row.J[5] * row.J[5];
		sums[28] += 1;
	}
}
