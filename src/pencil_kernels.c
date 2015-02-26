typedef unsigned int uint;
typedef unsigned short ushort;

/* OpenCL vector type definitions for PENCIL. */
struct float3 {
	float x;
	float y;
	float z;
};

struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct uchar3 {
	unsigned char x;
	unsigned char y;
	unsigned char z;
};

struct short2 {
	short x;
	short y;
};

struct float2 {
	float x;
	float y;
};

struct uint2 {
	uint x;
	uint y;
};

struct Matrix4 {
	struct float4 data[4];
};

struct TrackData {
	int result;
	float error;
	float J[6];
};

typedef struct float3 float3;
typedef struct float4 float4;
typedef struct uchar3 uchar3;
typedef struct short2 short2;
typedef struct float2 float2;
typedef struct uint2 uint2;
typedef struct Matrix4 Matrix4;
typedef struct TrackData TrackData;

static float3 make_float3(float x, float y, float z) {
	float3 ret;
	ret.x = x;
	ret.y = y;
	ret.z = z;
	return ret;
}

static float3 c_rotate(const Matrix4 M, const float3 v)
{
	return make_float3(M.data[0].x * v.x + M.data[0].y * v.y + M.data[0].z * v.z,
	                   M.data[1].x * v.x + M.data[1].y * v.y + M.data[1].z * v.z,
	                   M.data[2].x * v.x + M.data[2].y * v.y + M.data[2].z * v.z);
}

void bilateralFilter_core_summary(int x, int y, int size_x, int size_y,
                                  int r, int gaussianS, float e_d,
                                  const float in[restrict const static size_y][size_x],
                                  const float gaussian[restrict const static gaussianS])
{
	const float center = in[y][x];
	for (int i = 0; i <= gaussianS; ++i) {
		const float factor = gaussian[i + r];
	}
}

float bilateralFilter_core(int x, int y, int size_x, int size_y,
                           int r, int gaussianS, float e_d,
                           const float in[restrict const static size_y][size_x],
                           const float gaussian[restrict const static gaussianS])
      __attribute__((pencil_access(bilateralFilter_core_summary)));

void integrateKernel_core_summary(const uint vol_size_x, const uint vol_size_y,
                                  const uint vol_size_z, const float3 vol_dim,
                                  short2 vol_data[restrict const static vol_size_z][vol_size_y][vol_size_x],
                                  const uint x, const uint y, uint depthSize_x, uint depthSize_y,
                                  const float depth[restrict const static depthSize_y][depthSize_x],
                                  const Matrix4 invTrack, const Matrix4 K,
                                  const float mu, const float maxweight,
                                  const float3 delta, const float3 cameraDelta)
{
	for (int z = 0; z <= vol_size_z; ++z) {
		const float depthVal = depth[y][x];
		const short2 volVal = vol_data[z][y][x];
		vol_data[z][y][x] = volVal;
	}
	for (int i = 0; i < depthSize_y; i++)
	{
		for (int j = 0; j < depthSize_x; ++j)
		{
			const float val = depth[i][j];
		}
	}
}

void integrateKernel_core(const uint vol_size_x, const uint vol_size_y,
                          const uint vol_size_z, const float3 vol_dim,
                          short2 vol_data[restrict const static vol_size_z][vol_size_y][vol_size_x],
                          const uint x, const uint y, uint depthSize_x, uint depthSize_y,
                          const float depth[restrict const static depthSize_y][depthSize_x],
                          const Matrix4 invTrack, const Matrix4 K,
                          const float mu, const float maxweight,
                          const float3 delta, const float3 cameraDelta)
     __attribute__((pencil_access(integrateKernel_core_summary)));

void initVolume_core_summary(const uint x, const uint y, const uint z,
                             const uint v_size_x, const uint v_size_y, const uint v_size_z,
                             short2 v_data[restrict const static v_size_z][v_size_y][v_size_x],
                             const float dxVal, const float dyVal)
{
	short2 temp;
	temp.x = dxVal;
	temp.y = dyVal;
	v_data[z][y][x] = temp;
}

void initVolume_core(const uint x, const uint y, const uint z,
                     const uint v_size_x, const uint v_size_y, const uint v_size_z,
                     short2 v_data[restrict const static v_size_z][v_size_y][v_size_x],
                     const float dxVal, const float dyVal)
     __attribute__((pencil_access(initVolume_core_summary)));
void depth2vertex_core_summary(uint x, uint y, uint imageSize_x, uint imageSize_y,
                               const float depth[restrict const static imageSize_y][imageSize_x],
                               const Matrix4 invK)
{
	const float depth_val = depth[y][x];
}

float3 depth2vertex_core(const uint x, const uint y,
                         const uint imageSize_x, const uint imageSize_y,
                         const float depth[restrict const static imageSize_y][imageSize_x],
                         const Matrix4 invK)
       __attribute__((pencil_access(depth2vertex_core_summary)));

void vertex2normal_core_summary(const uint x, const uint y,
                                const uint imageSize_x, const uint imageSize_y,
                                const float3 in[restrict const static imageSize_y][imageSize_x])
{
	const float3 left = in[y][x];
}
float3 vertex2normal_core(const uint x, const uint y,
                          const uint imageSize_x, const uint imageSize_y,
                          const float3 in[restrict const static imageSize_y][imageSize_x])
       __attribute__((pencil_access(vertex2normal_core_summary)));

void halfSampleRobustImage_core_summary(const uint x, const uint y,
                                        const uint outSize_x, const uint outSize_y,
                                        const uint inSize_x, const uint inSize_y,
                                        const float in[restrict const static inSize_y][inSize_x],
                                        const float e_d, const int r)
{
	const float center = in[2*y][2*x];
}

float halfSampleRobustImage_core(const uint x, const uint y,
                                 const uint outSize_x, const uint outSize_y,
                                 const uint inSize_x, const uint inSize_y,
                                 const float in[restrict const static inSize_y][inSize_x],
                                 const float e_d, const int r)
      __attribute__((pencil_access(halfSampleRobustImage_core_summary)));

void renderNormal_core_summary(const uint x, const uint y,
                               const uint normalSize_x, const uint normalSize_y,
                               const float3 normal[restrict const static normalSize_y][normalSize_x])
{
	const float3 n = normal[y][x];
}

uchar3 renderNormal_core(const uint x, const uint y,
                         const uint normalSize_x, const uint normalSize_y,
                         const float3 normal[restrict const static normalSize_y][normalSize_x])
       __attribute__((pencil_access(renderNormal_core_summary)));

void renderDepth_core_summary(const uint x, const uint y,
                              const uint depthSize_x, const uint depthSize_y,
                              const float depth[restrict const static depthSize_y][depthSize_x],
                              const float nearPlane, const float farPlane,
                              const float rangeScale)
{
	const float d = depth[y][x];
}

uchar3 renderDepth_core(const uint x, const uint y,
                        const uint depthSize_x, const uint depthSize_y,
                        const float depth[restrict const static depthSize_y][depthSize_x],
                        const float nearPlane, const float farPlane,
                        const float rangeScale)
       __attribute__((pencil_access(renderDepth_core_summary)));

void renderTrack_core_summary(const uint x, const uint y,
                              const uint outSize_x, const uint outSize_y,
                              const TrackData data[restrict const static outSize_y][outSize_x])
{
	int test = data[y][x].result;
}

uchar3 renderTrack_core(const uint x, const uint y,
                        const uint outSize_x, const uint outSize_y,
                        const TrackData data[restrict const static outSize_y][outSize_x])
       __attribute__((pencil_access(renderTrack_core_summary)));

void renderVolume_core_summary(const uint x, const uint y,
                               const uint volume_size_x, const uint volume_size_y, const uint volume_size_z,
                               const short2 volume_data[restrict const static volume_size_z][volume_size_y][volume_size_x],
                               const float3 volume_dim, const Matrix4 view,
                               const float nearPlane, const float farPlane,
                               const float step, const float largestep,
                               const float3 light, const float3 ambient)
{
	const short2 d = volume_data[x+y][y][x];
}

uchar3 renderVolume_core(const uint x, const uint y,
                         const uint volume_size_x, const uint volume_size_y, const uint volume_size_z,
                         const short2 volume_data[restrict const static volume_size_z][volume_size_y][volume_size_x],
                         const float3 volume_dim, const Matrix4 view,
                         const float nearPlane, const float farPlane,
                         const float step, const float largestep,
                         const float3 light, const float3 ambient)
       __attribute__((pencil_access(renderVolume_core_summary)));

void raycast_core_summary(const uint x, const uint y,
                          const uint inputSize_x, const uint inputSize_y,
                          float3 vertex[restrict const static inputSize_y][inputSize_x],
                          float3 normal[restrict const static inputSize_y][inputSize_x],
                          const uint integration_size_x, const uint integration_size_y, const uint integration_size_z,
                          const short2 integration_data[restrict const static integration_size_z][integration_size_y][integration_size_x],
                          const float3 integration_dim, const Matrix4 view,
                          const float nearPlane, const float farPlane,
                          const float step, const float largestep)
{
	const float3 test;
	const float3 norm;
	const short2 integrVal = integration_data[x+y][y][x];
	vertex[y][x] = test;
	normal[y][x] = norm;
}

void raycast_core(const uint x, const uint y,
                  const uint inputSize_x, const uint inputSize_y,
                  float3 vertex[restrict const static inputSize_y][inputSize_x],
                  float3 normal[restrict const static inputSize_y][inputSize_x],
                  const uint integration_size_x, const uint integration_size_y, const uint integration_size_z,
                  const short2 integration_data[restrict const static integration_size_z][integration_size_y][integration_size_x],
                  const float3 integration_dim, const Matrix4 view,
                  const float nearPlane, const float farPlane,
                  const float step, const float largestep)
     __attribute__((pencil_access(raycast_core_summary)));

void track_core_summary(uint refSize_x, uint refSize_y, const TrackData output,
                        const float3 inVertex, const float3 inNormal,
                        const float3 refVertex[restrict const static refSize_y][refSize_x],
                        const float3 refNormal[restrict const static refSize_y][refSize_x],
                        const Matrix4 Ttrack, const Matrix4 view,
                        const float dist_threshold,
                        const float normal_threshold)
{
	const uint refx;
	const uint refy;
	const float3 vertVal = refVertex[refy][refx];
	const float3 normVal = refNormal[refy][refx];
}

TrackData track_core(uint refSize_x, uint refSize_y, const TrackData output,
                     const float3 inVertex, const float3 inNormal,
                     const float3 refVertex[restrict const static refSize_y][refSize_x],
                     const float3 refNormal[restrict const static refSize_y][refSize_x],
                     const Matrix4 Ttrack, const Matrix4 view,
                     const float dist_threshold,
                     const float normal_threshold)
          __attribute__((pencil_access(track_core_summary)));

void reduce_core_summary(float sums[restrict const static 32], TrackData row)
{
	for (int z = 0; z < 32; ++z) {
		const float adjustVal = row.J[z/6];
		const float tempVal = sums[z];
		sums[z] = adjustVal + tempVal;
	}
}
void reduce_core(float sums[restrict const static 32], TrackData row)
     __attribute__((pencil_access(reduce_core_summary)));


int mm2meters_pencil(uint outSize_x, uint outSize_y,
					 float out[restrict const static outSize_y][outSize_x],
					 uint inSize_x, uint inSize_y,
					 const ushort in[restrict const static inSize_y][inSize_x],
					 int ratio)
{
#pragma scop
	{
		__pencil_assume(outSize_y < 960);
		__pencil_assume(outSize_x < 1280);
		__pencil_assume(outSize_y % 120 == 0);
		__pencil_assume(outSize_x % 160 == 0);
		__pencil_assume(outSize_x > 0);
		__pencil_assume(outSize_y > 0);
		__pencil_assume(inSize_x > 0);
		__pencil_assume(inSize_y > 0);
		for (uint y = 0; y < outSize_y; y++) {
			for (uint x = 0; x < outSize_x; x++) {
				int xr = x * ratio;
				int yr = y * ratio;
				out[y][x] = in[yr][xr] / 1000.0f;
			}
		}
	}
#pragma endscop
	return 0;
}

int bilateralFilter_pencil(int size_x, int size_y,
						   float out[restrict const static size_y][size_x],
						   const float in[restrict const static size_y][size_x],
						   uint2 size, int gaussianS,
						   const float gaussian[restrict const static gaussianS],
						   float e_d, int r)
{
#pragma scop
	{
		__pencil_assume(size_y < 960);
		__pencil_assume(size_x < 1280);
		__pencil_assume(size_y % 120 == 0);
		__pencil_assume(size_x % 160 == 0);
		__pencil_assume(size_x > 0);
		__pencil_assume(size_y > 0);
		__pencil_assume(r > 0);
		__pencil_assume(r < 16);
		for (uint y = 0; y < size_y; y++) {
			for (uint x = 0; x < size_x; x++) {
				if (in[y][x] == 0) {
					out[y][x] = 0;
				}
				else {
					out[y][x] = bilateralFilter_core(x, y, size_x, size_y, r,
													 gaussianS, e_d, in, gaussian);
				}
			}
		}
	}
#pragma endscop
	return 0;
}

int initVolume_pencil(const uint v_size_x, const uint v_size_y, const uint v_size_z,
					  short2 v_data[restrict const static v_size_z][v_size_y][v_size_x],
					  const float2 d)
{
#pragma scop
	{
		__pencil_assume(v_size_x < 1024);
		__pencil_assume(v_size_y < 1024);
		__pencil_assume(v_size_z < 1024);
		__pencil_assume(v_size_x % 256 == 0);
		__pencil_assume(v_size_y % 256 == 0);
		__pencil_assume(v_size_z % 256 == 0);
		__pencil_assume(v_size_x > 0);
		__pencil_assume(v_size_y > 0);
		__pencil_assume(v_size_z > 0);
		for (unsigned int x = 0; x < v_size_x; x++) {
			for (unsigned int y = 0; y < v_size_y; y++) {
				for (unsigned int z = 0; z < v_size_z; z++) {
					initVolume_core(x, y, z, v_size_x, v_size_y, v_size_z, v_data, (d.x * 32766.0f), d.y);
				}
			}
		}
	}
#pragma endscop
	return 0;
}

int integrateKernel_pencil(const uint vol_size_x, const uint vol_size_y,
						   const uint vol_size_z, const float3 vol_dim,
						   short2 vol_data[restrict const static vol_size_z][vol_size_y][vol_size_x],
						   uint depthSize_x, uint depthSize_y,
						   const float depth[restrict const static depthSize_y][depthSize_x],
						   const Matrix4 invTrack, const Matrix4 K,
						   const float mu, const float maxweight)
{
	const float3 delta = c_rotate(invTrack,
								  make_float3(0, 0, vol_dim.z / vol_size_z));
	const float3 cameraDelta = c_rotate(K, delta);
#pragma scop
	{
		__pencil_assume(vol_size_x < 1024);
		__pencil_assume(vol_size_y < 1024);
		__pencil_assume(vol_size_z < 1024);
		__pencil_assume(vol_size_x % 256 == 0);
		__pencil_assume(vol_size_y % 256 == 0);
		__pencil_assume(vol_size_z % 256 == 0);
		__pencil_assume(depthSize_x > 0);
		__pencil_assume(depthSize_y > 0);
		__pencil_assume(vol_size_x > 0);
		__pencil_assume(vol_size_y > 0);
		__pencil_assume(vol_size_z > 0);
		for (unsigned int y = 0; y < vol_size_y; y++) {
			for (unsigned int x = 0; x < vol_size_x; x++) {
				integrateKernel_core(vol_size_x, vol_size_y, vol_size_z, vol_dim,
									 vol_data, x, y, depthSize_x, depthSize_y, depth,
									 invTrack, K, mu, maxweight, delta, cameraDelta);
			}
		}
	}
#pragma endscop
	return 0;
}

int depth2vertex_pencil(uint imageSize_x, uint imageSize_y,
						float3 vertex[restrict const static imageSize_y][imageSize_x],
						const float depth[restrict const static imageSize_y][imageSize_x],
						const Matrix4 invK)
{
#pragma scop
	{
		__pencil_assume(imageSize_y < 960);
		__pencil_assume(imageSize_x < 1280);
		__pencil_assume(imageSize_y % 60 == 0);
		__pencil_assume(imageSize_x % 80 == 0);
		__pencil_assume(imageSize_x > 0);
		__pencil_assume(imageSize_y > 0);
		for (unsigned int y = 0; y < imageSize_y; y++) {
			for (unsigned int x = 0; x < imageSize_x; x++) {
				vertex[y][x] = depth2vertex_core(x, y, imageSize_x,
												 imageSize_y, depth, invK);
			}
		}
	}
#pragma endscop
	return 0;
}

int vertex2normal_pencil(uint imageSize_x, uint imageSize_y,
						 float3 out[restrict const static imageSize_y][imageSize_x],
						 const float3 in[restrict const static imageSize_y][imageSize_x])
{
#pragma scop
	{
		__pencil_assume(imageSize_y < 960);
		__pencil_assume(imageSize_x < 1280);
		__pencil_assume(imageSize_y % 60 == 0);
		__pencil_assume(imageSize_x % 80 == 0);
		__pencil_assume(imageSize_x > 0);
		__pencil_assume(imageSize_y > 0);
		for (unsigned int y = 0; y < imageSize_y; y++) {
			for (unsigned int x = 0; x < imageSize_x; x++) {
				out[y][x] = vertex2normal_core(x, y, imageSize_x, imageSize_y, in);
			}
		}
	}
#pragma endscop
	return 0;
}

int halfSampleRobustImage_pencil(uint outSize_x, uint outSize_y,
								 uint inSize_x, uint inSize_y,
								 float out[restrict const static outSize_y][outSize_x],
								 const float in[restrict const static inSize_y][inSize_x],
								 const float e_d, const int r)
{
#pragma scop
	{
		__pencil_assume(outSize_y < 960);
		__pencil_assume(outSize_x < 1280);
		__pencil_assume(outSize_y % 60 == 0);
		__pencil_assume(outSize_x % 80 == 0);
		__pencil_assume(outSize_x > 0);
		__pencil_assume(outSize_y > 0);
		for (unsigned int y = 0; y < outSize_y; y++) {
			for (unsigned int x = 0; x < outSize_x; x++) {
				out[y][x] = halfSampleRobustImage_core(x, y, outSize_x, outSize_y,
													   inSize_x, inSize_y, in, e_d, r);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderNormal_pencil(uint normalSize_x, uint normalSize_y,
						uchar3 out[restrict const static normalSize_y][normalSize_x],
						const float3 normal[restrict const static normalSize_y][normalSize_x])
{
#pragma scop
	{
		__pencil_assume(normalSize_y < 960);
		__pencil_assume(normalSize_x < 1280);
		__pencil_assume(normalSize_y % 120 == 0);
		__pencil_assume(normalSize_x % 160 == 0);
		__pencil_assume(normalSize_x > 0);
		__pencil_assume(normalSize_y > 0);
		for (unsigned int y = 0; y < normalSize_y; y++) {
			for (unsigned int x = 0; x < normalSize_x; x++) {
				out[y][x] = renderNormal_core(x, y, normalSize_x, normalSize_y, normal);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderDepth_pencil(uint depthSize_x, uint depthSize_y,
					   uchar3 out[restrict const static depthSize_y][depthSize_x],
					   const float depth[restrict const static depthSize_y][depthSize_x],
					   const float nearPlane, const float farPlane)
{
	float rangeScale = 1 / (farPlane - nearPlane);
#pragma scop
	{
		__pencil_assume(depthSize_y < 960);
		__pencil_assume(depthSize_x < 1280);
		__pencil_assume(depthSize_y % 120 == 0);
		__pencil_assume(depthSize_x % 160 == 0);
		__pencil_assume(depthSize_x > 0);
		__pencil_assume(depthSize_y > 0);
		for (unsigned int y = 0; y < depthSize_y; y++) {
			for (unsigned int x = 0; x < depthSize_x; x++) {
				out[y][x] = renderDepth_core(x, y, depthSize_x, depthSize_y,
											 depth, nearPlane, farPlane, rangeScale);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderTrack_pencil(uint outSize_x, uint outSize_y,
					   uchar3 out[restrict const static outSize_y][outSize_x],
					   const TrackData data[restrict const static outSize_y][outSize_x])
{
#pragma scop
	{
		__pencil_assume(outSize_y < 960);
		__pencil_assume(outSize_x < 1280);
		__pencil_assume(outSize_y % 120 == 0);
		__pencil_assume(outSize_x % 160 == 0);
		__pencil_assume(outSize_x > 0);
		__pencil_assume(outSize_y > 0);
		for (unsigned int y = 0; y < outSize_y; y++) {
			for (unsigned int x = 0; x < outSize_x; x++) {
				out[y][x] = renderTrack_core (x, y, outSize_x, outSize_y, data);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderVolume_pencil(uint depthSize_x, uint depthSize_y,
						uchar3 out[restrict const static depthSize_y][depthSize_x],
						const uint volume_size_x, const uint volume_size_y, const uint volume_size_z,
						const short2 volume_data[restrict const static volume_size_z][volume_size_y][volume_size_x],
						const float3 volume_dim, const Matrix4 view,
						const float nearPlane, const float farPlane,
						const float step, const float largestep,
						const float3 light, const float3 ambient)
{
#pragma scop
	{
		__pencil_assume(depthSize_y < 960);
		__pencil_assume(depthSize_x < 1280);
		__pencil_assume(depthSize_y % 120 == 0);
		__pencil_assume(depthSize_x % 160 == 0);
		__pencil_assume(depthSize_x > 0);
		__pencil_assume(depthSize_y > 0);
		for (unsigned int y = 0; y < depthSize_y; y++) {
			for (unsigned int x = 0; x < depthSize_x; x++) {
				out[y][x] = renderVolume_core(x, y, volume_size_x, volume_size_y,
											  volume_size_z, volume_data, volume_dim,
											  view, nearPlane, farPlane, step,
											  largestep, light, ambient);
			}
		}
	}
#pragma endscop
	return 0;
}

int raycast_pencil(uint inputSize_x, uint inputSize_y,
				   float3 vertex[restrict const static inputSize_y][inputSize_x],
				   float3 normal[restrict const static inputSize_y][inputSize_x],
				   const uint integration_size_x, const uint integration_size_y, const uint integration_size_z,
				   const short2 integration_data[restrict const static integration_size_z][integration_size_y][integration_size_x],
				   const float3 integration_dim, const Matrix4 view,
				   const float nearPlane, const float farPlane,
				   const float step, const float largestep)
{
#pragma scop
	{
		__pencil_assume(inputSize_y < 960);
		__pencil_assume(inputSize_x < 1280);
		__pencil_assume(inputSize_y % 120 == 0);
		__pencil_assume(inputSize_x % 160 == 0);
		__pencil_assume(inputSize_x > 0);
		__pencil_assume(inputSize_y > 0);
		for (unsigned int y = 0; y < inputSize_y; y++) {
			for (unsigned int x = 0; x < inputSize_x; x++) {
				raycast_core(x, y, inputSize_x, inputSize_y, vertex, normal,
							 integration_size_x, integration_size_y, integration_size_z,
							 integration_data, integration_dim, view, nearPlane,
							 farPlane, step, largestep);
			}
		}
	}
#pragma endscop
	return 0;
}

int track_pencil(uint refSize_x, uint refSize_y, uint inSize_x, uint inSize_y,
				 TrackData output[restrict const static refSize_y][refSize_x],
				 const float3 inVertex[restrict const static inSize_y][inSize_x],
				 const float3 inNormal[restrict const static inSize_y][inSize_x],
				 const float3 refVertex[restrict const static refSize_y][refSize_x],
				 const float3 refNormal[restrict const static refSize_y][refSize_x],
				 const Matrix4 Ttrack, const Matrix4 view,
				 const float dist_threshold, const float normal_threshold)
{
#pragma scop
	{
		__pencil_assume(inSize_y < 960);
		__pencil_assume(inSize_x < 1280);
		__pencil_assume(inSize_y % 60 == 0);
		__pencil_assume(inSize_x % 80 == 0);
		__pencil_assume(inSize_x > 0);
		__pencil_assume(inSize_y > 0);
		for (unsigned int y = 0; y < inSize_y; y++) {
			for (unsigned int x = 0; x < inSize_x; x++) {
				output[y][x] = track_core(refSize_x, refSize_y, output[y][x],
										  inVertex[y][x], inNormal[y][x],
										  refVertex, refNormal, Ttrack, view,
										  dist_threshold, normal_threshold);
			}
		}
	}
#pragma endscop
	return 0;
}

int reduce_pencil(float sums[restrict const static 8][32], const uint Jsize_x, const uint Jsize_y,
				  TrackData J[restrict const static Jsize_y][Jsize_x],
				  const uint size_x, const uint size_y)
{
#pragma scop
	{
		__pencil_assume(size_y < 960);
		__pencil_assume(size_x < 1280);
		__pencil_assume(Jsize_y % 120 == 0);
		__pencil_assume(Jsize_x % 160 == 0);
		__pencil_assume(size_y % 60 == 0);
		__pencil_assume(size_x % 80 == 0);
		__pencil_assume(size_y > 0);
		__pencil_assume(size_x > 0);

		float intrmdSums[size_x][8][32];

		for (uint blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (uint i = 0; i < 32; ++i) {
				sums[blockIndex][i] = 0;
				for (uint x = 0; x < size_x; x++) {
					intrmdSums[x][blockIndex][i] = 0;
				}
			}
		}
		for (uint blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (uint y = blockIndex; y < size_y; y += 8) {
				for (uint x = 0; x < size_x; x++) {
					reduce_core (intrmdSums[x][blockIndex], J[y][x]);
				}
			}
		}
		for (uint blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (uint i = 0; i < 32; ++i) {
				for (uint x = 0; x < size_x; x++) {
					sums[blockIndex][i] += intrmdSums[x][blockIndex][i];
				}
			}
		}
	}
#pragma endscop
	return 0;
}
