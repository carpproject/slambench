#include <kernels.h>
#include <stdio.h>
#include <string.h>

#include <pencil_runtime.h>

#define TICK()    {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}
#define TOCK(str,size)  if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
	if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
	std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}

extern "C" {
	int bilateralFilter_pencil(int size_x, int size_y, float *out, const float *in, uint2 size, int gaussianS, const float *gaussian, float e_d, int r);

	int mm2meters_pencil(uint outSize_x, uint outSize_y, float *out, uint inSize_x, uint inSize_y, const ushort *in, int ratio);

	int initVolume_pencil(const uint v_size_x, const uint v_size_y, const uint v_size_z, short2 *v_data, const float2 d);

	int integrateKernel_pencil(const uint vol_size_x, const uint vol_size_y, const uint vol_size_z, const float3 vol_dim, short2 *vol_data, uint depthSize_x,
	                           uint depthSize_y, const float *depth, const Matrix4 invTrack, const Matrix4 K, const float mu, const float maxweight);

	int depth2vertex_pencil(uint imageSize_x, uint imageSize_y, float3 *vertex, const float *depth, const Matrix4 invK);

	int vertex2normal_pencil(uint imageSize_x, uint imageSize_y, float3 *out, const float3 *in);

	int track_pencil(uint refSize_x, uint refSize_y, uint inSize_x, uint inSize_y, TrackData *output, const float3 *inVertex,
	                 const float3 *inNormal, const float3 *refVertex, const float3 *refNormal, const Matrix4 Ttrack,
	                 const Matrix4 view, const float dist_threshold,	const float normal_threshold);

	int halfSampleRobustImage_pencil(uint outSize_x, uint outSize_y, uint inSize_x, uint inSize_y, float *out, const float *in, const float e_d, const int r);

	int renderNormal_pencil(uint normalSize_x, uint normalSize_y, uchar3 *out, const float3 *normal);

	int renderDepth_pencil(uint depthSize_x, uint depthSize_y, uchar3 *out, const float *depth, const float nearPlane, const float farPlane);

	int renderTrack_pencil(uint outSize_x, uint outSize_y, uchar3 *out, const TrackData *data);

	int renderVolume_pencil(uint depthSize_x, uint depthSize_y, uchar3 *out, const uint volume_size_x, const uint volume_size_y, const uint volume_size_z,
	                        const short2 *volume_data, const float3 volume_dim, const Matrix4 view, const float nearPlane, const float farPlane,
	                        const float step, const float largestep, const float3 light, const float3 ambient);

	int raycast_pencil(uint inputSize_x, uint inputSize_y, float3 *vertex, float3 *normal, const uint integration_size_x, const uint integration_size_y,
	                   const uint integration_size_z, const short2 *integration_data, const float3 integration_dim, const Matrix4 view,
	                   const float nearPlane, const float farPlane, const float step, const float largestep);

	int reduce_pencil(float *sums, const uint Jsize_x, const uint Jsize_y, TrackData *J, const uint size_x, const uint size_y);
}

float * gaussian;

Volume volume;
float3 *vertex;
float3 *normal;

TrackData *trackingResult;
float *reductionoutput;
float **ScaledDepth;
float *floatDepth;
Matrix4 oldPose;
Matrix4 raycastPose;
float3 **inputVertex;
float3 **inputNormal;

bool print_kernel_timing = false;
struct timespec tick_clockData;
struct timespec tock_clockData;

void Kfusion::languageSpecificConstructor()
{
	if (getenv("KERNEL_TIMINGS"))
		print_kernel_timing = true;

	pencil_init(PENCIL_TARGET_DEVICE_DYNAMIC);

	// internal buffers to initialize
	size_t reductionoutput_size = sizeof(float) * 8 * 32;
	reductionoutput = (float*) pencil_alloc(reductionoutput_size);
	memset(reductionoutput, 0, reductionoutput_size);

	ScaledDepth = (float**)  malloc(sizeof(float*)  * iterations.size());
	inputVertex = (float3**) malloc(sizeof(float3*) * iterations.size());
	inputNormal = (float3**) malloc(sizeof(float3*) * iterations.size());

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		size_t size = (computationSize.x * computationSize.y) / (int) pow(2, i);
		ScaledDepth[i] = (float*)  pencil_alloc(sizeof(float)  * size);
		memset(ScaledDepth[i], 0, sizeof(float) * size);

		inputVertex[i] = (float3*) pencil_alloc(sizeof(float3) * size);
		memset(inputVertex[i], 0, sizeof(float3) * size);

		inputNormal[i] = (float3*) pencil_alloc(sizeof(float3) * size);
		memset(inputNormal[i], 0, sizeof(float3) * size);
	}

	size_t size = computationSize.x * computationSize.y;
	floatDepth     = (float*)     pencil_alloc(sizeof(float)     * size);
	vertex         = (float3*)    pencil_alloc(sizeof(float3)    * size);
	normal         = (float3*)    pencil_alloc(sizeof(float3)    * size);
	trackingResult = (TrackData*) pencil_alloc(sizeof(TrackData) * size);

	memset(floatDepth, 0, sizeof(float) * size);
	memset(vertex, 0, sizeof(float3) * size);
	memset(normal, 0, sizeof(float3) * size);
	memset(trackingResult, 0, sizeof(TrackData) * size);

	// Start generating the gaussian.
	size_t gaussianS = radius * 2 + 1;
	gaussian = (float*) pencil_alloc(gaussianS * sizeof(float));
	int x;
	for (unsigned int i = 0; i < gaussianS; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}
	// Done generating the gaussian.

	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion()
{
	pencil_free(reductionoutput);
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		pencil_free(ScaledDepth[i]);
		pencil_free(inputVertex[i]);
		pencil_free(inputNormal[i]);
	}
	free(ScaledDepth);
	free(inputVertex);
	free(inputNormal);

	pencil_free(vertex);
	pencil_free(normal);
	pencil_free(gaussian);
	pencil_free(floatDepth);
	pencil_free(trackingResult);

	pencil_shutdown();

	volume.release();
}

void Kfusion::reset()
{
	initVolumeKernel(volume);
}

void init() {};

void clean() {};

void initVolumeKernel(Volume volume)
{
	TICK();
	initVolume_pencil(volume.size.x, volume.size.y, volume.size.z,
	                  volume.data, make_float2(1.0f, 0.0f));
	TOCK("initVolumeKernel", volume.size.x * volume.size.y * volume.size.z);
}

void bilateralFilterKernel(float *out, const float *in, uint2 size,
                           const float *gaussian, float e_d, int r)
{
	TICK();
	bilateralFilter_pencil(size.x, size.y, out, in, size,
	                       (radius * 2 + 1), gaussian, e_d, r);
	TOCK("bilateralFilterKernel", size.x * size.y);
}

void depth2vertexKernel(float3 *vertex, const float *depth,
                        uint2 imageSize, const Matrix4 invK)
{
	TICK();
	depth2vertex_pencil(imageSize.x, imageSize.y, vertex, depth, invK);
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void vertex2normalKernel(float3 *out, const float3 *in, uint2 imageSize)
{
	TICK();
	vertex2normal_pencil(imageSize.x, imageSize.y, out, in);
	TOCK("vertex2normalKernel", imageSize.x * imageSize.y);
}

void trackKernel(TrackData* output, const float3* inVertex,
                 const float3* inNormal, uint2 inSize, const float3* refVertex,
                 const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
                 const Matrix4 view, const float dist_threshold,
                 const float normal_threshold)
{
	TICK();
	track_pencil(refSize.x, refSize.y, inSize.x, inSize.y, output,
	             inVertex, inNormal, refVertex, refNormal, Ttrack,
	             view, dist_threshold, normal_threshold);
	TOCK("trackKernel", inSize.x * inSize.y);
}

void mm2metersKernel(float * out, uint2 outSize,
                     const ushort * in, uint2 inSize)
{
	TICK();
	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;

	mm2meters_pencil(outSize.x, outSize.y, out, inSize.x, inSize.y, in, ratio);
	TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 outSize,
                                 const float e_d, const int r)
{
	TICK();
	halfSampleRobustImage_pencil(outSize.x, outSize.y, 2*outSize.x, 2*outSize.y,
	                             out, in, e_d, r);
	TOCK("halfSampleRobustImageKernel", outSize.x * outSize.y);
}

void integrateKernel(Volume vol, const float* depth, uint2 depthSize,
                     const Matrix4 invTrack, const Matrix4 K, const float mu,
                     const float maxweight)
{
	TICK();
	integrateKernel_pencil(vol.size.x, vol.size.y, vol.size.z, vol.dim,
	                       vol.data, depthSize.x, depthSize.y, depth,
	                       invTrack, K, mu, maxweight);
	TOCK("integrateKernel", vol.size.x * vol.size.y);
}

void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
                   const Volume integration, const Matrix4 view,
                   const float nearPlane, const float farPlane,
                   const float step, const float largestep)
{
	TICK();
	raycast_pencil(inputSize.x, inputSize.y, vertex, normal, integration.size.x,
	               integration.size.y, integration.size.z, integration.data,
	               integration.dim, view, nearPlane, farPlane, step, largestep);
	TOCK("raycastKernel", inputSize.x * inputSize.y);
}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold)
{
	bool res = false;
	TICK();
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	if (norm(x) < icp_threshold)
		res = true;

	TOCK("updatePoseKernel", 1);
	return res;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
                     uint2 imageSize, float track_threshold)
{
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2) ||
	    (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
		pose = oldPose;
		return false;
	} else {
		return true;
	}

}

void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize)
{
	TICK();
	renderNormal_pencil(normalSize.x, normalSize.y, out, normal);
	TOCK("renderNormalKernel", normalSize.x * normalSize.y);
}

void renderDepthKernel(uchar3* out, float * depth, uint2 depthSize,
                       const float nearPlane, const float farPlane)
{
	TICK();
	renderDepth_pencil(depthSize.x, depthSize.y, out, depth, nearPlane, farPlane);
	TOCK("renderDepthKernel", depthSize.x * depthSize.y);
}

void renderTrackKernel(uchar3* out, const TrackData* data, uint2 outSize)
{
	TICK();
	renderTrack_pencil(outSize.x, outSize.y, out, data);
	TOCK("renderTrackKernel", outSize.x * outSize.y);
}

void renderVolumeKernel(uchar3* out, const uint2 depthSize,
                        const Volume volume, const Matrix4 view,
                        const float nearPlane, const float farPlane,
                        const float step, const float largestep,
                        const float3 light, const float3 ambient)
{
	TICK();
	renderVolume_pencil(depthSize.x, depthSize.y, out, volume.size.x,
	                    volume.size.y, volume.size.z, volume.data,
	                    volume.dim, view, nearPlane, farPlane, step,
	                    largestep, light, ambient);
	TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}

void reduceKernel(float * out, TrackData* J,
                  const uint2 Jsize, const uint2 size)
{
	TICK();
	reduce_pencil(out, Jsize.x, Jsize.y, J, size.x, size.y);
	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j)
		values[0] += values[j];
	TOCK("reduceKernel", 512);
}




bool Kfusion::preprocessing(const ushort * inputDepth, const uint2 inputSize)
{
	mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
	bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize,
	                      gaussian, e_delta, radius);
	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold,
                       uint tracking_rate, uint frame)
{
	if (frame % tracking_rate != 0)
		return false;

	for (unsigned int i = 1; i < iterations.size(); ++i) {
		halfSampleRobustImageKernel(ScaledDepth[i], ScaledDepth[i - 1],
		                            make_uint2(computationSize.x / (int) pow(2, i),
		                                       computationSize.y / (int) pow(2, i)),
		                            e_delta * 3, 1);
	}

	uint2 localimagesize = computationSize;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
		depth2vertexKernel(inputVertex[i], ScaledDepth[i], localimagesize, invK);
		vertex2normalKernel(inputNormal[i], inputVertex[i], localimagesize);
		localimagesize = make_uint2(localimagesize.x / 2, localimagesize.y / 2);
	}

	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	for (int level = iterations.size() - 1; level >= 0; --level) {
		uint2 localimagesize = make_uint2(computationSize.x / (int) pow(2, level),
		                                  computationSize.y / (int) pow(2, level));
		for (int i = 0; i < iterations[level]; ++i) {
			trackKernel(trackingResult, inputVertex[level], inputNormal[level],
			            localimagesize, vertex, normal, computationSize, pose,
			            projectReference, dist_threshold, normal_threshold);

			reduceKernel(reductionoutput, trackingResult,
			             computationSize, localimagesize);

			if (updatePoseKernel(pose, reductionoutput, icp_threshold))
				break;

		}
	}
	return checkPoseKernel(pose, oldPose, reductionoutput,
	                       computationSize, track_threshold);
}

bool Kfusion::raycasting(float4 k, float mu, uint frame)
{
	bool doRaycast = false;

	if (frame > 2) {
		raycastPose = pose;
		raycastKernel(vertex, normal, computationSize, volume,
		              raycastPose * getInverseCameraMatrix(k),
		              nearPlane, farPlane, step, 0.75f * mu);
	}

	return doRaycast;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame)
{
	bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput,
	                                   computationSize, track_threshold);

	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
		integrateKernel(volume, floatDepth, computationSize, inverse(pose),
		                getCameraMatrix(k), mu, maxweight);
		doIntegrate = true;
	} else {
		doIntegrate = false;
	}
	return doIntegrate;
}

void Kfusion::dumpVolume(std::string filename)
{
	std::ofstream fDumpFile;

	if (filename == "") {
		return;
	}

	std::cout << "Dumping the volumetric representation on file: "
	          << filename << std::endl;
	fDumpFile.open(filename.c_str(), std::ios::out | std::ios::binary);
	if (fDumpFile == NULL) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}

	for (unsigned int i = 0;
	     i < volume.size.x * volume.size.y * volume.size.z;
	     i++) {
		fDumpFile.write((char *) (volume.data + i), sizeof(short));
	}

	fDumpFile.close();
}

void Kfusion::renderVolume(uchar3 * out, uint2 outputSize, int frame,
                           int raycast_rendering_rate, float4 k,
                           float largestep)
{
	if (frame % raycast_rendering_rate == 0)
		renderVolumeKernel(out, outputSize, volume,
		                   *(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
		                   farPlane * 2.0f, step, largestep, light, ambient);
}

void Kfusion::renderTrack(uchar3 * out, uint2 outputSize)
{
	renderTrackKernel(out, trackingResult, outputSize);
}

void Kfusion::renderDepth(uchar3 * out, uint2 outputSize)
{
	renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
}

void synchroniseDevices()
{
	// Nothing to do in the C++ implementation
}
