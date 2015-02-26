include ../config

BENCH_SRC=${SLAMBENCH_DIR}/kfusion/src/pencil/
BENCH_BUILD_DIR=${SLAMBENCH_DIR}/build
TOON_INCLUDE_DIR=${TOON_INSTALL_DIR}/include

CPP=cpp


PENCIL_INCLUDE=${PENCIL_UTIL_HOME}/include/pencil.h
OPTIMIZER=${PENCIL_TOOLS_HOME}/optimizer
LINKER=${PENCIL_TOOLS_HOME}/linker

PPCG_BASE_OPTIONS=--isl-schedule-fuse=min --no-private-memory --no-shared-memory --target=opencl --opencl-pencil-runtime --opencl-include-file=cl_kernel_vector.cl --opencl-compiler-options='-I.' --no-opencl-print-kernel-types

define PPCG_SIZES
kernel[0]->tile[16,16]; kernel[0]->grid[3000,4000]; kernel[0]->block[16,16];
kernel[1]->tile[8,16]; kernel[1]->grid[3000,4000]; kernel[1]->block[8,16];
kernel[2]->tile[16,16]; kernel[2]->grid[1024,1024]; kernel[2]->block[16,16];
kernel[3]->tile[8,8]; kernel[3]->grid[1024,1024]; kernel[3]->block[8,8];
kernel[4]->tile[4,16]; kernel[4]->grid[3000,4000]; kernel[4]->block[4,16];
kernel[5]->tile[4,16]; kernel[5]->grid[3000,4000]; kernel[5]->block[4,16];
kernel[6]->tile[4,16]; kernel[6]->grid[3000,4000]; kernel[6]->block[4,16];
kernel[7]->tile[8,32]; kernel[7]->grid[3000,4000]; kernel[7]->block[8,32];
kernel[8]->tile[8,32]; kernel[8]->grid[3000,4000]; kernel[8]->block[8,32];
kernel[9]->tile[8,32]; kernel[9]->grid[3000,4000]; kernel[9]->block[8,32];
kernel[10]->tile[4,16]; kernel[10]->grid[3000,4000]; kernel[10]->block[4,16];
kernel[11]->tile[4,16]; kernel[11]->grid[3000,4000]; kernel[11]->block[4,16];
kernel[12]->tile[4,16]; kernel[12]->grid[3000,4000]; kernel[12]->block[4,16];
kernel[13]->tile[16,16]; kernel[13]->grid[3000,4000]; kernel[13]->block[16,16];
kernel[14]->tile[4,16]; kernel[14]->grid[3000,4000]; kernel[14]->block[4,16];
kernel[15]->tile[16,16]; kernel[15]->grid[1000,4000]; kernel[15]->block[16,16];
kernel[16]->tile[16,16]; kernel[16]->grid[1000,4000]; kernel[16]->block[16,16]
endef

pencil_kernels_prep.c: pencil_kernels.c
	${CPP} -P -D__PENCIL__ -D__SKIPCPP__ -include ${PENCIL_INCLUDE} $< -o $@

pencil_kernels_opt.c: pencil_kernels_prep.c
	${OPTIMIZER} $< -o $@

export PPCG_SIZES
ppcg: pencil_kernels_opt.c
	${PPCG_PATH}/ppcg ${PPCG_BASE_OPTIONS} --sizes="{$${PPCG_SIZES}}" $<
	sed -e 's/struct float2/float2/g' -e 's/struct short2/short2/g' -e 's/struct float4/float4/g' -i pencil_kernels_opt_kernel.cl

export PENCIL_UTIL_HOME
export PPCG_PATH
prepare:
	rm -rf ${BENCH_BUILD_DIR}
	mkdir -p ${BENCH_BUILD_DIR}
	cd ${BENCH_BUILD_DIR} && cmake ${SLAMBENCH_DIR} -DTOON_INCLUDE_PATH=${TOON_INCLUDE_DIR}

build:
	rm -rf ${BENCH_SRC}
	mkdir -p ${BENCH_SRC}
	cp kernels.cpp ${BENCH_SRC}
	cp pencil_kernels.c ${BENCH_SRC}
	cp cl_kernel_vector.cl ${BENCH_SRC}
	cp pencil_kernels_opt_host.c pencil_kernels_opt_kernel.cl ${BENCH_SRC}
	PENCIL_UTIL_HOME=${PENCIL_UTIL_HOME} make -C ${BENCH_BUILD_DIR}

clean:
	rm -f pencil_kernels_prep.c
	rm -f pencil_kernels_opt.c
	rm -f pencil_kernels_opt_host.c pencil_kernels_opt_kernel.cl
