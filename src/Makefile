include ../config

BENCH_SRC=${SLAMBENCH_DIR}/kfusion/src/pencil/
BENCH_BUILD_DIR=${SLAMBENCH_DIR}/build
TOON_INCLUDE_DIR=${TOON_INSTALL_DIR}/include

CPP=cpp

PENCIL_INCLUDE=${PENCIL_UTIL_HOME}/include/pencil.h
OPTIMIZER=${PENCIL_TOOLS_HOME}/pencil-optimizer
LINKER=${PENCIL_TOOLS_HOME}/linker

PPCG_BASE_OPTIONS=--isl-schedule-fuse=min --no-private-memory --no-shared-memory --target=opencl --opencl-include-file=cl_kernel_vector.cl --opencl-compiler-options='-I.' --no-opencl-print-kernel-types

define PPCG_SIZES
kernel[0]->tile[16,16]; kernel[0]->grid[3000,4000]; kernel[0]->block[16,16];
kernel[1]->tile[8,16]; kernel[1]->grid[3000,4000]; kernel[1]->block[8,16];
kernel[2]->tile[16,16]; kernel[2]->grid[1024,1024]; kernel[2]->block[16,16];
kernel[3]->tile[4,4]; kernel[3]->grid[1024,1024]; kernel[3]->block[4,4];
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

pencil_kernels_opt.c: pencil_kernels.c
	${OPTIMIZER} $< -o $@

export PPCG_SIZES
ppcg: pencil_kernels_opt.c
	${PPCG_PATH}/ppcg ${PPCG_BASE_OPTIONS} --sizes="{$${PPCG_SIZES}}" $<

export PENCIL_UTIL_HOME
export PPCG_PATH

build:
	rm -rf ${BENCH_SRC}
	mkdir -p ${BENCH_SRC}
	cp kernels.cpp ${BENCH_SRC}
	cp pencil_kernels.c ${BENCH_SRC}
	cp pencil_kernels_opt_host.c ${BENCH_SRC}
	cp cl_kernel_vector.cl ${SLAMBENCH_DIR}
	cp pencil_kernels_opt_kernel.cl ${SLAMBENCH_DIR}
	PRL_PATH=${PRL_PATH} PENCIL_UTIL_HOME=${PENCIL_UTIL_HOME} CC=${ARCH}gcc CXX=${ARCH}g++ OCLROOT=${OPENCL_SDK} make -C ${SLAMBENCH_DIR}

clean:
	rm -f pencil_kernels_prep.c
	rm -f pencil_kernels_opt.c
	rm -f pencil_kernels_opt_host.c pencil_kernels_opt_kernel.cl
