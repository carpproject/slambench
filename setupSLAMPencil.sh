#!/bin/bash
set -e

source config

if [ -z "$SLAMBENCH_DIR" ]; then
    echo "Error: Need to set SLAMBENCH_DIR"
    exit 1
fi

if [ -z "$OPENCL_SDK" ]; then
    echo "Error: Need to set OPENCL_SDK"
    exit 1
fi

if [ -z "$PATH_TO_REPO" ]; then
    echo "Error: Need to set PATH_TO_REPO"
    exit 1
fi

if [ -z "$PENCIL_TOOLS_HOME" ]; then
    echo "Error: Need to set PENCIL_TOOLS_HOME"
    exit 1
fi

if [ -z "$PENCIL_UTIL_HOME" ]; then
    echo "Error: Need to set PENCIL_UTIL_HOME"
    exit 1
fi

if [ -z "$PPCG_PATH" ]; then
    echo "Error: Need to set PPCG_PATH"
    exit 1
fi

if [ -z "$PRL_PATH" ]; then
    echo "Error: Need to set PRL_PATH"
    exit 1
fi

echo "Applying patch to SLAMBench."
cd ${SLAMBENCH_DIR}
patch -p1 < ${PATH_TO_REPO}/patches/0001-Enable-PENCIL-OpenCL-benchmark.patch

echo "Running Pencil Optimizer and PPCG..."
cd ${PATH_TO_REPO}
make -C src ppcg
echo "PPCG run completed. host and kernel.cl files generated."

echo "Copying generated pencil sources to SLAMBench, and building pencil executable"
make -C src build
echo "SLAMBench Pencil build successful. You may now run the SLAMBench Pencil benchmark as described in README.txt"
