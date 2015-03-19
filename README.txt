Installation.

1. Download the SLAMBench benchmark suite, as well as the
   'living_room_traj2_loop.raw' test data. You must have a working copy of
   SLAMBench that correctly compiles the native OpenCL version.

2. Create 'config' file
   % cd ${PATH_TO_REPO}
   % cp config.orig config

3. Within this 'config' file, specify all of the paths required:
   $PENCIL_TOOLS_HOME - location of 'pencil-optimizer'
   $PENCIL_UTIL_HOME  - location of 'pencil.h', 'prl.h' and associated headers
   $PPCG_PATH         - location of 'ppcg' executable
   $PRL_PATH          - location of 'libprl.so'
   $ARCH              - for cross compilation, specifies target compiler prefix
                        (e.g. 'arm-linux-gnueabihf-' for ARM)
   $OPENCL_SDK        - path to OpenCL SDK, needed to cover the case when OpenCL
                        is not a system-wide install (e.g. when cross compiling
                        on x86 for a Mali Target):
      % ls ${OPENCL_SDK}
      include/  lib/
      % ls ${OPENCL_SDK}/include/CL
      opencl.h

   $SLAMBENCH_DIR     - configuration variable must point to the directory which
                        contains the unpacked benchmark from step 2 above:
      % ls ${SLAMBENCH_DIR}
      build/  cmake/  CMakeLists.txt  gui_doc/  kfusion/  LICENSE  Makefile  README.md

   $TOON_INSTALL_DIR  - configuration variable must point to the directory where
                        toon is installed:
      % ls ${TOON_INSTALL_DIR}
      include/  lib/
      % ls ${TOON_INSTALL_DIR}/include
      TooN/

   $PATH_TO_REPO      - configuration variable must point to this directory,
                        i.e. where this repository lies
      % ls ${PATH_TO_REPO}
      config  config.orig  README.txt  /patches  /src

4. Once 'config' file is appropriately filled you may run the provided
   'setupSLAMPencil.sh' script. The script will do the following:
      4.1.  apply a patch to the SLAMBench benchmark in folder ${SLAMBENCH_DIR}
      4.2.  run 'ppcg' on ${PATH_TO_REPO}/src/pencil_kernels.c
      4.3.  create a new '${SLAMBENCH_DIR}/kfusion/src/pencil' folder
      4.4.  copy the generated host.c and kernel.cl files to this folder, along
            with ${PATH_TO_REPO}/src/kernels.cpp
      4.5.  run 'make' in ${SLAMBENCH_DIR} to build the Pencil-based executable

   Or, the aforementioned steps may be performed manually, as directed below in
   steps 5 and 6. Note that the script above should only be used the first time.

5. Apply patches/0001-Enable-PENCIL-OpenCL-benchmark.patch patch in the ${SLAMBENCH_DIR}:
   % cd ${SLAMBENCH_DIR}
   % patch -p1 < ${PATH_TO_REPO}/patches/0001-Enable-PENCIL-OpenCL-benchmark.patch

6. Build PENCIL module:
   % cd ${PATH_TO_REPO}
   % make -C src ppcg       # Generate OpenCL from PENCIL
   % make -C src build      # Build SLAMBench with PENCIL

   Rebuild PENCIL module, if needed (i.e. if PENCIL code was changed), in the
   same way:
   % make -C src ppcg       # If PENCIL code was changed
   % make -C src build      # Rebuild the PENCIL OpenCL module

7. Run the pencil executable as follows:
   % cd ${SLAMBENCH_DIR}
   % ./build/kfusion/kfusion-benchmark-pencilCL -s 4.8 -p 0.34,0.5,0.24 -z 4 \
     -c 2 -r 1 -k 481.2,480,320,240 -i ../living_room_traj2_loop.raw \
     -o benchmark.2.PencilCL.log 2> kernels.2.PencilCL.log

   Also set LD_LIBRARY_PATH variable if there is a need to specify the location
   of libOpenCL.so or libprl.so, etc.
