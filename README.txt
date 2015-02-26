Installation.

1. Download the SLAMBench benchmark suite.
   The SLAMBENCH_DIR configuration variable must point to the directory
   which contains the unpacked benchmark:

   % ls ${SLAMBENCH_DIR}
   build/  cmake/  CMakeLists.txt  gui_doc/  kfusion/  LICENSE  Makefile  README.md

2. Download and install the Toon library (see SLAMBench Readme.txt):
   The TOON_INSTALL_DIR configuration variable must point to the directory
   where toon is installed (i.e. prefix):

   % ls ${TOON_INSTALL_DIR}
   include/  lib/

   % ls ${TOON_INSTALL_DIR}/include
   TooN/

3. Apply patches/0001-Enable-PENCIL-OpenCL-benchmark.patch patch in the ${SLAMBENCH_DIR}:
   % cd ${SLAMBENCH_DIR}
   % patch -p1 < ${PATH_TO_REPO}/patches/0001-Enable-PENCIL-OpenCL-benchmark.patch

4. Create 'config' file
   % cd ${PATH_TO_REPO}
   % cp config.orig config

5. Within this 'config' file, specify all of the paths required:
   $PENCIL_TOOLS_HOME - location of pencil optimizer (optimizer executable)
   $PENCIL_UTIL_HOME  - location of pencil runtime (installation folder)
   $PPCG_PATH         - location of ppcg executable
   $ARCH              - for cross compilation, specifies target compiler prefix
   $OPENCL_SDK        - path to OpenCL SDK, needed to cover the case when OpenCL is not a system-wide install (e.g. when cross compiling on x86 for a Mali Target).

6. Build PENCIL module:
   % cd ${PATH_TO_REPO}
   % make -C src prepare    # Initialize SLAMBench.
   % make -C src ppcg       # Generate OpenCL from PENCIL
   % make -C src build      # Build SLAMBench with PENCIL

7. Rebuild PENCIL module (if needed):
   % make -C src ppcg       # If PENCIL code was changed
   % make -C src build      # Rebuild the PENCIL OpenCL module
