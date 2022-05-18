#! /bin/bash
#
# build.sh
# Copyright (C) 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.
#


MKLLIBS="${PREFIX}/lib/libmkl_intel_lp64.so"
MKLLIBS="${MKLLIBS};${PREFIX}/lib/libmkl_core.so"
MKLLIBS="${MKLLIBS};${PREFIX}/lib/libmkl_sequential.so"

export OMPI_CC=${CC}
export OMPI_CXX=${CXX}

mkdir build
cd build

cmake \
    -Denable_double=ON \
    -Denable_single=OFF \
    -Denable_complex16=OFF \
    -Denable_tests=OFF \
    -Denable_examples=OFF \
    -DXSDK_ENABLE_Fortran=OFF \
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
    -DTPL_BLAS_LIBRARIES="${MKLLIBS}" \
    -DTPL_ENABLE_PARMETISLIB=ON \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PREFIX}/include" \
    -DTPL_PARMETIS_LIBRARIES="${PREFIX}/lib/libparmetis.so" \
    -DTPL_ENABLE_LAPACKLIB=ON \
    -DTPL_LAPACK_LIBRARIES="${MKLLIBS}" \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_CUDALIB=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON \
    -DUSE_XSDK_DEFAULTS=ON \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    ../

#-DCMAKE_C_FLAGS=${CFLAGS_USED} \
#-DCMAKE_CXX_FLAGS=${CXXFLAGS_USED} \
#-DCMAKE_LINKER=mpicxx \
#-DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS_USED}"\

make all -j ${CPU_COUNT}
make install
cd ..
